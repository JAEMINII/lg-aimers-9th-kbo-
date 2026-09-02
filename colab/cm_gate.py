# -*- coding: utf-8 -*-
"""cm 계열 3열의 창 의존(누출)을 as-of 로 고쳐 관문에 올린다.

발견 경로
    "올바른 as-of 피처는 학습 창을 바꿔도 그 행의 값이 안 변한다" 는 이진 검사로
    배치 전처리 44열을 전부 훑었더니 4열이 걸렸다.
        plat_dev   4.69e-02   이미 고침 (LB +11)
        cm_rel     1.62e-01
        p_adj_cm   3.27e-02
        lg_cm_eff  1.52e-02
    plat_dev 가 걸린 것이 검사가 작동한다는 양성 대조군이다.

크기는 plat_dev 보다 작다
    plat_dev 는 투수당(792칸, 칸당 1,860행)이라 그 투수 자신의 미래가 들어갔다.
    cm 은 48칸, 칸당 20,605행이라 리그 수준이다.
        열          누출판     as-of     누출분    두 판 상관
        lg_cm_eff  +0.03118  +0.02838  -0.00280    0.8488
        cm_rel     -0.00572  -0.00117  +0.00455    0.7204
        p_adj_cm   +0.10264  +0.10142  -0.00122    0.9911
        plat_dev   +0.02368  +0.01478  -0.00890    0.7447
    p_adj_cm 은 두 판 상관 0.9911 로 사실상 같은 열이다.

그래도 재는 이유
    부류가 **결함 수정**이다. 전이율 1.1(스케줄러) / 0.5(plat_dev) 로 이 프로젝트에서
    실제로 통한 유일한 부류다. 피처 추가(0.08~0.16)와 기대치가 다르다.

팔
    base    현행 (누출판)
    cmfix   3열 전부 as-of      p_adj_cm 이 cm_rel 에서 파생되므로 같이 고치는 게 일관된다
    lgonly  lg_cm_eff 만 as-of  상대 누출이 제일 큰 열 (9%)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
DL = "/workspace/aimers/colab/_dl"
SEEDS = (42, 1, 777)
LR1 = 3e-3
OLD_F_MAX, OLD_W = 2022, 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
FOLDS = (2022, 2024)
CM3 = ["lg_cm_eff", "cm_rel", "p_adj_cm"]
ARMS = (("base", []), ("cmfix", CM3), ("lgonly", ["lg_cm_eff"]))

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def one(seed, tr_idx, t_isf, w, season, VS, gate, isf_g):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"VS{VS} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        pr = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=pr, seed=seed + e,
                    tag=f"VS{VS} {br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(isf_g, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


if __name__ == "__main__":
    d0 = F.build(DATA, VS=2024)
    season = d0["season"].astype(np.float64)
    isf, y = d0["is_f"], d0["y"].astype(np.float64)
    F44 = list(d0["F44"])
    rid = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id"])["row_id"].to_numpy()
    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(rid).to_numpy()

    # as-of 판은 누적이라 창에 안 의존한다 -> 폴드마다 같은 파일을 쓴다
    ca = pd.read_csv(os.path.join(DL, "cm_asof.csv.gz")
                     ).set_index("row_id").reindex(rid)
    if ca.isna().any().any():
        raise ValueError("cm_asof row_id 정렬 실패")
    AS = {c: ca[f"asof_{c}"].to_numpy(np.float32) for c in CM3}

    ALL = {}
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        G.gate, G.yv = gate, yv
        m_tr = season < VS
        hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
        Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
        cols = list(Xs.columns)
        Xbase = Xs.to_numpy(dtype=np.float32)[pos_map]
        ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        dv = F.build(DATA, VS=VS)
        Xbase[:, cols.index("plat_dev")] = \
            dv["X44"][:, F44.index("plat_dev")].astype(np.float32)
        for c in CM3:
            o = Xbase[:, cols.index(c)]
            G.log(f"  {c:12s} 누출판 vs as-of  상관 "
                  f"{np.corrcoef(o[m_tr], AS[c][m_tr])[0,1]:.4f}  "
                  f"최대차 {np.abs(o - AS[c]).max():.4e}")

        old = season <= OLD_F_MAX
        c4 = np.where(old & isf, 0.0,
                      np.where(old & ~isf, 1.0,
                               np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}")

        for nm, repl in ARMS:
            Xfr = Xbase.copy()
            for c in repl:
                Xfr[:, cols.index(c)] = AS[c]
            Xin = np.concatenate([Xfr, c4], 1)
            Xn, Xc, cards = G.prep(Xin, m_tr, ci + [Xfr.shape[1]])
            G.Xn, G.cards = Xn, cards
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"cmg_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:7s} ({len(repl)}열 교체) 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 92)
    print("  cm 계열 창 의존 수정 — plat 고친 기준선 위, 두 폴드")
    print("=" * 92)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm, repl in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:7s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)
    print("\n  plat_dev 는 +19.5(t=5.5) / +24.2(t=8.2) 였다. 누출분이 1/3 이라")
    print("  그 비례면 +6~8 이 기대치다. 결함 수정은 전이율이 0.5~1.1 이다.")
