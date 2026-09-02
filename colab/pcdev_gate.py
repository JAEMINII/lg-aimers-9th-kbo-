# -*- coding: utf-8 -*-
"""편차 피처를 관문에 올린다. as-of 집계, plat 고친 기준선 위, 두 폴드.

무엇을 재나
    f(선수, 상황) - f(선수) 형태의 트랙맨 편차. 선수당 상수형은 ID 임베딩과
    겹쳐 0/21 이었는데 편차형은 6/21 통과했다. 세 폴드 전부에서 임계를 넘겼다.
        VS=2022  pc_ivb +0.00958  pc_rel_speed +0.00842  pc_br -0.00814
        VS=2024  pc_br  -0.00716  pc_ivb       +0.00715  pc_ext +0.00711
        VS=2025  pc_br  -0.00767  pc_ivb       +0.00703

팔 — 상호상관이 팔을 정했다
    pc_* 다섯은 서로 |r| 0.64~0.86 이다. 한 축이다(구종 선택).
    bh_spin_rate 만 독립이다 (0.011).
        base      현행
        pcbr      + pc_br                      단독 최강
        pair      + pc_br, bh_spin_rate        독립축 둘
        bundle    + pc_* 다섯 + bh_spin_rate   묶으면 나쁜지 확인

판정선 — 두 폴드에서 +10 이상 & 3/3
    피처 추가는 관문이 여섯 번 틀린 부류다. 기준을 높인다.
    참고로 실제로 통했던 plat_dev 는 +19.5(t=5.5) / +24.2(t=8.2) 였다.
    그 근처가 아니면 안 싣는다.

집계는 폴드별로 as-of 다 (pcdev_{VS}.csv.gz, colab/pcdev_build.py 가 만든다).
2025 배치용은 트랙맨 전 구간(2019~2024)이며 그건 정의상 과거다.
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
PC5 = ["pc_br", "pc_induced_vert_break", "pc_extension", "pc_rel_speed", "pc_fb"]
ARMS = (("base", []), ("pcbr", ["pc_br"]),
        ("pair", ["pc_br", "bh_spin_rate"]),
        ("bundle", PC5 + ["bh_spin_rate"]))

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
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id"])
    rid = raw["row_id"].to_numpy()

    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(rid).to_numpy()

    ALL = {}
    for VS in FOLDS:
        # ---- 그 폴드의 as-of 편차표. row_id 로 명시 정렬한다
        pcd = pd.read_csv(os.path.join(DL, f"pcdev_{VS}.csv.gz")
                          ).set_index("row_id").reindex(rid)
        if pcd.isna().any().any():
            raise ValueError(f"pcdev_{VS} row_id 정렬 실패")
        EX = {c: pcd[c].to_numpy(np.float32) for c in pcd.columns}
        G.log(f"\n  VS={VS}  편차열 {list(EX)}  "
              f"비영 {float((EX['pc_br']!=0).mean())*100:.1f}%")

        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        G.gate, G.yv = gate, yv
        m_tr = season < VS

        hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
        Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
        cols = list(Xs.columns)
        Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
        ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        dv = F.build(DATA, VS=VS)
        Xfr[:, cols.index("plat_dev")] = \
            dv["X44"][:, F44.index("plat_dev")].astype(np.float32)

        old = season <= OLD_F_MAX
        c4 = np.where(old & isf, 0.0,
                      np.where(old & ~isf, 1.0,
                               np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"  학습 {len(tr_idx):,}  검증 {len(gate):,}")

        for nm, adds in ARMS:
            Xin = np.concatenate([Xfr, c4] + [EX[a][:, None] for a in adds], 1)
            Xn, Xc, cards = G.prep(Xin, m_tr, ci + [Xfr.shape[1]])
            G.Xn, G.cards = Xn, cards
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"pcd_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:7s} (+{len(adds)}열) 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 92)
    print("  트랙맨 편차 피처 — as-of 집계, plat 고친 기준선, 두 폴드")
    print("=" * 92)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm, adds in ARMS:
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
    print("\n  판정선 두 폴드 +10 & 3/3. plat_dev 는 +19.5/+24.2 였다.")
    print("  bundle 이 pcbr 보다 나쁘면 '묶지 말라' 는 규율이 또 확인된 것이다.")
