# -*- coding: utf-8 -*-
"""1군 경로에 체제 열(c4)이 있고 없고를 직접 잰다. 여기가 아직 불일치다.

배경
    submit_34 로 1069 -> 1072 (+3). 바꾼 것은 퓨처스 경로의 체제 열을
    이진 abs_regime -> 4단계 c4 로 맞춘 것과 Stage2 e4 였다.
    관문 +2.0 이 리더보드 +3 이 됐다 — 전이율 1.5.
    구성 불일치는 튜닝이 아니라 **결함**이었다.

아직 남은 불일치 — 이쪽이 7.5배 크다
    퓨처스 경로 11.8%   c4 있음                 방금 맞췄다
    1군   경로 88.2%   체제 열이 **아예 없다**    지인 코드 산출물이라 44열이다
    그런데 우리 관문은 1군 브랜치에도 늘 c4 를 붙여 쟀다.
    **88.2% 가 검증된 적 없는 구성으로 돌고 있다.**

배치의 실제 구조 (submit_34)
    1군  행 = 0.6 x all_tabm(44열)    + 0.4 x regular_tabm(44열)
    퓨처스행 = 0.6 x f2b_all(45열,c4) + 0.4 x f2b_futures(45열,c4)
    즉 **all 모델을 두 벌** 쓴다. 하나는 c4 없이, 하나는 c4 있게.

설정
    deploy   1군 = c4 없는 all/regular      <- 지금 배치. 한 번도 안 재봤다
    allc4    1군 = c4 있는 all/regular      <- 관문이 늘 쓰던 것
    퓨처스 경로는 두 설정 모두 c4 있는 all/futures 로 고정한다.
    그래야 **바뀌는 것이 1군 경로 하나**가 된다.

불변량 검사
    퓨처스 행 예측은 두 설정에서 **완전히 동일**해야 한다. 안 그러면 라우팅이
    틀린 것이다 (fut_stage2 때 그 검사를 안 넣어 결론을 두 번 뒤집었다).
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
SEEDS = (42, 1, 777)
FOLDS = (2022, 2024)
OLD_F_MAX, OLD_W, LR1 = 2022, 0.1, 3e-3
EP2 = {"all": 1, "regular": 1, "futures": 4}

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def fit(idx, ep2, w, season, VS, gate, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, LR1, w=w, seed=seed, tag=f"VS{VS} s{seed}")
    s2 = idx[season[idx] == VS - 1]
    pr = G.stage2_params(m)
    for e in range(ep2):
        G.train(m, s2, 1, 2e-4, params=pr, seed=seed + e,
                tag=f"VS{VS} s{seed} S2e{e+1}")
    p = G.predict(m, gate)
    del m
    torch.cuda.empty_cache()
    return p


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

    RES = {}
    for VS in FOLDS:
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
        fut_idx, reg_idx = tr_idx[t_isf], tr_idx[~t_isf]
        w_all = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        w_f = np.where(old[fut_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  all {len(tr_idx):,}  regular {len(reg_idx):,}  "
              f"futures {len(fut_idx):,}")

        P = {}
        for tag, Xin, cidx in (("c4", np.concatenate([Xfr, c4], 1),
                                ci + [Xfr.shape[1]]),
                               ("noc4", Xfr, ci)):
            Xn, Xc, cards = G.prep(Xin, m_tr, cidx)
            G.Xn, G.cards, G.m_tr = Xn, cards, m_tr
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            G.log(f"    [{tag}] 수치 {Xn.shape[1]}열  범주 {len(cards)}개")
            t0 = time.time()
            for br, idxb, wb in (("all", tr_idx, w_all),
                                 ("regular", reg_idx, None),
                                 ("futures", fut_idx, w_f)):
                if tag == "noc4" and br == "futures":
                    continue          # 퓨처스는 c4 판만 쓴다 (배치가 그렇다)
                P[(tag, br)] = [fit(idxb, EP2[br], wb, season, VS, gate, sd)
                                for sd in SEEDS]
            G.log(f"    [{tag}] 끝 {time.time()-t0:.0f}s")

        for i, sd in enumerate(SEEDS):
            for k, v in P.items():
                np.save(os.path.join(OUT, f"c4g_{VS}_{k[0]}_{k[1]}_s{sd}.npy"),
                        v[i])
        RES[VS] = P

    def sc(p, yv, m):
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 92)
    print("  1군 경로의 체제 열 — 배치(c4 없음) vs 관문(c4 있음)")
    print("=" * 92)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        allm = np.ones(len(yv), bool)
        P = RES[VS]
        blends = {}
        for tag in ("noc4", "c4"):
            blends[tag] = [
                np.where(isf_g,
                         0.6 * P[("c4", "all")][i] + 0.4 * P[("c4", "futures")][i],
                         0.6 * P[(tag, "all")][i] + 0.4 * P[(tag, "regular")][i])
                for i in range(len(SEEDS))]
        inv = max(float(np.abs(a[isf_g] - b[isf_g]).max())
                  for a, b in zip(blends["noc4"], blends["c4"]))
        print(f"\n  VS={VS}   퓨처스 행 불변량 최대차 {inv:.2e} "
              f"{'(정상)' if inv == 0 else '**라우팅 이상**'}")
        print(f"    {'설정':8s} {'전체':>9s} {'1군':>9s} {'퓨처스':>9s}   "
              f"{'1군 짝차이 (noc4 기준)':>30s}")
        for tag in ("noc4", "c4"):
            ps = blends[tag]
            p = np.mean(ps, 0)
            line = (f"    {tag:8s} {sc(p, yv, allm):9.1f} {sc(p, yv, ~isf_g):9.1f} "
                    f"{sc(p, yv, isf_g):9.1f}")
            if tag == "c4":
                for m, nm in ((~isf_g, "1군"), (allm, "전체")):
                    dd = [sc(a, yv, m) - sc(b, yv, m)
                          for a, b in zip(ps, blends["noc4"])]
                    mu = float(np.mean(dd))
                    se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                    line += (f"   {nm} {mu:+6.1f}+-{se:4.1f} "
                             f"t={mu/max(se,1e-9):5.2f} "
                             f"{sum(1 for v in dd if v>0)}/3")
            print(line)
    print("\n  c4 가 양수면 지금 배치가 그만큼 흘리고 있다는 뜻이다.")
    print("  퓨처스 경로 11.8% 를 맞춰 리더보드 +3 이었다. 1군은 88.2% 다.")
