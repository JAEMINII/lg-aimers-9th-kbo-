# -*- coding: utf-8 -*-
"""버려진 표본 크기 열 셋을 되살린다. 44열에 없는 가장 큰 정보다.

발견
    features44 의 DUP 목록이 이 셋을 '중복' 으로 제거했다.
        asof_pitcher_n            투수 통산 투구수
        asof_batter_n             타자 통산 타석수
        asof_pitcher_pitchmix_n   구종 표본수

    그런데 44열에는 asof_pitcher_success_rate(비율)만 있고 **그게 몇 구에서
    나온 값인지가 없다.** 50구짜리 0.55 와 5000구짜리 0.55 를 구분 못 한다.

증상이 이미 찍혀 있었다 (오차 해부)
        투수경험      비중     실제      예측      편향
        ~100        2.7%   0.4540   0.4658   +0.0118   <- 10배
        100~300     3.6%   0.4741   0.4705   -0.0036
        3000~      62.0%   0.4894   0.4895   +0.0001
    저표본 투수에서 편향이 10배다. 모델이 못 믿을 비율을 그대로 믿고 있고,
    얼마나 믿을지 판단할 재료가 없다.

누출이 아니다
    asof_* 는 주최측이 각 행 시점까지로 계산해준 값이고 test.csv 에도 있다.
    2025 시험 행에 정상적으로 들어온다. 그냥 버려진 정보다.

    참고: pn_cur(인시즌 투구수)는 44열에 있다. 하지만 그건 **이번 시즌만**이라
    통산 표본을 대신하지 못한다. 시즌 초 베테랑과 신인이 똑같이 pn_cur 가 작다.

팔 (plat 고친 기준선 위, 두 폴드)
    base   현행 44열
    pn     + asof_pitcher_n
    n3     + asof_pitcher_n, asof_batter_n, asof_pitcher_pitchmix_n
    logn3  + 위 셋의 log1p        (분포가 심하게 치우쳐 있어 로그가 나을 수 있다)

    '통과분은 하나씩 넣어라' 규율대로 pn 단독을 먼저 본다.

기대
    피처 추가는 관문이 여섯 번 틀린 부류다. 다만 이건 '새 정보' 가 아니라
    **버려진 정보의 복구**라, 오늘 통한 plat_dev(틀린 계산의 수정)와 성격이
    가깝다. 그리고 기전이 오차 해부의 실측 편향으로 뒷받침된다.
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
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
FOLDS = (2022, 2024)
NCOLS = ["away_win_expectancy", "run_total_before", "score_diff_home",
         "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b"]
ARMS = (("base", [], False), ("runners", NCOLS[3:], False),
        ("lin3", NCOLS[:3], False), ("all7", NCOLS, False))

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
                      usecols=["row_id"] + NCOLS)
    NV = {c: raw[c].to_numpy(np.float64) for c in NCOLS}
    for c in NCOLS:
        v = NV[c]
        print(f"  {c:26s} 결측 {int(np.isnan(v).sum()):>8,}  "
              f"중앙값 {np.nanmedian(v):>8.0f}  최대 {np.nanmax(v):>8.0f}")
        print(f"  {'':26s} 이 열이 44열에 없다: "
              f"{c not in F44}")

    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(raw["row_id"].to_numpy()).to_numpy()

    ALL = {}
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
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}")

        for nm, adds, use_log in ARMS:
            ex = []
            for c in adds:
                v = NV[c].copy()
                if use_log:
                    v = np.log1p(np.where(np.isnan(v), 0.0, v))
                ex.append(v.astype(np.float32)[:, None])
            Xin = np.concatenate([Xfr, c4] + ex, 1)
            Xn, Xc, cards = G.prep(Xin, m_tr, ci + [Xfr.shape[1]])
            G.Xn, G.cards = Xn, cards
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"dp_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:6s} (+{len(adds)}열{' log' if use_log else ''}) "
                  f"끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 92)
    print("  DUP 로 버려진 나머지 7열 복구 — plat 고친 기준선 위, 두 폴드")
    print("=" * 92)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm, adds, _ in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:6s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)

    print("\n  저표본 투수(asof_pitcher_n < 500) 구간에서 특히 봐야 한다 —")
    print("  거기가 편향 +0.0118 로 문제가 잡힌 자리다.")
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        yv = y[gate]
        low = NV["asof_pitcher_n"][gate] < 500
        if low.sum() < 500:
            continue
        b = np.mean(ALL[(VS, "base")], 0)
        print(f"\n  VS={VS}  저표본 {int(low.sum()):,}행 "
              f"({low.mean()*100:.1f}%)")
        for nm, *_ in ARMS:
            p = np.mean(ALL[(VS, nm)], 0)
            print(f"    {nm:6s} {sc(p, yv, low):8.1f}"
                  + ("" if nm == "base"
                     else f"   {sc(p, yv, low) - sc(b, yv, low):+7.1f}"))
