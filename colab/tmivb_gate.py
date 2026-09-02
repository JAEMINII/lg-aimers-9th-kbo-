# -*- coding: utf-8 -*-
"""트랙맨 마지막 후보 tmc_ivb 를 8시드 짝비교로 재고 끝낸다.

tmc_ivb = 그 투수가 '이 카운트에서' 상하무브를 평소보다 얼마나 바꾸는가
    (투수, 카운트12) 평균 - (투수) 전체 평균.  시즌 < Y 누적으로만 만든다.
    행마다 값이 변한다 — 같은 투수라도 카운트가 다르면 다른 값이다.

편상관 스크린 결과 (2024, 통제 27개)
    tmc_ivb  단순 -0.0036  편상관 +0.0092   기준선 0.010 에 살짝 미달
    tmc_spd                       +0.0069
    tmc_fb                        +0.0040
비교: 오늘 확정한 등판 강도가 +0.0111, 기존 asof_batter_middle_rate 가 -0.008.

왜 그래도 재보나
    애매한 구간이라 편상관만으로 못 자른다. 열 하나뿐이라 결측 플래그도 하나뿐이고,
    오늘 반복해서 본 '열을 늘릴수록 나빠진다'(2열 +0.8 / 13열 +0.3 / 21열 -7.1)
    함정을 피한다. 40분이면 트랙맨 건이 어느 쪽으로든 끝난다.

기준선은 오늘 확정한 최선 구성이다 — 시즌가중 3.5 + 등판 강도 + 저카디널리티 범주화.
그 위에 더하는지가 문제이기 때문이다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777, 2, 3, 5, 11, 23)
DECAY = 3.5
LOWCARD = ["game_month", "game_dayofweek", "inning",
           "balls_before", "strikes_before", "outs_before"]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402
from load_gate import load_cols                                 # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)
    F44 = list(d["F44"])
    base = np.concatenate([d["X44"], load_cols(F44, d["X44"])], 1)   # 등판 강도 포함
    names = F44 + ["load_rate", "load_lin"]

    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["pitcher_id", "season", "balls_before",
                               "strikes_before"])
    raw["cnt12"] = raw.balls_before.astype(int) * 3 + raw.strikes_before.astype(int)
    prof = pd.read_csv(os.path.join(SC, "tm_cnt_profile.csv"))
    m = raw.merge(prof[["pitcher_id", "season", "cnt12", "tmc_ivb"]],
                  on=["pitcher_id", "season", "cnt12"], how="left")
    add = m[["tmc_ivb"]].to_numpy(dtype=np.float32)
    X2 = np.concatenate([base, add], 1)
    G.log(f"  추가 1열  전체 커버리지 {np.isfinite(add).mean()*100:.1f}%   "
          f"관문 {np.isfinite(add[gate]).mean()*100:.1f}%")

    ci = sorted(set(d["cat_idx"]) | {names.index(c) for c in LOWCARD})
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    def blend(r):
        return sc(0.10 * CB + 0.30 * MLPF + 0.60 * r)

    def load(X):
        Xn, Xc, cards = G.prep(X, G.m_tr, ci)
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    P0, P1 = {}, {}
    G.log("")
    G.log("  시드   기준혼합   +tmc_ivb   차이")
    for sd in SEEDS:
        t0 = time.time()
        load(base)
        P0[sd] = one(sd, decay=DECAY)
        load(X2)
        P1[sd] = one(sd, decay=DECAY)
        G.log(f"  {sd:4d}  {blend(P0[sd]):9.1f}  {blend(P1[sd]):9.1f}  "
              f"{blend(P1[sd]) - blend(P0[sd]):+7.1f}   {time.time()-t0:.0f}s")

    dif = [blend(P1[s]) - blend(P0[s]) for s in SEEDS]
    mm, sd_ = float(np.mean(dif)), float(np.std(dif, ddof=1))
    se = sd_ / np.sqrt(len(dif))
    G.log("")
    G.log(f"  혼합 짝차이  평균 {mm:+.1f}  표준편차 {sd_:.1f}  표준오차 {se:.1f}  "
          f"양수 {sum(1 for x in dif if x > 0)}/{len(dif)}  t={mm/se:.2f}  "
          f"{'확정' if abs(mm) > 2 * se else '미확정'}")
    r0 = np.mean([P0[s] for s in SEEDS], 0)
    r1 = np.mean([P1[s] for s in SEEDS], 0)
    G.log(f"  8시드평균  기준 {blend(r0):7.1f}   +tmc_ivb {blend(r1):7.1f}")
    G.log("\n  미확정이면 트랙맨은 여기서 접는다. 여섯 번째 시도다.")
