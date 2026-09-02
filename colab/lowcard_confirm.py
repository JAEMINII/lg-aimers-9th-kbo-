# -*- coding: utf-8 -*-
"""저카디널리티 범주화를 8시드 짝비교로 확인한다. 혼합 기준으로.

4시드 스크리닝에서 나온 것
    단독 908.3 -> 905.6  (-4.2 ± 2.4, 양수 1/4)
    혼합 934.6 -> 936.6  (+2.0, 짝비교 표준오차 미측정)

방향이 지금까지와 반대다. 다른 실험은 전부 단독이 오르고 혼합은 제자리였는데
이건 단독이 내리고 혼합이 오른다.

짐작되는 기제
    이닝(13종) 카운트(4/3) 월(8) 요일(7) 아웃(3) 을 범주로 넣으면 모델이
    다른 방식으로 틀리게 되고, CatBoost·flatMLP 와 덜 겹친다.
    우리 앙상블의 한계가 상관 0.9 였으니 정확히 그 지점이다.
    그렇다면 단독이 내려가도 혼합이 오르는 게 자연스럽다.

여기서는 혼합 기준 짝차이를 직접 잰다. 상관도 같이 찍어 기제를 확인한다.
학습량이 같은 변형이라 관문을 믿을 수 있는 축이다.
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777, 2, 3, 5, 11, 23)
DECAY = 3.5
VS = 2024
LOWCARD = ["game_month", "game_dayofweek", "inning",
           "balls_before", "strikes_before", "outs_before"]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    F44 = list(d["F44"])
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    M = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))
    ci_base = list(d["cat_idx"])
    ci_low = sorted(set(ci_base) | {F44.index(c) for c in LOWCARD})

    def blend(r):
        return sc(0.10 * CB + 0.30 * M + 0.60 * r)

    def load(ci):
        Xn, Xc, cards = G.prep(d["X44"], G.m_tr, ci)
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        return cards

    G.log(f"  기준 범주 {len(ci_base)}개  ->  범주화 {len(ci_low)}개")
    P0, P1 = {}, {}
    G.log("")
    G.log("  시드    기준단독  범주단독   기준혼합  범주혼합   혼합차이")
    for sd in SEEDS:
        t0 = time.time()
        load(ci_base)
        P0[sd] = one(sd, decay=DECAY)
        load(ci_low)
        P1[sd] = one(sd, decay=DECAY)
        G.log(f"  {sd:4d}  {sc(P0[sd]):9.1f} {sc(P1[sd]):9.1f}  "
              f"{blend(P0[sd]):9.1f} {blend(P1[sd]):9.1f}  "
              f"{blend(P1[sd]) - blend(P0[sd]):+9.1f}   {time.time()-t0:.0f}s")

    for tag, f in (("단독", sc), ("혼합", blend)):
        dif = [f(P1[s]) - f(P0[s]) for s in SEEDS]
        m, sd_ = float(np.mean(dif)), float(np.std(dif, ddof=1))
        se = sd_ / np.sqrt(len(dif))
        G.log(f"\n  {tag} 짝차이  평균 {m:+.1f}  표준편차 {sd_:.1f}  표준오차 {se:.1f}  "
              f"양수 {sum(1 for x in dif if x > 0)}/{len(dif)}  t={m/se:.2f}  "
              f"{'확정' if abs(m) > 2 * se else '미확정'}")

    r0 = np.mean([P0[s] for s in SEEDS], 0)
    r1 = np.mean([P1[s] for s in SEEDS], 0)
    np.save(os.path.join(OUT, "lowcard_tabm.npy"), r1)
    G.log(f"\n  8시드평균  기준 단독 {sc(r0):7.1f} 혼합 {blend(r0):7.1f}   "
          f"범주화 단독 {sc(r1):7.1f} 혼합 {blend(r1):7.1f}")
    G.log("\n  상관 (기제 확인 — 낮아졌으면 다양성이 늘어난 것)")
    for tag, r in (("기준", r0), ("범주화", r1)):
        G.log(f"    {tag:6s} CatBoost {np.corrcoef(r, CB)[0,1]:.4f}   "
              f"flatMLP {np.corrcoef(r, M)[0,1]:.4f}")
    G.log(f"    기준↔범주화 {np.corrcoef(r0, r1)[0,1]:.4f}")

    G.log("\n  비중을 다시 훑으면 (범주화 판본)")
    for w in (0.5, 0.55, 0.6, 0.65, 0.7):
        rest = 1 - w
        G.log(f"    TabM {w:.2f}   {sc(rest*0.25*CB + rest*0.75*M + w*r1):7.1f}")
