# -*- coding: utf-8 -*-
"""TabM 비중을 올렸을 때 관문이 어떻게 움직이는지 시드 6개로 잰다.

문제
    시즌가중이 TabM 단독을 +36 올리는데 혼합에서는 +3.9 만 남는다.
    이득의 90%를 CatBoost·flatMLP 가 흡수한다. 비중이 0.60 에 묶여 있어서다.

    그런데 비중을 관문으로 정하려던 시도가 세 번 다 틀렸다. 가장 최근이
    submit_15 로, 절편이 계열 사이에서 보간된다고 가정해 1076 을 추정했는데
    실제는 1046 이었다.

이번엔 조건이 다르다
    지금까지 관문이 비중을 틀리게 정한 근본 원인은 관문이 CatBoost 를 TabM 보다
    높게 쳤기 때문이다(903.2 vs 867.5). 리더보드는 반대였다(1021 vs 1041).
    그래서 관문은 늘 "CatBoost 를 더 넣어라" 고 말했다.

    시즌가중을 걸면 TabM 단독이 911.2 로 CatBoost 를 앞선다. 두 지표가 처음으로
    같은 방향을 가리킨다. 그러면 관문의 비중 판단도 처음으로 믿을 근거가 생긴다.

    다만 '근거가 생겼다' 이지 '맞다' 가 아니다. 곡선의 모양만 보고, 절대값을
    리더보드로 환산하지 않는다.

무엇을 찍나
    시드 6개 평균 TabM 으로 비중을 훑는다.
    CatBoost:flatMLP 비율은 현 제출본과 같은 1:3 으로 유지한다.
    각 비중에서 최적 시프트로 채점한다.
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
SEEDS = (42, 1, 777, 2, 3, 5)
DECAY = 3.5

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)
    Xn, Xc, cards = G.prep(d["X44"], G.m_tr, d["cat_idx"])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    M = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    # 시즌가중 3.5 를 8시드로 확정할 때 만든 평균이 이미 있다(confirm_sw2.py).
    # 같은 설정·같은 라우팅이므로 그대로 쓴다. 재학습 37분을 아낀다.
    cached = os.path.join(OUT, "sw2conf_sw3.5.npy")
    if os.path.exists(cached):
        T = np.load(cached)
        G.log("  기존 8시드 평균(sw3.5) 재사용")
    else:
        ps = []
        for sd in SEEDS:
            t0 = time.time()
            ps.append(one(sd, decay=DECAY))
            G.log(f"  seed {sd} {time.time() - t0:.0f}s")
        T = np.mean(ps, 0)
        np.save(os.path.join(OUT, "wsweep_tabm.npy"), T)

    G.log("")
    G.log(f"  단독   TabM(sw3.5) {sc(T):7.1f}   CatBoost {sc(CB):7.1f}   "
          f"flatMLP {sc(M):7.1f}")
    G.log("")
    G.log("  TabM비중   CB    MLP     점수     현재대비")
    ref = None
    for w in np.arange(0.4, 1.001, 0.05):
        rest = 1.0 - w
        cb, ml = rest * 0.25, rest * 0.75      # 현 제출본과 같은 1:3
        s = sc(cb * CB + ml * M + w * T)
        if abs(w - 0.6) < 1e-9:
            ref = s
        G.log(f"    {w:.2f}     {cb:.2f}  {ml:.2f}  {s:8.1f}"
              + ("" if ref is None else f"   {s - ref:+6.1f}"))

    G.log("")
    G.log("  CatBoost 를 빼고 flatMLP 만 남기면")
    for w in (0.6, 0.7, 0.8, 0.9, 1.0):
        s = sc((1 - w) * M + w * T)
        G.log(f"    TabM {w:.2f} / MLP {1-w:.2f}   {s:8.1f}")
    G.log("")
    G.log("  곡선의 모양만 본다. 절대값을 리더보드로 환산하지 않는다.")
    G.log("  비중을 관문으로 정하려던 시도가 세 번 다 틀렸다.")
