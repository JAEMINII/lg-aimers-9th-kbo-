# -*- coding: utf-8 -*-
"""시즌가중 최적값을 8시드 짝비교로 확정하고, 더 센 값까지 훑는다.

왜 8개인가
    tm_sd2 가 3시드에서 +11.6 이었는데 6시드에서 +0.8 로 사라졌다. 짝지은 차이의
    표준편차가 22.3 이다. 즉 필요한 판정선은
        n=3  ±25.7      n=6  ±18.2      n=8  ±15.8      n=10  ±14.1
    '3시드 평균이면 표준오차 3점' 은 틀린 계산이었다. 그건 평균 점수의 산포지
    짝지은 차이의 산포가 아니다.

왜 3.5 인가
    decay 를 훑으니 곡선이 단조로 올라가다 3~4 에서 포화했다.
        1.0(없음) 867.5 / 1.5 882.8 / 2.0 895.5 / 3.0 906.0 / 4.0 907.6
    개별 비교가 아니라 곡선 전체가 근거라 노이즈로 보기 어렵다. 다만 확정하려면
    짝비교가 필요하다. 최적점 부근인 3.5 로 8시드를 돌린다.

    CatBoost 최적값은 2.0 이었고 그 위는 손해였다. TabM 은 3~4 다. CatBoost 에서
    나온 값을 그대로 쓰던 게 실수였다.

짝지어 비교한다. 같은 시드에서 base 와 sw3.5 를 각각 학습해 차이를 본다.
시드 간 편차가 24점이라 짝을 안 지으면 효과가 묻힌다.
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

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    d = F.build(DATA, VS=int(os.environ.get("VS", 2024)))
    Xn, Xc, cards = G.prep(d["X44"], G.m_tr, d["cat_idx"])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    P0, P2, dif = {}, {}, []
    G.log("")
    G.log("  시드    base     sw3.5     차이")
    for sd in SEEDS:
        t0 = time.time()
        P0[sd] = one(sd)
        P2[sd] = one(sd, decay=3.5)
        a, b = sc(P0[sd]), sc(P2[sd])
        dif.append(b - a)
        G.log(f"  {sd:4d}  {a:7.1f}  {b:7.1f}  {b - a:+7.1f}   {time.time() - t0:.0f}s")

    m, s = float(np.mean(dif)), float(np.std(dif, ddof=1))
    se = s / np.sqrt(len(dif))
    G.log("")
    G.log(f"  짝지은 차이  평균 {m:+.1f}  표준편차 {s:.1f}  표준오차 {se:.1f}  "
          f"양수 {sum(1 for x in dif if x > 0)}/{len(dif)}   t={m / se:.2f}")
    G.log(f"  판정: {'확정' if abs(m) > 2 * se else '미확정'}")
    for tag, P in (("base", P0), ("sw3.5", P2)):
        r = np.mean([P[k] for k in SEEDS], 0)
        np.save(os.path.join(OUT, f"sw2conf_{tag}.npy"), r)
        blend = 0.10 * CB + 0.30 * MLPF + 0.60 * r
        G.log(f"  {tag:6s} 8시드평균 단독 {sc(r):7.1f}   제출비중 {sc(blend):7.1f}")

    # 포화 이후를 확인한다. 4.0 에서 이미 평평했으니 더 세게 걸면 꺾여야 정상이다.
    # 극단(최근 시즌만)은 CatBoost 에서 손해였다(849.0 -> 823.3). TabM 도 그런지 본다.
    G.log("")
    G.log("  더 센 가중 (3시드, 방향만)")
    b3 = sc(np.mean([P0[k] for k in SEEDS[:3]], 0))
    for tag, kw in (("sw5.0", dict(decay=5.0)), ("sw8.0", dict(decay=8.0))):
        r = np.mean([one(sd, **kw) for sd in SEEDS[:3]], 0)
        G.log(f"    {tag:8s} {sc(r):7.1f}   기준대비 {sc(r) - b3:+6.1f}")
