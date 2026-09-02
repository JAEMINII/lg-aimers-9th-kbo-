# -*- coding: utf-8 -*-
"""CALIB_T 가 아직도 맞는 값인가. 재학습 없이 저장된 예측으로 잰다.

의심
    배치본은 p_out = sigmoid(T x logit(p) + shift) 를 쓰고 T = 1.0279 다.
    그런데 그 T 는 **plat_dev 누출을 고치기 전** 모델에 맞춰 잡은 값이다.
    submit_30 에서 예측 표준편차가 0.03128 -> 0.02992 로 바뀌었다.

    T 는 분산을 조절하는 장치다 (T>1 이면 예측을 날카롭게). 모델의 분산이
    바뀌면 최적 T 도 움직인다. 안 맞으면 그냥 손해다.

    그리고 기억에 이렇게 남아 있다:
        '온도 보정은 전이가 안 된다 — 3폴드 중 둘에서 -744 / -144. 절편만 쓸 것.'
    배치본 주석도 같은 얘기를 하면서 '이득 +0.6 이라 사실상 중립' 이라고 적어뒀다.
    이득이 0.6 인데 위험이 그만큼이면 빼는 게 낫다.

무엇을 재나
    T 를 훑으면서 **각 T 에서 시프트를 다시 최적화**한다. 둘이 상호작용하기
    때문에 시프트를 고정하고 T 만 바꾸면 T 의 효과가 아니라 보정 오류를 잰다.

    두 폴드(VS=2022, 2024)에서 최적 T 가 같은 쪽을 가리키는지 본다.
    폴드마다 다르면 T 는 그 해에만 맞는 값이고, 빼는 게 옳다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/aimers")
import features44 as F                                          # noqa: E402

OUT = "/workspace/aimers/out"
SEEDS_T = (42, 1, 777)
SEEDS_C = (42, 1234, 2025)
W_CB = 0.30
TS = (1.00, 1.01, 1.0279, 1.05, 1.08, 1.12, 1.20)


def load(pat, seeds):
    return [np.load(os.path.join(OUT, pat.format(s))) for s in seeds
            if os.path.exists(os.path.join(OUT, pat.format(s)))]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def score(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))


def best_shift_at(p, y, T):
    from scipy.optimize import minimize_scalar
    z = T * logit(p)
    r = minimize_scalar(lambda c: -score(sig(z + c), y),
                        bounds=(-0.6, 0.6), method="bounded")
    return -r.fun, r.x


import tabm_gate_gpu as G                                       # noqa: E402
yv_2024 = G.yv

# VS=2024 — plat 고친 TabM + CatBoost
tm = np.mean(load("pf_2024_asof_s{}.npy", SEEDS_T), 0)
cb = np.mean(load("cbpf_leaky_s{}.npy", SEEDS_C), 0)   # 배치본은 누출판 CB 를 쓴다
blend = W_CB * cb + (1 - W_CB) * tm
print(f"\n  VS=2024   혼합 (CB 0.30 / TabM 0.70)   "
      f"예측 SD {blend.std():.5f}")
print(f"  {'T':>8s} {'최적시프트':>10s} {'점수':>9s}   T=1.0 대비")
base = None
for T in TS:
    s, c = best_shift_at(blend, yv_2024, T)
    if base is None:
        base = s
    mark = "  <- 배치본" if abs(T - 1.0279) < 1e-9 else ""
    print(f"  {T:8.4f} {c:+10.4f} {s:9.1f}   {s-base:+7.1f}{mark}")

# 폴드 2 — VS=2022 (platfix 실험이 남긴 예측)
p22 = [os.path.join(OUT, f"pf_2022_asof_s{s}.npy") for s in SEEDS_T]
if all(os.path.exists(p) for p in p22):
    import pandas as pd
    d = F.build("/workspace/aimers/data", VS=2022)
    y22 = d["y"][d["m_va"]].astype(np.float64)
    tm22 = np.mean([np.load(p) for p in p22], 0)
    print(f"\n  VS=2022   TabM 단독 (CatBoost 예측이 이 폴드엔 없다)   "
          f"예측 SD {tm22.std():.5f}")
    print(f"  {'T':>8s} {'최적시프트':>10s} {'점수':>9s}   T=1.0 대비")
    b2 = None
    for T in TS:
        s, c = best_shift_at(tm22, y22, T)
        if b2 is None:
            b2 = s
        mark = "  <- 배치본" if abs(T - 1.0279) < 1e-9 else ""
        print(f"  {T:8.4f} {c:+10.4f} {s:9.1f}   {s-b2:+7.1f}{mark}")

print("\n  각 T 에서 시프트를 다시 최적화했다 — 둘이 상호작용해서")
print("  시프트를 고정하면 T 효과가 아니라 보정 오류를 재게 된다.")
print("  두 폴드가 같은 T 를 가리키지 않으면 온도는 그 해에만 맞는 값이다.")
