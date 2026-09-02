# -*- coding: utf-8 -*-
"""시드 흔들림의 크기를 역산한다. 시드를 늘려 얻을 점수의 상한을 미리 본다.

원리
    한 번 학습한 예측 = 이상적 예측 + 흔들림 (초기화 난수, 배치 순서, 드롭아웃).
    흔들림은 정답과 무관하므로 순수 손해다. k개를 평균하면 그 분산이 1/k 로 준다.

        Brier(k) = Brier(이상) + 흔들림분산 / k
        점수(k)  = 천장 - c/k,     c = 100000 x 흔들림분산 / r(1-r)

    c 가 곧 '시드 1개로 내면서 버리는 점수' 다.

흔들림분산 재는 법
    두 시드의 예측 차이 (p_i - p_j) 의 분산은 흔들림분산의 **2배**다.
    두 흔들림이 독립이라 분산이 더해지기 때문이다. 그래서 2로 나눈다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/aimers")
import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

OUT = "/workspace/aimers/out"
S = (42, 1, 777)
is_f, yv = G.is_f[G.gate], G.yv
P = [np.where(is_f, np.load(f"{OUT}/pm_base_f_s{s}.npy"),
              np.load(f"{OUT}/pm_base_r_s{s}.npy")) for s in S]


def sc(p):
    return F.best_shift(p, yv)[0]


solo = [sc(p) for p in P]
diffs = [P[i] - P[j] for i in range(len(S)) for j in range(i + 1, len(S))]
sig2 = float(np.mean([np.var(x) for x in diffs])) / 2.0
r = float(yv.mean())
c = 100000.0 * sig2 / (r * (1 - r))
s3 = sc(np.mean(P, 0))
ceil = s3 + c / len(S)                      # 3시드 관측에서 천장 역산

print(f"\n  시드별 단독 점수  {[round(v, 1) for v in solo]}   "
      f"(편차 {np.std(solo):.1f})")
print(f"  3시드 평균        {s3:.1f}")
print(f"  흔들림 표준편차   {np.sqrt(sig2):.5f}   "
      f"(예측 자체의 표준편차 {P[0].std():.5f})")
print(f"  c = {c:.1f}   -> 시드 1개는 천장보다 {c:.1f}점 아래에 있다\n")
print(f"  {'시드수':>6s} {'예측 점수':>10s} {'k=1 대비':>10s}")
for k in (1, 2, 3, 4, 8, 16, 32, 999):
    lab = "무한" if k == 999 else str(k)
    v = ceil - (0.0 if k == 999 else c / k)
    print(f"  {lab:>6s} {v:10.1f} {v - (ceil - c):+10.1f}")
print("\n  3시드 관측에서 역산한 근사다. seed_curve.py 가 8시드로 실측한다.")
print("  지금 배치는 시드 1개다 — 표의 k=1 자리에 있다.")
