# -*- coding: utf-8 -*-
"""TabPFN 을 1057 구성 위에 얹었을 때의 증분. TabR 과 같은 잣대로 잰다.

단독이 낮아도 상관이 낮으면 값어치가 있을 수 있다. 그 가능성까지 닫고 끝낸다.
평가 행이 관문 전체가 아니라 2만 행 부분표본이라 절대값은 관문 표와 다르다.
같은 행에서 기준 혼합도 다시 채점해 **증분만** 읽는다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/aimers")
import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

OUT = "/workspace/aimers/out"
gate, is_f_all, yv_all = G.gate, G.is_f[G.gate], G.yv

ev = np.load(f"{OUT}/tabpfn_smoke_idx.npy")
pf = np.load(f"{OUT}/tabpfn_smoke.npy")
# 관문 배열에서 ev 행이 어디인지
pos = {int(v): i for i, v in enumerate(gate)}
sel = np.array([pos[int(v)] for v in ev])
y = yv_all[sel]
isf = is_f_all[sel]

tm = np.mean([np.where(is_f_all,
                       np.load(f"{OUT}/emb2_linear_relu_f_s{s}.npy"),
                       np.load(f"{OUT}/emb2_linear_relu_r_s{s}.npy"))
              for s in (42, 1, 777)], 0)
HG = np.load(f"{OUT}/hg_route_d2.0.npy")
REF = (0.10 * G.CB + 0.10 * HG + 0.80 * tm)[sel]


def sc(p):
    return F.best_shift(p, y)[0]


r0 = sc(REF)
print(f"\n  평가 {len(sel):,}행 (관문 부분표본)   실제 성공률 {y.mean():.4f}")
print(f"  기준 혼합 {r0:8.1f}   TabM 단독 {sc(tm[sel]):8.1f}")
print(f"  TabPFN    {sc(pf):8.1f}   기준상관 {np.corrcoef(pf, REF)[0,1]:.4f}"
      f"   TabM상관 {np.corrcoef(pf, tm[sel])[0,1]:.4f}")
print(f"  예측 평균 {pf.mean():.4f}  표준편차 {pf.std():.4f}  "
      f"(TabM {tm[sel].std():.4f})")
print(f"\n  {'비중 a':>8s} {'점수':>9s} {'증분':>8s}")
for a in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
    v = sc((1 - a) * REF + a * pf)
    print(f"  {a:8.2f} {v:9.1f} {v - r0:+8.1f}")
print("\n  판정선은 TabR 때와 같다 — 증분 +5 미만이면 안 싣는다.")
