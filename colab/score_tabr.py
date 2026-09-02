# -*- coding: utf-8 -*-
"""저장된 TabR 예측을 1057 구성 위에서 채점한다. 학습을 기다리지 않고 돌린다."""
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/aimers")
import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

OUT = "/workspace/aimers/out"
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


tm = np.mean([np.where(is_f,
                       np.load(f"{OUT}/emb2_linear_relu_f_s{s}.npy"),
                       np.load(f"{OUT}/emb2_linear_relu_r_s{s}.npy"))
              for s in (42, 1, 777)], 0)
HG = np.load(f"{OUT}/hg_route_d2.0.npy")
REF = 0.10 * G.CB + 0.10 * HG + 0.80 * tm
r0 = sc(REF)
print(f"\n  기준 혼합(submit_20 구성) {r0:8.1f}   "
      f"1군 {sc(REF, R):.1f}   퓨처스 {sc(REF, is_f):.1f}")
print(f"  TabM 단독                {sc(tm):8.1f}\n")

AL = (0.05, 0.10, 0.15, 0.20, 0.30)
head = "  ".join([f"a={a:.2f}" for a in AL])
print("  {:24s} {:>7s} {:>8s}   {}".format("팔", "단독", "기준상관", head))
print("  " + "-" * 76)
for f in sorted(os.listdir(OUT)):
    if not f.startswith("tabr_") or not f.endswith(".npy"):
        continue
    p = np.load(os.path.join(OUT, f))
    inc = [sc((1 - a) * REF + a * p) - r0 for a in AL]
    body = "  ".join([f"{v:+6.1f}" for v in inc])
    print("  {:24s} {:7.1f} {:8.4f}   {}   최적 {:+.1f}".format(
        f[:-4], sc(p), np.corrcoef(p, REF)[0, 1], body, max(inc)))
print("\n  판정: 증분 +5 미만이면 안 싣는다.")
