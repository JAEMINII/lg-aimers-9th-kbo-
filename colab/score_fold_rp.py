# -*- coding: utf-8 -*-
"""이미 나온 npy 로 VS=2022 만 먼저 채점한다 (본 실행은 VS=2024 를 도는 중)."""
import os, sys
import numpy as np
sys.path.insert(0, "/workspace/aimers/colab"); sys.path.insert(0, "/workspace/aimers")
OUT, DATA = "/workspace/aimers/out", "/workspace/aimers/data"
SEEDS = (42, 1, 777)
ARMS = ("base", "rplat", "rboth")
import features44 as F
d0 = F.build(DATA, VS=2024)
season, isf, y = d0["season"].astype(np.float64), d0["is_f"], d0["y"].astype(np.float64)
VS = 2022
gate = np.where(season == VS)[0]
isf_g, yv = isf[gate], y[gate]
def sc(p, m=None):
    m = np.ones(len(yv), bool) if m is None else m
    return F.best_shift(p[m], yv[m])[0]
P = {a: [np.load(f"{OUT}/rp_{VS}_{a}_s{s}.npy") for s in SEEDS] for a in ARMS}
print("=" * 92)
print(f"  reverse 플래툰 편차 — VS={VS}   판정선: 두 폴드 +10 이상 & 3/3")
print("=" * 92)
for a in ARMS:
    ps = P[a]; p = np.mean(ps, 0)
    line = f"  {a:7s} 전체 {sc(p):8.1f}  1군 {sc(p, ~isf_g):8.1f}  퓨처스 {sc(p, isf_g):8.1f}"
    if a != "base":
        dd = [sc(x) - sc(b) for x, b in zip(ps, P["base"])]
        mu = float(np.mean(dd)); se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
        line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f} "
                 f"{sum(1 for v in dd if v>0)}/3   [{' '.join(f'{v:+.1f}' for v in dd)}]")
    print(line)
print("\n  참고  plat_dev 는 +19.5(t=5.5) / +24.2(t=8.2) 였다.")
