# -*- coding: utf-8 -*-
"""앙상블 비중을 훑는다. 저장된 관문 예측만 쓰므로 CPU 로 돈다.

재료 (전부 배치 구성에서 나온 관문 예측)
    CB      cb_gate.npy                     CatBoost
    HG      hg_route_d2.0.npy               HistGB (submit_27 에서 뺐다)
    TM8     sc_s{8시드}.npy 평균             TabM, 시드 곡선이 만든 것
    TM1     sc_s42.npy                      TabM 시드 1개 = 지금 배치 상태
    MN      mnca2_c65536_s{3시드}.npy 평균   MNCA

무엇을 답하나
    ① 시드 8개가 비중 최적점을 옮기나
       TabM 이 좋아지면 GBDT 비중이 줄어드는 게 자연스럽다. submit_27 의
       0.30 이 시드 1개 기준이라 8시드에서는 다를 수 있다.
    ② MNCA 를 넣으면 최적 배합이 어떻게 되나
    ③ HistGB 를 되살릴 이유가 생기나 (지금은 뺀 상태)

관문을 얼마나 믿나 — 조심할 것
    혼합 비중은 관문이 여섯 번 틀린 '조합' 부류다. 계열 간 순위를 못 매기고
    (CatBoost 관문 903 > TabM 868 인데 리더보드는 반대), 브랜치 구성도
    관문 -5.4 / 0-4 인데 리더보드 +3 이었다.

    그래서 이 표는 **지도**로만 쓴다. 최적점을 그대로 옮기지 않고,
    '현행 0.30 이 고원 안에 있나' 와 'MNCA 자리가 있나' 만 읽는다.
    고원이 넓으면 현행을 유지하는 게 안전하다.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/aimers")
import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

OUT = "/workspace/aimers/out"
is_f, yv = G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(G.gate), bool)
SEEDS8 = (42, 1, 777, 2, 7, 13, 99, 2024)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def load(name):
    p = os.path.join(OUT, name)
    return np.load(p) if os.path.exists(p) else None


CB = G.CB
HG = load("hg_route_d2.0.npy")
tm = [load(f"sc_s{s}.npy") for s in SEEDS8]
tm = [t for t in tm if t is not None]
TM8 = np.mean(tm, 0) if tm else None
TM1 = tm[0] if tm else None
mn = [load(f"mnca2_c65536_s{s}.npy") for s in (42, 1, 777)]
mn = [m for m in mn if m is not None]
MN = np.mean(mn, 0) if mn else None

print(f"\n  재료   CB {CB is not None}  HG {HG is not None}  "
      f"TabM 시드 {len(tm)}개  MNCA 시드 {len(mn)}개")
for nm, p in (("CatBoost", CB), ("HistGB", HG), ("TabM x1", TM1),
              ("TabM x8", TM8), ("MNCA", MN)):
    if p is not None:
        print(f"  {nm:10s} 단독 {sc(p):7.1f}   1군 {sc(p, R):7.1f}   "
              f"퓨처스 {sc(p, is_f):7.1f}")

print("\n" + "=" * 74)
print("  [A] CatBoost 비중 스윕.  나머지는 TabM.  시드 1개 vs 8개")
print("=" * 74)
print(f"  {'w_CB':>6s} {'TabM x1':>10s} {'TabM x8':>10s}   차이")
for w in (0.0, 0.1, 0.2, 0.25, 0.30, 0.35, 0.4, 0.5):
    a = sc((1 - w) * TM1 + w * CB)
    b = sc((1 - w) * TM8 + w * CB)
    mark = "  <- submit_27" if abs(w - 0.30) < 1e-9 else ""
    print(f"  {w:6.2f} {a:10.1f} {b:10.1f}   {b-a:+6.1f}{mark}")

if MN is not None:
    print("\n" + "=" * 74)
    print("  [B] 3원 격자.  w_CB + w_MNCA + w_TabM8 = 1")
    print("=" * 74)
    best = []
    for wc in np.arange(0.0, 0.45, 0.05):
        row = []
        for wm in np.arange(0.0, 0.45, 0.05):
            if wc + wm > 0.6:
                row.append(None)
                continue
            p = wc * CB + wm * MN + (1 - wc - wm) * TM8
            v = sc(p)
            row.append(v)
            best.append((v, round(float(wc), 2), round(float(wm), 2)))
        print(f"  w_CB={wc:4.2f}  " + " ".join(
            f"{v:7.1f}" if v is not None else "      ." for v in row))
    print(f"  {'':11s}" + " ".join(f"w_MN={w:4.2f}"[-7:].rjust(7)
                                   for w in np.arange(0.0, 0.45, 0.05)))
    best.sort(reverse=True)
    print(f"\n  상위 5개")
    for v, wc, wm in best[:5]:
        print(f"    CB {wc:4.2f} / MNCA {wm:4.2f} / TabM {1-wc-wm:4.2f}  -> {v:7.1f}")
    ref = sc(0.30 * CB + 0.70 * TM8)
    print(f"\n  현행 배합(CB 0.30 / TabM8 0.70) {ref:7.1f}")
    print(f"  최적 대비 {best[0][0]-ref:+.1f}")

if HG is not None:
    print("\n" + "=" * 74)
    print("  [C] HistGB 를 되살릴 이유가 있나 (GBDT 총량 0.30 고정)")
    print("=" * 74)
    ref = sc(0.30 * CB + 0.70 * TM8)
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        g = (1 - a) * CB + a * HG
        v = sc(0.30 * g + 0.70 * TM8)
        print(f"  CB {1-a:4.2f} / HistGB {a:4.2f}   {v:7.1f}   {v-ref:+6.1f}")

print("\n  이 표는 지도다. 관문은 조합 부류에서 여섯 번 틀렸다.")
print("  '현행이 고원 안에 있나' 와 'MNCA 자리가 있나' 만 읽는다.")
