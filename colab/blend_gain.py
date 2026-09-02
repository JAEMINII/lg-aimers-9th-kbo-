# -*- coding: utf-8 -*-
"""약하지만 상관 낮은 모델이 혼합에 실제로 도움이 되는가. 정확히 계산한다.

질문
    "점수가 낮아도 TabM 과 상관 낮은 모델을 앙상블하면?"
    이건 추정할 필요가 없다. 저장된 예측이 전부 **같은 폴드·같은 253,507행**이라
    비중을 훑어 실제 혼합 점수를 직접 재면 된다.

방법
    각 후보 X 에 대해  p = (1-w) x TabM + w x X  를 w=0..0.6 으로 훑고
    **최적 시프트에서** 채점한다. TabM 단독 대비 최대 이득과 그 때의 w 를 낸다.

    마지막에 전 후보를 동시에 쓰는 최적 비중도 낸다. 이건 그 해 정답을 보고
    맞춘 값이라 **상한**이지 쓸 수 있는 값이 아니다.

주의
    여기서 나온 이득은 **관문 이득**이다. 기억의 전이율 표를 보면 조합 변경은
    관문 +12.1 -> 리더보드 +1.0 (0.08배) 였다. 관문에서 +10 이 나와도 실제로는
    +1 이하로 봐야 한다. MNCA 는 관문 +4.3 이었는데 리더보드 -23 이었다.
"""
import os
import sys

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(SC, "_dl")
VS = 2024

import features44 as F                                          # noqa: E402

NAMES = ["TabM", "MNCA", "FT-Transformer", "ResNet", "TabR(lin_relu)",
         "TabR(PLR)", "CatBoost d2.0", "CatBoost d1.0", "HistGB",
         "XGBoost", "LightGBM", "로지스틱", "투수성공률 1열"]
P = np.load(os.path.join(DL, "table_preds.npy")).astype(np.float64)
assert P.shape[0] == len(NAMES), f"{P.shape} vs {len(NAMES)}"

d = F.build(DATA, VS=VS)
season = d["season"].astype(int)
gate = np.where(season == VS)[0]
yv = d["y"].astype(np.float64)[gate]
isf_g = d["is_f"][gate]
print(f"  {P.shape[0]}개 모델 x {P.shape[1]:,}행   실제 {yv.mean():.4f}\n")

tabm = P[0]
base = F.best_shift(tabm, yv)[0]
WS = np.arange(0.0, 0.65, 0.02)
print(f"  TabM 단독 {base:.1f}\n")
print(f"  {'모델':16s} {'단독':>8s} {'상관':>7s} {'최적w':>6s} {'혼합':>9s} "
      f"{'이득':>7s}   {'w=0.2':>7s}")
rows = []
for i in range(1, len(NAMES)):
    x = P[i]
    solo = F.best_shift(x, yv)[0]
    r = float(np.corrcoef(x, tabm)[0, 1])
    ss = [F.best_shift((1 - w) * tabm + w * x, yv)[0] for w in WS]
    k = int(np.argmax(ss))
    g20 = F.best_shift(0.8 * tabm + 0.2 * x, yv)[0] - base
    rows.append((NAMES[i], solo, r, WS[k], ss[k], ss[k] - base, g20))
    print(f"  {NAMES[i]:16s} {solo:8.1f} {r:7.4f} {WS[k]:6.2f} {ss[k]:9.1f} "
          f"{ss[k]-base:+7.1f}   {g20:+7.1f}")

print("\n  상관 낮은 순으로 다시 (재민님 가설: 낮을수록 유리한가)")
for nm, solo, r, w, s, g, g20 in sorted(rows, key=lambda t: t[2]):
    print(f"    상관 {r:.4f}  단독 {solo:7.1f}  ->  최대이득 {g:+6.1f} (w={w:.2f})")

# 전 후보 동시 최적 (상한)
from scipy.optimize import minimize                             # noqa: E402

idx = [i for i in range(1, len(NAMES)) if NAMES[i] != "TabR(PLR)"]
M = np.vstack([tabm] + [P[i] for i in idx])


def neg(v):
    a = np.abs(v)
    a = a / a.sum()
    return -F.best_shift(a @ M, yv)[0]


r0 = minimize(neg, np.r_[1.0, np.zeros(len(idx))], method="Nelder-Mead",
              options={"maxiter": 6000, "fatol": 1e-6})
a = np.abs(r0.x)
a = a / a.sum()
print(f"\n  [상한] 전 후보 동시 최적 비중  {-r0.fun:.1f}   "
      f"TabM 단독 대비 {-r0.fun - base:+.1f}")
for nm, wgt in sorted(zip(["TabM"] + [NAMES[i] for i in idx], a),
                      key=lambda t: -t[1]):
    if wgt > 0.01:
        print(f"    {nm:16s} {wgt:.3f}")
print("\n  이건 그 해 정답을 보고 맞춘 값이라 쓸 수 있는 값이 아니다.")
print("  그리고 조합 변경의 관문->리더보드 전이율은 0.08 이었다 (+12.1 -> +1.0).")
