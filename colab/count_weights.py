# -*- coding: utf-8 -*-
"""카운트별로 비중을 다르게 주면 실제로 얼마나 남는가.

오라클 상한은 카운트12 기준 +10.1 이었다. 그건 정답을 보고 칸마다 최적 비중을
고른 값이라 실현 불가능하다. 여기서는 정직하게 잰다.

    관문 행을 무작위 반으로 가른다
    A 에서 칸별 최적 비중을 구한다
    B 에서 채점한다  (그리고 A/B 를 바꿔 한 번 더, 평균)

비교 대상은 '전역 비중도 똑같이 A 에서 구해 B 에 적용' 이다. 그래야 칸별로
쪼갠 것의 순수 이득이 나온다.

수축도 같이 본다
    칸이 작으면 A 에서 구한 비중이 노이즈다. 칸 크기 n 에 대해
        w = (n * w_cell + LAM * w_global) / (n + LAM)
    로 전역 쪽으로 당긴다. 실무에서 칸별 비중을 쓸 때의 표준 처리다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
WS = np.arange(0.0, 1.001, 0.02)

import features44 as F                                          # noqa: E402


def opt_shift(p, t):
    from scipy.optimize import minimize_scalar
    r = t.mean()
    def sc(c):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))
        return ((q - t) ** 2).mean()
    c = minimize_scalar(sc, bounds=(-0.3, 0.3), method="bounded").x
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))


def best_w(T, O, y):
    """제곱오차를 최소화하는 TabM 비중."""
    se = [(((w * T + (1 - w) * O) - y) ** 2).sum() for w in WS]
    return WS[int(np.argmin(se))]


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)
    gate = np.where(d["m_va"])[0]
    y = d["y"][gate].astype(np.float64)
    R = y.mean()
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig").iloc[gate]
    cell = (raw.balls_before.astype(int) * 3 + raw.strikes_before.astype(int)).to_numpy()

    T = opt_shift(np.load(os.path.join(OUT, "hp_ep2.npy")), y)
    C = opt_shift(np.load(os.path.join(SC, "cb_gate.npy")), y)
    M = opt_shift(np.load(os.path.join(OUT, "mlpnew_flat_s3.npy")), y)
    O = 0.25 * C + 0.75 * M          # 현 제출본의 CB:MLP 비율 (0.10:0.30)

    rng = np.random.default_rng(0)
    half = rng.random(len(y)) < 0.5
    res = {"전역": [], "칸별": [], "칸별+수축": []}
    LAM = 20000.0
    for A, B in ((half, ~half), (~half, half)):
        wg = best_w(T[A], O[A], y[A])
        pg = wg * T[B] + (1 - wg) * O[B]
        res["전역"].append(((pg - y[B]) ** 2).sum() / B.sum())
        for tag, lam in (("칸별", 0.0), ("칸별+수축", LAM)):
            p = np.empty(B.sum())
            idxB = np.flatnonzero(B)
            for c in range(12):
                mA, mB = A & (cell == c), B & (cell == c)
                if mB.sum() == 0:
                    continue
                w = best_w(T[mA], O[mA], y[mA]) if mA.sum() > 50 else wg
                n = mA.sum()
                w = (n * w + lam * wg) / (n + lam) if lam else w
                sel = np.isin(idxB, np.flatnonzero(mB))
                p[sel] = w * T[mB] + (1 - w) * O[mB]
            res[tag].append(((p - y[B]) ** 2).sum() / B.sum())
        if A is half:
            print(f"  전역 최적 비중 {wg:.2f}   칸별 최적 비중:")
            for c in range(12):
                mA = A & (cell == c)
                if mA.sum() > 50:
                    print(f"    {c//3}-{c%3}  n={mA.sum():6,}  w={best_w(T[mA],O[mA],y[mA]):.2f}")

    print()
    base = np.mean(res["전역"])
    for k, v in res.items():
        mse = np.mean(v)
        print(f"  {k:10s} 점수 {100000*(1-mse/(R*(1-R))):8.1f}   "
              f"전역대비 {100000*(base-mse)/(R*(1-R)):+6.1f}")
    print("\n  오라클 상한은 +10.1 이었다. 여기 남는 값이 실제로 얻을 수 있는 몫이다.")
    print("  ※ 이 비중들은 관문에서 학습된 것이라 관문의 편향을 그대로 물려받는다.")
    print("    관문은 CatBoost 를 TabM 보다 높게 치는데(903.2 vs 867.5)")
    print("    리더보드는 반대다(1021 vs 1041). 즉 여기서 나온 비중은")
    print("    CatBoost 쪽으로 과하게 기울어 있을 것이다.")
