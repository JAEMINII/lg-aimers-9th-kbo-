# -*- coding: utf-8 -*-
"""TabM 이 어디서 틀리는지, 구간별로 섞으면 얼마나 남는지 본다.

지금 앙상블은 전 구간 같은 비중(0.10/0.30/0.60)이다. 만약 TabM 이 특정 구간에서만
약하고 다른 모델이 그 구간을 메운다면, 구간별 비중이 전역 비중보다 낫다.
그게 아니라 어디서나 고르게 이긴다면 전역 비중이 맞고 앙상블에서 더 짜낼 게 없다.

찍는 것
    1. 신뢰도(캘리브레이션)  예측 구간별 실제 성공률. 어느 확률대에서 어긋나는가
    2. 구간별 손실 분해       각 구간이 전체 오차의 몇 %를 차지하고, 거기서 누가 나은가
    3. 오라클 상한            구간마다 최적 비중을 썼을 때의 점수. 전역 비중 대비 얼마인가
                             (오라클은 정답을 보고 고르는 것이라 실제로는 그만큼 못 얻는다.
                              상한이 작으면 구간별 비중은 애초에 가망이 없다는 뜻이다)
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"

import features44 as F                                          # noqa: E402


def bss(p, t, r):
    """전체 성공률 r 로 정규화. 구간끼리 비교되도록 분모를 고정한다."""
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


def opt_shift(p, t):
    from scipy.optimize import minimize_scalar
    r = t.mean()
    def f(c):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))
        return -bss(q, t, r)
    res = minimize_scalar(f, bounds=(-0.3, 0.3), method="bounded")
    c = res.x
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)
    gate = np.where(d["m_va"])[0]
    y = d["y"][gate].astype(np.float64)
    R = y.mean()
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig").iloc[gate]

    T = np.load(os.path.join(OUT, "hp_ep2.npy"))
    C = np.load(os.path.join(SC, "cb_gate.npy"))[gate] if len(
        np.load(os.path.join(SC, "cb_gate.npy"))) != len(gate) else np.load(
        os.path.join(SC, "cb_gate.npy"))
    M = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))
    if len(T) != len(gate):
        T = T[gate]
    if len(M) != len(gate):
        M = M[gate]
    # 각 모델을 자기 최적 시프트에 맞춰 놓고 비교한다. 수준 보정과 판별력을 섞지 않기 위함
    T, C, M = opt_shift(T, y), opt_shift(C, y), opt_shift(M, y)
    B = opt_shift(0.10 * C + 0.30 * M + 0.60 * T, y)
    print(f"관문 {len(y):,}행  실제 성공률 {R:.4f}")
    print(f"단독  TabM {bss(T,y,R):7.1f}   CatBoost {bss(C,y,R):7.1f}   "
          f"flatMLP {bss(M,y,R):7.1f}   혼합 {bss(B,y,R):7.1f}\n")

    print("=" * 78)
    print("1. 신뢰도 — TabM 예측 구간별 실제 성공률")
    print("=" * 78)
    qs = np.quantile(T, np.linspace(0, 1, 11))
    qs[0], qs[-1] = -1, 2
    for i in range(10):
        m = (T >= qs[i]) & (T < qs[i + 1])
        if m.sum() == 0:
            continue
        print(f"  {T[m].min():.3f}~{T[m].max():.3f}  n={m.sum():6,}  "
              f"예측 {T[m].mean():.4f}  실제 {y[m].mean():.4f}  "
              f"차 {T[m].mean()-y[m].mean():+.4f}   "
              f"TabM오차 {((T[m]-y[m])**2).mean():.4f}")

    print("\n" + "=" * 78)
    print("2. 구간별 — 오차 비중과 모델 우열 (음수 = TabM 보다 나음)")
    print("=" * 78)
    cnt = raw.balls_before.astype(int) * 3 + raw.strikes_before.astype(int)
    slices = {
        "game_type": raw.game_type.astype(str).values,
        "strikes": raw.strikes_before.values,
        "balls": raw.balls_before.values,
        "카운트12": cnt.values,
        "이닝": np.clip(raw.inning.values, 1, 10),
        "투수손x타자손": (raw.pitcher_hand.astype(str) + "-"
                        + raw.batter_hand.astype(str)).values,
        "투수경험": pd.cut(raw.asof_pitcher_n, [-1, 500, 2000, 8000, 1e9],
                        labels=["~500", "~2k", "~8k", "8k+"]).values,
        "주자": raw.base_state.astype(str).values,
    }
    tot = ((T - y) ** 2).sum()
    for name, key in slices.items():
        print(f"\n  [{name}]")
        ks = pd.Series(key)
        for v, idx in ks.groupby(ks, observed=True).groups.items():
            m = np.zeros(len(y), bool)
            m[np.asarray(idx)] = True
            if m.sum() < 2000:
                continue
            et = ((T[m] - y[m]) ** 2)
            print(f"    {str(v):10s} n={m.sum():7,}  오차몫 {et.sum()/tot*100:5.1f}%  "
                  f"실제 {y[m].mean():.3f}  TabM예측 {T[m].mean():.3f}  "
                  f"TabM {bss(T[m],y[m],R):8.1f}   "
                  f"CB {bss(C[m],y[m],R)-bss(T[m],y[m],R):+8.1f}  "
                  f"MLP {bss(M[m],y[m],R)-bss(T[m],y[m],R):+8.1f}")

    print("\n" + "=" * 78)
    print("3. 오라클 상한 — 구간별로 최적 비중을 썼다면")
    print("=" * 78)
    ws = np.arange(0.0, 1.01, 0.05)
    for name, key in slices.items():
        ks = pd.Series(key)
        se = 0.0
        glob = ((B - y) ** 2).sum()
        for v, idx in ks.groupby(ks, observed=True).groups.items():
            m = np.zeros(len(y), bool)
            m[np.asarray(idx)] = True
            best = min((((w * T[m] + (1 - w) * (0.25 * C[m] + 0.75 * M[m])) - y[m]) ** 2).sum()
                       for w in ws)
            se += best
        g = 100000 * (glob - se) / (len(y) * R * (1 - R))
        print(f"  {name:14s} 오라클 이득 {g:+7.1f}")
    print("\n  오라클은 정답을 보고 비중을 고른 값이다. 실제로는 이보다 훨씬 적게 얻는다.")
    print("  이 값이 작으면 구간별 비중은 시도할 가치가 없다.")
