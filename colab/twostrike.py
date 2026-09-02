# -*- coding: utf-8 -*-
"""'TabM 이 2스트라이크에서 약하다' 가 사실인지 다시 잰다.

앞선 분석의 결함
    구간별 BSS 를 전체 성공률 R 로 나눠서 계산했다.
        BSS = 100000 * (1 - Brier / (R(1-R)))
    이러면 구간의 '내재 난이도' 가 모델 성능처럼 보인다. 어떤 구간의 결과가
    본래 예측하기 어려우면 어느 모델이든 BSS 가 낮게 나온다.

    그 값으로 "2스트라이크에서 TabM 이 무너진다(622.6)" 고 했는데, 같은 표에서
    CatBoost 와의 격차는 스트라이크 0 이 +60.7 로 가장 크고 2 는 +24.9 였다.
    절대값과 상대격차가 반대 방향을 가리킨다.

여기서 바로잡는 것
    1. 구간 자체의 분모로 잰다 (그 구간의 성공률). 구간 간 비교가 가능해진다.
    2. 세 모델을 같은 구간 안에서 맞대본다. 이게 '누가 약한가' 의 답이다.
    3. 시즌가중을 건 새 TabM 도 같이 본다. 약점이 그대로인지 바뀌었는지.

각 모델은 자기 최적 시프트로 맞춰 놓고 비교한다. 수준 보정과 판별력을 섞지 않는다.
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


def opt_shift(p, t):
    from scipy.optimize import minimize_scalar
    def mse(c):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))
        return ((q - t) ** 2).mean()
    c = minimize_scalar(mse, bounds=(-0.3, 0.3), method="bounded").x
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))


def bss_local(p, t):
    """구간 자체의 성공률을 분모로 쓴다. 구간 간 비교가 되게."""
    r = t.mean()
    if r <= 0 or r >= 1:
        return float("nan")
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)
    gate = np.where(d["m_va"])[0]
    y = d["y"][gate].astype(np.float64)
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig").iloc[gate]

    M = {}
    M["TabM(기존)"] = np.load(os.path.join(OUT, "hp_ep2.npy"))
    p = os.path.join(OUT, "sw2conf_sw3.5.npy")
    if os.path.exists(p):
        M["TabM(가중3.5)"] = np.load(p)
    M["CatBoost"] = np.load(os.path.join(SC, "cb_gate.npy"))
    M["flatMLP"] = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))
    M = {k: opt_shift(v, y) for k, v in M.items()}
    names = list(M)

    print(f"관문 {len(y):,}행   전체 성공률 {y.mean():.4f}")
    print("전체 (구간 자체 분모)")
    for n in names:
        print(f"  {n:14s} {bss_local(M[n], y):8.1f}")

    cnt = (raw.balls_before.astype(int) * 3 + raw.strikes_before.astype(int)).to_numpy()
    st = raw.strikes_before.to_numpy()

    print("\n" + "=" * 78)
    print("스트라이크별 — 구간 자체 분모로 잰 판별력")
    print("=" * 78)
    print(f"  {'구간':8s} {'n':>8s} {'성공률':>7s} " + "".join(f"{n:>15s}" for n in names))
    for v in (0, 1, 2):
        m = st == v
        row = "".join(f"{bss_local(M[n][m], y[m]):15.1f}" for n in names)
        print(f"  {'S=' + str(v):8s} {m.sum():8,} {y[m].mean():7.3f} {row}")

    print("\n  TabM(기존) 대비 격차 (양수 = TabM 보다 나음)")
    for v in (0, 1, 2):
        m = st == v
        base = bss_local(M["TabM(기존)"][m], y[m])
        row = "".join(f"{bss_local(M[n][m], y[m]) - base:15.1f}"
                      for n in names if n != "TabM(기존)")
        print(f"  {'S=' + str(v):8s} " + row)

    print("\n" + "=" * 78)
    print("카운트 12칸 — 구간 자체 분모")
    print("=" * 78)
    print(f"  {'카운트':8s} {'n':>8s} {'성공률':>7s} " + "".join(f"{n:>15s}" for n in names))
    for c in range(12):
        m = cnt == c
        if m.sum() < 2000:
            continue
        row = "".join(f"{bss_local(M[n][m], y[m]):15.1f}" for n in names)
        print(f"  {str(c // 3) + '-' + str(c % 3):8s} {m.sum():8,} "
              f"{y[m].mean():7.3f} {row}")

    print("\n  결론 판단 기준")
    print("    'TabM 이 2스트라이크에 약하다' 가 맞으려면, S=2 에서 TabM 이")
    print("    다른 모델보다 더 크게 뒤져야 한다. 절대값이 낮은 것만으로는")
    print("    구간이 어려운 것과 구별되지 않는다.")
