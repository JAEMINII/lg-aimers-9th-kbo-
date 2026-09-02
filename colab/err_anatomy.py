# -*- coding: utf-8 -*-
"""우리 예측이 **어디서** 틀리는지 해부한다. 부품 비교가 아니라 오차 구조를 본다.

왜 지금 이걸 하나
    지금까지는 부품을 바꿔가며 관문 점수를 비교했다. 우리 예측이 어느 구간에서
    체계적으로 어긋나는지는 한 번도 안 봤다. 200팀이 1100 이상이면 우리가 통째로
    놓치는 구간이 있을 수 있고, 그건 부품 교체로는 안 보인다.

무엇을 재나
    전역 최적 시프트를 **먼저** 걸어 전체 절편을 맞춘 뒤, 남는 구간별 편향만 본다.
    안 그러면 전역 미보정이 모든 구간에 묻어 들어가 전부 편향처럼 보인다.

    구간 k 에서
        n_k    행수
        y_k    실제 성공률
        p_k    예측 평균
        편향   p_k - y_k
    구간별 절편을 완벽히 고쳤을 때 회수되는 점수는
        100000 x sum_k (n_k/N) (p_k - y_k)^2 / (r(1-r))

부풀림 방지 — 반반 검사
    같은 자료에서 구간 절편을 구하고 그 자리에서 채점하면 표본 잡음까지 편향으로
    세어진다. 구간이 잘게 쪼개질수록 심해진다. 그래서
        A 절반에서 구간 절편을 구하고  ->  B 절반에서 적용해 점수 변화를 잰다
    양쪽을 바꿔 평균낸다. 이 값이 진짜 회수 가능액이다. 둘의 차이가 곧 잡음이다.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/aimers")
import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
N = len(gate)
R = float(yv.mean())
DEN = R * (1 - R)


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def score(p, y):
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))


# ---------------------------------------------------------------- 예측 만들기
tm = np.mean([np.where(is_f,
                       np.load(f"{OUT}/emb2_linear_relu_f_s{s}.npy"),
                       np.load(f"{OUT}/emb2_linear_relu_r_s{s}.npy"))
              for s in (42, 1, 777)], 0)
HG = np.load(f"{OUT}/hg_route_d2.0.npy")
P = 0.10 * G.CB + 0.10 * HG + 0.80 * tm
s0, c0 = F.best_shift(P, yv)
P = sig(logit(P) + c0)                       # 전역 절편을 먼저 맞춘다
print(f"\n  관문 {N:,}행   실제 성공률 {R:.4f}   전역 시프트 {c0:+.4f}")
print(f"  1057 구성 점수 {s0:.1f}   (= 전역 보정 후 판별력)")
print(f"  예측 평균 {P.mean():.4f}   표준편차 {P.std():.4f}")

# ---------------------------------------------------------------- 구간 정의
raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["balls_before", "strikes_before", "inning",
                           "num_runners_on", "base_state", "asof_pitcher_n",
                           "game_month", "game_type", "outs_before",
                           "asof_batter_n", "li", "score_diff_pitcher_team"])
raw = raw.iloc[gate].reset_index(drop=True)
an = raw["asof_pitcher_n"].to_numpy(np.float64)
bn = raw["asof_batter_n"].to_numpy(np.float64)

SLICES = {
    "카운트": (raw["balls_before"].astype(str) + "-"
             + raw["strikes_before"].astype(str)).to_numpy(),
    "이닝": np.clip(raw["inning"].to_numpy(), 1, 10).astype(str),
    "아웃": raw["outs_before"].to_numpy().astype(str),
    "주자상태": raw["base_state"].to_numpy().astype(str),
    "주자수": raw["num_runners_on"].to_numpy().astype(str),
    "월": raw["game_month"].to_numpy().astype(str),
    "리그": np.where(raw["game_type"].to_numpy() == "F", "퓨처스", "1군"),
    "투수경험": pd.cut(an, [-1, 100, 300, 700, 1500, 3000, 1e9],
                   labels=["~100", "100~300", "300~700", "700~1500",
                           "1500~3000", "3000~"]).astype(str),
    "타자경험": pd.cut(bn, [-1, 100, 300, 700, 1500, 3000, 1e9],
                   labels=["~100", "100~300", "300~700", "700~1500",
                           "1500~3000", "3000~"]).astype(str),
    "LI": pd.qcut(raw["li"].to_numpy(), 5, labels=False,
                  duplicates="drop").astype(str),
    "점수차": pd.cut(raw["score_diff_pitcher_team"].to_numpy(),
                  [-99, -5, -2, -1, 0, 1, 2, 5, 99]).astype(str),
    "예측십분위": pd.qcut(P, 10, labels=False, duplicates="drop").astype(str),
}

# ---------------------------------------------------------------- 표
print("\n" + "=" * 96)
print("  구간별 실제 vs 예측   (전역 절편은 이미 맞춘 상태)")
print("=" * 96)
rng = np.random.default_rng(0)
half = rng.random(N) < 0.5
summary = []
for name, key in SLICES.items():
    key = np.asarray(key)
    vals = sorted(set(key.tolist()))
    print(f"\n  [{name}]  {len(vals)}구간")
    print(f"    {'구간':12s} {'행수':>9s} {'비중':>6s} {'실제':>7s} "
          f"{'예측':>7s} {'편향':>8s} {'구간점수':>9s}")
    raw_gain = 0.0
    for v in vals:
        m = key == v
        n = int(m.sum())
        if n < 30:
            continue
        ya, pa = float(yv[m].mean()), float(P[m].mean())
        b = pa - ya
        raw_gain += (n / N) * b * b
        sl = (100000.0 * (1 - np.mean((P[m] - yv[m]) ** 2) / (ya * (1 - ya)))
              if 0 < ya < 1 else float("nan"))
        print(f"    {str(v):12s} {n:9,d} {n/N*100:5.1f}% {ya:7.4f} "
              f"{pa:7.4f} {b:+8.4f} {sl:9.1f}")
    # 반반 검사: A 에서 구간 절편 -> B 에서 적용
    got = []
    for tr_m, te_m in ((half, ~half), (~half, half)):
        Q = P.copy()
        for v in vals:
            m = key == v
            a = m & tr_m
            if a.sum() < 30:
                continue
            d = F.best_shift(P[a], yv[a])[1]      # 그 구간의 최적 절편
            Q[m] = sig(logit(P[m]) + d)
        got.append(score(Q[te_m], yv[te_m]) - score(P[te_m], yv[te_m]))
    summary.append((name, len(vals), 100000.0 * raw_gain / DEN,
                    float(np.mean(got)), got))

print("\n" + "=" * 96)
print("  구간별 절편을 고치면 얼마를 회수하나")
print("=" * 96)
print(f"  {'구간':12s} {'개수':>5s} {'그자리 계산':>12s} "
      f"{'반반 검증':>10s}   {'양쪽':>16s}")
print("  " + "-" * 70)
for name, k, rawg, hh, got in sorted(summary, key=lambda x: -x[3]):
    print(f"  {name:12s} {k:5d} {rawg:12.1f} {hh:+10.1f}   "
          f"[{got[0]:+.1f}, {got[1]:+.1f}]")
print("\n  '그자리 계산' 은 잡음까지 편향으로 세므로 항상 부풀려진다.")
print("  판정은 '반반 검증' 으로 한다. 이게 +로 크면 구조적 편향이 실재하고,")
print("  0 근처면 우리 구조는 멀쩡한데 정보가 부족한 것이다 — 대응이 완전히 다르다.")
