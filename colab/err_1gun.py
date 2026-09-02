# -*- coding: utf-8 -*-
"""1군에서 어디를 제일 못 맞히나. 구간별로 '남은 점수' 를 센다.

자료
    관문 VS=2024 (2019~2023 학습 -> 2024 예측). 전부 아웃샘플이다.
    TabM     0.6*c4g_all + 0.4*c4g_regular   seed 42 — 배치와 같은 구성
    CatBoost ta2024_base 3시드 평균          50열(pc_c12+pc_cmh 포함)
    혼합     0.30*CatBoost + 0.70*TabM,  최적 시프트로 수준만 맞춘다

무엇을 세나
    점수 = 100000 * (1 - Brier / (r(1-r)))   r 은 **전체** 성공률
    그래서 구간 i 가 총점에서 깎는 몫은
        손실_i = 100000 * (n_i/N) * Brier_i / (r(1-r))
    그 구간의 자기 기저율만 알아도 Brier 는 r_i(1-r_i) 까지 내려간다. 차이가
        남은점수_i = 100000 * (n_i/N) * (Brier_i - r_i(1-r_i)) / (r(1-r))
    이게 "이 구간을 완벽히 보정하면 벌 수 있는 점수" 다. 음수면 모델이
    기저율보다 잘하고 있다는 뜻이다.

주의
    구간을 잘게 쪼갤수록 r_i(1-r_i) 는 자동으로 내려간다(과적합). n 이 작은 칸은
    믿지 말 것. 그래서 n 과 같이 본다.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "colab"))
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(ROOT, "colab", "_dl")
import features44 as F                                          # noqa: E402

d = F.build(DATA, VS=2024, return_frame=True)
fr = d["frame"]
season = d["season"]
gate = np.where(season == 2024)[0]
y = d["y"].astype(np.float64)[gate]
isf = d["is_f"][gate]
tab = 0.6 * np.load(f"{DL}/c4g_2024_c4_all_s42.npy") \
    + 0.4 * np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")
cb = np.load(f"{DL}/ta2024_base.npy").mean(0)
assert len(tab) == len(y) == len(cb), (len(tab), len(y), len(cb))
p = 0.30 * cb + 0.70 * tab
sc, sh = F.best_shift(p[~isf], y[~isf])
q = np.clip(p, 1e-9, 1 - 1e-9)
p = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + sh)))

g = fr.iloc[gate].reset_index(drop=True)
m = ~isf                                    # 1군만
y, p, g = y[m], p[m], g[m].reset_index(drop=True)
N = len(y)
r = y.mean()
DEN = r * (1 - r)
br = (p - y) ** 2
print(f"  1군 2024  {N:,}행   성공률 {r:.4f}   전체점수 "
      f"{100000*(1-br.mean()/DEN):.1f}   시프트 {sh:+.4f}")
print(f"  예측 평균 {p.mean():.4f}  sd {p.std():.4f}   실제 sd {y.std():.4f}\n")

cnt = g.balls_before.astype(int) * 3 + g.strikes_before.astype(int)
pn = g.asof_pitcher_n.fillna(0).to_numpy()
bn = g.asof_batter_n.fillna(0).to_numpy() if "asof_batter_n" in g else np.zeros(N)
AX = {
    "카운트(B-S)": g.balls_before.astype(str) + "-" + g.strikes_before.astype(str),
    "이닝": np.clip(g.inning.fillna(1).astype(int), 1, 10),
    "아웃": g.outs_before.astype(int),
    "주자상황": g.base_state.astype(str),
    "손 조합(P/B)": g.pitcher_hand.astype(str) + "/" + g.batter_hand.astype(str),
    "투수 누적투구수": pd.cut(pn, [-1, 200, 1000, 3000, 8000, 1e9],
                        labels=["<200", "200-1k", "1k-3k", "3k-8k", "8k+"]),
    "월": g.game_month.astype(int),
    "예측 십분위": pd.qcut(p, 10, labels=[f"D{i+1}" for i in range(10)],
                       duplicates="drop"),
}
for nm, key in AX.items():
    k = pd.Series(np.asarray(key)).astype(str)
    t = pd.DataFrame({"k": k, "y": y, "p": p, "b": br})
    a = t.groupby("k", sort=False).agg(n=("y", "size"), rate=("y", "mean"),
                                       pred=("p", "mean"), brier=("b", "mean"))
    a["편차"] = a["pred"] - a["rate"]
    a["최선"] = a["rate"] * (1 - a["rate"])
    a["남은점수"] = 100000 * (a["n"] / N) * (a["brier"] - a["최선"]) / DEN
    a = a.sort_values("남은점수", ascending=False)
    print(f"  {nm}")
    print(f"    {'칸':12s} {'n':>8s} {'실제':>7s} {'예측':>7s} {'편차':>8s} "
          f"{'남은점수':>9s}")
    for k2, rw in a.head(4).iterrows():
        print(f"    {k2:12s} {int(rw['n']):>8,} {rw['rate']:>7.4f} "
              f"{rw['pred']:>7.4f} {rw['편차']:>+8.4f} {rw['남은점수']:>9.1f}")
    print(f"    {'... 합계':12s} {N:>8,} {'':7s} {'':7s} {'':8s} "
          f"{a['남은점수'].sum():>9.1f}")
    print()
