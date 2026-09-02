# -*- coding: utf-8 -*-
"""1군에서 아직 안 쓴 신호가 어디 있나. 구간별 '치우침' 과 그 값어치를 잰다.

앞선 분석에서 모든 구간이 자기 기저율보다 나았다 — 못 맞히는 구간은 없다.
그러면 남은 질문은 **체계적으로 치우친 구간**이다.

    치우침 b_i = 평균예측_i - 실제_i
    그 구간의 예측을 b_i 만큼 통째로 옮기면 Brier 가 (n_i/N)*b_i^2 만큼 준다
    점수로는  100000 * (n_i/N) * b_i^2 / (r(1-r))

주의 — 치우침은 표본잡음으로도 생긴다. n_i 행의 표준오차가 sqrt(r(1-r)/n_i) 라
    |b| 가 그 2배를 못 넘으면 잡음이다. 그래서 t 를 같이 낸다.
    그리고 이 값은 **상한**이다. 2024 에서 잰 치우침이 2025 에도 있어야 쓸모가 있다.
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
fr, season = d["frame"], d["season"]
gate = np.where(season == 2024)[0]
y, isf = d["y"].astype(np.float64)[gate], d["is_f"][gate]
tab = 0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy") + 0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")
cb = np.load(f"{DL}/ta2024_base.npy").mean(0)
p = 0.30*cb + 0.70*tab
_, sh = F.best_shift(p[~isf], y[~isf])
q = np.clip(p, 1e-9, 1-1e-9)
p = 1/(1+np.exp(-(np.log(q/(1-q)) + sh)))
g = fr.iloc[gate].reset_index(drop=True)
m = ~isf
y, p, g = y[m], p[m], g[m].reset_index(drop=True)
N, r = len(y), y.mean()
DEN = r*(1-r)
print(f"  1군 2024  {N:,}행  성공률 {r:.4f}  "
      f"점수 {100000*(1-((p-y)**2).mean()/DEN):.1f}\n")

pn = g.asof_pitcher_n.fillna(0).to_numpy()
AX = {
    "월": g.game_month.astype(int).astype(str),
    "카운트": g.balls_before.astype(str) + "-" + g.strikes_before.astype(str),
    "주자상황": g.base_state.astype(str),
    "이닝": np.clip(g.inning.fillna(1).astype(int), 1, 10).astype(str),
    "아웃": g.outs_before.astype(int).astype(str),
    "손 P/B": g.pitcher_hand.astype(str) + "/" + g.batter_hand.astype(str),
    "투수경험": pd.cut(pn, [-1, 200, 1000, 3000, 8000, 1e9],
                   labels=["<200", "200-1k", "1k-3k", "3k-8k", "8k+"]).astype(str),
    "투수팀": g.pitcher_team_id.astype(str),
}
rows = []
print(f"  {'축':10s} {'보정 상한':>10s}   {'가장 치우친 칸':>34s}")
for nm, key in AX.items():
    k = pd.Series(np.asarray(key)).astype(str)
    t = pd.DataFrame({"k": k, "y": y, "p": p})
    a = t.groupby("k", sort=False).agg(n=("y","size"), rate=("y","mean"), pred=("p","mean"))
    a["b"] = a["pred"] - a["rate"]
    a["se"] = np.sqrt(DEN / a["n"])
    a["t"] = a["b"] / a["se"]
    a["pt"] = 100000 * (a["n"]/N) * a["b"]**2 / DEN
    tot = a["pt"].sum()
    sig = a[a["t"].abs() > 2]
    w = a.reindex(a["pt"].sort_values(ascending=False).index).iloc[0]
    rows.append((nm, tot, float(sig["pt"].sum()), len(a), len(sig)))
    print(f"  {nm:10s} {tot:10.1f}   {a['pt'].idxmax():>10s} n={int(w['n']):>6,} "
          f"치우침 {w['b']:+.4f} t={w['t']:+5.2f} -> {w['pt']:.1f}점")
print(f"\n  {'축':10s} {'전체 상한':>10s} {'t>2 칸만':>10s}  {'칸수':>5s} {'t>2':>4s}")
for nm, tot, s, nk, ns in sorted(rows, key=lambda x: -x[2]):
    print(f"  {nm:10s} {tot:10.1f} {s:10.1f}  {nk:5d} {ns:4d}")

print("\n  === 월별 전체 (시즌 내 흐름) ===")
k = g.game_month.astype(int)
t = pd.DataFrame({"k": k, "y": y, "p": p}).groupby("k").agg(
    n=("y","size"), rate=("y","mean"), pred=("p","mean"))
t["b"] = t["pred"] - t["rate"]
t["t"] = t["b"] / np.sqrt(DEN/t["n"])
t["pt"] = 100000*(t["n"]/N)*t["b"]**2/DEN
print(f"  {'월':>4s} {'n':>8s} {'실제':>8s} {'예측':>8s} {'치우침':>9s} {'t':>6s} {'점수':>6s}")
for kk, rw in t.iterrows():
    print(f"  {kk:>4d} {int(rw['n']):>8,} {rw['rate']:>8.4f} {rw['pred']:>8.4f} "
          f"{rw['b']:>+9.4f} {rw['t']:>+6.2f} {rw['pt']:>6.1f}")
