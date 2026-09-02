# -*- coding: utf-8 -*-
"""OPS 역산이 가능한가. 타석 경계와 결과를 복원할 수 있는지부터 본다.

문제
    train.csv 49열에 안타/볼넷/타수가 없다. 있는 건 상태열뿐이다 —
    이닝, 아웃, 주자(1/2/3루), 득점, 볼카운트, 타자 id.
    타석 결과는 **연속한 두 투구 사이의 상태 변화**로만 알 수 있다.

되려면 두 가지가 필요하다
    1. 투구가 시간 순으로 정렬돼 있어야 한다 (train.csv 에 pitch_no 가 없다)
    2. 타석 경계를 찾을 수 있어야 한다

여기서 확인
    row_id 순서가 곧 경기 내 투구 순서인가. 한 경기를 뽑아 이닝/아웃/카운트가
    앞으로만 진행하는지 본다. 뒤로 가면 정렬이 아니라는 뜻이고 역산은 못 한다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
KEY = ["season", "game_month", "game_dayofweek", "pitcher_team_id",
       "batter_team_id"]

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["row_id", "game_type", "inning", "top_bottom",
                          "outs_before", "balls_before", "strikes_before",
                          "batter_id", "num_runners_on", "run_total_before"]
                 + KEY)
print(f"  {len(tr):,}행")
print(f"  row_id 예시 {tr['row_id'].iloc[:3].tolist()}")
suf = tr["row_id"].astype(str).str.extract(r"(\d+)$", expand=False).astype(np.int64)
print(f"  row_id 접미 숫자가 단조증가 {bool((np.diff(suf) > 0).all())}")

tr = tr[tr.game_type == "R"].copy()
tr["_k"] = list(zip(*[tr[c] for c in KEY]))
sz = tr.groupby("_k").size()
k = sz[(sz > 200) & (sz < 320)].index[0]
g = tr[tr["_k"] == k].copy()
print(f"\n  표본 경기 {k}  투구 {len(g)}개")

# 한 경기 안에서 이닝/하프이닝이 앞으로만 가는가
half = g["inning"] * 2 + (g["top_bottom"].astype(str).str[:1].str.upper() == "B")
print(f"  하프이닝이 뒤로 가지 않는다  {bool((np.diff(half.to_numpy()) >= 0).all())}")
print(f"  하프이닝 {half.nunique()}개  (정상 경기면 18 근처)")

# 타석 경계 — 타자가 바뀌거나 카운트가 되감기면 새 타석
b = g["batter_id"].to_numpy()
bb = g["balls_before"].to_numpy()
ss = g["strikes_before"].to_numpy()
new = np.r_[True, (b[1:] != b[:-1]) | (bb[1:] < bb[:-1]) | (ss[1:] < ss[:-1])]
pa = np.cumsum(new)
print(f"  추정 타석 {pa.max()}개   투구/타석 {len(g)/pa.max():.2f}"
      f"   (실제 KBO 는 3.8~3.9)")

# 타석 안에서 카운트가 0-0 에서 시작하고 한 칸씩 오르는가
ok_start = int(sum(1 for i in np.where(new)[0] if bb[i] == 0 and ss[i] == 0))
step = 0
tot = 0
for i in range(1, len(g)):
    if new[i]:
        continue
    tot += 1
    if (bb[i] - bb[i-1]) + (ss[i] - ss[i-1]) == 1 and bb[i] >= bb[i-1] \
            and ss[i] >= ss[i-1]:
        step += 1
print(f"  0-0 으로 시작한 타석 {ok_start}/{pa.max()} ({ok_start/pa.max()*100:.0f}%)")
print(f"  타석 내 카운트가 정확히 1 오른 투구 {step}/{tot} "
      f"({step/max(tot,1)*100:.0f}%)")

print("\n  판정")
if ok_start / pa.max() > 0.9 and step / max(tot, 1) > 0.9:
    print("  투구 순서가 살아 있다. 타석 경계와 볼/스트라이크 결과를 복원할 수 있다.")
    print("  다음 단계는 타석 **결과**(안타/볼넷/아웃) 를 주자·아웃·득점 변화로")
    print("  역산하는 것이다. 볼넷은 4번째 볼로 특정되지만, 안타 종류(1/2/3루타)")
    print("  는 주자 진루로 추정해야 해서 오차가 남는다.")
else:
    print("  **순서가 안 산다.** row_id 가 시간 순이 아니거나 투구가 섞여 있다.")
    print("  타석 경계를 못 잡으면 OPS 역산은 불가능하다.")
