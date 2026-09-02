# -*- coding: utf-8 -*-
"""타자 매핑을 만든다. trackman_map 에 투수만 있고 타자는 아무도 안 했다.

왜 되는가
    투수 매핑 README 에 이렇게 적혀 있다.
        경기 키 = (season, game_month, game_dayofweek, pitcher_team_id, batter_team_id)
                  trackman 10,981 키 중 10,095개(91.9%)가 정확히 1경기
        상태열  = (inning, top_bottom, balls_before, strikes_before, outs_before)
                  매칭된 투수의 상태열 일치율 **0.9956** (무작위 대조군 0.2918)

    즉 (경기 키 + 투수 + 상태열) 이면 특정 투구가 거의 확정된다. 그 투구의
    batter_trackman_id 를 읽어 train 의 batter_id 와 짝지으면 타자 매핑이 나온다.
    투수 매핑을 만든 사람이 왜 안 했는지는 모르겠지만 안 돼 있다.

왜 하는가
    44열의 타자 정보가 3개뿐이다 (asof_batter_n / success_rate / middle_rate).
    투수는 16개다. 5배 비대칭이다.

    트랙맨은 투구 물리량이라 **타자별로 집계하면 '이 타자에게 투수들이 어떻게
    접근하는가'** 가 나온다 — 구종 배합, 평균 구속, 무브먼트, 존 공략.
    승부를 피하는 타자, 변화구를 많이 보는 타자는 투수의 제구 성공률에 직접
    걸린다. 44열에 전혀 없는 축이다.

절차
    1  팀 매핑을 읽는다 (team_map.csv, 이미 있다)
    2  trackman 에서 단일 경기로 특정되는 키만 남긴다
    3  그 키 안에서 (투수, 상태열) 로 train 행과 trackman 투구를 맞춘다
    4  맞은 투구의 batter_trackman_id 를 train batter_id 별로 모아 다수결
    5  독립 검증 — 타자 좌우(batter_hand)가 일치하는지. 매칭에 안 쓴 열이다

채택 기준 (투수 매핑과 같은 규율)
    표  >= 100       충분한 투구에서 관찰
    1위 비율 >= 0.7  2위와 확실히 갈림
    hand 일치        독립 검증 통과
    trackman_id 중복 아님
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
MAPDIR = os.path.join(ROOT, "trackman_map")
OUT = os.path.join(MAPDIR, "batter_map.csv")
MIN_N, MIN_SHARE = 100, 0.70

KEY = ["season", "game_month", "game_dayofweek", "pt", "bt"]
STATE = ["inning", "top_bottom", "balls_before", "strikes_before", "outs_before"]

team = pd.read_csv(os.path.join(MAPDIR, "team_map.csv"), encoding="utf-8-sig")
tmap = dict(zip(team["trackman_team"], team["train_team_id"]))
pm = pd.read_csv(os.path.join(MAPDIR, "pitcher_map.csv"), encoding="utf-8-sig")
p2t = dict(zip(pm["pitcher_id"], pm["trackman_id"]))
print(f"  팀 매핑 {len(tmap)}개   투수 매핑 {len(p2t)}명")

tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek",
                          "pitcher_team", "batter_team", "trackman_game_id",
                          "pitcher_trackman_id", "batter_trackman_id",
                          "batter_hand"] + STATE)
# 표기 정규화 — train 은 'B'/'T' 와 1/2, trackman 은 'Bottom'/'Top' 와 'Left'/'Right'
tm["top_bottom"] = tm["top_bottom"].astype(str).str[:1].str.upper()
tm["batter_hand"] = tm["batter_hand"].astype(str).str[:1].str.upper().map(
    {"L": 1, "R": 2})
tm["pt"] = tm["pitcher_team"].map(tmap)
tm["bt"] = tm["batter_team"].map(tmap)
tm = tm.dropna(subset=["pt", "bt"]).copy()
tm["pt"] = tm["pt"].astype(int)
tm["bt"] = tm["bt"].astype(int)
print(f"  trackman 1군 매핑 가능 {len(tm):,}행")

# 2단계 — 단일 경기로 특정되는 키만
g = tm.groupby(KEY)["trackman_game_id"].nunique()
uniq = set(g[g == 1].index)
tm["_k"] = list(zip(*[tm[c] for c in KEY]))
tm = tm[tm["_k"].isin(uniq)].copy()
print(f"  단일경기 키 {len(uniq):,}개 -> trackman {len(tm):,}행")

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek", "game_type",
                          "pitcher_team_id", "batter_team_id", "pitcher_id",
                          "batter_id", "batter_hand"] + STATE)
tr = tr[tr["game_type"] == "R"].copy()
tr["top_bottom"] = tr["top_bottom"].astype(str).str[:1].str.upper()
tr["pt"] = tr["pitcher_team_id"]
tr["bt"] = tr["batter_team_id"]
tr["ptm"] = tr["pitcher_id"].map(p2t)
tr = tr.dropna(subset=["ptm"]).copy()
tr["ptm"] = tr["ptm"].astype(int)
tr["_k"] = list(zip(*[tr[c] for c in KEY]))
tr = tr[tr["_k"].isin(uniq)].copy()
print(f"  train 1군 · 투수매핑 있음 · 단일경기 키  {len(tr):,}행")

# 3단계 — (키, 투수, 상태열) 로 조인
jk = ["_k", "ptm"] + STATE
L = tr[jk + ["batter_id", "batter_hand"]].rename(
    columns={"batter_hand": "bh_tr"})
R = tm[["_k", "pitcher_trackman_id"] + STATE +
       ["batter_trackman_id", "batter_hand"]].rename(
    columns={"pitcher_trackman_id": "ptm", "batter_hand": "bh_tm"})
# 상태열이 같은 투구가 한 경기에 여럿일 수 있다 -> 유일한 것만 쓴다
R2 = R.drop_duplicates(subset=jk, keep=False)
L2 = L.drop_duplicates(subset=jk, keep=False)
J = L2.merge(R2, on=jk, how="inner")
print(f"  유일하게 맞은 투구 {len(J):,}쌍 "
      f"({len(J)/max(len(tr),1)*100:.1f}% of train 후보)")

# 4단계 — 타자별 다수결
cnt = J.groupby(["batter_id", "batter_trackman_id"]).size().rename("n")
tot = cnt.groupby(level=0).sum().rename("tot")
top = cnt.groupby(level=0).idxmax()
rows = []
for bid, idx in top.items():
    n = int(cnt.loc[idx])
    t = int(tot.loc[bid])
    sub = J[(J.batter_id == bid) & (J.batter_trackman_id == idx[1])]
    hand_ok = float((sub["bh_tr"].astype(str).str[:1].str.upper()
                     == sub["bh_tm"].astype(str).str[:1].str.upper()).mean()) \
        if len(sub) else np.nan
    rows.append((bid, int(idx[1]), n, t, n / max(t, 1), hand_ok))
res = pd.DataFrame(rows, columns=["batter_id", "trackman_id", "n_top",
                                  "n_total", "share", "hand_match"])
print(f"\n  후보가 있는 타자 {len(res):,}명 / train 830명")
print(f"  표본 분위 {res.n_total.quantile([0, .25, .5, .75, 1]).astype(int).tolist()}")
print(f"  1위 비율 분위 {res.share.quantile([0, .25, .5, .75, 1]).round(3).tolist()}")
print(f"  hand 일치 중앙 {res.hand_match.median():.4f}   "
      f"(무작위면 0.5 근처여야 한다)")

ok = res[(res.n_total >= MIN_N) & (res.share >= MIN_SHARE)
         & (res.hand_match >= 0.95)].copy()
ok = ok.sort_values("n_top", ascending=False).drop_duplicates("trackman_id")
print(f"\n  채택 {len(ok):,}명   기준: 표본>={MIN_N}, 1위비율>={MIN_SHARE}, "
      f"hand>=0.95, trackman_id 중복 제거")
print(f"  채택자 1위비율 중앙 {ok.share.median():.4f}   "
       f"hand 일치 중앙 {ok.hand_match.median():.4f}")
ok[["batter_id", "trackman_id"]].to_csv(OUT, index=False, encoding="utf-8-sig")
res.to_csv(os.path.join(MAPDIR, "batter_map_full.csv"), index=False,
           encoding="utf-8-sig")
print(f"\n  저장  {OUT}")
print("  hand 일치가 0.99 근처면 매핑이 맞다 — 매칭에 안 쓴 열로 검증한 것이다.")
