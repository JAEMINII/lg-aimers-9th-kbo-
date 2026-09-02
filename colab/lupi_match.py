# -*- coding: utf-8 -*-
"""train 행 <-> 현재 투구의 트랙맨 물리량을 행 단위로 잇는다 (LUPI 교사용).

batter_map.py 가 검증한 조인법 그대로 (상태열 일치율 0.9956):
    경기키 = (season, game_month, game_dayofweek, 투수팀, 타자팀)  91.9% 단일경기
    + (투수tm, 타자tm, inning, top_bottom, balls, strikes, outs)
    양쪽에서 유일한 조합만 남긴다 (같은 PA 안에서 파울로 반복된 카운트는 버려짐)

여기서 만드는 것
    lupi_match.csv.gz — row_id + 현재 투구의 물리량 9열 + 구종군
    교사는 이걸로 학습하고, 학생(44열)에게 증류한다. 추론에는 절대 안 쓴다.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "open (1)", "data")
MAPDIR = os.path.join(ROOT, "trackman_map")
OUT = os.path.join(ROOT, "colab", "_dl", "lupi_match.csv.gz")
STATE = ["inning", "top_bottom", "balls_before", "strikes_before", "outs_before"]
KEY = ["season", "game_month", "game_dayofweek", "pt", "bt"]
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]

team = pd.read_csv(os.path.join(MAPDIR, "team_map.csv"), encoding="utf-8-sig")
tmap = dict(zip(team["trackman_team"], team["train_team_id"]))
pm = pd.read_csv(os.path.join(MAPDIR, "pitcher_map.csv"), encoding="utf-8-sig")
bm = pd.read_csv(os.path.join(MAPDIR, "batter_map.csv"), encoding="utf-8-sig")
p2t = dict(zip(pm["pitcher_id"], pm["trackman_id"]))
b2t = dict(zip(bm["batter_id"], bm["trackman_id"]))
print(f"  매핑: 투수 {len(p2t)}  타자 {len(b2t)}  팀 {len(tmap)}")

tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek",
                          "pitcher_team", "batter_team", "trackman_game_id",
                          "pitcher_trackman_id", "batter_trackman_id",
                          "pitch_type_group"] + STATE + PHYS)
tm["top_bottom"] = tm["top_bottom"].astype(str).str[:1].str.upper()
tm["pt"] = tm["pitcher_team"].map(tmap)
tm["bt"] = tm["batter_team"].map(tmap)
tm = tm.dropna(subset=["pt", "bt"]).copy()
tm["pt"] = tm["pt"].astype(int)
tm["bt"] = tm["bt"].astype(int)
g = tm.groupby(KEY)["trackman_game_id"].nunique()
uniq = set(g[g == 1].index)
tm["_k"] = list(zip(*[tm[c] for c in KEY]))
tm = tm[tm["_k"].isin(uniq)].copy()
print(f"  trackman 단일경기 {len(tm):,}행")

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["row_id", "season", "game_month", "game_dayofweek",
                          "game_type", "pitcher_team_id", "batter_team_id",
                          "pitcher_id", "batter_id"] + STATE)
tr = tr[tr["game_type"] == "R"].copy()
tr["top_bottom"] = tr["top_bottom"].astype(str).str[:1].str.upper()
tr["pt"] = tr["pitcher_team_id"]
tr["bt"] = tr["batter_team_id"]
tr["ptm"] = tr["pitcher_id"].map(p2t)
tr["btm"] = tr["batter_id"].map(b2t)
tr = tr.dropna(subset=["ptm", "btm"]).copy()
tr["ptm"] = tr["ptm"].astype(int)
tr["btm"] = tr["btm"].astype(int)
tr["_k"] = list(zip(*[tr[c] for c in KEY]))
tr = tr[tr["_k"].isin(uniq)].copy()
print(f"  train 후보 {len(tr):,}행 (1군, 두 매핑, 단일경기)")

jk = ["_k", "ptm", "btm"] + STATE
R = tm.rename(columns={"pitcher_trackman_id": "ptm",
                       "batter_trackman_id": "btm"})
R2 = R.drop_duplicates(subset=jk, keep=False)
L2 = tr.drop_duplicates(subset=jk, keep=False)
J = L2[jk + ["row_id"]].merge(
    R2[jk + ["pitch_type_group"] + PHYS], on=jk, how="inner")
n_tr_1g = 1314088
print(f"  유일 매칭 {len(J):,}쌍  (후보의 {len(J)/max(len(tr),1)*100:.1f}%, "
      f"1군 전체의 {len(J)/n_tr_1g*100:.1f}%)")
keep = ["row_id", "pitch_type_group"] + PHYS
J[keep].to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"  저장 {OUT}")
print(f"  물리량 결측률: " + "  ".join(
    f"{c}:{J[c].isna().mean()*100:.1f}%" for c in PHYS[:4]))
print(f"  구종군 분포:\n{J['pitch_type_group'].value_counts().head(6).to_string()}")
