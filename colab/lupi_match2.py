# -*- coding: utf-8 -*-
"""행 매칭 v2 — v1 이 버린 것을 구제한다.

v1 (lupi_match.py): 835,713쌍, 1군의 63.6%. 버린 곳 셋:
    ① 같은 (경기,투수,타자,상태) 중복 -> keep=False 로 전부 폐기 (~13.8%)
       파울로 같은 카운트가 반복되면 생긴다. **순서로 구제 가능** —
       train 은 행 순서(=투구 순서), 트랙맨은 pitch_no 로 n번째끼리 짝지으면 된다.
       단 양쪽 중복 수가 같을 때만 (다르면 어긋날 수 있어 버린다).
    ② 미매핑 투수 (443/792) / 타자 (390/831)
       부트스트랩 한 바퀴: 이미 매핑된 타자+상태로 투수를 투표하고, 그 반대도.
       채택 기준: 손 일치>=0.95, 1위 비율>=0.85, 표본>=30.
    ③ 복수경기 키 (8.1%) — 이번엔 건드리지 않는다.

품질 검증은 매칭에 안 쓴 열로: batter_hand 일치율 (v1 방식 그대로).
산출: colab/_dl/lupi_match2.csv.gz (row_id + 물리량 8 + 구종군)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "open (1)", "data")
MAPDIR = os.path.join(ROOT, "trackman_map")
DL = os.path.join(ROOT, "colab", "_dl")
STATE = ["inning", "top_bottom", "balls_before", "strikes_before", "outs_before"]
KEY = ["season", "game_month", "game_dayofweek", "pt", "bt"]
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
N_1GUN = 1314088

team = pd.read_csv(os.path.join(MAPDIR, "team_map.csv"), encoding="utf-8-sig")
tmap = dict(zip(team["trackman_team"], team["train_team_id"]))
pm = pd.read_csv(os.path.join(MAPDIR, "pitcher_map.csv"), encoding="utf-8-sig")
bm = pd.read_csv(os.path.join(MAPDIR, "batter_map.csv"), encoding="utf-8-sig")
p2t = dict(zip(pm["pitcher_id"], pm["trackman_id"]))
b2t = dict(zip(bm["batter_id"], bm["trackman_id"]))

tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["season", "game_month", "game_dayofweek",
                          "pitcher_team", "batter_team", "trackman_game_id",
                          "pitch_no", "pitcher_trackman_id",
                          "batter_trackman_id", "batter_hand", "pitcher_hand",
                          "pitch_type_group"] + STATE + PHYS)
tm["top_bottom"] = tm["top_bottom"].astype(str).str[:1].str.upper()
tm["bh"] = tm["batter_hand"].astype(str).str[:1].str.upper().map({"L": 1, "R": 2})
tm["ph"] = tm["pitcher_hand"].astype(str).str[:1].str.upper().map({"L": 1, "R": 2})
tm["pt"] = tm["pitcher_team"].map(tmap)
tm["bt"] = tm["batter_team"].map(tmap)
tm = tm.dropna(subset=["pt", "bt"]).copy()
tm["pt"] = tm["pt"].astype(int)
tm["bt"] = tm["bt"].astype(int)
g = tm.groupby(KEY)["trackman_game_id"].nunique()
uniq = set(g[g == 1].index)
tm["_k"] = list(zip(*[tm[c] for c in KEY]))
tm = tm[tm["_k"].isin(uniq)].copy()

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["row_id", "season", "game_month", "game_dayofweek",
                          "game_type", "pitcher_team_id", "batter_team_id",
                          "pitcher_id", "batter_id", "batter_hand",
                          "pitcher_hand"] + STATE)
tr = tr[tr["game_type"] == "R"].copy()
tr["_ord"] = np.arange(len(tr))
tr["top_bottom"] = tr["top_bottom"].astype(str).str[:1].str.upper()
tr["pt"] = tr["pitcher_team_id"]
tr["bt"] = tr["batter_team_id"]
tr["_k"] = list(zip(*[tr[c] for c in KEY]))
tr = tr[tr["_k"].isin(uniq)].copy()
maj = tr["pitcher_hand"].value_counts().idxmax()   # train 손 코드 확인용
print(f"  trackman 단일경기 {len(tm):,}   train 후보풀 {len(tr):,}")


def vote_expand(tr, tm, ent_tr, ent_tm, known_tr2tm, other_tr, other_tm,
                other_map, hand_tr, hand_tm, min_n=30, min_share=0.85):
    """이미 매핑된 '상대편' + 상태로 미매핑 개체를 투표한다."""
    L = tr[~tr[ent_tr].isin(known_tr2tm)].copy()
    L["otm"] = L[other_tr].map(other_map)
    L = L.dropna(subset=["otm"])
    L["otm"] = L["otm"].astype(int)
    jk = ["_k", "otm"] + STATE
    R = tm.rename(columns={other_tm: "otm"})
    L2 = L.drop_duplicates(subset=jk, keep=False)
    R2 = R.drop_duplicates(subset=jk, keep=False)
    J = L2[jk + [ent_tr, hand_tr]].merge(
        R2[jk + [ent_tm, hand_tm]], on=jk, how="inner")
    if not len(J):
        return {}
    cnt = J.groupby([ent_tr, ent_tm]).size().rename("n").reset_index()
    tot = cnt.groupby(ent_tr)["n"].sum().rename("tot")
    cnt = cnt.merge(tot, on=ent_tr)
    cnt["share"] = cnt["n"] / cnt["tot"]
    top = cnt.sort_values("n", ascending=False).drop_duplicates(ent_tr)
    out = {}
    used = set()
    for _, r in top.sort_values("n", ascending=False).iterrows():
        if r["tot"] < min_n or r["share"] < min_share:
            continue
        if r[ent_tm] in used:
            continue
        sub = J[(J[ent_tr] == r[ent_tr]) & (J[ent_tm] == r[ent_tm])]
        # 손 일치 — 매칭에 안 쓴 독립 검증
        a = sub[hand_tr].to_numpy()
        b = sub[hand_tm].to_numpy()
        if np.mean(a == b) < 0.95:
            continue
        out[r[ent_tr]] = int(r[ent_tm])
        used.add(r[ent_tm])
    return out


# ---- 부트스트랩 1바퀴: 투수 확장(타자 기준) -> 타자 확장(투수 기준)
tr["hand_p"] = np.where(tr["pitcher_hand"] == maj, 2, 1)   # R 다수 가정 검증 아래서
# train pitcher_hand 코드가 1/2 인지 문자인지 자동 감지
if tr["pitcher_hand"].dtype.kind in "iuf":
    tr["hand_p"] = tr["pitcher_hand"].astype(int)
    tr["hand_b"] = tr["batter_hand"].astype(int)
else:
    tr["hand_b"] = tr["batter_hand"]

new_p = vote_expand(tr, tm, "pitcher_id", "pitcher_trackman_id", p2t,
                    "batter_id", "batter_trackman_id", b2t,
                    "hand_p", "ph")
p2t.update(new_p)
new_b = vote_expand(tr, tm, "batter_id", "batter_trackman_id", b2t,
                    "pitcher_id", "pitcher_trackman_id", p2t,
                    "hand_b", "bh")
b2t.update(new_b)
print(f"  부트스트랩 확장: 투수 +{len(new_p)} (총 {len(p2t)})   "
      f"타자 +{len(new_b)} (총 {len(b2t)})")

# ---- 본 매칭: 순서 기반 중복 구제
tr["ptm"] = tr["pitcher_id"].map(p2t)
tr["btm"] = tr["batter_id"].map(b2t)
L = tr.dropna(subset=["ptm", "btm"]).copy()
L["ptm"] = L["ptm"].astype(int)
L["btm"] = L["btm"].astype(int)
jk = ["_k", "ptm", "btm"] + STATE
R = tm.rename(columns={"pitcher_trackman_id": "ptm",
                       "batter_trackman_id": "btm"})
L = L.sort_values("_ord")
R = R.sort_values("pitch_no")
L["occ"] = L.groupby(jk).cumcount()
R["occ"] = R.groupby(jk).cumcount()
nL = L.groupby(jk).size().rename("nL")
nR = R.groupby(jk).size().rename("nR")
sz = pd.concat([nL, nR], axis=1)
same = sz[(sz["nL"] == sz["nR"])].index
L2 = L.set_index(jk).loc[L.set_index(jk).index.isin(same)].reset_index()
R2 = R.set_index(jk).loc[R.set_index(jk).index.isin(same)].reset_index()
J = L2[jk + ["occ", "row_id", "hand_b"]].merge(
    R2[jk + ["occ", "bh", "pitch_type_group"] + PHYS], on=jk + ["occ"],
    how="inner")
agree = float((J["hand_b"].astype(str).str[:1] == J["bh"].astype(str).str[:1]
               ).mean()) if J["hand_b"].dtype.kind not in "iuf" else \
    float((J["hand_b"] == J["bh"]).mean())
print(f"  v2 매칭 {len(J):,}쌍  (1군의 {len(J)/N_1GUN*100:.1f}%)   "
      f"타자손 일치 {agree:.4f}")
dup_rescued = int((sz[sz['nL'] == sz['nR']]['nL'] > 1).sum())
print(f"  중복 상태키 구제 셀 {dup_rescued:,}개")
keep = ["row_id", "pitch_type_group"] + PHYS
J[keep].to_csv(os.path.join(DL, "lupi_match2.csv.gz"), index=False,
               encoding="utf-8-sig")
pd.DataFrame({"pitcher_id": list(p2t), "trackman_id": list(p2t.values())}) \
    .to_csv(os.path.join(MAPDIR, "pitcher_map_v2.csv"), index=False,
            encoding="utf-8-sig")
pd.DataFrame({"batter_id": list(b2t), "trackman_id": list(b2t.values())}) \
    .to_csv(os.path.join(MAPDIR, "batter_map_v2.csv"), index=False,
            encoding="utf-8-sig")
print(f"  저장: lupi_match2.csv.gz, pitcher_map_v2.csv, batter_map_v2.csv")
print(f"  v1 대비 {len(J) - 835713:+,}쌍")
