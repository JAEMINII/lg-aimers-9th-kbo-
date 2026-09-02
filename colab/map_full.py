# -*- coding: utf-8 -*-
"""트랙맨을 train.csv 에 투구 단위로 붙인다. 매핑을 끝까지 밀어본다.

지금까지
    투수 매핑 443명 (train 행의 93.7%)
    기본 9키로 일대일 정합 49.3%, 타자손을 더해 60.6%
    팀을 키에 넣으면 오히려 떨어졌다 — 팀 코드 26개 중 11개만 매핑돼 있어서
    나머지 23.4% 행이 통째로 탈락한다.

여기서 두 가지를 한다
    1. 퓨처스 팀 12개를 매핑한다. MIN_DOO -> 12 처럼 접두사만 떼면 대응된다.
       올스타(KBO_ARM/KBO_POL)와 이벤트(ACE_MEX)는 train 에 없으니 버린다.
    2. 남는 중복을 '경기 내 순서' 로 푼다.
       train 은 row_id 가 시간순이다(asof_* 가 누적이라 검증됨).
       트랙맨은 pitch_no 가 경기 내 순서다.
       (투수, 경기) 묶음 안에서 양쪽을 순서대로 세워 위치끼리 맞춘다.
       상태가 같은 투구도 순서가 다르면 구별된다.

경기를 어떻게 특정하나
    train 에는 game_id 가 없다. (season, month, dayofweek, 투수팀, 타자팀) 으로
    묶으면 같은 달 같은 요일에 같은 카드가 여러 번 있을 수 있어 완전하지 않다.
    그래서 그 묶음 안에서 투구수가 양쪽 같을 때만 순서 정렬을 인정한다.
    수가 다르면 경기가 섞인 것이므로 붙이지 않는다.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
OUT = "colab/pitch_join.csv.gz"   # parquet 엔진이 없어 gzip csv 로 둔다
BASE = ["pitcher_id", "season", "game_month", "game_dayofweek", "inning",
        "top_bottom", "balls_before", "strikes_before", "outs_before"]
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]


def build_team_map():
    """1군 11개(기존) + 퓨처스 12개. MIN_XXX 는 모팀과 같은 id 로 보낸다."""
    base = pd.read_csv("trackman_map/team_map.csv")
    m = dict(zip(base["trackman_team"], base["train_team_id"]))
    # 퓨처스 코드는 모팀 약자를 담고 있다. 1군 코드의 앞 세 글자와 맞춘다.
    stem = {}
    for code, tid in m.items():
        stem[code.split("_")[0][:3]] = tid
    extra = {"MIN_DOO": "DOO", "MIN_HAN": "HAN", "MIN_HAW": None, "MIN_HER": "KIW",
             "MIN_KIA": "KIA", "MIN_KTW": "KT", "MIN_LGT": "LG", "MIN_LOT": "LOT",
             "MIN_NCD": "NC", "MIN_SAM": "SAM", "MIN_SKW": "SK", "MIN_SSG": "SSG"}
    for code, parent in extra.items():
        if parent is None:
            continue                      # MIN_HAW(고양 히어로즈 구명 등)는 뒤에서 처리
        key = parent[:3]
        if key in stem:
            m[code] = stem[key]
    # MIN_HAW 는 키움 2군(고양)이다. KIW 와 같은 곳으로 보낸다.
    if "KIW" in stem:
        m["MIN_HAW"] = stem["KIW"]
    return m


if __name__ == "__main__":
    mp = pd.read_csv("trackman_map/pitcher_map.csv")
    tmap = build_team_map()
    print(f"팀 매핑 {len(tmap)}개")
    for k in sorted(tmap):
        print(f"  {k:10s} -> {tmap[k]}")

    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"), encoding="utf-8-sig")
    tm = tm.merge(mp, left_on="pitcher_trackman_id", right_on="trackman_id")
    tm["top_bottom"] = tm["top_bottom"].astype(str).str[0].str.upper()
    for c in ("pitcher_hand", "batter_hand"):
        tm[c + "_c"] = tm[c].astype(str).str[0].str.upper()
    for c in ("pitcher_hand", "batter_hand"):
        big = tr[c].value_counts().index[0]
        tr[c + "_c"] = np.where(tr[c] == big, "R", "L")
    tm["pitcher_team_id"] = tm["pitcher_team"].map(tmap)
    tm["batter_team_id"] = tm["batter_team"].map(tmap)
    lost = tm.pitcher_team_id.isna() | tm.batter_team_id.isna()
    print(f"\n팀 변환 실패 {lost.mean() * 100:.1f}%  "
          f"(남은 코드: {sorted(set(tm.loc[lost, 'pitcher_team']) | set(tm.loc[lost, 'batter_team']))[:6]})")
    tm = tm[~lost].copy()
    for c in ("pitcher_team_id", "batter_team_id"):
        tm[c] = tm[c].astype("int64")

    sub = tr[tr.pitcher_id.isin(set(mp.pitcher_id))].copy()
    sub["_ord"] = sub["row_id"].astype(str).str.extract(r"(\d+)$").astype("int64")
    sub = sub.sort_values("_ord").reset_index(drop=True)
    tm = tm.sort_values(["season", "game_month", "game_dayofweek",
                         "trackman_game_id", "pitch_no"]).reset_index(drop=True)
    print(f"대상 train 행 {len(sub):,}   트랙맨 행 {len(tm):,}")

    KEYS = BASE + ["batter_hand_c", "pitcher_team_id", "batter_team_id"]

    # --- 1단계: 키가 양쪽에서 유일한 행
    a = sub.groupby(KEYS, dropna=False).size().rename("na")
    b = tm.groupby(KEYS, dropna=False).size().rename("nb")
    cnt = pd.concat([a, b], axis=1).fillna(0)
    uniq = cnt[(cnt.na == 1) & (cnt.nb == 1)].index
    print(f"\n1단계 유일키 정합  {len(uniq):,} 행 ({len(uniq) / len(sub) * 100:.1f}%)")

    # --- 2단계: 같은 키에 양쪽 개수가 같으면 순서로 짝짓는다
    same = cnt[(cnt.na == cnt.nb) & (cnt.na > 1)]
    print(f"2단계 대상(개수 일치, 2개 이상)  키 {len(same):,}  "
          f"행 {int(same.na.sum()):,} ({same.na.sum() / len(sub) * 100:.1f}%)")

    sub["_k"] = list(map(tuple, sub[KEYS].to_numpy()))
    tm["_k"] = list(map(tuple, tm[KEYS].to_numpy()))
    ok = set(uniq) | set(same.index)
    sa = sub[sub["_k"].isin(ok)].copy()
    tb = tm[tm["_k"].isin(ok)].copy()
    sa["_r"] = sa.groupby("_k").cumcount()
    tb["_r"] = tb.groupby("_k").cumcount()
    j = sa.merge(tb[["_k", "_r"] + PHYS + ["pitch_type_group", "pitch_of_pa"]],
                 on=["_k", "_r"], how="inner")
    print(f"\n최종 정합 {len(j):,} 행 = train 전체의 {len(j) / len(tr) * 100:.1f}% / "
          f"대상의 {len(j) / len(sub) * 100:.1f}%")
    print(f"  시즌별:", j.groupby('season').size().to_dict())
    j[["row_id", "control_success", "season"] + PHYS +
      ["pitch_type_group", "pitch_of_pa"]].to_csv(OUT, index=False, compression="gzip")
    print(f"저장 {OUT}  {os.path.getsize(OUT) / 1e6:.1f} MB")
