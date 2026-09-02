# -*- coding: utf-8 -*-
"""키를 하나씩 더해가며 투구 단위 정합률이 어디까지 오르는지 잰다.

기본 9개 키만으로 49.3% 가 일대일 대응했다. 아직 안 쓴 키가 있다.
    양손 정보     pitcher_hand, batter_hand
    팀            pitcher_team, batter_team  (team_map.csv 로 코드 변환)
    타자          batter_trackman_id — 매핑이 없어서 못 쓴다

'일대일' 기준
    같은 키 값을 갖는 행이 train 에도 1개, trackman 에도 1개일 때만 센다.
    2개씩 있으면 어느 쪽이 어느 쪽인지 모르므로 붙이면 안 된다.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
BASE = ["pitcher_id", "season", "game_month", "game_dayofweek", "inning",
        "top_bottom", "balls_before", "strikes_before", "outs_before"]

if __name__ == "__main__":
    mp = pd.read_csv("trackman_map/pitcher_map.csv")
    team = pd.read_csv("trackman_map/team_map.csv")
    print("team_map 열:", list(team.columns), " 행", len(team))

    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"), encoding="utf-8-sig",
                     usecols=["season", "game_month", "game_dayofweek", "inning",
                              "top_bottom", "balls_before", "strikes_before",
                              "outs_before", "pitcher_trackman_id", "pitcher_hand",
                              "batter_hand", "pitcher_team", "batter_team"])
    tm = tm.merge(mp, left_on="pitcher_trackman_id", right_on="trackman_id")
    tm["top_bottom"] = tm["top_bottom"].astype(str).str[0].str.upper()

    print("\ntrain 손 표기:", sorted(tr.pitcher_hand.unique()),
          sorted(tr.batter_hand.unique()))
    print("trackman 손 표기:", sorted(tm.pitcher_hand.unique()),
          sorted(tm.batter_hand.unique()))

    # 손: trackman 은 Right/Left. train 이 정수면 대응을 데이터로 추정한다.
    for col in ("pitcher_hand", "batter_hand"):
        tm[col + "_c"] = tm[col].astype(str).str[0].str.upper()   # R / L
    # train 쪽 정수 -> R/L 대응을 빈도로 맞춘다 (좌투가 더 적다)
    for col in ("pitcher_hand", "batter_hand"):
        vc = tr[col].value_counts()
        big = vc.index[0]
        tr[col + "_c"] = np.where(tr[col] == big, "R", "L")
    print("추정 대응 후 train:", tr.pitcher_hand_c.value_counts().to_dict(),
          " trackman:", tm.pitcher_hand_c.value_counts().to_dict())

    tcol = [c for c in team.columns if "team" in c.lower()]
    print("\nteam_map 예시:\n", team.head(3).to_string())
    # team_map.csv 는 trackman_team -> train_team_id 두 열로 되어 있다.
    tmap = dict(zip(team["trackman_team"], team["train_team_id"]))
    tm["pitcher_team_id"] = tm["pitcher_team"].map(tmap)
    tm["batter_team_id"] = tm["batter_team"].map(tmap)
    print(f"팀 변환 실패율  투수팀 {tm.pitcher_team_id.isna().mean()*100:.1f}%  "
          f"타자팀 {tm.batter_team_id.isna().mean()*100:.1f}%")

    sub = tr[tr.pitcher_id.isin(set(mp.pitcher_id))].copy()
    print(f"\n대상 train 행 {len(sub):,} ({len(sub)/len(tr)*100:.1f}%)\n")

    steps = [
        ("기본 9키", BASE),
        ("+ 타자손", BASE + ["batter_hand_c"]),
        ("+ 투수팀", BASE + ["batter_hand_c", "pitcher_team_id"]),
        ("+ 타자팀", BASE + ["batter_hand_c", "pitcher_team_id", "batter_team_id"]),
    ]
    print(f"  {'키':22s} {'일대일 행':>12s} {'비율':>8s}  {'중복탈락':>10s}")
    for name, keys in steps:
        ok = [k for k in keys if k in sub.columns and k in tm.columns]
        if len(ok) != len(keys):
            print(f"  {name:22s} 건너뜀 (없는 키: {set(keys) - set(ok)})")
            continue
        a = sub.groupby(ok, dropna=False).size().rename("na")
        b = tm.groupby(ok, dropna=False).size().rename("nb")
        j = pd.concat([a, b], axis=1).fillna(0)
        u = (j.na == 1) & (j.nb == 1)
        n1 = int(j.loc[u, "na"].sum())
        dup = int(j.loc[(j.na > 0) & (j.nb > 0) & ~u, "na"].sum())
        print(f"  {name:22s} {n1:12,} {n1/len(sub)*100:7.1f}%  {dup:10,}")
    print("\n  '중복탈락' 은 양쪽에 다 있지만 여러 행이 같은 키를 가져")
    print("  어느 것과 짝인지 정할 수 없는 행이다. 키를 더하면 이쪽이 줄어든다.")
