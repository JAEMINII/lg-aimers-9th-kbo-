# -*- coding: utf-8 -*-
"""트랙맨을 투수x시즌 집계표로 만든다. 353MB 를 서버로 옮기지 않기 위함.

왜 트랙맨인가
    지금 44열은 전부 train.csv 안에서 나온 것이다. 오늘 잰 후보들(runners, n_cols)이
    효과가 없었던 것도 이미 있는 정보를 재배열한 것이기 때문이다.
    트랙맨은 구속·회전수·무브먼트·릴리스포인트 — 44열 어디에도 없는 물리량이다.

누출 위험이 구조적으로 없다
    트랙맨에는 control_success 가 없다. 순수 공변량이라 bxh/pxc 를 망친
    '자기 정답이 자기 피처에 들어가는' 문제가 일어날 수 없다.
    다만 시간 누출은 막아야 하므로 집계를 시즌별로 쪼개 둔다.
    쓰는 쪽에서 '해당 시즌보다 이전' 만 누적해 쓴다.

피처 선정 근거
    제구(control)를 맞히는 과제다. 그래서 구속·회전수 같은 '구위' 뿐 아니라
    릴리스포인트의 흔들림(rel_height/rel_side 의 표준편차)을 넣는다. 릴리스가
    일정한 투수가 커맨드가 좋다는 건 야구에서 표준적인 관찰이다.
    패스트볼과 오프스피드의 구속차(터널링/기만)도 넣는다.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
COLS = ["season", "pitcher_trackman_id", "pitch_type_group",
        "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
MEANS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
         "extension", "rel_height", "rel_side", "zone_speed"]

if __name__ == "__main__":
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"),
                     encoding="utf-8-sig", usecols=COLS)
    mp = pd.read_csv("trackman_map/pitcher_map.csv")
    tm = tm.merge(mp, left_on="pitcher_trackman_id", right_on="trackman_id",
                  how="inner")
    print(f"매핑 후 {len(tm):,}행  투수 {tm.pitcher_id.nunique()}명")

    g = tm.groupby(["pitcher_id", "season"], sort=False)
    out = g[MEANS].mean()
    out.columns = ["tm_" + c for c in out.columns]
    # 릴리스 흔들림 — 제구 과제에서 구위보다 직접적일 수 있다
    out["tm_relh_sd"] = g["rel_height"].std()
    out["tm_rels_sd"] = g["rel_side"].std()
    out["tm_n"] = g.size()

    # 구종군별 구속. 패스트볼 기준선과 오프스피드와의 격차
    for grp in ("fastball", "offspeed"):
        s = (tm[tm.pitch_type_group == grp]
             .groupby(["pitcher_id", "season"])["rel_speed"].mean())
        out[f"tm_{grp}_speed"] = s
    out["tm_speed_gap"] = out["tm_fastball_speed"] - out["tm_offspeed_speed"]
    out = out.drop(columns=["tm_offspeed_speed"])
    out = out.reset_index()

    print(f"집계표 {out.shape}   피처 {len([c for c in out.columns if c.startswith('tm_')])}개")
    print(out.columns.tolist())
    print(out.head(3).to_string())
    print("\n결측률")
    for c in out.columns:
        if c.startswith("tm_"):
            print(f"  {c:20s} {out[c].isna().mean()*100:5.1f}%")
    out.to_csv("colab/trackman_pitcher_season.csv", index=False)
    print(f"\n저장 colab/trackman_pitcher_season.csv  "
          f"{os.path.getsize('colab/trackman_pitcher_season.csv')/1e6:.2f} MB")
