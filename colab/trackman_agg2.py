# -*- coding: utf-8 -*-
"""트랙맨 집계 v2 — 2스트라이크 국면을 따로 뽑는다.

왜 2스트라이크인가
    오차 분석에서 TabM 이 유독 그 국면에서 무너진다.
        스트라이크 0  BSS  896.8
        스트라이크 1      1062.0
        스트라이크 2       622.6    <- 오차의 28.9%
        0-2 카운트         239.4    <- 최악
    지금 44열에 카운트는 balls_before / strikes_before 두 숫자뿐이다. 투수가
    2스트라이크에서 접근을 바꾸는지 — 유인구로 가는지 계속 존을 공략하는지 —
    를 나타내는 값이 하나도 없다.

    트랙맨엔 투구마다 balls_before/strikes_before 가 있으므로 투수별로
    '2스트라이크일 때의 구종 배합과 구속 변화' 를 만들 수 있다.
    plate_x/plate_z 는 없어서 존 공략률은 못 만든다. 배합과 구속으로 대신한다.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
BASE = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
COLS = ["season", "pitcher_trackman_id", "pitch_type_group", "strikes_before",
        "balls_before"] + BASE

if __name__ == "__main__":
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"),
                     encoding="utf-8-sig", usecols=COLS)
    mp = pd.read_csv("trackman_map/pitcher_map.csv")
    tm = tm.merge(mp, left_on="pitcher_trackman_id", right_on="trackman_id",
                  how="inner")
    key = ["pitcher_id", "season"]
    g = tm.groupby(key, sort=False)

    out = g[BASE].mean()
    out.columns = ["tm_" + c for c in out.columns]
    out["tm_relh_sd"] = g["rel_height"].std()      # 릴리스 일관성 = 커맨드 대리지표
    out["tm_rels_sd"] = g["rel_side"].std()
    out["tm_n"] = g.size()
    for grp in ("fastball", "offspeed"):
        out[f"tm_{grp}_speed"] = (tm[tm.pitch_type_group == grp]
                                  .groupby(key)["rel_speed"].mean())
    out["tm_speed_gap"] = out["tm_fastball_speed"] - out["tm_offspeed_speed"]
    out = out.drop(columns=["tm_offspeed_speed"])

    # ---- 2스트라이크 전용
    two = tm[tm.strikes_before == 2]
    g2 = two.groupby(key, sort=False)
    n2 = g2.size()
    out["tm2s_n"] = n2
    mix2 = (two.groupby(key + ["pitch_type_group"]).size()
            .unstack(fill_value=0))
    mix2 = mix2.div(mix2.sum(1), axis=0)
    for c in ("fastball", "breaking", "offspeed"):
        if c in mix2.columns:
            out[f"tm2s_{c}_rate"] = mix2[c]
    # 전체 대비 배합 변화 — '2스트라이크에서 접근을 바꾸는가'
    mixall = (tm.groupby(key + ["pitch_type_group"]).size().unstack(fill_value=0))
    mixall = mixall.div(mixall.sum(1), axis=0)
    for c in ("fastball", "breaking"):
        if c in mix2.columns and c in mixall.columns:
            out[f"tm2s_{c}_shift"] = mix2[c] - mixall[c]
    out["tm2s_speed_delta"] = g2["rel_speed"].mean() - out["tm_rel_speed"]
    out["tm2s_relh_sd"] = g2["rel_height"].std()

    out = out.reset_index()
    feats = [c for c in out.columns if c.startswith("tm")]
    print(f"집계표 {out.shape}   피처 {len(feats)}개")
    print("  전체국면:", [c for c in feats if not c.startswith("tm2s")])
    print("  2스트라이크:", [c for c in feats if c.startswith("tm2s")])
    print("\n결측률")
    for c in feats:
        print(f"  {c:22s} {out[c].isna().mean()*100:5.1f}%")
    out.to_csv("colab/trackman_pitcher_season.csv", index=False)
    print(f"\n저장  {os.path.getsize('colab/trackman_pitcher_season.csv')/1e6:.2f} MB")
