# -*- coding: utf-8 -*-
"""train.csv 와 trackman 을 '투구 단위'로 붙일 수 있는지 확인한다.

왜 이걸 보나
    매핑 파일을 보니 채택된 443명의 state_match 가 0.999, ratio 가 0.939 다.
    투구 상태(이닝·카운트·아웃)가 거의 완전히 일치하고, 트랙맨이 train 투구의
    94% 를 담고 있다는 뜻이다. 두 파일이 같은 투구를 다르게 기록한 것이다.

    그렇다면 투수×시즌 평균 같은 뭉갠 집계 대신, 투구 하나하나에 물리량을
    붙일 수 있다. 지금까지 쓴 것은 '이 투수의 평균 구속' 이었는데,
    '이 투수가 이 카운트에서 어떤 공을 던지는가' 로 내려갈 수 있다.

주의 — 2025 에는 트랙맨이 없다
    트랙맨은 2019~2024 다. 평가 대상인 2025 행에는 물리량을 직접 못 붙인다.
    그래서 투구 단위 정합은 '피처를 직접 쓰는' 용도가 아니라
      - 어떤 물리량이 제구 성공과 실제로 관계있는지 투구 단위로 확인하고
      - 그걸 바탕으로 투수×상황 프로파일을 정확하게 만드는
    용도다. 무엇을 집계할지 정하는 근거가 된다.

여기서 확인하는 것
    1. 경기를 특정할 수 있는가 (train 에는 game_id 가 없다)
    2. 경기 안에서 투구를 특정할 수 있는가
    3. 붙인 결과가 얼마나 되는가
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before"]

if __name__ == "__main__":
    mp = pd.read_csv("trackman_map/pitcher_map.csv")
    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"), encoding="utf-8-sig",
                     usecols=["season", "game_month", "game_dayofweek", "inning",
                              "top_bottom", "balls_before", "strikes_before",
                              "outs_before", "pitch_of_pa", "pitcher_trackman_id",
                              "batter_trackman_id", "trackman_game_id",
                              "rel_speed", "spin_rate", "rel_height", "rel_side",
                              "pitch_type_group"])
    tm = tm.merge(mp, left_on="pitcher_trackman_id", right_on="trackman_id")
    # train 은 'T'/'B', 트랙맨은 'Top'/'Bottom' 이다. 앞 글자로 맞춘다.
    tm["top_bottom"] = tm["top_bottom"].astype(str).str[0].str.upper()
    print(f"train {len(tr):,}   trackman(매핑됨) {len(tm):,}")
    print(f"train 의 top_bottom 값: {sorted(tr.top_bottom.unique())}")
    print(f"trackman 변환 후: {sorted(tm.top_bottom.dropna().unique())}")

    # 매핑된 투수의 train 행만 대상으로
    sub = tr[tr.pitcher_id.isin(set(mp.pitcher_id))].copy()
    print(f"\n매핑된 투수의 train 행 {len(sub):,} ({len(sub)/len(tr)*100:.1f}%)")

    # 1) (투수, 시즌) 단위 투구수가 맞는가
    a = sub.groupby(["pitcher_id", "season"]).size().rename("train_n")
    b = tm.groupby(["pitcher_id", "season"]).size().rename("tm_n")
    j = pd.concat([a, b], axis=1).dropna()
    print(f"\n(투수,시즌) 쌍 {len(j):,}   트랙맨/train 투구수 비율")
    print(f"  중앙값 {(j.tm_n / j.train_n).median():.3f}   "
          f"1.0 인 쌍 {(np.abs(j.tm_n / j.train_n - 1) < 0.01).mean() * 100:.1f}%")

    # 2) 상태 조합 단위로 붙여 본다. 유일하게 대응되는 비율이 관건이다.
    k2 = ["pitcher_id"] + KEY + ["pitch_of_pa"]
    have = [c for c in k2 if c in sub.columns and c in tm.columns]
    print(f"\n양쪽에 다 있는 키: {have}")
    ca = sub.groupby(have).size().rename("n_train")
    cb = tm.groupby(have).size().rename("n_tm")
    jj = pd.concat([ca, cb], axis=1).fillna(0)
    uniq = (jj.n_train == 1) & (jj.n_tm == 1)
    print(f"  키 조합 {len(jj):,}")
    print(f"  양쪽 모두 유일한 조합 {uniq.sum():,}  "
          f"= train 행의 {jj.loc[uniq, 'n_train'].sum() / len(sub) * 100:.1f}%")
    print(f"  train 에만 있는 조합의 행 {jj.loc[jj.n_tm == 0, 'n_train'].sum():,}")
    print(f"  trackman 에만 있는 조합의 행 {jj.loc[jj.n_train == 0, 'n_tm'].sum():,.0f}")
    print("\n  유일 대응 비율이 높으면 투구 단위 정합이 가능하다.")
