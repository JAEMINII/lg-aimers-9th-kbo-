# -*- coding: utf-8 -*-
"""남은 미정합 행이 왜 남았는지 보고, 붙일 수 있는 만큼 더 붙인다.

현재 82.1% (121만 행). 남은 것의 성격을 먼저 나눈다.
    A. 투수 자체가 미매핑        약 6.3%  — 매핑을 늘려야 함
    B. train 에만 있는 키        키가 트랙맨에 없음. 트랙맨이 그 경기를 안 담았거나
                                 팀 변환에서 빠졌거나
    C. 개수 불일치               같은 키에 train 3개 / 트랙맨 5개 같은 경우.
                                 지금은 통째로 버리는데, 순서 정렬로 일부는 붙는다.

C 를 푸는 방법
    지금은 '개수가 같을 때만' 순서 정렬을 인정했다. 안전하지만 보수적이다.
    개수가 달라도, 트랙맨 쪽이 더 많은 건 train 이 일부 투구를 걸러냈기 때문일 수
    있다(train 이 147만, 트랙맨이 179만). 그렇다면 train 쪽 순서를 트랙맨 쪽
    부분수열에 맞추는 문제가 된다.

    다만 어느 것을 버릴지 모르면 잘못 짝지을 위험이 있다. 그래서 여기서는
    더 안전한 쪽만 시도한다 — 키를 더 잘게 쪼개서 애초에 중복을 줄인다.
    아직 안 쓴 키가 남아 있다: 주자 상태, 점수차, 이닝 내 아웃카운트 조합.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
BASE = ["pitcher_id", "season", "game_month", "game_dayofweek", "inning",
        "top_bottom", "balls_before", "strikes_before", "outs_before"]


def prep():
    mp = pd.read_csv("trackman_map/pitcher_map.csv")
    base = pd.read_csv("trackman_map/team_map.csv")
    m = dict(zip(base["trackman_team"], base["train_team_id"]))
    stem = {c.split("_")[0][:3]: t for c, t in m.items()}
    for code, parent in {"MIN_DOO": "DOO", "MIN_HAN": "HAN", "MIN_HAW": "KIW",
                         "MIN_HER": "KIW", "MIN_KIA": "KIA", "MIN_KTW": "KT",
                         "MIN_LGT": "LG", "MIN_LOT": "LOT", "MIN_NCD": "NC",
                         "MIN_SAM": "SAM", "MIN_SKW": "SK", "MIN_SSG": "SSG"}.items():
        if parent[:3] in stem:
            m[code] = stem[parent[:3]]

    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
    tm = pd.read_csv(os.path.join(D, "trackman_history.csv"), encoding="utf-8-sig")
    tm = tm.merge(mp, left_on="pitcher_trackman_id", right_on="trackman_id")
    tm["top_bottom"] = tm["top_bottom"].astype(str).str[0].str.upper()
    for c in ("pitcher_hand", "batter_hand"):
        tm[c + "_c"] = tm[c].astype(str).str[0].str.upper()
        big = tr[c].value_counts().index[0]
        tr[c + "_c"] = np.where(tr[c] == big, "R", "L")
    tm["pitcher_team_id"] = tm["pitcher_team"].map(m)
    tm["batter_team_id"] = tm["batter_team"].map(m)
    tm = tm[tm.pitcher_team_id.notna() & tm.batter_team_id.notna()].copy()
    for c in ("pitcher_team_id", "batter_team_id"):
        tm[c] = tm[c].astype("int64")
    return tr, tm, mp


def match(sub, tm, keys, allow_unequal=False):
    """유일키 + 개수일치 순서정렬. allow_unequal 이면 min(개수)만큼 앞에서 짝짓는다."""
    a = sub.groupby(keys, dropna=False).size().rename("na")
    b = tm.groupby(keys, dropna=False).size().rename("nb")
    cnt = pd.concat([a, b], axis=1).fillna(0)
    good = cnt[(cnt.na > 0) & (cnt.nb > 0)]
    if not allow_unequal:
        good = good[good.na == good.nb]
    ok = set(good.index)
    sa = sub[sub.set_index(keys).index.isin(ok)].copy()
    tb = tm[tm.set_index(keys).index.isin(ok)].copy()
    sa["_k"] = list(map(tuple, sa[keys].to_numpy()))
    tb["_k"] = list(map(tuple, tb[keys].to_numpy()))
    sa["_r"] = sa.groupby("_k").cumcount()
    tb["_r"] = tb.groupby("_k").cumcount()
    return sa.merge(tb[["_k", "_r"]], on=["_k", "_r"], how="inner")


if __name__ == "__main__":
    tr, tm, mp = prep()
    sub = tr[tr.pitcher_id.isin(set(mp.pitcher_id))].copy()
    sub["_ord"] = sub["row_id"].astype(str).str.extract(r"(\d+)$").astype("int64")
    sub = sub.sort_values("_ord").reset_index(drop=True)
    tm = tm.sort_values(["season", "game_month", "game_dayofweek",
                         "trackman_game_id", "pitch_no"]).reset_index(drop=True)
    N = len(tr)
    print(f"train {N:,}   매핑된 투수의 행 {len(sub):,} ({len(sub)/N*100:.1f}%)")

    K0 = BASE + ["batter_hand_c", "pitcher_team_id", "batter_team_id"]
    plans = [
        ("현재 (개수일치만)", K0, False),
        ("+ 주자상태", K0 + ["base_state"], False),
        ("+ 주자상태 + 점수차", K0 + ["base_state", "score_diff_pitcher_team"], False),
        ("+ 위 전부, 개수불일치 허용", K0 + ["base_state", "score_diff_pitcher_team"], True),
    ]
    print(f"\n  {'구성':30s} {'정합 행':>12s} {'대상비':>8s} {'전체비':>8s}")
    for name, keys, ue in plans:
        miss = [k for k in keys if k not in sub.columns or k not in tm.columns]
        if miss:
            print(f"  {name:30s} 건너뜀 (트랙맨에 없는 키: {miss})")
            continue
        j = match(sub, tm, keys, ue)
        print(f"  {name:30s} {len(j):12,} {len(j)/len(sub)*100:7.1f}% "
              f"{len(j)/N*100:7.1f}%")

    # 남은 것의 성격
    a = sub.groupby(K0, dropna=False).size().rename("na")
    b = tm.groupby(K0, dropna=False).size().rename("nb")
    cnt = pd.concat([a, b], axis=1).fillna(0)
    only_tr = int(cnt.loc[cnt.nb == 0, "na"].sum())
    uneq = cnt[(cnt.na > 0) & (cnt.nb > 0) & (cnt.na != cnt.nb)]
    print(f"\n  남은 미정합의 성격 (기본 키 기준)")
    print(f"    투수 미매핑                {N - len(sub):9,} ({(N-len(sub))/N*100:5.1f}%)")
    print(f"    트랙맨에 그 키가 아예 없음  {only_tr:9,} ({only_tr/N*100:5.1f}%)")
    print(f"    개수 불일치로 버림          {int(uneq.na.sum()):9,} "
          f"({uneq.na.sum()/N*100:5.1f}%)")
    print(f"      그중 train<=트랙맨 인 키의 행 "
          f"{int(uneq.loc[uneq.na <= uneq.nb, 'na'].sum()):,}")
