# -*- coding: utf-8 -*-
"""'상황 변조' 피처를 만든다. 투수x시즌 누적표로 저장한다.

측정된 근거
    투구 단위로 붙인 121만 행에서
      물리량이 기존 성공률 너머로 남는다
        induced_vert_break 편상관 +0.0535, rel_speed +0.0463 (성공률 자체가 +0.0763)
      2스트라이크 변조가 시즌을 넘어 이어진다
        지속성 구속 +0.670, 상하무브 +0.711  (연도쌍 5개 전부 0.59~0.76)
      그 성향이 제구 성공과 관계있다
        구속 변조 +0.219, 상하무브 변조 +0.232
        전체 수준을 통제해도 안 줄어든다(+0.198 / +0.235)

    44열에는 투수의 전체 성적만 있고 '상황에 따라 어떻게 변하는가' 가 없다.
    어제 넣은 물리량 평균 13개는 성공률과 중복이라 실패했는데, 변조는 다르다.

만드는 것은 둘뿐이다
    tm_mod_speed   2스트라이크 구속 - 전체 구속
    tm_mod_ivb     2스트라이크 상하무브 - 전체 상하무브

    열을 늘릴수록 나빠지는 것을 어제 확인했다(2열 +11.6 / 13열 +0.3 / 21열 -7.1).
    근거가 있는 둘만 넣는다.

시간 규약
    시즌 Y 행에는 시즌 < Y 의 투구만 쓴다. 누적 평균이다.
    2025 제출 때는 2019~2024 전부가 쓰인다.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
OUT = "colab/mod_pitcher_season.csv"
MIN_N = 300          # 이보다 적으면 변조 추정이 노이즈다


if __name__ == "__main__":
    j = pd.read_csv("colab/pitch_join.csv.gz")
    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id", "pitcher_id", "strikes_before"])
    j = j.merge(tr, on="row_id", how="left")
    j["two"] = (j.strikes_before == 2).astype(int)

    # (투수, 시즌) 단위 합계. 뒤에서 시즌을 누적하며 '이전까지' 로 만든다.
    rows = []
    for c, tag in (("rel_speed", "speed"), ("induced_vert_break", "ivb")):
        g = j.groupby(["pitcher_id", "season"])[c]
        s_all, n_all = g.sum(), g.size()
        g2 = j[j.two == 1].groupby(["pitcher_id", "season"])[c]
        s_two, n_two = g2.sum(), g2.size()
        rows.append(pd.DataFrame({f"s_all_{tag}": s_all, f"n_all_{tag}": n_all,
                                  f"s_two_{tag}": s_two, f"n_two_{tag}": n_two}))
    t = pd.concat(rows, axis=1).fillna(0.0).reset_index()
    print(f"투수x시즌 {len(t):,}")

    seasons = sorted(t.season.unique())
    out = []
    for pid, d in t.groupby("pitcher_id"):
        d = d.set_index("season")
        acc = {k: 0.0 for k in d.columns}
        for s in seasons:
            rec = {"pitcher_id": pid, "season": s}
            for tag in ("speed", "ivb"):
                na, nt = acc[f"n_all_{tag}"], acc[f"n_two_{tag}"]
                if na >= MIN_N and nt >= MIN_N * 0.2:
                    rec[f"tm_mod_{tag}"] = (acc[f"s_two_{tag}"] / nt
                                            - acc[f"s_all_{tag}"] / na)
                else:
                    rec[f"tm_mod_{tag}"] = np.nan
            out.append(rec)
            if s in d.index:
                for k in d.columns:
                    acc[k] += float(d.loc[s, k])
    res = pd.DataFrame(out)
    print(f"누적표 {res.shape}")
    for c in ("tm_mod_speed", "tm_mod_ivb"):
        v = res[c].dropna()
        print(f"  {c:14s} 유효 {len(v):,}  평균 {v.mean():+.3f}  표준편차 {v.std():.3f}")

    # 커버리지 확인
    full = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig",
                       usecols=["pitcher_id", "season"])
    m = full.merge(res, on=["pitcher_id", "season"], how="left")
    for c in ("tm_mod_speed", "tm_mod_ivb"):
        print(f"  train 행 커버리지 {c:14s} {m[c].notna().mean()*100:5.1f}%")
    for s in sorted(full.season.unique()):
        k = m[m.season == s]
        print(f"    {s}  {k.tm_mod_speed.notna().mean()*100:5.1f}%")
    res.to_csv(OUT, index=False)
    print(f"\n저장 {OUT}")
