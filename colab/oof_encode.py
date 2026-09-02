# -*- coding: utf-8 -*-
"""상호작용 인코딩을 out-of-fold 로 제대로 만든다. GPU 없이 검증까지.

지난번 실패
    batter x pitcher_hand 를 넣었더니 관문 단독 -41.9, pitcher x count 는 -731.
    원인은 피처 자체가 아니라 만드는 방식이었다.
      학습 행의 자기 정답이 자기 셀 평균에 들어갔다
      학습구간 상관 +0.117 -> 관문 +0.038 로 3배 꺾였다
      관문 행의 20%가 학습구간에 없는 조합이라 중앙값으로 메워졌다
    즉 그 피처들은 아직 시험된 적이 없다.

이 피처가 다른 이유
    트랙맨 파생은 전부 (투수, 시즌) 단위 상수였고, pitcher_id 가 이미 793개짜리
    임베딩이라 중복이었다. batter x pitcher_hand 는 같은 투수라도 상대 타자에
    따라 값이 달라진다 — 행마다 다른 값이다.

제대로 만드는 법
    1. out-of-fold: 학습 행의 피처는 그 행이 빠진 데이터로 만든다.
       시간 순서를 지키려면 폴드를 시즌으로 나누는 게 자연스럽다.
       시즌 Y 행의 피처는 시즌 < Y 로만 만든다. 관문/배치와 같은 구조다.
    2. 없는 조합은 중앙값이 아니라 상위 그룹으로 후퇴시킨다.
       (타자, 투수손) 이 없으면 -> 타자 전체 -> 리그 평균
    3. 표본 수도 같이 준다. 모델이 신뢰도를 알 수 있게.

여기서는 인코딩이 제대로 됐는지만 검증한다.
    학습구간 상관과 검증구간 상관이 비슷해야 한다. 크게 꺾이면 아직 새고 있다.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
OUT = "colab/oof_bxh.csv.gz"
ALPHA = 50.0


def build(tr, keys, name, alpha=ALPHA):
    """시즌 < Y 누적으로 만든 평활 성공률과 log 표본수. 자기 정답은 절대 안 들어간다.

    후퇴 경로: 전체키 -> 첫 키만 -> 리그 평균
    """
    seasons = sorted(tr.season.unique())
    y = tr["control_success"]
    out_r = np.full(len(tr), np.nan)
    out_n = np.zeros(len(tr))
    for s in seasons:
        past = tr.season < s
        cur = (tr.season == s).to_numpy()
        if not past.any():
            continue                      # 첫 시즌은 과거가 없다. NaN 으로 둔다.
        prior = float(y[past].mean())
        g = y[past].groupby([tr.loc[past, k] for k in keys], sort=False)
        n, ssum = g.size(), g.sum()
        rate = (ssum + alpha * prior) / (n + alpha)
        # 후퇴용: 첫 키만으로 집계
        g1 = y[past].groupby(tr.loc[past, keys[0]], sort=False)
        n1, s1 = g1.size(), g1.sum()
        rate1 = (s1 + alpha * prior) / (n1 + alpha)

        idx = pd.MultiIndex.from_frame(tr.loc[cur, keys])
        r = rate.reindex(idx).to_numpy()
        nn = n.reindex(idx).fillna(0).to_numpy()
        r1 = rate1.reindex(tr.loc[cur, keys[0]]).to_numpy()
        r = np.where(np.isnan(r), np.where(np.isnan(r1), prior, r1), r)
        out_r[cur] = r
        out_n[cur] = nn
    return pd.DataFrame({f"{name}_rate": out_r,
                         f"{name}_logn": np.log1p(out_n)})


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
    tr["cnt12"] = tr.balls_before.astype(int) * 3 + tr.strikes_before.astype(int)
    y = tr["control_success"].to_numpy(dtype=np.float64)

    specs = [(["batter_id", "pitcher_hand"], "bxh"),
             (["pitcher_id", "cnt12"], "pxc"),
             (["batter_id", "cnt12"], "bxc")]
    cols = {}
    for keys, name in specs:
        f = build(tr, keys, name)
        cols[name] = f
        r = f[f"{name}_rate"].to_numpy()
        print(f"\n[{name}]  키 {keys}")
        print(f"  NaN {np.isnan(r).mean()*100:5.1f}%  "
              f"(첫 시즌 {(tr.season == tr.season.min()).mean()*100:.1f}% 는 과거가 없어 NaN)")
        for s in sorted(tr.season.unique()):
            m = (tr.season == s).to_numpy() & np.isfinite(r)
            if m.sum() < 1000:
                print(f"    {s}  표본 부족")
                continue
            print(f"    {s}  n={m.sum():7,}  피처-정답 상관 {np.corrcoef(r[m], y[m])[0,1]:+.4f}")

    print("\n  시즌별 상관이 고르면 제대로 만들어진 것이다.")
    print("  옛 판본은 학습구간 +0.117 / 관문 +0.038 로 3배 꺾였다 — 그게 누출이었다.")
    all_f = pd.concat(list(cols.values()), axis=1)
    all_f.insert(0, "row_id", tr["row_id"])
    all_f.to_csv(OUT, index=False, compression="gzip")
    print(f"\n저장 {OUT}  {os.path.getsize(OUT)/1e6:.1f} MB  열 {list(all_f.columns)}")
