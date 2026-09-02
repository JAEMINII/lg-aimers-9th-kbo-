# -*- coding: utf-8 -*-
"""붙인 121만 투구에서 물리량과 제구 성공의 관계를 직접 잰다.

지금까지는 '투수 평균 구속' 같은 뭉갠 값으로만 봤다. 그래서 관계가 있어도
투수 간 차이에 묻혔다. 이제 투구 단위로 볼 수 있다.

두 가지를 나눠 본다
    투수 간 (between)   구속이 빠른 투수가 제구가 좋은가
    투수 내 (within)    이 투수가 평소보다 릴리스가 어긋난 공이 실패하는가

    두 번째가 '커맨드' 가설이다. 그리고 이게 실제로 있다면, 2025 행에는
    그 공의 릴리스를 모르니 직접 못 쓴다. 대신 '이 투수가 얼마나 흔들리는가'
    를 쓰게 되는데, 그건 이미 재봤고 효과가 없었다(6시드 +0.8).

    그래서 여기서 확인할 것은 순서가 이렇다.
      1) 투수 내 관계가 실제로 있는가. 없으면 이 방향 자체가 끝이다.
      2) 있다면 그 크기가 어느 정도인가. 기존 44피처가 이미 담고 있는가.
      3) 2025 에 쓸 수 있는 형태로 무엇이 남는가.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]


def deciles(x, y, k=10):
    """x 십분위별 y 평균. 관계의 모양을 본다."""
    q = pd.qcut(x, k, labels=False, duplicates="drop")
    g = pd.DataFrame({"q": q, "y": y}).groupby("q", observed=True)["y"]
    return g.mean(), g.size()


if __name__ == "__main__":
    j = pd.read_csv("colab/pitch_join.csv.gz")
    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id", "pitcher_id", "balls_before",
                              "strikes_before", "game_type"])
    j = j.merge(tr, on="row_id", how="left")
    y = j["control_success"].to_numpy(dtype=np.float64)
    print(f"정합 {len(j):,} 행   성공률 {y.mean():.4f}   "
          f"(train 전체 {pd.read_csv(os.path.join(D, 'train.csv'), encoding='utf-8-sig', usecols=['control_success']).control_success.mean():.4f})")

    # 투수x시즌 평균을 빼서 '투수 내 편차' 를 만든다
    g = j.groupby(["pitcher_id", "season"])
    for c in PHYS:
        j[c + "_dev"] = j[c] - g[c].transform("mean")
    # 릴리스 편차의 크기(2차원 거리) — 커맨드 가설의 핵심 변수
    j["rel_dist"] = np.sqrt(j["rel_height_dev"] ** 2 + j["rel_side_dev"] ** 2)

    print("\n" + "=" * 72)
    print("  변수                  상관(원값)   상관(투수내 편차)")
    print("=" * 72)
    rows = []
    for c in PHYS:
        ok = j[c].notna()
        r_raw = np.corrcoef(j.loc[ok, c], y[ok.to_numpy()])[0, 1]
        ok2 = j[c + "_dev"].notna()
        r_dev = np.corrcoef(j.loc[ok2, c + "_dev"], y[ok2.to_numpy()])[0, 1]
        rows.append((c, r_raw, r_dev))
        print(f"  {c:20s} {r_raw:+11.4f}   {r_dev:+15.4f}")
    ok = j["rel_dist"].notna()
    rd = np.corrcoef(j.loc[ok, "rel_dist"], y[ok.to_numpy()])[0, 1]
    print(f"  {'rel_dist(편차크기)':20s} {'':11s}   {rd:+15.4f}")

    print("\n" + "=" * 72)
    print("  릴리스 편차 크기 십분위별 성공률  (커맨드 가설)")
    print("=" * 72)
    m, n = deciles(j.loc[ok, "rel_dist"], y[ok.to_numpy()])
    lo = j.loc[ok, "rel_dist"].quantile(np.linspace(0, 1, 11)).to_numpy()
    for i in range(len(m)):
        print(f"  {i+1:2d}분위  {lo[i]:.4f}~{lo[i+1]:.4f}  n={n.iloc[i]:7,}  "
              f"성공률 {m.iloc[i]:.4f}")
    print(f"  최저-최고 차이 {m.max() - m.min():+.4f}")

    print("\n" + "=" * 72)
    print("  2스트라이크 여부로 나눠서 (국면에 따라 다른가)")
    print("=" * 72)
    for tag, msk in (("스트라이크 0-1", j.strikes_before < 2),
                     ("스트라이크 2", j.strikes_before == 2)):
        mm = msk.to_numpy() & ok.to_numpy()
        r = np.corrcoef(j.loc[mm, "rel_dist"], y[mm])[0, 1]
        d, _ = deciles(j.loc[mm, "rel_dist"], y[mm])
        print(f"  {tag:14s} n={mm.sum():8,}  상관 {r:+.4f}  "
              f"십분위 최저-최고 {d.max() - d.min():+.4f}")

    print("\n  상관이 0.01 수준이면 신호가 거의 없다는 뜻이다.")
    print("  기존 44피처의 asof_pitcher_success_rate 는 상관 0.05 안팎이다.")
