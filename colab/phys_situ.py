# -*- coding: utf-8 -*-
"""물리량이 기존 44피처 '너머로' 무엇을 더하는지 본다.

앞에서 나온 것
    투구 단위 상관   induced_vert_break +0.055, rel_speed +0.040
    투수 내 편차     구속 +0.054 로 원값보다 크다
    릴리스 편차      -0.001. 커맨드 가설은 죽었다.

그런데 어제 이 물리량들의 투수 평균을 피처로 넣었을 때 효과가 0 이었다.
이유가 짐작된다 — 물리량은 제구 성공의 '원인' 이고, asof_pitcher_success_rate 는
그 '결과' 를 이미 측정한 값이다. 원인을 더해도 결과가 이미 있으면 새 정보가 아니다.

그래서 여기서는 두 가지를 본다
    1. 편상관   투수의 기존 성공률을 통제한 뒤에도 물리량이 남는가
    2. 상황 변조  투수가 카운트에 따라 구속·무브먼트를 바꾸는 성향.
                 이건 전체 성공률이 담을 수 없는 정보다.
                 단, 쓸모가 있으려면 시즌을 넘어 이어지는 '성향' 이어야 한다.
                 2025 에 쓰려면 과거에서 뽑은 값이 그대로 통해야 하기 때문이다.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
PHYS = ["rel_speed", "induced_vert_break", "zone_speed", "extension",
        "horz_break", "spin_rate"]


def partial_corr(x, y, z):
    """z 를 선형 제거한 뒤의 x-y 상관."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    Z = np.c_[np.ones(len(z)), z]
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(x - Z @ bx, y - Z @ by)[0, 1]


if __name__ == "__main__":
    j = pd.read_csv("colab/pitch_join.csv.gz")
    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id", "pitcher_id", "balls_before",
                              "strikes_before", "asof_pitcher_success_rate",
                              "asof_pitcher_n"])
    j = j.merge(tr, on="row_id", how="left")
    y = j["control_success"].to_numpy(dtype=np.float64)
    sr = j["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)

    print("=" * 74)
    print("  1. 편상관 — 투수의 기존 성공률을 통제한 뒤")
    print("=" * 74)
    print(f"  {'변수':22s} {'단순상관':>10s} {'편상관':>10s}")
    for c in PHYS:
        x = j[c].to_numpy(dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(sr)
        r0 = np.corrcoef(x[ok], y[ok])[0, 1]
        r1 = partial_corr(x, y, sr)
        print(f"  {c:22s} {r0:+10.4f} {r1:+10.4f}")
    print(f"  {'(참고) 성공률 자체':22s} {np.corrcoef(sr[np.isfinite(sr)], y[np.isfinite(sr)])[0,1]:+10.4f}")

    # ---- 2. 상황 변조: 2스트라이크에서 구속/무브를 얼마나 바꾸는가
    print("\n" + "=" * 74)
    print("  2. 상황 변조 — 투수x시즌별 '2스트라이크 - 전체' 차이")
    print("=" * 74)
    j["two"] = (j.strikes_before == 2).astype(int)
    key = ["pitcher_id", "season"]
    mod = {}
    for c in ("rel_speed", "induced_vert_break"):
        g = j.groupby(key)[c].mean().rename("all")
        g2 = j[j.two == 1].groupby(key)[c].mean().rename("two")
        m = pd.concat([g, g2], axis=1)
        m["mod"] = m["two"] - m["all"]
        n = j.groupby(key).size().rename("n")
        m = m.join(n)
        m = m[m.n >= 300]
        mod[c] = m
        print(f"  {c:22s} 투수x시즌 {len(m):,}   변조 평균 {m['mod'].mean():+.4f}  "
              f"표준편차 {m['mod'].std():.4f}")

        # 시즌 간 이어지는 성향인가
        w = m.reset_index().pivot(index="pitcher_id", columns="season", values="mod")
        pairs = []
        for s in range(2019, 2024):
            if s in w.columns and s + 1 in w.columns:
                d = w[[s, s + 1]].dropna()
                if len(d) > 30:
                    pairs.append((s, len(d), np.corrcoef(d[s], d[s + 1])[0, 1]))
        for s, n_, r in pairs:
            print(f"      {s}->{s+1}  n={n_:4d}  상관 {r:+.3f}")
        if pairs:
            print(f"      평균 지속성 {np.mean([p[2] for p in pairs]):+.3f}")

    # ---- 3. 변조가 제구 성공과 관계있는가
    print("\n" + "=" * 74)
    print("  3. 변조 성향이 그 투수의 제구 성공과 관계있는가")
    print("=" * 74)
    for c, m in mod.items():
        sy = j.groupby(key)["control_success"].mean().rename("y")
        d = m.join(sy).dropna()
        r = np.corrcoef(d["mod"], d["y"])[0, 1]
        rp = partial_corr(d["mod"].to_numpy(), d["y"].to_numpy(),
                          d["all"].to_numpy())
        print(f"  {c:22s} 변조↔성공률 상관 {r:+.4f}   "
              f"(전체수준 통제 후 {rp:+.4f})   n={len(d):,}")
    print("\n  지속성이 낮으면 2025 에 못 쓴다. 관계가 없으면 써도 소용없다.")
