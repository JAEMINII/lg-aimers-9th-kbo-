# -*- coding: utf-8 -*-
"""v5 투수 클러스터/트랙맨 프로파일을 편상관으로 거른다. 로컬 CPU.

무엇이 예전과 다른가
    트랙맨 파생은 다섯 번 기각됐다. 그런데 그건 전부 **투수 커리어 상수**였고,
    pitcher_id 임베딩(카디널리티 793)과 중복이라 0 이 나왔다.
    v5 는 집계 단위가 **(pitcher_trackman_id, season)** 이라 투수 안에서
    시즌마다 값이 바뀐다. 성장·노화·구종 변경이 잡힌다. 다시 볼 값어치가 있다.

    그리고 max_speed / fb_hb_norm / fb_ivb 는 44열에 **아예 없는 축**이다.
    44열은 결과 집계(성공률, 구종비율)뿐이고 구속·무브먼트·릴리스는 없다.

후보
    연속   max_speed        최고 구속
           fb_hb_norm       속구 수평 무브먼트(정규화)
           fb_ivb           속구 유도 수직 무브먼트
           n_pitch_types    구종 수
           n_breaking_types 변화구 수
    범주   cluster_v5       10칸 라벨 (R_0_few ... L_1_many)
           fb_shape         Sink / Cut / Balanced

    라벨은 원재료를 압축한 것이라 보통 원재료가 낫다. 둘 다 잰다.
    범주형은 순서가 없으므로 편상관에 넣을 때 성공률 순 정렬 대신
    one-hot 잔차의 다중상관(R)으로 본다.

커버리지가 문제다
    2019~2021 은 0% (트랙맨 클러스터가 2022~2024 만 있다).
    전체 매칭률 42.8%. 결측을 어떻게 두느냐가 결과를 좌우한다.
    여기서는 결측을 '학습 구간 중앙값'으로 채우고, 매칭된 행만으로도
    따로 잰다. 둘이 크게 다르면 그 피처는 결측 패턴을 배우는 것이다.

판정
    |편상관| > 0.005 면 관문으로. 어제와 같은 기준선이다.
    비교 대상: basic 후보 12개는 전부 |r| < 0.0025 였고,
    통과한 lg_f_share 는 전체 -0.0174 / 퓨처스 -0.0325 였다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
V5DIR = os.environ.get("V5DIR", "")
VS = 2024
THRESH = 0.005

import features44 as F                                          # noqa: E402


def resid(x, C):
    return x - C @ np.linalg.lstsq(C, x, rcond=None)[0]


def partial(x, y, C):
    rx, ry = resid(x, C), resid(y, C)
    sx, sy = rx.std(), ry.std()
    return 0.0 if sx < 1e-12 or sy < 1e-12 else float((rx * ry).mean() / (sx * sy))


def multi_partial(M, y, C):
    """one-hot 행렬 M 이 통제 후 y 에 대해 갖는 다중상관 R."""
    RM = np.column_stack([resid(M[:, j], C) for j in range(M.shape[1])])
    ry = resid(y, C)
    keep = RM.std(0) > 1e-9
    if not keep.any():
        return 0.0
    RM = RM[:, keep]
    beta = np.linalg.lstsq(RM, ry, rcond=None)[0]
    fit = RM @ beta
    return 0.0 if ry.std() < 1e-12 else float(np.sqrt(max(fit.var() / ry.var(), 0)))


if __name__ == "__main__":
    t0 = time.time()
    v5 = pd.read_csv(os.path.join(V5DIR, "pitcher_cluster_v5.csv"))
    fb = pd.read_csv(os.path.join(V5DIR, "fastball_shape_features.csv"))
    v5 = v5[v5["pitcher_id"].notna()].copy()
    v5["pitcher_id"] = v5["pitcher_id"].astype(int)
    fb = fb.merge(v5[["pitcher_trackman_id", "season", "pitcher_id"]],
                  on=["pitcher_trackman_id", "season"], how="inner")

    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id", "season", "game_type", "pitcher_id"])
    cont = ["max_speed", "fb_hb_norm", "fb_ivb", "n_pitch_types",
            "n_breaking_types"]
    m = raw.merge(v5[["pitcher_id", "season"] + cont + ["cluster_v5"]],
                  on=["pitcher_id", "season"], how="left")
    m = m.merge(fb[["pitcher_id", "season", "fb_shape"]],
                on=["pitcher_id", "season"], how="left")

    d = F.build(DATA, VS=VS)
    X44 = d["X44"].astype(np.float64)
    y = d["y"].astype(np.float64)
    m_tr, isf = d["m_tr"], d["is_f"]
    assert len(m) == len(X44), "행 수가 안 맞는다"

    med = np.nanmedian(X44[m_tr], 0)
    Xf = np.where(np.isnan(X44), med, X44)
    ok = Xf[m_tr].std(0) > 1e-9
    C = np.c_[np.ones(m_tr.sum()), Xf[m_tr][:, ok]]
    yt = y[m_tr]

    hit = m["cluster_v5"].notna().to_numpy()
    m_hit = m_tr & hit
    Ch = np.c_[np.ones(m_hit.sum()), Xf[m_hit][:, ok]]
    m_hf = m_tr & hit & isf
    Cf = np.c_[np.ones(m_hf.sum()), Xf[m_hf][:, ok]]
    print(f"  학습 {m_tr.sum():,}행   매칭 {m_hit.sum():,} "
          f"({m_hit.sum()/m_tr.sum()*100:.1f}%)   매칭+퓨처스 {m_hf.sum():,}")
    print(f"  기준선 |r| > {THRESH}   (basic 후보 12개는 전부 <0.0025, "
          f"통과한 lg_f_share 는 -0.0174)\n")
    print(f"  {'후보':18s} {'전체(결측채움)':>14s} {'매칭행만':>10s} "
          f"{'매칭+퓨처스':>12s}  판정")

    keep = []
    for c in cont:
        v = m[c].to_numpy(np.float64)
        vm = np.where(np.isnan(v), np.nanmedian(v[m_tr & hit]), v)
        a = partial(vm[m_tr], yt, C)
        b = partial(v[m_hit], y[m_hit], Ch)
        f_ = partial(v[m_hf], y[m_hf], Cf)
        good = max(abs(a), abs(b), abs(f_)) > THRESH
        if good:
            keep.append(c)
        print(f"  {c:18s} {a:+14.4f} {b:+10.4f} {f_:+12.4f}  "
              f"{'통과' if good else ''}")

    for c in ("cluster_v5", "fb_shape"):
        s = m[c].fillna("__none__").astype(str)
        M = pd.get_dummies(s, drop_first=True).to_numpy(np.float64)
        a = multi_partial(M[m_tr], yt, C)
        sh = pd.get_dummies(m.loc[hit, c].astype(str),
                            drop_first=True).to_numpy(np.float64)
        b = multi_partial(sh[m_tr[hit]], y[m_hit], Ch)
        shf = pd.get_dummies(m.loc[hit & isf, c].astype(str),
                             drop_first=True).to_numpy(np.float64)
        f_ = multi_partial(shf[m_tr[hit & isf]], y[m_hf], Cf)
        good = max(a, b, f_) > THRESH
        if good:
            keep.append(c)
        print(f"  {c+' (다중R)':18s} {a:+14.4f} {b:+10.4f} {f_:+12.4f}  "
              f"{'통과' if good else ''}")

    print(f"\n  통과 {len(keep)}개: {keep if keep else '없음'}   {time.time()-t0:.0f}s")
    print("  '전체' 와 '매칭행만' 이 크게 다르면 그 피처는 결측 패턴을 배우는 것이다.")
    print("  2019~2021 이 통째로 결측이라(트랙맨 클러스터가 2022~ 만 있다)")
    print("  결측 지시자가 곧 '최근 3시즌인가' 가 되고, season 이 이미 그걸 담는다.")
