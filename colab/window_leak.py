# -*- coding: utf-8 -*-
"""배치 전처리의 모든 열에 창 의존성이 있는지 이진 판정한다.

원리
    올바른 as-of 피처는 그 행보다 **앞선 데이터만** 본다. 그러므로 학습 창을
    2023 까지 잡든 2024 까지 잡든, **2022년 행의 값은 똑같아야 한다**.
    달라지면 그 열은 미래를 본 것이다.

    관문처럼 잡음 섞인 판정이 아니다. 값이 같거나 다르거나 둘 중 하나다.
    plat_dev 누출(LB +11)을 찾은 것도 결국 이 성질이었다.

검사 대상은 **배치 경로**다
    features44 가 아니라 train_chan_3.preprocess 다. 제출 패키지가 쓰는 코드다.
        A = transform_features(전체행, fit_history_tables(season < 2023))
        B = transform_features(전체행, fit_history_tables(season < 2024))
    season < 2023 인 행에서 A 와 B 를 비교한다.

기대
    plat_dev 는 지인 판이 전 구간 통계를 얼려 쓰므로 **걸려야 한다**.
    안 걸리면 검사가 잘못된 것이다 (양성 대조군).
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"

from train_chan_3 import preprocess as PP                       # noqa: E402

tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                   encoding="utf-8-sig"))
season = tr["season"].to_numpy()
print(f"  {len(tr):,}행   시즌 {season.min()}~{season.max()}")

out = {}
for VS in (2023, 2024):
    h = PP.fit_history_tables(tr[tr.season < VS])
    out[VS] = PP.transform_features(tr, h, train_mode=True)
    print(f"  창 season<{VS} 로 생성 완료  {out[VS].shape}")

cols = list(out[2023].columns)
assert cols == list(out[2024].columns), "열 구성이 다르다"
m = season < 2023                       # 두 창 모두에서 '과거' 인 행
print(f"  비교 대상 {int(m.sum()):,}행 (season < 2023)\n")

A = out[2023].to_numpy(np.float64)[m]
B = out[2024].to_numpy(np.float64)[m]
bad = []
print(f"  {'열':32s} {'최대차':>12s} {'다른 행':>10s}")
for j, c in enumerate(cols):
    a, b = A[:, j], B[:, j]
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
    mx = float(d.max())
    nd = int((d > 1e-9).sum())
    if nd:
        bad.append((c, mx, nd))
        print(f"  {c:32s} {mx:12.4e} {nd:10,}")
if not bad:
    print("  (전부 동일)")
print(f"\n  창 의존 열 {len(bad)}개 / {len(cols)}개")
if bad:
    print("\n  판정")
    for c, mx, nd in sorted(bad, key=lambda t: -t[2]):
        share = nd / int(m.sum()) * 100
        print(f"    {c:30s} {share:5.1f}% 의 행에서 값이 변한다  최대차 {mx:.4e}")
    print("\n  plat_dev 가 목록에 있으면 검사가 작동한 것이다 (양성 대조군).")
    print("  그 외에 뭔가 더 있으면 그게 아직 안 고친 누출이다.")
else:
    print("  plat_dev 조차 안 걸렸다 -> 검사가 잘못됐다. 판정 보류.")
