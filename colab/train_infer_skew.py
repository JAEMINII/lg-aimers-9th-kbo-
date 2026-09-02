# -*- coding: utf-8 -*-
"""학습 경로와 추론 경로가 같은 행에 같은 값을 내는가.

왜
    plat_dev 누출 하나가 리더보드 +11 이었다 (1058 -> 1069). 그건 두 전처리
    구현을 대조해서 찾았다. 같은 방법을 쓸 수 있는 자리가 하나 더 있다 —
    **같은 파이프라인 안의 두 경로**다.

        학습   transform_features(train, history, train_mode=True)
        추론   build_inference_features(ordered, history)

    배치에서 학습 행은 앞의 경로로, 2025 시험 행은 뒤의 경로로 간다.
    두 경로가 같은 개념의 열에 다른 값을 내면 모델이 학습한 것과 다른 걸
    보게 된다. 그 자체가 train/test 불일치이고 점수를 깎는다.

설계 — 배치 구조를 그대로 흉내낸다
    history 를 2019~2023 으로 고정하고, **2024 행**을 두 경로에 각각 넣는다.
        학습 경로  2024 를 학습 프레임의 일부로 보고 transform_features
        추론 경로  2024 를 시험셋처럼 보고 build_inference_features
    배치에서 2025 가 겪는 것과 같은 위치다.

읽는 법
    열마다 상관과 최대차를 본다. 상관 1.0 이 아닌 열이 용의자다.
    plat_dev 는 이미 아는 문제라 그것 말고 다른 게 나오는지가 관건이다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"
VS = 2024

from train_chan_3 import preprocess as PP                       # noqa: E402

tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                   encoding="utf-8-sig"))
hist = PP.fit_history_tables(tr[tr.season < VS])

# 학습 경로 — 2024 를 학습 프레임의 일부로
X_tr = PP.transform_features(tr, hist, train_mode=True)
m = (tr["season"] == VS).to_numpy()
A = X_tr.loc[m].reset_index(drop=True)

# 추론 경로 — 2024 를 시험셋처럼
te = tr.loc[m].drop(columns=[PP.TARGET]).reset_index(drop=True)
B = PP.build_inference_features(PP.sort_by_row_id(te), hist)

print(f"\n  {VS} 행 {len(A):,}개   history 는 season < {VS} 로 고정")
print(f"  학습경로 {A.shape}   추론경로 {B.shape}")
ca, cb = list(A.columns), list(B.columns)
print(f"  열 이름 동일: {ca == cb}")
if set(ca) != set(cb):
    print(f"    학습에만: {sorted(set(ca)-set(cb))}")
    print(f"    추론에만: {sorted(set(cb)-set(ca))}")

rows = []
for c in ca:
    if c not in cb:
        continue
    x = pd.to_numeric(A[c], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(B[c], errors="coerce").to_numpy(np.float64)
    ok = ~(np.isnan(x) | np.isnan(z))
    if ok.sum() < 100:
        rows.append((c, np.nan, np.nan, np.nan))
        continue
    r = (np.corrcoef(x[ok], z[ok])[0, 1]
         if x[ok].std() > 1e-12 and z[ok].std() > 1e-12 else 1.0)
    rows.append((c, r, float(np.abs(x[ok] - z[ok]).max()),
                 float(np.mean(x[ok] - z[ok]))))

rows.sort(key=lambda t: (1.0 if np.isnan(t[1]) else t[1]))
print(f"\n  {'열':30s} {'두 경로 상관':>12s} {'최대차':>11s} {'평균차':>11s}")
print("  " + "-" * 68)
bad = []
for c, r, md, mn in rows:
    flag = ""
    if np.isnan(r) or r < 0.9999 or (md and md > 1e-6):
        flag = "  <- 다름"
        bad.append(c)
    print(f"  {c:30s} {r:12.6f} {md:11.5f} {mn:+11.5f}{flag}")

print(f"\n  두 경로가 다른 열 {len(bad)} / {len(rows)}: {bad}")
print("\n  배치에서 학습 행은 학습경로로, 2025 시험 행은 추론경로로 간다.")
print("  다른 열이 있으면 모델이 학습한 것과 다른 값을 시험 때 받는다는 뜻이다.")
