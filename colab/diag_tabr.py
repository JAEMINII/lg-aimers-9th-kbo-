# -*- coding: utf-8 -*-
"""TabR 진단. (1) 범주열 카디널리티 (2) 예측 분포 포화 여부.

의심
    범주열에 pitcher_id / batter_id 가 들어간다. 카디널리티가 700~800 대라
    임베딩 표가 크고, 2에폭 학습에서 드물게 등장하는 선수의 벡터는 초기값에
    가깝게 남는다. TabM 은 그래도 괜찮다 — 그 벡터가 뒷단 MLP 를 거쳐 확률로
    바뀔 뿐이다.

    TabR 은 다르다. 그 벡터가 **검색 거리에 그대로 들어간다**. 거의 무작위인
    좌표 16개가 L2 거리에 섞이면 이웃 선정이 오염된다. 이웃을 잘못 고르면
    그 뒤가 전부 무의미하다.

    포화 검사로 갈린다 — softmax 가 argmax 로 붕괴해 최근접 라벨을 복사하면
    예측이 0/1 로 몰린다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"


def show(name, p):
    q = np.percentile(p, [0, 1, 25, 50, 75, 99, 100])
    print(f"  {name:24s} 평균 {p.mean():.4f}  표준편차 {p.std():.4f}")
    print(f"  {'':24s} 분위 " + " ".join(f"{v:.3f}" for v in q))
    print(f"  {'':24s} <0.02 {np.mean(p < 0.02) * 100:5.1f}%   "
          f">0.98 {np.mean(p > 0.98) * 100:5.1f}%")


print("\n===== 범주열 =====")
from train_chan_3 import preprocess as PP                       # noqa: E402
import pandas as pd                                             # noqa: E402
cats = list(PP.TABM_CATEGORICAL_FEATURES)
print(f"  지인 전처리 범주 지정 {len(cats)}개 (+ abs_regime = {len(cats)+1})")
tr = pd.read_csv("/workspace/aimers/data/train.csv", encoding="utf-8-sig",
                 usecols=lambda c: c in set(cats) | {"season"})
for c in cats:
    if c in tr.columns:
        n = tr[c].nunique(dropna=True)
        print(f"    {c:32s} 고유값 {n:6,d}"
              + ("   <- 고카디널리티" if n > 100 else ""))
    else:
        print(f"    {c:32s} (파생열, train.csv 에 없음)")

print("\n===== 예측 분포 =====")
for f in sorted(os.listdir(OUT)):
    if f.startswith("tabr_") and f.endswith(".npy"):
        show(f[:-4], np.load(os.path.join(OUT, f)))
tm = np.load(f"{OUT}/emb2_linear_relu_r_s42.npy")
show("TabM (참고)", tm)
print("\n  극단 비율이 TabM 보다 크게 높으면 최근접 라벨 복사다.")
print("  softmax 가 argmax 로 붕괴한 것이고, 원인은 거리 척도다.")
