# -*- coding: utf-8 -*-
"""배치된 numpy 트리 순회가 정상인가. 원본 모델이 없어 우회로 검증한다.

배경
    TabM 은 원본 PyTorch 모델과 npz 재구현을 직접 맞춰봤다 (최대차 1.19e-07).
    CatBoost 는 **원본 모델 파일이 없어** 같은 방법을 못 쓴다.
        trees.npz 에 cb_feat / cb_thr / cb_leaf / cb_nan_left / sp_idx 만 있고
        script.py 가 12,000 트리를 numpy 로 직접 순회한다.

    순회에 결함이 있으면 조용히 틀린 확률을 낸다. 혼합의 0.20~0.30 이다.

우회 검증
    같은 피처·같은 설정으로 CatBoost 를 **새로 학습**해서 배치본과 대조한다.
    둘 다 정상이면 상관이 0.98 이상이어야 한다 (같은 알고리즘, 같은 데이터,
    시드만 다름). 순회가 망가져 있으면 크게 벌어진다.

        배치본   script.py 의 predict_cb (trees.npz, numpy 순회)
        기준     catboost 라이브러리로 새로 학습 -> 라이브러리 predict_proba

    피처는 script.py 의 build_features 로 통일한다. 그래야 순회만 남는다.

    다만 배치본은 2019~2024 로 학습됐고 시드도 다르다. 그래서 '완전 일치' 는
    기대할 수 없고, **정상 범위인지**만 본다.
        0.98 이상  순회 정상
        0.90 부근  의심스럽다
        0.7 이하   결함
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"
PKG = "submit_30"
VS = 2024
DECAY = 2.0

from catboost import CatBoostClassifier                         # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
import importlib.util                                            # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pkgscript", os.path.join(PKG, "script.py"))
S = importlib.util.module_from_spec(spec)
S.__dict__["__name__"] = "pkgscript"
try:
    spec.loader.exec_module(S)
except SystemExit:
    pass

tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                   encoding="utf-8-sig"))
y = tr[PP.TARGET].to_numpy(np.float64)
season = tr["season"].to_numpy()
isf = tr["game_type"].astype(str).to_numpy() == "F"

# script.py 의 피처 생성기로 전 행을 만든다 (시험 행 취급)
os.chdir(PKG)
hist, z, shift = S.load_model()
X = S.build_features(tr.drop(columns=[PP.TARGET]), hist)
os.chdir("..")
Xv = X.to_numpy(dtype=np.float64)
print(f"  피처 {Xv.shape}   결측 {int(np.isnan(Xv).sum()):,}개\n")

va = season == VS
tr_m = season < VS
gate = np.where(va)[0]
yv = y[gate]

# ---- 배치본: numpy 순회
os.chdir(PKG)
# 배치본은 라우팅(1군/퓨처스)을 쓴다. 순회 자체만 보려고 is_f 를 넘긴다.
p_dep = S.predict_numpy(Xv[gate], z, chunk=4096)
os.chdir("..")
p_dep = np.asarray(p_dep, dtype=np.float64).ravel()
print(f"  배치본(numpy 순회)  평균 {p_dep.mean():.5f}  SD {p_dep.std():.5f}")

# ---- 기준: 같은 피처로 새로 학습
w = DECAY ** (season[tr_m].astype(np.float64) - 2019)
med = np.nanmedian(Xv[tr_m], 0)
Xf = np.where(np.isnan(Xv), med, Xv)
m = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4,
                       l2_leaf_reg=1.0, verbose=0, random_seed=42,
                       allow_writing_files=False, thread_count=6)
m.fit(Xf[tr_m], y[tr_m].astype(int), sample_weight=w)
p_ref = m.predict_proba(Xf[gate])[:, 1].astype(np.float64)
print(f"  기준(라이브러리)    평균 {p_ref.mean():.5f}  SD {p_ref.std():.5f}")

r = float(np.corrcoef(p_dep, p_ref)[0, 1])
print(f"\n  상관 {r:.5f}")
import features44 as F                                          # noqa: E402
print(f"  점수  배치본 {F.best_shift(p_dep, yv)[0]:7.1f}   "
      f"기준 {F.best_shift(p_ref, yv)[0]:7.1f}")
print()
if r >= 0.98:
    print("  순회 정상. 같은 알고리즘·같은 데이터라 이 정도가 나와야 한다.")
elif r >= 0.90:
    print("  의심스럽다. 학습 구간·시드 차이로 설명되는지 더 봐야 한다.")
else:
    print("  **결함이다.** 두 CatBoost 가 이렇게 벌어질 이유가 없다 —")
    print("  numpy 순회가 trees.npz 를 잘못 읽고 있다.")
print("\n  주의: 배치본은 2019~2024 로, 기준은 2019~2023 으로 학습됐다.")
print("  2024 가 배치본에겐 인샘플이라 점수는 배치본이 높게 나오는 게 정상이다.")
print("  여기서 읽는 건 **상관**이다.")
