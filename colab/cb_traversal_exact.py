# -*- coding: utf-8 -*-
"""numpy 트리 순회를 **같은 모델** 위에서 정확히 검증한다.

앞선 시도가 왜 결론이 안 났나
    배치본(2019~2024 학습, 30모델)과 새 CatBoost(2019~2023, 1모델)를 비교해
    상관 0.842 가 나왔다. 그런데 2024 가 한쪽엔 인샘플이고 모델 수도 다르다.
    그 교란만으로 0.84 는 설명된다. 결함이라고 단정할 근거가 못 된다.

이번 방법 — 학습 차이를 없앤다
    CatBoost 모델 **하나**를 만들고
        A  라이브러리 predict_proba
        B  그 모델의 트리를 뽑아 script.py 의 predict_numpy 로 순회
    를 비교한다. 같은 모델이므로 차이가 있으면 **순회 로직의 결함**뿐이다.

    TabM 에서 PyTorch 모델 vs npz 재구현을 맞춰 1.19e-07 을 확인한 것과
    같은 종류의 검사다. CatBoost 만 안 하고 있었다.

트리 뽑기
    save_model(format="json") 이 대칭 트리를 그대로 내준다.
        splits           트리별 분할 (float_feature_index, border)
        leaf_values      2^depth 개
    잎 인덱스 규약이 관건이다. CatBoost 는
        leaf = sum_k (x[feat_k] > border_k) << k
    로 두는데, script.py 도 같은 규약을 쓴다고 주석에 적혀 있다.
    그게 맞는지가 이 검사의 핵심이다.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"
PKG = "submit_30"
N_TR, N_TE = 200000, 20000
DEPTH, ITERS = 4, 200

from catboost import CatBoostClassifier                         # noqa: E402
import features44 as F                                          # noqa: E402
import importlib.util                                            # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pkgscript", os.path.join(PKG, "script.py"))
S = importlib.util.module_from_spec(spec)
S.__dict__["__name__"] = "pkgscript"
try:
    spec.loader.exec_module(S)
except SystemExit:
    pass

d = F.build(DATA, VS=2024)
X = d["X44"].astype(np.float64)
y = d["y"].astype(int)
med = np.nanmedian(X[d["m_tr"]], 0)
X = np.where(np.isnan(X), med, X)          # NaN 을 없애 nan 규약을 논외로 둔다
rng = np.random.default_rng(0)
tr_i = rng.choice(np.where(d["m_tr"])[0], N_TR, replace=False)
te_i = rng.choice(np.where(d["m_va"])[0], N_TE, replace=False)

m = CatBoostClassifier(iterations=ITERS, learning_rate=0.05, depth=DEPTH,
                       l2_leaf_reg=1.0, verbose=0, random_seed=7,
                       allow_writing_files=False, thread_count=6)
m.fit(X[tr_i], y[tr_i])
pa = m.predict_proba(X[te_i])[:, 1].astype(np.float64)
print(f"  A 라이브러리   {len(pa):,}행  평균 {pa.mean():.6f}")

# ---- 트리를 뽑아 trees.npz 형식으로
with tempfile.TemporaryDirectory() as td:
    jp = os.path.join(td, "m.json")
    m.save_model(jp, format="json")
    js = json.load(open(jp, encoding="utf-8"))
fb = js["features_info"]["float_features"]
oblivious = js["oblivious_trees"]
print(f"  트리 {len(oblivious)}개  깊이 {DEPTH}  "
      f"float 피처 {len(fb)}개")

T = len(oblivious)
cb_feat = np.zeros((T, DEPTH), np.int32)
cb_thr = np.zeros((T, DEPTH), np.float32)
cb_leaf = np.zeros((T, 2 ** DEPTH), np.float64)
for t, tree in enumerate(oblivious):
    sp = tree["splits"]
    assert len(sp) == DEPTH, f"깊이 불일치 {len(sp)}"
    for k, s in enumerate(sp):
        cb_feat[t, k] = int(fb[s["float_feature_index"]]["flat_feature_index"])
        cb_thr[t, k] = float(s["border"])
    lv = tree["leaf_values"]
    assert len(lv) == 2 ** DEPTH
    cb_leaf[t] = np.asarray(lv, np.float64)

scale = float(js.get("scale_and_bias", [1.0, [0.0]])[0])
bias = float(js.get("scale_and_bias", [1.0, [0.0]])[1][0])
print(f"  scale {scale}  bias {bias}")

z = {"cb_feat": cb_feat, "cb_thr": cb_thr, "cb_leaf": cb_leaf,
     "cb_depth": np.int32(DEPTH), "n_models": np.int32(1),
     "cb_nan_left": np.zeros(X.shape[1], np.uint8),
     "gt_col": np.int32(-1),
     "sp_feat": cb_feat.ravel().astype(np.int32),
     "sp_thr": cb_thr.ravel().astype(np.float32),
     "sp_idx": np.arange(T * DEPTH, dtype=np.int32).reshape(T, DEPTH)}


class _Z(dict):
    @property
    def files(self):
        return list(self.keys())


pb = S.predict_numpy(X[te_i], _Z(z), chunk=4096)
pb = np.asarray(pb, np.float64).ravel()
if abs(bias) > 1e-12 or abs(scale - 1.0) > 1e-12:
    print("  주의: scale/bias 가 1/0 이 아니다. 순회는 그걸 안 쓴다.")
print(f"  B numpy 순회   {len(pb):,}행  평균 {pb.mean():.6f}")

dd = np.abs(pa - pb)
print(f"\n  최대차 {dd.max():.3e}   평균차 {dd.mean():.3e}")
print(f"  1e-5 초과 {int((dd > 1e-5).sum()):,} / {len(dd):,}")
print(f"  상관 {np.corrcoef(pa, pb)[0,1]:.10f}")
if dd.max() < 1e-5:
    print("\n  순회 정확하다. 잎 인덱스 규약이 CatBoost 와 같다.")
else:
    print("\n  **어긋난다.** 같은 모델인데 다르게 나온다 — 순회 결함이다.")
    bad = np.argsort(-dd)[:3]
    for i in bad:
        print(f"    행 {i}: 라이브러리 {pa[i]:.6f}  순회 {pb[i]:.6f}")
