# -*- coding: utf-8 -*-
"""평가 경로 피처 생성이 학습 경로와 같은 값을 내는지 확인한다.

train.csv 의 2024 행을 '평가 데이터인 척' 하고 as-of-2024 표로 다시 만든 뒤
prep_cache.py 가 만든 2024 구간(mlp_feat_ref2024.npz)과 비교한다.
여기가 어긋나면 2025 예측이 통째로 틀어진다.
"""
import os, sys
import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
D = os.path.join(ROOT, "open (1)", "data")
from mlp_features import load_feat, build_mlp_input      # noqa: E402

F = load_feat(os.path.join(SC, "mlp_feat.npz"))
ref = np.load(os.path.join(SC, "mlp_feat_ref2024.npz"), allow_pickle=False)
Xn_ref, Xc_ref = ref["Xn"], ref["Xc"]

test_cols = pd.read_csv(os.path.join(D, "test.csv"), encoding="utf-8-sig", nrows=0).columns
tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
tr["_r"] = tr.row_id.str.slice(6).astype("int32")
tr = tr.sort_values("_r").reset_index(drop=True)
d24 = tr[tr.season == 2024].reset_index(drop=True)
# 평가 데이터에 있는 열만 남긴다 (control_success 등은 없다)
d24 = d24[[c for c in test_cols if c in d24.columns]]
print(f"2024 행 {len(d24):,}  열 {len(d24.columns)} (test.csv 열 {len(test_cols)})")
assert len(d24.columns) == len(test_cols), \
    f"test 열 중 없는 것: {[c for c in test_cols if c not in d24.columns]}"

got = build_mlp_input(d24, F, tag="t24")
Xn_got, Xc_got = got[:, :44], got[:, 44:]
print(f"생성 {got.shape}")

print("\n범주 16열")
eq = (Xc_got.astype(np.int64) == Xc_ref.astype(np.int64))
print(f"  완전일치={eq.all()}  불일치 행비율={1-eq.all(1).mean():.6f}")
if not eq.all():
    for j in range(16):
        bad = (~eq[:, j]).sum()
        if bad:
            print(f"    열{j} {F['cat_names'][j]:24s} 불일치 {bad:,}")

print("\n수치 44열")
d = np.abs(Xn_got.astype(np.float64) - Xn_ref.astype(np.float64))
print(f"  완전일치={np.array_equal(Xn_got, Xn_ref)}  최대차이={d.max():.3e}")
order = np.argsort(-d.max(0))
names = F["con_names"] + [f"na_{F['con_names'][i]}" for i in np.where(F["has_nan"])[0]]
for j in order[:6]:
    print(f"    열{j:2d} {names[j]:34s} 최대차이 {d[:, j].max():.3e}  "
          f"평균차이 {d[:, j].mean():.3e}")

# 예측까지 비교
print("\n예측 비교 (MLP)")
from mlp_numpy import mlp_predict                        # noqa: E402
z = np.load(os.path.join(SC, "mlp.npz"), allow_pickle=False)
p_ref = mlp_predict(np.concatenate([Xn_ref, Xc_ref], 1).astype(np.float32), z)
p_got = mlp_predict(got, z)
dd = np.abs(p_ref - p_got)
print(f"  n={len(p_got):,}  최대차이={dd.max():.3e}  중앙차이={np.median(dd):.3e}  "
      f"평균 {p_ref.mean():.6f} / {p_got.mean():.6f}")
