# -*- coding: utf-8 -*-
"""실전 규모(253,507행)에서 세 모델 중 어디서 1.1e-09 가 나오는지 찾는다.

6만 행에서는 세 경로 모두 차이가 0 이었다. 그런데 25만 행 감사에서는 1.104e-09 가
나왔다. 규모가 커지면 BLAS 가 행렬을 다르게 쪼개므로 누산 순서가 바뀔 수 있다.
그렇다면 이웃 행의 '정보' 와는 무관하다.

각 모델을 전체 프레임과 20% 부분집합에 각각 돌려 같은 행끼리 비교한다.
차이를 내는 모델이 무엇인지, 크기가 얼마인지 본다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SUB = os.environ["ENS_SUB"]
ROOT = os.environ["ENS_ROOT"]
N = int(os.environ.get("CHK_N", "253507"))
sys.path.insert(0, SUB)
os.chdir(SUB)

import script as S                                              # noqa: E402


def report(tag, full, part, idx):
    d = np.abs(full[idx] - part)
    n = int((d > 0).sum())
    print(f"  {tag:10s} 최대차 {d.max():.3e}   다른 행 {n:,}/{len(d):,} "
          f"({n / len(d) * 100:.3f}%)")
    return d.max()


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), encoding="utf-8-sig")
    test = tr.sample(n=N, random_state=0).reindex(columns=list(te.columns))
    test["row_id"] = [f"TEST_{i:06d}" for i in range(len(test))]
    hist, z, shift = S.load_model()
    X = S.build_features(test, hist)
    idx = np.sort(np.random.default_rng(1).choice(len(X), len(X) // 5, replace=False))
    Xs = X.iloc[idx]
    print(f"{len(X):,}행 전체 vs {len(idx):,}행 부분집합\n")

    # 피처부터. 부분집합으로 만들어도 같은 값이어야 한다.
    Xs2 = S.build_features(test.iloc[idx], hist)
    fd = np.abs(X.values[idx].astype(np.float64) - Xs2.values.astype(np.float64))
    print(f"  {'피처':10s} 최대차 {np.nanmax(fd):.3e}")

    A = X.values.astype(np.float64)
    B = A[idx]

    cb_f = S.predict_numpy(A, z)
    cb_p = S.predict_numpy(B, z)
    report("CatBoost", cb_f, cb_p, idx)

    st = np.load(S.resolve("model/prep_vs2025.npz"), allow_pickle=False)
    arcs = [np.load(S.resolve(f"model/vs2025_seed{s_}.npz"), allow_pickle=False)
            for s_ in (42, 1, 777)]
    Xn, Xc = S._fm_prep(A, st)
    Xn2, Xc2 = S._fm_prep(B, st)
    report("flatMLP", S._fm_predict(Xn, Xc, arcs), S._fm_predict(Xn2, Xc2, arcs), idx)

    z0 = np.load(S.resolve("model/tabm_all_seed42.npz"), allow_pickle=False)
    meta = json.loads(str(z0["meta"].item()))
    meta["cat_cardinalities"] = meta["cards"]
    Tn, Tc = S._fm_prep(A, z0)
    Tn2, Tc2 = S._fm_prep(B, z0)

    def tabm(n_, c_):
        acc = None
        for sd in S.TABM_SEEDS:
            zz = np.load(S.resolve(f"model/tabm_all_seed{sd}.npz"), allow_pickle=False)
            q = S._tabm_forward(n_, c_, meta, zz, 4096)
            acc = q if acc is None else acc + q
        return acc / len(S.TABM_SEEDS)

    report("TabM(all)", tabm(Tn, Tc), tabm(Tn2, Tc2), idx)
    print("\n  차이를 내는 경로가 하나면 그 안의 행렬 연산 규모 때문이다.")
    print("  1e-9 는 확률값 정밀도로 의미가 없고, 정보를 나를 수 있는 크기도 아니다.")
