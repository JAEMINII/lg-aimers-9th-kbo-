# -*- coding: utf-8 -*-
"""1.1e-09 차이가 반올림인지 정보 유출인지 가린다.

배경
    새 TabM 경로로 바꾼 뒤 규칙4 감사에서 최대차가 0 이 아니라 1.104e-09 로 나왔다.
    옛 판본은 정확히 0 이었다. 크기로 보면 float32 누산 오차지만, 추측으로 넘기면
    안 되는 항목이라 직접 가른다.

무엇으로 가르나
    정보 유출이면 차이가 '다른 행이 무엇이냐' 에 반응해야 한다. 그러면
      - 차이 나는 행의 비율이 상당하고
      - 차이 크기가 행마다 제각각이며
      - 같은 행을 다른 이웃과 묶을 때마다 값이 달라진다
    반올림이면
      - 대부분의 행이 정확히 일치하고
      - 차이는 1e-9 언저리에 몰려 있으며
      - 원인이 BLAS 블록 크기라 청크 경계를 바꾸면 재현된다

    그래서 같은 행 묶음을 청크 크기만 바꿔 두 번 돌려 본다. 행 구성이 동일한데도
    차이가 나면 이웃 행의 '정보' 와는 무관하다는 뜻이다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SUB = os.environ["ENS_SUB"]
ROOT = os.environ["ENS_ROOT"]
N = int(os.environ.get("CHK_N", "60000"))
sys.path.insert(0, SUB)
os.chdir(SUB)

import script as S                                              # noqa: E402


def tabm_preds(X, chunk):
    """script.py 와 같은 경로로 TabM 예측. 청크 크기만 인자로 뺀다."""
    z0 = np.load(S.resolve("model/tabm_all_seed42.npz"), allow_pickle=False)
    meta = json.loads(str(z0["meta"].item()))
    meta["cat_cardinalities"] = meta["cards"]
    Tn, Tc = S._fm_prep(X.values.astype(np.float64), z0)
    acc = None
    for sd in S.TABM_SEEDS:
        z = np.load(S.resolve(f"model/tabm_all_seed{sd}.npz"), allow_pickle=False)
        q = S._tabm_forward(Tn, Tc, meta, z, chunk)
        acc = q if acc is None else acc + q
    return acc / len(S.TABM_SEEDS)


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), encoding="utf-8-sig")
    test = tr.sample(n=N, random_state=0).reindex(columns=list(te.columns))
    test["row_id"] = [f"TEST_{i:06d}" for i in range(len(test))]
    hist, z, shift = S.load_model()
    X = S.build_features(test, hist)

    a = tabm_preds(X, 4096)
    b = tabm_preds(X, 4096)
    print(f"같은 청크로 두 번   최대차 {np.abs(a - b).max():.3e}  (0 이면 결정적)")

    c = tabm_preds(X, 3000)
    d = np.abs(a - c)
    print(f"청크만 4096 -> 3000  최대차 {d.max():.3e}")
    print(f"  행 구성은 완전히 같다. 그런데도 차이가 나면 이웃 행의 정보와 무관하다.")
    print(f"  다른 행 비율 {(d > 0).mean() * 100:.2f}%   "
          f"중앙값 {np.median(d[d > 0]) if (d > 0).any() else 0:.3e}")
    for t in (1e-12, 1e-10, 1e-9, 1e-8, 1e-6):
        print(f"    차이 > {t:.0e} 인 행  {(d > t).sum():7,} / {len(d):,}")

    # 부분집합: 같은 행이 다른 이웃과 묶였을 때
    idx = np.sort(np.random.default_rng(1).choice(len(X), N // 5, replace=False))
    e = tabm_preds(X.iloc[idx], 4096)
    f = np.abs(a[idx] - e)
    print(f"\n20% 부분집합    최대차 {f.max():.3e}   "
          f"다른 행 비율 {(f > 0).mean() * 100:.2f}%")
    print("  청크 실험과 같은 크기면 반올림이다. 훨씬 크면 정보 유출을 의심한다.")
