# -*- coding: utf-8 -*-
"""HistGB / LightGBM 을 지금 배치 구성에서 다시 잰다. 로컬 CPU.

왜 다시 재나
    2026-08-16 에 쟀고 기각했다.
        LightGBM  단독 849.7  CB상관 0.9617  최적비중 0.10  현제출대비 +0.5
        HistGB    단독 850.5  CB상관 0.9628  최적비중 0.05  현제출대비 +0.4
    그런데 그 측정은 지금과 두 군데가 다르다.
        1. 배치가 CatBoost + 0.2 MLP 였다. 지금은 0.14 CatBoost + 0.86 TabM 이다.
           HistGB 와 **TabM** 의 상관은 한 번도 안 쟀다.
        2. 1군만 채점했다. 지금은 전체 R+F 다.
    부품의 값어치는 나머지 앙상블이 뭐냐에 달려 있다. 오늘 regular 브랜치에서
    그걸 비싸게 배웠다 (4시드 평균 위에서 재고 뺐는데 배치는 1시드였다).

무엇을 하나
    관문(2019~2023 학습 -> 2024 채점, 전체 R+F)에서 예측을 만들어 저장한다.
    TabM 관문 예측(fb_base.npy)이 서버에서 내려오면 blend_gbdt.py 로 3원 비중을 훑는다.
    여기서는 단독 점수와 CatBoost 상관까지만 낸다.
"""
import os
import sys
import time

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
DL = os.path.join(SC, "_dl")
VS = 2024
DECAY = 2.0
NTH = 6

import features44 as F                                          # noqa: E402


def mk_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6, max_leaf_nodes=15,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
        random_state=42)


def mk_lgb():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                          min_child_samples=200, reg_lambda=1.0, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=NTH, random_state=42,
                          verbose=-1)


def mk_lgb_dart():
    """DART 는 부스팅 방식 자체가 달라 오차가 더 떨어질 수 있다."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(boosting_type="dart", n_estimators=400,
                          learning_rate=0.05, num_leaves=15,
                          min_child_samples=200, reg_lambda=1.0,
                          drop_rate=0.1, n_jobs=NTH, random_state=42,
                          verbose=-1)


if __name__ == "__main__":
    os.makedirs(DL, exist_ok=True)
    d = F.build(DATA, VS=VS)
    X, y, m_tr, m_va = d["X44"], d["y"], d["m_tr"], d["m_va"]
    gate = np.where(m_va)[0]
    yv = y[gate].astype(np.float64)
    isf = d["is_f"][gate]
    w = DECAY ** (d["season"][m_tr].astype(np.float64) - 2019)
    Xtr = X[m_tr].astype(np.float64)
    ytr = y[m_tr].astype(int)
    Xva = X[gate].astype(np.float64)

    cb = np.load(os.path.join(SC, "cb_gate.npy")).astype(np.float64)
    print(f"  학습 {m_tr.sum():,}  관문 {len(gate):,} (퓨처스 {isf.mean()*100:.1f}%)"
          f"  시즌가중 {DECAY}")
    print(f"  CatBoost 관문 단독 {F.best_shift(cb, yv)[0]:.1f}  "
          f"(전체 R+F 채점)\n")
    print(f"  {'모델':12s} {'단독':>8s} {'1군':>8s} {'퓨처스':>8s} "
          f"{'CB상관':>8s} {'시간':>7s}")

    for name, mk in (("HistGB", mk_hgb), ("LightGBM", mk_lgb),
                     ("LGBM_DART", mk_lgb_dart)):
        t0 = time.time()
        try:
            mdl = mk()
            mdl.fit(Xtr, ytr, sample_weight=w)
            p = mdl.predict_proba(Xva)[:, 1].astype(np.float64)
        except Exception as e:
            print(f"  {name:12s} 실패: {type(e).__name__} {e}")
            continue
        np.save(os.path.join(DL, f"gb_{name}.npy"), p)
        print(f"  {name:12s} {F.best_shift(p, yv)[0]:8.1f} "
              f"{F.best_shift(p[~isf], yv[~isf])[0]:8.1f} "
              f"{F.best_shift(p[isf], yv[isf])[0]:8.1f} "
              f"{np.corrcoef(p, cb)[0,1]:8.4f} {time.time()-t0:7.0f}s")

    np.save(os.path.join(DL, "gate_y.npy"), yv)
    np.save(os.path.join(DL, "gate_isf.npy"), isf)
    print(f"\n  저장 {DL}")
    print("  TabM 관문 예측(fb_base.npy)이 내려오면 blend_gbdt.py 로 3원 비중을 훑는다.")
