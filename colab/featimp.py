# -*- coding: utf-8 -*-
"""AGATa 를 우리 설정에 옮기기 위한 열 중요도. 로컬 CPU 로 돈다.

AGATa (NeurIPS 2024) 는 Transformer self-attention 으로 열 중요도를 재고
**하위 40%** 만 증강 대상으로 삼는다. TabM 에는 attention 이 없으므로
같은 자리에 CatBoost 중요도를 쓴다. 어차피 매번 학습하는 모델이라 공짜다.

중요도는 학습 구간(2019~2023)에서만 잰다. 검증 연도를 보면 규칙 4 위반이고,
증강 대상 선정 자체가 검증에 맞춰지면 관문 점수를 못 믿는다.
"""
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
VS = 2024
DECAY = 2.0                       # 배치 CatBoost 와 같은 시즌가중
HP = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, thread_count=6, random_seed=42)

import features44 as F                                          # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    d = F.build(DATA, VS=VS)
    X, y, m_tr, season = d["X44"], d["y"], d["m_tr"], d["season"]
    F44 = list(d["F44"])
    w = DECAY ** (season[m_tr].astype(np.float64) - 2019)
    mdl = CatBoostClassifier(**HP)
    mdl.fit(X[m_tr].astype(np.float64), y[m_tr].astype(int), sample_weight=w)
    imp = np.asarray(mdl.get_feature_importance())
    order = np.argsort(-imp)
    k = int(round(0.40 * len(F44)))
    low = set(order[-k:].tolist())

    print(f"  학습 {m_tr.sum():,}행  {time.time()-t0:.0f}s   시즌가중 {DECAY}")
    print(f"  하위 40% = {k}개 / {len(F44)}개\n")
    print(f"  {'순위':>4s} {'열':34s} {'중요도':>8s}  대상")
    for r, j in enumerate(order):
        print(f"  {r+1:4d} {F44[j]:34s} {imp[j]:8.3f}  {'<= 증강' if j in low else ''}")
    names = [F44[j] for j in sorted(low)]
    print(f"\n  증강 대상 {len(names)}개")
    print("  AUG_COLS = " + repr(names))
    np.save(os.path.join(SC, "featimp.npy"), imp)
