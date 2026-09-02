# -*- coding: utf-8 -*-
"""cb_gate{VS}.npy 를 다시 만든다. 서버를 갈아서 없어졌다.

tabm_gate_gpu 가 import 시점에 이 파일을 읽는다. 2024 폴드 것만 로컬에
남아 있어 2022/2023 폴드가 아예 안 돈다. 배치 CatBoost 와 같은 설정으로
만든다 — 어차피 기준선 출력용이라 정확히 같을 필요는 없지만, 다르면
과거 로그와 비교가 안 된다.
"""
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "/workspace/aimers/data"
DECAY = 2.0
HP = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, random_seed=42)

import features44 as F                                          # noqa: E402

if __name__ == "__main__":
    for VS in (2022, 2023):
        out = os.path.join(SC, f"cb_gate{VS}.npy")
        if os.path.exists(out):
            print(f"  VS={VS} 이미 있음")
            continue
        t0 = time.time()
        d = F.build(DATA, VS=VS)
        X, y, m_tr, m_va = d["X44"], d["y"], d["m_tr"], d["m_va"]
        w = DECAY ** (d["season"][m_tr].astype(np.float64) - 2019)
        mdl = CatBoostClassifier(**HP)
        mdl.fit(X[m_tr].astype(np.float64), y[m_tr].astype(int), sample_weight=w)
        gate = np.where(m_va)[0]
        p = mdl.predict_proba(X[gate].astype(np.float64))[:, 1]
        np.save(out, p)
        yv = y[gate].astype(np.float64)
        print(f"  VS={VS}  학습 {m_tr.sum():,}  관문 {len(gate):,}  "
              f"단독 {F.best_shift(p, yv)[0]:7.1f}  {time.time()-t0:.0f}s")
