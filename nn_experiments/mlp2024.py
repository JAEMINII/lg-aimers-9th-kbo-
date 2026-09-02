# -*- coding: utf-8 -*-
"""제출용 MLP: 2024 한 시즌만 학습해 2025 를 예측한다.

관문 검증 (2023만 학습 -> 2024 예측)
    MLP 단독            773.3
    CatBoost 전체 단독  884.5
    0.2 MLP + 0.8 CB    896.7      <- +12.2
  비중은 0.15~0.30 구간이 895~897 로 평평해 0.20 이 안전하다.
  예측 상관 0.8884 (GBDT 끼리는 0.982) — 학습 데이터를 다르게 준 게 다양성의 원천이다.

제출은 한 시즌 밀어 2024 만 학습한다.
학습된 모델을 pickle 로 저장하고, 다음 단계에서 가중치를 numpy 로 추출한다.
"""
import os, time, pickle
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(SC, "nn_cache.npz"), allow_pickle=False)
Xc, Xn, y, season = z["Xc"], z["Xn"], z["y"], z["season"]
X = np.concatenate([Xc.astype(np.float32), Xn], 1)
cat_idx = list(range(Xc.shape[1]))
del Xc, Xn; z.close()
i_fit = np.where(season == 2024)[0]
print(f"학습 2024 {len(i_fit):,}행  (성공률 {y[i_fit].mean()*100:.2f}%)", flush=True)
from pytabkit import MLP_PLR_D_Classifier as M
t0 = time.time()
m = M(random_state=1, n_threads=4, verbosity=0, device="cpu", batch_size=512)
m.fit(X[i_fit], y[i_fit].astype(int), cat_col_names=cat_idx)
print(f"학습 완료 {time.time()-t0:.0f}s", flush=True)
with open(os.path.join(SC, "mlp2024.pkl"), "wb") as f:
    pickle.dump(m, f)
p = m.predict_proba(X[i_fit][:5000])[:, 1]
print(f"저장 완료. 표본예측 평균 {p.mean():.4f}  범위 [{p.min():.4f}, {p.max():.4f}]")
