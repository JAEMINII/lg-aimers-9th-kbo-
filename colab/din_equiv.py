# -*- coding: utf-8 -*-
"""DIN 의 torch(관문) <-> numpy(배치) 동치 검증. 제출 전에 건너뛴 검사다.

관문 점수는 torch 로 쟀고, 1089 를 낸 배치는 numpy 손구현이다. 둘이 다르면
배치는 검증 안 된 다른 모델이다 — MNCA 가 정확히 이 유형으로 -24 였다.
실데이터가 필요 없다. 배치 가중치를 torch 모델에 되넣고 같은 입력을 먹인다.
"""
import os
import sys

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
import ctr_zoo as Z                                             # noqa: E402
import din_numpy as DN                                          # noqa: E402

ROOT = os.path.dirname(SC)
M = np.load(os.path.join(ROOT, "submit_44", "model", "din_meta.npz"),
            allow_pickle=False)
cards = M["cards"].tolist()
n_num, n_state, n_cell = int(M["n_num"]), int(M["n_state"]), int(M["n_cell"])
K = int(M["kseq"])
rng = np.random.default_rng(0)
B = 4096
Xn = rng.standard_normal((B, n_num)).astype(np.float32)
Xc = np.stack([rng.integers(0, c, B) for c in cards], 1).astype(np.int64)
S = rng.integers(0, n_state, (B, K)).astype(np.int64)
C = rng.integers(0, n_cell, (B, K)).astype(np.int64)
Mk = rng.random((B, K)) < 0.7
Mk[~Mk.any(1), 0] = True
S[~Mk] = 0; C[~Mk] = 0
cur = rng.integers(1, n_cell, B).astype(np.int64)

worst = 0.0
for br in ("all", "regular", "futures"):
    for sd in (42, 1, 777):
        W = np.load(os.path.join(ROOT, "submit_44", "model",
                                 f"din_{br}_s{sd}.npz"), allow_pickle=False)
        m = Z.DINBST(n_num, cards, n_state, n_cell, mode="din")
        sd_t = {k.replace("__", "."): torch.from_numpy(W[k]) for k in W.files}
        m.load_state_dict(sd_t)
        m.eval()
        with torch.no_grad():
            pt = torch.sigmoid(m(torch.from_numpy(Xn), torch.from_numpy(Xc),
                                 torch.from_numpy(S), torch.from_numpy(C),
                                 torch.from_numpy(Mk), torch.from_numpy(cur))
                               ).numpy().astype(np.float64)
        pn = DN.din_forward(Xn, Xc, S, C, Mk, cur, W)
        d = float(np.abs(pt - pn).max())
        worst = max(worst, d)
        print(f"  {br:8s} s{sd:<4d} 최대차 {d:.3e}  "
              f"상관 {np.corrcoef(pt, pn)[0,1]:.6f}")
print(f"\n  최악 {worst:.3e}   "
      + ("동치 (float32 반올림 수준)" if worst < 1e-4 else "**불일치 — 배치가 딴 모델이다**"))
