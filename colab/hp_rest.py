# -*- coding: utf-8 -*-
"""HP 스윕에서 못 돌린 나머지. epoch 축은 관문이 못 맞히므로 뺐다.

    ep4 를 관문은 ep2 보다 +5.5 로 쳤는데 리더보드는 -61 이었다.
    학습량이 다른 설정끼리는 관문이 순위를 못 매긴다.
    아래 셋은 학습량이 아니라 구조·가중을 바꾸는 것이라 그 함정에서 자유롭다.
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)

CASES = {
    "ep2":       dict(),
    "ep2_k64":   dict(k=64),          # BatchEnsemble 멤버 2배
    "ep2_nos2":  dict(stage2=False),  # 마지막 시즌 미세조정이 정말 필요한가
    "ep2_sw2":   dict(decay=2.0),     # 우리 CatBoost 를 +22 올렸던 시즌가중
}

if __name__ == "__main__":
    d = F.build(DATA, VS=int(os.environ.get("VS", 2024)))
    Xn, Xc, cards = G.prep(d["X44"], G.m_tr, d["cat_idx"])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    G.log("\n" + "=" * 72)
    G.log("  설정         시드평균 단독   제출비중   개별 시드")
    G.log("=" * 72)
    ref = None
    for name, kw in CASES.items():
        t0 = time.time()
        ps = [one(sd, **kw) for sd in SEEDS]
        solos = [F.best_shift(p[FULL], yv[FULL])[0] for p in ps]
        r = np.mean(ps, 0)
        solo = F.best_shift(r[FULL], yv[FULL])[0]
        bl = F.best_shift((0.10 * CB + 0.30 * MLPF + 0.60 * r)[FULL], yv[FULL])[0]
        np.save(os.path.join(OUT, f"hp_{name}.npy"), r)
        dl = "" if ref is None else f"   기준대비 {solo-ref[0]:+6.1f} / {bl-ref[1]:+6.1f}"
        if ref is None:
            ref = (solo, bl)
        G.log(f"  {name:12s} {solo:7.1f}      {bl:7.1f}   "
              f"[{', '.join(f'{v:.0f}' for v in solos)}]   {time.time()-t0:.0f}s{dl}")
