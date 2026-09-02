# -*- coding: utf-8 -*-
"""릴리스 흔들림 2열(tm_relh_sd, tm_rels_sd)을 시드 6개로 확인한다.

3시드에서 단독 +11.6 이 나왔다. 판정선(+5)의 두 배지만 유보가 하나 있다 —
시드별로 짝지으면 -3 / +9 / +21 로 부호가 갈린다. 시드를 늘려 확인한다.

같은 시드끼리 짝지어 비교한다(paired). 시드 간 편차가 24점이라 짝을 안 지으면
효과가 노이즈에 묻힌다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777, 2, 3, 5)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from trackman_gate2 import trackman_matrix, one                 # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)

if __name__ == "__main__":
    d = F.build(DATA, VS=int(os.environ.get("VS", 2024)))
    base = d["X44"]
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["pitcher_id", "season"])
    M, names = trackman_matrix(raw.season.to_numpy(), raw.pitcher_id.to_numpy(),
                               cols={"tm_relh_sd", "tm_rels_sd"})
    G.log(f"\n  추가 열 {names}   값 있는 행 {(~np.isnan(M[:,0])).mean()*100:.1f}%")
    X2 = np.concatenate([base, M], 1)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    P0, P2 = {}, {}
    G.log("\n  시드   base    +sd2    차이")
    for sd in SEEDS:
        t0 = time.time()
        P0[sd] = one(base, d["cat_idx"], sd)
        P2[sd] = one(X2, d["cat_idx"], sd)
        a = F.best_shift(P0[sd][FULL], yv[FULL])[0]
        b = F.best_shift(P2[sd][FULL], yv[FULL])[0]
        G.log(f"  {sd:4d}  {a:7.1f}  {b:7.1f}  {b-a:+7.1f}   {time.time()-t0:.0f}s")

    dif = [F.best_shift(P2[s][FULL], yv[FULL])[0] - F.best_shift(P0[s][FULL], yv[FULL])[0]
           for s in SEEDS]
    G.log(f"\n  짝지은 차이  평균 {np.mean(dif):+.1f}  표준편차 {np.std(dif, ddof=1):.1f}  "
          f"표준오차 {np.std(dif, ddof=1)/np.sqrt(len(dif)):.1f}  "
          f"양수 {sum(1 for x in dif if x>0)}/{len(dif)}")
    for tag, P in (("base", P0), ("+sd2", P2)):
        r = np.mean([P[s] for s in SEEDS], 0)
        solo = F.best_shift(r[FULL], yv[FULL])[0]
        G.log(f"  {tag:6s} 6시드평균 단독 {solo:7.1f}   "
              f"제출비중 {F.best_shift((0.10*CB+0.30*MLPF+0.60*r)[FULL], yv[FULL])[0]:7.1f}")
        np.save(os.path.join(OUT, f"sd2conf_{tag.strip('+')}.npy"), r)
    G.log("\n  TabM 비중을 올렸을 때 (비중은 관문으로 정하면 안 되지만 방향은 본다)")
    r2 = np.mean([P2[s] for s in SEEDS], 0)
    for w in (0.6, 0.7, 0.8, 0.9, 1.0):
        rest = 1 - w
        q = (rest / 4) * CB + (rest * 3 / 4) * MLPF + w * r2
        G.log(f"    TabM {w:.1f}   {F.best_shift(q[FULL], yv[FULL])[0]:7.1f}")
