# -*- coding: utf-8 -*-
"""등판 강도와 저카디널리티 범주화가 더해지는지 겹치는지 8시드로 잰다.

각각의 결과 (기준 = 시즌가중 3.5, 혼합 기준, 8시드 짝비교)
    등판 강도 2열   +2.4 ± 0.7   8/8 양수   t=3.41   확정
    범주화 6열      +2.0 ± 1.2   7/8 양수   t=1.77   준확정

둘 다 상관이 거의 안 변했다(기준↔변형 0.997). 다양성으로 이기는 게 아니라
같은 모델을 조금 더 잘 맞게 만드는 쪽이다. 그러면 같은 것을 건드려서
겹칠 가능성이 있다.

네 벌을 같은 시드로 학습해 짝지어 본다.
    기준 / 부하만 / 범주화만 / 둘 다
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
SEEDS = (42, 1, 777, 2, 3, 5, 11, 23)
DECAY = 3.5
LOWCARD = ["game_month", "game_dayofweek", "inning",
           "balls_before", "strikes_before", "outs_before"]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402
from load_gate import load_cols                                 # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)
    F44 = list(d["F44"])
    base = d["X44"]
    Xl = np.concatenate([base, load_cols(F44, base)], 1)
    ci_b = list(d["cat_idx"])
    ci_c = sorted(set(ci_b) | {F44.index(c) for c in LOWCARD})
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    M = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    def blend(r):
        return sc(0.10 * CB + 0.30 * M + 0.60 * r)

    def load(X, ci):
        Xn, Xc, cards = G.prep(X, G.m_tr, ci)
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    plans = [("기준", base, ci_b), ("부하", Xl, ci_b),
             ("범주화", base, ci_c), ("둘다", Xl, ci_c)]
    P = {n: {} for n, _, _ in plans}
    for sd in SEEDS:
        t0 = time.time()
        for name, X, ci in plans:
            load(X, ci)
            P[name][sd] = one(sd, decay=DECAY)
        G.log(f"  seed {sd}  " + "  ".join(
            f"{n} {blend(P[n][sd]):.1f}" for n, _, _ in plans) +
            f"   {time.time()-t0:.0f}s")

    G.log("")
    G.log("  구성      혼합 8시드평균   짝차이(기준대비)")
    b0 = np.mean([P["기준"][s] for s in SEEDS], 0)
    G.log(f"  {'기준':8s} {blend(b0):14.1f}")
    for name in ("부하", "범주화", "둘다"):
        r = np.mean([P[name][s] for s in SEEDS], 0)
        dif = [blend(P[name][s]) - blend(P["기준"][s]) for s in SEEDS]
        m, sd_ = float(np.mean(dif)), float(np.std(dif, ddof=1))
        se = sd_ / np.sqrt(len(dif))
        G.log(f"  {name:8s} {blend(r):14.1f}   {m:+6.1f} ± {se:.1f}  "
              f"{sum(1 for x in dif if x>0)}/{len(dif)}  t={m/se:.2f}  "
              f"{'확정' if abs(m) > 2*se else '미확정'}")
        np.save(os.path.join(OUT, f"stack_{name}.npy"), r)

    a = np.mean([blend(P["부하"][s]) - blend(P["기준"][s]) for s in SEEDS])
    b = np.mean([blend(P["범주화"][s]) - blend(P["기준"][s]) for s in SEEDS])
    c = np.mean([blend(P["둘다"][s]) - blend(P["기준"][s]) for s in SEEDS])
    G.log("")
    G.log(f"  따로 더하면 {a:+.1f} + {b:+.1f} = {a+b:+.1f}   같이 걸면 {c:+.1f}")
    G.log(f"  -> {'더해진다' if c > a + b - 1.0 else '겹친다'}")
