# -*- coding: utf-8 -*-
"""tab(지인 피처 경로) 시즌감쇠 쌍대조 — 배치 f2b 는 감쇠가 없다 (구퓨처스 0.1 만).

MS/CB 는 감쇠가 +20급, DIN 은 -6 — 계열별이라 tab 은 직접 재야 한다.
같은 시드로 base(현행 가중) vs decay15 를 쌍으로 학습해 잡음을 상쇄한다.
저장: td_{VS}_{arm}_s{sd}.npy (라우팅 0.6/0.4 완성본)."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
os.environ.setdefault("AIMERS_ROOT", "/root")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
SEEDS = (42, 1)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
FOLDS = (2024, 2022, 2023)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def one(seed, tr_idx, t_isf, w, season, VS, gate, isf_g):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"VS{VS} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        pr = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=pr, seed=seed + e,
                    tag=f"VS{VS} {br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(isf_g, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


d0 = F.build(DATA, VS=2024)
season = d0["season"].astype(np.float64)
isf = d0["is_f"]
y = d0["y"].astype(np.float64)
tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                          encoding="utf-8-sig"))
tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id"])
pos = pd.Series(np.arange(len(tr_sorted)), index=tr_sorted["row_id"].to_numpy())
pos_map = pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()
old_all = season <= OLD_F_MAX

for VS in FOLDS:
    gate = np.where(season == VS)[0]
    isf_g = isf[gate]
    yv = y[gate]
    G.gate, G.yv = gate, yv
    m_tr = season < VS
    hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
    Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
    cols = list(Xs.columns)
    Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
    ci_fr = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    c4 = np.where(old_all & isf, 0.0,
                  np.where(old_all & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xn, Xc, cards = G.prep(np.concatenate([Xfr, c4], 1), m_tr,
                           ci_fr + [Xfr.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    tr_idx = np.where(m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old_all[tr_idx], OLD_W, 1.0).astype(np.float64)
    wdec = w10 * 1.5 ** (season[tr_idx] - 2019.0)
    for arm, w in (("base", w10), ("dec15", wdec)):
        for sd in SEEDS:
            t0 = time.time()
            p = one(sd, tr_idx, t_isf, w, season, VS, gate, isf_g)
            np.save(DL + f"/td_{VS}_{arm}_s{sd}.npy", p)
            print(f"  td VS{VS} {arm} s{sd}  {time.time()-t0:.0f}s  "
                  f"단독 {F.best_shift(p, yv)[0]:.1f}", flush=True)
print("tab 감쇠 쌍대조 끝", flush=True)
