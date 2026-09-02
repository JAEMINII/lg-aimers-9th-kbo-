# -*- coding: utf-8 -*-
"""DIN/tab 퓨처스브랜치 신체제(2023+) 한정 학습 — VS=2024 폴드, F조각 판정용.
저장: dtn_2024_din.npy (3시드 라우팅 완성본), dtn_2024_tab.npy (2시드)."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
sys.path.insert(0, ROOT + "/nn_experiments")
os.environ.setdefault("AIMERS_ROOT", "/root")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
VS = 2024

# ---------- DIN (ctr_zoo)
import ctr_zoo as Z                                             # noqa: E402
Z.VS = VS
D = Z.build_inputs()
season, isf = D["season"], D["isf"]
gate = np.where(season == VS)[0]
tr = np.where(D["m_tr"])[0]
w = np.where(isf & (season <= 2022), 0.1, 1.0)
isf_g = isf[gate]
P = []
for sd in (42, 1, 777):
    ps_all = Z.fit("DIN", D, tr, w, sd, gate)
    ps_reg = Z.fit("DIN", D, tr[~isf[tr]], w, sd, gate)
    newf = tr[isf[tr] & (season[tr] >= 2023)]
    ps_fut = Z.fit("DIN", D, newf, w, sd, gate)
    P.append(np.where(isf_g, 0.6 * ps_all + 0.4 * ps_fut,
                      0.6 * ps_all + 0.4 * ps_reg))
    print(f"DIN newF s{sd} 완료", flush=True)
np.save(DL + f"/dtn_{VS}_din.npy", np.mean(P, 0))

# ---------- tab (지인 피처 경로, tab_decay 기계)
import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
LR1 = 3e-3
EP2 = {"all": 1, "regular": 1, "futures": 4}
d0 = F.build(DATA, VS=VS)
season2 = d0["season"].astype(np.float64)
isf2 = d0["is_f"]
y2 = d0["y"].astype(np.float64)
tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                          encoding="utf-8-sig"))
tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id"])
pos = pd.Series(np.arange(len(tr_sorted)), index=tr_sorted["row_id"].to_numpy())
pos_map = pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()
old_all = season2 <= 2022
gate2 = np.where(season2 == VS)[0]
isf_g2 = isf2[gate2]
G.gate, G.yv = gate2, y2[gate2]
hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
cols = list(Xs.columns)
Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
ci_fr = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
c4 = np.where(old_all & isf2, 0.0,
              np.where(old_all & ~isf2, 1.0,
                       np.where(isf2, 2.0, 3.0))).astype(np.float32)[:, None]
Xn, Xc, cards = G.prep(np.concatenate([Xfr, c4], 1),
                       season2 < VS, ci_fr + [Xfr.shape[1]])
G.Xn, G.cards = Xn, cards
G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
tr_idx = np.where(season2 < VS)[0]
t_isf = isf2[tr_idx]
w10 = np.where(t_isf & old_all[tr_idx], 0.1, 1.0).astype(np.float64)
P2 = []
for sd in (42, 1):
    Pb = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)), ("regular", ~t_isf),
                    ("futures", t_isf & (season2[tr_idx] >= 2023))):
        idx, ww = tr_idx[sel], w10[sel]
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=sd, tag=f"tabNF {br} s{sd}")
        s2 = idx[season2[idx] == VS - 1]
        pr = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=pr, seed=sd + e,
                    tag=f"tabNF {br} s{sd} S2e{e+1}")
        Pb[br] = G.predict(m, gate2)
        del m
        torch.cuda.empty_cache()
    P2.append(np.where(isf_g2, 0.6 * Pb["all"] + 0.4 * Pb["futures"],
                       0.6 * Pb["all"] + 0.4 * Pb["regular"]))
    print(f"tab newF s{sd} 완료", flush=True)
np.save(DL + f"/dtn_{VS}_tab.npy", np.mean(P2, 0))
print("dt_newf 끝", flush=True)
