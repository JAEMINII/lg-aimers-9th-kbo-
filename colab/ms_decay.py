# -*- coding: utf-8 -*-
"""recency 가중의 마지막 미검 칸 — MS 감사판 + 시즌감쇠. VS=2024, 3시드.

전적: CatBoost 감쇠 배치중(+25) / TabM 감쇠 0/3 / DIN 감쇠 0/3. MS 만 미검.
"""
import os, sys, time
import numpy as np, pandas as pd, torch

os.environ.setdefault("AUDIT_MODE", "all"); os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
ROOT = "/root/aimers"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/colab")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
import features44 as F
import multistate_softmax as M
import multistate_auditfeat as A
import tabm_gate_gpu as G
from train_chan_3 import preprocess as PP

raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
aux = M.recover_state(raw)
VS = int(os.environ.get("MSD_VS", "2024"))
built = F.build(DATA, VS=VS)
X44 = built["X44"].astype(np.float32); names = list(built["F44"])
xnum, _, xcat, _ = A.audit_features(raw, X44, names)
c4 = np.where(old & isf, 0., np.where(old & ~isf, 1., np.where(isf, 2., 3.))).astype(np.float32)[:, None]
Xin = np.concatenate([X44, xnum, c4, xcat], 1)
ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
c4i = X44.shape[1] + xnum.shape[1]
cat_idx = ci + [c4i] + list(range(c4i + 1, Xin.shape[1]))
m_tr = season < VS
Xn, Xc, cards = G.prep(Xin, m_tr, cat_idx)
G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
tr = np.flatnonzero(m_tr); va = np.flatnonzero(season == VS)
yv = y[va].astype(float)
base_w = np.ones(len(y), np.float32)
base_w[tr] = np.where(isf[tr] & old[tr], .1, 1.)
if VS == 2024:
    cb = np.load(DL + "/ta2024_base.npy").mean(0)
    tab = 0.6*np.load(DL + "/c4g_2024_c4_all_s42.npy") + 0.4*np.load(DL + "/c4g_2024_c4_regular_s42.npy")
else:
    cb = np.load(DL + "/cb50fixed_2022.npy").astype(np.float64)
    tab = np.load(DL + "/h2h_2022_friend_s42.npy").astype(np.float64)
din = np.load(DL + f"/dg_{VS}_DIN.npy").mean(0)
sc = lambda p: M.best_shift(p, yv)[0]
for nm, dec in (("base", 1.0), ("decay15", 1.5)):
    ps = []
    for sd in M.SEEDS:
        w = base_w.copy()
        w[tr] = w[tr] * (dec ** (season[tr].astype(np.float32) - 2019.0)) if dec > 1 else w[tr]
        m = M.train_model(M.make_model(Xn.shape[1], cards, sd), tr, y, aux, w, sd, f"DEC-{nm} s{sd}")
        d, _ = M.predict(m, va); ps.append(d)
        del m; torch.cuda.empty_cache()
    p = np.mean(ps, 0)
    mix = sc(0.31*cb + 0.13*tab + 0.31*din + 0.25*p)
    print(f"ARM {nm:8s} MS단독 {sc(p):8.1f}   4원혼합 {mix:8.1f}", flush=True)
