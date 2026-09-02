# -*- coding: utf-8 -*-
"""+20 탐색 2부 — CatBoost 감쇠 강화 (2.0->2.5/3.0) + DIN 감쇠 1.5. VS=2024."""
import os, sys, time
import numpy as np, pandas as pd, torch
ROOT = "/root/aimers"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/colab")
sys.path.insert(0, ROOT + "/colab")
os.environ.setdefault("AIMERS_ROOT", "/root")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
import features44 as F
import train_c12_submit as T
from catboost import CatBoostClassifier

# ---------- CatBoost 감쇠 스윕 (배치 50열 그대로)
built = F.build(DATA, VS=2024, return_frame=True)
fr = built["frame"]; base_f = list(built["F44"])
fr, _ = T.add_c12(fr, return_tables=True)
fr, _ = T.add_cmh(fr, return_tables=True)
feats = base_f + ["pc_c12_rate","pc_c12_dev","pc_c12_n","pc_cmh_rate","pc_cmh_dev","pc_cmh_n"]
X = fr[feats].to_numpy(np.float32)
y = fr["control_success"].to_numpy(np.float64)
season = fr["season"].to_numpy(); gt = fr["game_type"].to_numpy()
tr_m, te_m = season < 2024, season == 2024
isf_t = gt[te_m] == 1; yv = y[te_m]
hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
masks = {"all": np.ones(int(tr_m.sum()), bool), "regular": gt[tr_m]==0, "futures": gt[tr_m]==1}
sc = lambda p: F.best_shift(p, yv)[0]
for dec in (2.0, 2.5, 3.0):
    w = dec ** (season[tr_m].astype(np.float64) - 2019.0)
    pg = {g: [] for g in masks}
    for g, m in masks.items():
        for sd in (1, 42, 777):
            mo = CatBoostClassifier(random_seed=sd, **hp)
            mo.fit(X[tr_m][m], y[tr_m][m], sample_weight=w[m])
            pg[g].append(mo.predict_proba(X[te_m])[:, 1])
    ps = [np.where(isf_t, .6*pg["all"][i]+.4*pg["futures"][i],
                   .6*pg["all"][i]+.4*pg["regular"][i]) for i in range(3)]
    p = np.mean(ps, 0)
    np.save(DL + f"/cbdec_{str(dec).replace('.','')}.npy", p)
    print(f"ARM CB decay{dec}  단독 {sc(p):8.1f}", flush=True)

# ---------- DIN 감쇠 1.5 (ctr_zoo)
import ctr_zoo as Z
Z.VS = 2024
D = Z.build_inputs()
season2, isf2 = D["season"], D["isf"]
gate = np.where(season2 == 2024)[0]
tr = np.where(D["m_tr"])[0]
isf_g = isf2[gate]
yv2 = D["y"].astype(np.float64)[gate]
sc2 = lambda p: F.best_shift(p, yv2)[0]
for nm, dec in (("din_d15", 1.5),):
    w = np.where(isf2 & (season2 <= 2022), 0.1, 1.0)
    w = w * (dec ** (season2.astype(np.float64) - 2019.0))
    P = []
    for sd in (42, 1, 777):
        ps = [Z.fit("DIN", D, tr[sel], w, sd, gate)
              for sel in (np.ones(len(tr), bool), ~isf2[tr], isf2[tr])]
        P.append(np.where(isf_g, .6*ps[0]+.4*ps[2], .6*ps[0]+.4*ps[1]))
    p = np.mean(P, 0)
    np.save(DL + f"/{nm}.npy", np.asarray(P))
    print(f"ARM {nm}  단독 {sc2(p):8.1f}", flush=True)
