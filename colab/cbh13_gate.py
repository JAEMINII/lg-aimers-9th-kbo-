# -*- coding: utf-8 -*-
"""CB 강규제 스윕 — border_count x l2_leaf_reg, VS2024. 대조 cbh4_2024_b58."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
sys.path.insert(0, ROOT + "/nn_experiments")
os.environ.setdefault("AIMERS_ROOT", "/root")
os.environ.setdefault("C12_OUT", "/root/_c12logs")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402


def bss(p, t):
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


VS = 2024
built = F.build(DATA, VS=VS, return_frame=True)
fr = built["frame"]
base_f = list(built["F44"])
fr, _ = T.add_c12(fr, return_tables=True)
fr, _ = T.add_cmh(fr, return_tables=True)
feats50 = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                    "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
for ph in (1, 2):
    for bh in (1, 2):
        fr[f"hand_match_{ph}_{bh}"] = (
            (fr["pitcher_hand"] == ph) & (fr["batter_hand"] == bh)
        ).astype(np.float32)
car = fr["asof_pitcher_success_rate"]
fr["pitcher_prev1_success_dev"] = (
    fr["asof_pitcher_prev1_game_success_rate"] - car).astype(np.float32)
fr["pitcher_prev3_success_dev"] = (
    fr["asof_pitcher_prev3_game_success_rate"] - car).astype(np.float32)
fr["pitcher_success_trend_1v5"] = (
    fr["asof_pitcher_prev1_game_success_rate"]
    - fr["asof_pitcher_prev5_game_success_rate"]).astype(np.float32)
fr["pitcher_middle_trend_1v5"] = (
    fr["asof_pitcher_prev1_game_middle_rate"]
    - fr["asof_pitcher_prev5_game_middle_rate"]).astype(np.float32)
feats58 = feats50 + [f"hand_match_{p}_{b}" for p in (1, 2) for b in (1, 2)] \
    + ["pitcher_prev1_success_dev", "pitcher_prev3_success_dev",
       "pitcher_success_trend_1v5", "pitcher_middle_trend_1v5"]
season = fr["season"].to_numpy()
gtr = fr["game_type"].to_numpy()
isf_all = (gtr == 1) | (gtr == "1") | (gtr == "F")
tr_m = season < VS
va_m = season == VS
y = fr["control_success"].to_numpy(np.float64)
yv = y[va_m]
isf_t = isf_all[va_m]
isf_tr = isf_all[tr_m]
w = 2.0 ** (season[tr_m].astype(np.float64) - 2019.0)
X = fr[feats58].to_numpy(np.float32)
b = np.load(DL + "/cbh4_2024_b58.npy")
for bc in (32, 16):
    for l2 in (100.0, 300.0):
        hp = dict(iterations=400, learning_rate=0.05, depth=4,
                  l2_leaf_reg=l2, border_count=bc, verbose=0,
                  allow_writing_files=False, task_type="GPU", devices="0")
        ps = []
        for sd_ in (1, 42, 777):
            pg = {}
            for g, mk in (("all", np.ones(int(tr_m.sum()), bool)),
                          ("regular", ~isf_tr), ("futures", isf_tr)):
                mo_ = CatBoostClassifier(random_seed=sd_, **hp)
                mo_.fit(X[tr_m][mk], y[tr_m][mk], sample_weight=w[mk])
                pg[g] = mo_.predict_proba(X[va_m])[:, 1]
            ps.append(np.where(isf_t, .6 * pg["all"] + .4 * pg["futures"],
                               .6 * pg["all"] + .4 * pg["regular"]))
        p = np.mean(ps, 0)
        np.save(DL + f"/cbh13_bc{bc}_l2{int(l2)}.npy", p)
        print(f"bc{bc} l2{int(l2)}  CB단독 {bss(p, yv) - bss(b, yv):+.2f}",
              flush=True)
print("cbh13 끝", flush=True)
