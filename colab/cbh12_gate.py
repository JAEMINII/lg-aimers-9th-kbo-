# -*- coding: utf-8 -*-
"""최근폼4 제거 CB 클론 (feats54 = 58 - prev1/3dev - trend1v5x2). VS2024/2023."""
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

hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
DROP = ["pitcher_prev1_success_dev", "pitcher_prev3_success_dev",
        "pitcher_success_trend_1v5", "pitcher_middle_trend_1v5"]


def bss(p, t):
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


for VS in (2024, 2023):
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
    feats54 = feats50 + [f"hand_match_{p}_{b}" for p in (1, 2) for b in (1, 2)]
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
    X = fr[feats54].to_numpy(np.float32)
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
    np.save(DL + f"/cbh12_{VS}_nrf.npy", p)
    b = np.load(DL + f"/cbh4_{VS}_b58.npy")
    m = ~isf_t if VS == 2023 else np.ones(len(yv), bool)
    print(f"VS{VS} CB단독 nrf-b58  {bss(p[m], yv[m]) - bss(b[m], yv[m]):+.2f}",
          flush=True)
print("cbh12 끝", flush=True)
