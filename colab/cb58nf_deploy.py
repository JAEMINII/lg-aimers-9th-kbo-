# -*- coding: utf-8 -*-
"""71 배치 CB — 58피처, 퓨처스 브랜치만 신체제(2023+) F 로 학습. trees58nf.npz"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
sys.path.insert(0, ROOT + "/nn_experiments")
os.environ.setdefault("AIMERS_ROOT", "/root")
os.environ.setdefault("C12_OUT", "/root/_c12logs")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
import features44                                                # noqa: E402
from train_c12_submit import add_c12, add_cmh, export_models     # noqa: E402
from catboost import CatBoostClassifier                          # noqa: E402

built = features44.build(DATA, VS=2025, return_frame=True)
frame, c12 = add_c12(built["frame"].copy(), return_tables=True)
frame, cmh = add_cmh(frame, return_tables=True)
base = list(built["F44"]) + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                             "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
for ph in (1, 2):
    for bh in (1, 2):
        frame[f"hand_match_{ph}_{bh}"] = (
            (frame["pitcher_hand"] == ph) & (frame["batter_hand"] == bh)
        ).astype(np.float32)
car = frame["asof_pitcher_success_rate"]
frame["pitcher_prev1_success_dev"] = (
    frame["asof_pitcher_prev1_game_success_rate"] - car).astype(np.float32)
frame["pitcher_prev3_success_dev"] = (
    frame["asof_pitcher_prev3_game_success_rate"] - car).astype(np.float32)
frame["pitcher_success_trend_1v5"] = (
    frame["asof_pitcher_prev1_game_success_rate"]
    - frame["asof_pitcher_prev5_game_success_rate"]).astype(np.float32)
frame["pitcher_middle_trend_1v5"] = (
    frame["asof_pitcher_prev1_game_middle_rate"]
    - frame["asof_pitcher_prev5_game_middle_rate"]).astype(np.float32)
added = [f"hand_match_{p}_{b}" for p in (1, 2) for b in (1, 2)] + [
    "pitcher_prev1_success_dev", "pitcher_prev3_success_dev",
    "pitcher_success_trend_1v5", "pitcher_middle_trend_1v5"]
features = base + added
x = frame[features].to_numpy(np.float32)
y = frame["control_success"].to_numpy(np.float32)
season = frame["season"].to_numpy(np.float64)
gt = frame["game_type"].to_numpy()
w = 2.0 ** (season - 2019.0)
hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
isf_all = (gt == 1) | (gt == "F")
masks = {"all": np.ones(len(y), bool), "regular": ~isf_all,
         "futures": isf_all & (season >= 2023)}
models = {k: [] for k in masks}
for br, mk in masks.items():
    for sd in (1, 42, 777):
        t0 = time.time()
        mo = CatBoostClassifier(random_seed=sd, **hp)
        mo.fit(x[mk], y[mk], sample_weight=w[mk])
        models[br].append(mo)
        print(f"{br} s{sd} {time.time()-t0:.0f}s", flush=True)
tree_keys = {"cb_feat", "cb_thr", "cb_leaf", "cb_nan_left", "cb_depth",
             "sp_feat", "sp_thr", "sp_idx", "n_models", "n_per_group",
             "blend_full", "gt_col", "features", "calib_logit_shift"}
with np.load("/root/trees_1129_original.npz", allow_pickle=False) as z:
    hist = {k: z[k] for k in z.files if k not in tree_keys}
hist.update(c12)
hist.update(cmh)
export_models(models, features, hist, "/root/trees58nf.npz")
print("EXPORT DONE", flush=True)
