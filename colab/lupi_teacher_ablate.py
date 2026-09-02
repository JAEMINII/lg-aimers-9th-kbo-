# -*- coding: utf-8 -*-
"""Teacher ablation: isolate current pitch-type versus continuous physics."""
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
MATCH = os.environ.get("LUPI_MATCH", "lupi_match_v2.csv.gz")
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]

import features44 as F                                          # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402


def main():
    d = F.build(DATA, VS=2024, return_frame=True)
    fr, x44 = d["frame"], d["X44"].astype(np.float32)
    season, y = fr.season.to_numpy(), fr.control_success.to_numpy(np.float64)
    m = pd.read_csv(os.path.join(DL, MATCH), encoding="utf-8-sig")
    m["ptg"] = m.pitch_type_group.map({"fastball": 0, "breaking": 1, "offspeed": 2}).fillna(3)
    pos = pd.Series(np.arange(len(fr)), index=fr.row_id).reindex(m.row_id).to_numpy()
    m, pos = m[np.isfinite(pos)], pos[np.isfinite(pos)].astype(np.int64)
    phys = np.full((len(fr), len(PHYS)), np.nan, np.float32)
    ptg = np.full((len(fr), 4), np.nan, np.float32)
    phys[pos] = m[PHYS].to_numpy(np.float32)
    g = m.ptg.to_numpy(np.int64)
    ptg[pos] = np.eye(4, dtype=np.float32)[g]
    has = np.zeros(len(fr), bool); has[pos] = True
    tr, va = (season < 2024) & has, (season == 2024) & has
    arms = {"base": x44, "type": np.c_[x44, ptg], "phys": np.c_[x44, phys],
            "both": np.c_[x44, phys, ptg]}
    hp = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
              verbose=0, allow_writing_files=False)
    for name, x in arms.items():
        out, t0 = [], time.time()
        for seed in (1, 42, 777):
            model = CatBoostClassifier(random_seed=seed, **hp)
            model.fit(x[tr], y[tr])
            out.append(model.predict_proba(x[va])[:, 1])
        print(f"{name:6s} {F.best_shift(np.mean(out, 0), y[va])[0]:8.1f} {time.time()-t0:5.0f}s", flush=True)


if __name__ == "__main__":
    main()
