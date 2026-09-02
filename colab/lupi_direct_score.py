"""Score saved LUPI arms by themselves on a held-out season."""
import os
import sys

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
VS = int(os.environ.get("SCORE_VS", "2023"))
ARMS = tuple(os.environ.get("SCORE_ARMS", "base,type02").split(","))

import features44 as F

d = F.build(DATA, VS=VS)
gate = np.flatnonzero(d["season"] == VS)
y = d["y"][gate]
ref = np.load(os.path.join(DL, f"lt_{VS}_base.npy"))
for arm in ARMS:
    p = np.load(os.path.join(DL, f"lt_{VS}_{arm}.npy"))
    scores = np.asarray([F.best_shift(x, y)[0] for x in p])
    msg = f"{arm:8s} direct {[round(x, 3) for x in scores]} mean={scores.mean():.3f}"
    if arm != "base":
        base = np.asarray([F.best_shift(x, y)[0] for x in ref])
        delta = scores - base
        msg += f"  delta={delta.mean():+.3f} {int((delta > 0).sum())}/3 {[round(x,3) for x in delta]}"
    print(msg, flush=True)
