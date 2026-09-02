# -*- coding: utf-8 -*-
"""Score a saved LUPI auxiliary tabular branch in the 44/45/47 mixtures."""
import os
import sys
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
VS = int(os.environ.get("SCORE_VS", "2024"))
ARMS = tuple(os.environ.get("SCORE_ARMS", "base,type005,type02").split(","))

import features44 as F                                          # noqa: E402

d = F.build(DATA, VS=VS)
gate = np.flatnonzero(d["season"] == VS)
y = d["y"][gate]
if VS == 2024:
    cb = .7*np.load(os.path.join(DL, "pcgpu2024_c12_cmh_10.npy")) + .3*np.load(os.path.join(DL, "pcgpu2024_base44_10.npy"))
    ms = np.load(os.path.join(DL, "ms24_audit_direct.npy"))
else:
    cb, ms = np.load(os.path.join(DL, f"cb50fixed_{VS}.npy")), np.load(os.path.join(DL, "ms22_audit_direct.npy"))
din = np.load(os.path.join(DL, f"dg_{VS}_DIN.npy")).mean(0)
pred = {a: np.load(os.path.join(DL, f"lt_{VS}_{a}.npy")) for a in ARMS}
ref = pred["base"]
for name, w in {"44": (.225,.525,.25,0), "45": (.4,.2,.4,0), "47": (.31,.13,.31,.25)}.items():
    base = [F.best_shift(w[0]*cb+w[1]*p+w[2]*din+w[3]*ms, y)[0] for p in ref]
    print(name, "base", [round(x,2) for x in base], flush=True)
    for a in ARMS:
        if a == "base": continue
        now = [F.best_shift(w[0]*cb+w[1]*p+w[2]*din+w[3]*ms, y)[0] for p in pred[a]]
        dd = np.asarray(now)-base
        print(f"  {a:8s} {dd.mean():+.3f} {int((dd>0).sum())}/3  {[round(x,3) for x in dd]}", flush=True)
