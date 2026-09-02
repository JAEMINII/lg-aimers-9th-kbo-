import os, sys
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar

sys.path.insert(0, "submit_35")
import features44

def score(p, y):
    r = y.mean()
    return 1e5 * (1.0 - ((p-y)**2).mean() / (r*(1-r)))

def cal(p, s=-.007):
    return expit(1.0279 * logit(np.clip(p, 1e-6, 1-1e-6)) + s)

d = features44.build(os.path.join(".", "open (1)", "data"), VS=2024, return_frame=True)
frame = d["frame"]
m = d["m_va"]
y = frame.loc[m, "control_success"].to_numpy(float)
reg = frame.loc[m, "game_type"].to_numpy() == 0
tab = np.load("nn_experiments/tabm_base_gate.npy")
old = np.load("nn_experiments/tm2024_base44_10.npy")
tm = np.load("nn_experiments/tm2024_c12_cmh_tm_sd2_10.npy")

for n, p in [("35eq", old), ("tm", tm), ("41tm", .3*old + .7*tm)]:
    raw = .7*tab + .3*p
    b = minimize_scalar(lambda s: -score(cal(raw, s)[reg], y[reg]),
                        bounds=(-.15, .15), method="bounded")
    print(n, "fixed", round(score(cal(raw)[reg], y[reg]), 3),
          "best", round(-b.fun, 3), "shift", round(b.x, 5))
for w in np.arange(0, 1.01, .1):
    raw = .7*tab + .3*((1-w)*old + w*tm)
    print("tmweight", round(float(w), 1), "fixed", round(score(cal(raw)[reg], y[reg]), 3))
