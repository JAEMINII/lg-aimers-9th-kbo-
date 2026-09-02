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
f = d["frame"]; m = d["m_va"]; y = f.loc[m, "control_success"].to_numpy(float)
reg = f.loc[m, "game_type"].to_numpy() == 0
tab = np.load("nn_experiments/tabm_base_gate.npy")
old = np.load("nn_experiments/tmfull2024_base44.npy")
cur_new = np.load("nn_experiments/pcgpu2024_c12_cmh.npy")
new = np.load("nn_experiments/tmfull2024_c12_cmh_tm_all.npy")
for name, p in [("35eq", old), ("41eq", .3*old + .7*cur_new),
                ("tmfull", new), ("41tmfull", .3*old + .7*new)]:
    raw = .7*tab + .3*p
    print(name)
    for label, mask in [("all", np.ones(len(y),bool)), ("regular",reg), ("futures",~reg)]:
        b = minimize_scalar(lambda s: -score(cal(raw,s)[mask], y[mask]),
                            bounds=(-.15,.15), method="bounded")
        print(" ", label, "fixed", round(score(cal(raw)[mask], y[mask]),3),
              "best", round(-b.fun,3), "shift", round(b.x,5))
print("candidate weight inside new CatBoost")
for w in np.arange(0, 1.01, .1):
    p = .3*old + .7*((1-w)*cur_new + w*new)
    raw = .7*tab + .3*p
    print(round(float(w),1), round(score(cal(raw)[reg], y[reg]),3),
          round(score(cal(raw), y),3))
print("overall CatBoost weight")
for wc in np.arange(0, .61, .05):
    pcur = (1-wc)*tab + wc*(.3*old + .7*cur_new)
    ptm = (1-wc)*tab + wc*(.3*old + .7*new)
    print(round(float(wc),2), round(score(cal(pcur)[reg], y[reg]),3),
          round(score(cal(ptm)[reg], y[reg]),3))
