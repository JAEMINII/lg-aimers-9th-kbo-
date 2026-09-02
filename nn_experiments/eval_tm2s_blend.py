import os, sys
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
sys.path.insert(0, "submit_35")
import features44

def score(p,y):
    r=y.mean()
    return 1e5*(1-((p-y)**2).mean()/(r*(1-r)))
def cal(p,s=-.007):
    return expit(1.0279*logit(np.clip(p,1e-6,1-1e-6))+s)
d=features44.build(os.path.join(".", "open (1)", "data"),VS=2024,return_frame=True)
f=d["frame"]; m=d["m_va"]; y=f.loc[m,"control_success"].to_numpy(float)
reg=f.loc[m,"game_type"].to_numpy()==0
tab=np.load("nn_experiments/tabm_base_gate.npy")
base=np.load("nn_experiments/pcgpu2024_base44.npy")
cur=np.load("nn_experiments/pcgpu2024_c12_cmh.npy")
tm=np.load("nn_experiments/tm2s2024_c12_cmh_tm_2s.npy")
for name,p in [("41eq",.3*base+.7*cur),("tm2s",tm),("41tm2s",.3*base+.7*tm)]:
    raw=.7*tab+.3*p
    print(name)
    for lab,mask in [("all",np.ones(len(y),bool)),("reg",reg),("fut",~reg)]:
        b=minimize_scalar(lambda s:-score(cal(raw,s)[mask],y[mask]),bounds=(-.15,.15),method="bounded")
        print(lab,"fixed",round(score(cal(raw)[mask],y[mask]),3),"best",round(-b.fun,3),"shift",round(b.x,5))
print("inside-new tm2s weight")
for w in np.arange(0,1.01,.1):
    p=.3*base+.7*((1-w)*cur+w*tm)
    raw=.7*tab+.3*p
    print(round(float(w),1),round(score(cal(raw)[reg],y[reg]),3),round(score(cal(raw),y),3))
print("overall CB weight baseline/candidate")
for wc in np.arange(0,.61,.05):
    q0=(1-wc)*tab+wc*(.3*base+.7*cur)
    q1=(1-wc)*tab+wc*(.3*base+.7*tm)
    print(round(float(wc),2),round(score(cal(q0)[reg],y[reg]),3),round(score(cal(q1)[reg],y[reg]),3))
print("10-seed exact current-vs-tm2s")
base10=np.load("nn_experiments/pcgpu2024_base44_10.npy")
cur10=np.load("nn_experiments/pcgpu2024_c12_cmh_10.npy")
tm10=np.load("nn_experiments/tm2s2024_c12_cmh_tm_2s_10.npy")
for name,p in [("41_10",.3*base10+.7*cur10),("41tm2s_10",.3*base10+.7*tm10)]:
    raw=.7*tab+.3*p
    print(name,"all",round(score(cal(raw),y),3),"reg",round(score(cal(raw)[reg],y[reg]),3),"fut",round(score(cal(raw)[~reg],y[~reg]),3))
