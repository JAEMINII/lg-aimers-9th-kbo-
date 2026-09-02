import os,sys
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
sys.path.insert(0,"submit_35")
import features44
def score(p,y):
 r=y.mean(); return 1e5*(1-((p-y)**2).mean()/(r*(1-r)))
def cal(p,s=-.007): return expit(1.0279*logit(np.clip(p,1e-6,1-1e-6))+s)
d=features44.build(os.path.join(".","open (1)","data"),VS=2024,return_frame=True)
f=d["frame"]; m=d["m_va"]; y=f.loc[m,"control_success"].to_numpy(float); reg=f.loc[m,"game_type"].to_numpy()==0
tab0=np.load("nn_experiments/tabm_base_gate.npy")
tab1=np.load("nn_experiments/tabm_tm2s_ep4_gate.npy")
cb0=np.load("nn_experiments/pcgpu2024_base44_10.npy")
cb1=np.load("nn_experiments/pcgpu2024_c12_cmh_10.npy")
cb_tm=np.load("nn_experiments/tm2s2024_c12_cmh_tm_2s_10.npy")
pcur=.3*cb0+.7*cb1
ptm=.3*cb0+.7*cb_tm
for n,p in [("tab0",tab0),("tab_tm2s_ep4",tab1)]:
 print(n,"all",round(score(cal(p),y),3),"reg",round(score(cal(p)[reg],y[reg]),3),"fut",round(score(cal(p)[~reg],y[~reg]),3))
for n,t in [("current",tab0),("tm_tab",tab1)]:
 for c in [pcur,ptm,np.where(reg,pcur,ptm)]:
  raw=.7*t+.3*c
  print(n,"cb",("tm" if c is ptm else "mixed"),"all",round(score(cal(raw),y),3),"reg",round(score(cal(raw)[reg],y[reg]),3),"fut",round(score(cal(raw)[~reg],y[~reg]),3))
print("blend tab weight current CB fixed")
for w in np.arange(0,1.01,.1):
 raw=w*tab1+(1-w)*tab0
 print(round(float(w),1),round(score(cal(raw),y),3),round(score(cal(raw)[reg],y[reg]),3),round(score(cal(raw)[~reg],y[~reg]),3))

