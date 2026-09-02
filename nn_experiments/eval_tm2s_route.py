import os,sys
import numpy as np
from scipy.special import expit,logit
sys.path.insert(0,"submit_35")
import features44
def score(p,y):
 r=y.mean(); return 1e5*(1-((p-y)**2).mean()/(r*(1-r)))
def cal(p,s=-.007): return expit(1.0279*logit(np.clip(p,1e-6,1-1e-6))+s)
d=features44.build(os.path.join(".","open (1)","data"),VS=2024,return_frame=True)
f=d["frame"]; m=d["m_va"]; y=f.loc[m,"control_success"].to_numpy(float); reg=f.loc[m,"game_type"].to_numpy()==0
tab=np.load("nn_experiments/tabm_base_gate.npy")
base=np.load("nn_experiments/pcgpu2024_base44_10.npy")
cur=np.load("nn_experiments/pcgpu2024_c12_cmh_10.npy")
tm=np.load("nn_experiments/tm2s2024_c12_cmh_tm_2s_10.npy")
pcur=.3*base+.7*cur; ptm=.3*base+.7*tm
for name,p in [("current_all",pcur),("tm_all",ptm),("tm_fut_only",np.where(reg,pcur,ptm)),("tm_reg_only",np.where(reg,ptm,pcur))]:
 raw=.7*tab+.3*p
 print(name,"all",round(score(cal(raw),y),3),"reg",round(score(cal(raw)[reg],y[reg]),3),"fut",round(score(cal(raw)[~reg],y[~reg]),3))
print("separate CB blend")
for wf in np.arange(0,1.01,.1):
 p=np.where(reg,pcur,.3*base+.7*((1-wf)*cur+wf*tm))
 raw=.7*tab+.3*p
 print(round(float(wf),1),round(score(cal(raw),y),3),round(score(cal(raw)[reg],y[reg]),3),round(score(cal(raw)[~reg],y[~reg]),3))
print("CB weight only on futures candidate")
for wc in np.arange(0,.71,.05):
 raw=np.where(reg,(1-wc)*tab+wc*pcur,(1-wc)*tab+wc*ptm)
 print(round(float(wc),2),round(score(cal(raw),y),3),round(score(cal(raw)[reg],y[reg]),3),round(score(cal(raw)[~reg],y[~reg]),3))
