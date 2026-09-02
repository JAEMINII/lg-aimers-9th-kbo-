import numpy as np,pandas as pd
def bss(p,y):
 r=y.mean(); return 1e5*(1-((p-y)**2).mean()/(r*(1-r)))
def best(p,y):
 best=(-1,None)
 for c in np.linspace(-.12,.04,161):
  q=np.clip(p,1e-6,1-1e-6); q=1/(1+np.exp(-(np.log(q/(1-q))+c))); v=bss(q,y)
  if v>best[0]:best=(v,c)
 return best
tr=pd.read_csv('/root/data/train.csv',encoding='utf-8-sig');tr=tr.sort_values('row_id',key=lambda s:s.str.slice(6).astype(int)).reset_index(drop=True);m=tr.season.to_numpy()==2024;y=tr.control_success.to_numpy(float)[m];f=tr.game_type.astype(str).to_numpy()[m]=='F'; st=np.load('/root/ms_k32d512_3090/ms_vs2024_direct.npy');cb=np.load('/root/aimers/cb_gate.npy');tab=np.mean([np.load('/root/arch3090_2024/ar_tabm_s'+str(s)+'.npy') for s in [42,1,777,2]],0);p=.7*st+.27*cb+.03*tab
for name,ix in [('all',np.ones(len(y),bool)),('first',np.arange(len(y))<len(y)//2),('second',np.arange(len(y))>=len(y)//2),('R',~f),('F',f)]:print(name,len(y[ix]),best(p[ix],y[ix]))
