import numpy as np, pandas as pd
def bss(p,y):
 r=y.mean(); return 1e5*(1-((p-y)**2).mean()/(r*(1-r)))
def shift(p,c):
 q=np.clip(p,1e-6,1-1e-6); z=np.log(q/(1-q)); return 1/(1+np.exp(-(z+c)))
def score(p,y,c=-.065): return bss(shift(p,c),y)
d='/root/data'; tr=pd.read_csv(d+'/train.csv',encoding='utf-8-sig'); tr=tr.sort_values('row_id',key=lambda s:s.str.slice(6).astype(int)).reset_index(drop=True); m=tr.season.to_numpy()==2024; y=tr.control_success.to_numpy(float)[m]
tab=np.mean([np.load('/root/arch3090_2024/ar_tabm_s'+str(s)+'.npy') for s in [42,1,777,2]],0); cb=np.load('/root/aimers/cb_gate.npy'); st=np.load('/root/ms_k32d512_3090/ms_vs2024_direct.npy')
best=(-1,None)
for ws in np.linspace(0,1,11):
 for wc in np.linspace(0,1-ws,11):
  p=ws*st+wc*cb+(1-ws-wc)*tab
  v=score(p,y,-.065)
  if v>best[0]: best=(v,(ws,wc,1-ws-wc,-.065))
ws,wc,_wt,_=best[1]
best=(max((score(ws*st+wc*cb+(1-ws-wc)*tab,y,c),c) for c in np.linspace(-.085,-.045,17))[0], (ws,wc,1-ws-wc,max((score(ws*st+wc*cb+(1-ws-wc)*tab,y,c),c) for c in np.linspace(-.085,-.045,17))[1]))
print('best',best)
for name,p in [('state+tab',.75*st+.25*tab),('state+incumbent',.75*st+.25*(.3*cb+.7*tab)),('state+cb+tab',best[1][0]*st+best[1][1]*cb+best[1][2]*tab)]: print(name,max((score(p,y,c),c) for c in np.linspace(-.085,-.045,17)))
