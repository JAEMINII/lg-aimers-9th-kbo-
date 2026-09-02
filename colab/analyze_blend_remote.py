import numpy as np, pandas as pd

def bss(p,y):
    r=y.mean(); return 1e5*(1-((p-y)**2).mean()/(r*(1-r)))
def shift(p,c):
    q=np.clip(p,1e-6,1-1e-6); z=np.log(q/(1-q)); return 1/(1+np.exp(-(z+c)))
def best(p,y):
    cs=np.linspace(-.15,.15,121); vals=[bss(shift(p,c),y) for c in cs]; i=int(np.argmax(vals)); return vals[i],cs[i]
d='/root/data'; tr=pd.read_csv(d+'/train.csv',encoding='utf-8-sig'); tr=tr.sort_values('row_id',key=lambda s:s.str.slice(6).astype(int)).reset_index(drop=True)
m=tr.season.to_numpy()==2024; y=tr.control_success.to_numpy(float)[m]; f=tr.game_type.astype(str).to_numpy()[m]=='F'
P={}
for k in ['ar_tabm','ar_k64','ar_d512','ar_blocks4']:
 P[k]=np.mean([np.load('/root/arch3090_2024/'+k+'_s'+str(s)+'.npy') for s in [42,1,777,2]],axis=0)
P['state']=np.load('/root/ms_k32d512_3090/ms_vs2024_direct.npy')
for k in ['cb','mlp']: P[k]=np.load('/root/aimers/'+k+'_gate.npy')
print('n',len(y),'f',f.sum())
for k,p in P.items(): print(k,'overall',best(p,y),'R',best(p[~f],y[~f]),'F',best(p[f],y[f]))
base=P['ar_tabm']
for k,p in P.items():
 if k=='ar_tabm': continue
 vals=[]
 for w in np.linspace(0,1,21):
  q=(1-w)*base+w*p; vals.append((best(q,y)[0],w,best(q,y)[1]))
 print('blend',k,max(vals))
best3=(-1,None)
for ws in np.linspace(0,1,21):
 for wc in np.linspace(0,1-ws,21):
  q=ws*P['state']+wc*P['cb']+(1-ws-wc)*base; v,sh=best(q,y)
  if v>best3[0]: best3=(v,(ws,wc,1-ws-wc,sh))
print('blend3 state/cb/tabm',best3)
