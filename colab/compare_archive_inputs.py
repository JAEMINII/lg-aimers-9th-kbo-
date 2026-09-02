import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/root/aimers'); sys.path.insert(0,'/root')
import features44 as FF
from train_chan_3 import preprocess as PP
import importlib.util
sp=importlib.util.spec_from_file_location('sub','/root/submission_script.py'); sub=importlib.util.module_from_spec(sp); sp.loader.exec_module(sub)
DATA='/root/data'; raw=PP.sort_by_row_id(pd.read_csv(DATA+'/train.csv',encoding='utf-8-sig')); season=raw.season.to_numpy(); m=season==2024; y=raw.control_success.to_numpy(float)[m]
ff=FF.build(DATA,VS=2024); xff=ff['X44']; names=list(ff['F44']); isf=raw.game_type.astype(str).to_numpy()=='F'; old=season<=2022; c4=np.where(old&isf,0.,np.where(old&~isf,1.,np.where(isf,2.,3.))).astype(np.float32)[:,None]; xff=np.c_[xff,c4]
h=PP.fit_history_tables(raw[season<2024]); xc=PP.transform_features(raw,h,train_mode=True).to_numpy(np.float32); xc=np.c_[xc,c4]
print('diff max/mean',np.nanmax(np.abs(xff[:,:44]-xc[:,:44])),np.nanmean(np.abs(xff[:,:44]-xc[:,:44])))
for typ,xx in [('ff',xff),('canon',xc),('canon_zero_pdev',xc.copy())]:
 if typ=='canon_zero_pdev': xx[:,43]=0.0
 xx=xx[m]
 ps=[]
 for s in [42,1,777]:
  ar=np.load('/root/ms_export/ms_seed'+str(s)+'.npz',allow_pickle=False); meta=json.loads(str(ar['metadata'].item()))
  df=pd.DataFrame(xx,columns=meta['feature_names']); ps.append(sub.predict_frame(df,meta,ar,8192)); ar.close()
 p=np.mean(ps,0)[m]; q=np.clip(p,1e-6,1-1e-6); z=np.log(q/(1-q)); best=(-1,None)
 for c in np.linspace(-.12,.04,161):
  qq=1/(1+np.exp(-(z+c))); r=y.mean(); sc=1e5*(1-((qq-y)**2).mean()/(r*(1-r)))
  if sc>best[0]:best=(sc,c)
 print(typ,best,p.mean(),p.min(),p.max())
