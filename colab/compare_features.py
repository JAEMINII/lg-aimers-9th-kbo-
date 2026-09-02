import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
sys.path.insert(0,'colab'); sys.path.insert(0,'train_chan_3')
from features44 import build
from train_chan_3 import preprocess as p
d=build('open (1)/data',VS=2024)
tr=p.sort_by_row_id(pd.read_csv('open (1)/data/train.csv',encoding='utf-8-sig'))
h=p.fit_history_tables(tr[tr.season<2024]); x=p.transform_features(tr,h,train_mode=True)
names=d['F44']; a=d['X44']; b=x[names].to_numpy(np.float32)
print('shape',a.shape,b.shape,'max',np.nanmax(np.abs(a-b)),'mean',np.nanmean(np.abs(a-b)))
for i,n in enumerate(names):
 z=np.nanmax(np.abs(a[:,i]-b[:,i]))
 if z>1e-4: print(n,z,np.nanmean(np.abs(a[:,i]-b[:,i])))
