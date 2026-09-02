import numpy as np
p=np.mean([np.load('/root/aux3090/aux_vs2024_s'+str(s)+'.npy') for s in [42,1,777]],0)
print(p.min(),p.max(),p.mean(),np.quantile(p,[.01,.5,.99]))
q=np.load('/root/ms_k32d512_3090/ms_vs2024_direct.npy'); print('ms',q.min(),q.max(),q.mean(),np.quantile(q,[.01,.5,.99]))
cb=np.load('/root/aimers/cb_gate.npy'); tab=np.mean([np.load('/root/arch3090_2024/ar_tabm_s'+str(s)+'.npy') for s in [42,1,777,2]],0); import pandas as pd; tr=pd.read_csv('/root/data/train.csv',usecols=['season','control_success']); print('means blend',(.7*q+.27*cb+.03*tab).mean(), 'y',tr.loc[tr.season==2024,'control_success'].mean())
