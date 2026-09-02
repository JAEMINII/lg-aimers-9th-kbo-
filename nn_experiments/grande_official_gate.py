"""One fair GRANDE run through the official high-level preprocessing path."""
import os, sys, types, time
import numpy as np
import pandas as pd
import torch

# Minimal replacement for the AutoGluon label encoder used by GRANDE.py.
ag = types.ModuleType('autogluon'); agf = types.ModuleType('autogluon.features'); agg = types.ModuleType('autogluon.features.generators')
class LabelEncoderFeatureGenerator:
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, X=None, **kw):
        self.features_in = [c for c in X.columns if str(X[c].dtype) == 'category' or X[c].dtype == object]
        self.maps = {}
        for c in self.features_in:
            vals = X[c].astype('category').cat.categories.tolist()
            self.maps[c] = {v: i+1 for i, v in enumerate(vals)}
        return self
    def transform(self, X=None, **kw):
        Z = X.copy()
        for c in self.features_in:
            Z[c] = Z[c].map(self.maps[c]).fillna(0).astype(np.int64)
        return Z[self.features_in]
agg.LabelEncoderFeatureGenerator = LabelEncoderFeatureGenerator
sys.modules.update({'autogluon':ag,'autogluon.features':agf,'autogluon.features.generators':agg})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '/root/GRANDE')
import features44 as F
from GRANDE.GRANDE import GRANDE

VS=int(os.environ.get('VS','2024')); DATA=os.environ.get('AIMERS_DATA','/root/open (1)/data')
d=F.build(DATA,VS=VS); cols=d['F44']; X=pd.DataFrame(d['X44'],columns=cols)
cat=[cols[i] for i in d['cat_idx']]
for c in cat:
    # Use global categories so train/validation share the same integer coding.
    X[c]=pd.Categorical(X[c], categories=np.unique(X[c][np.isfinite(X[c])]))
tr=d['m_tr']; va=d['m_va']; y=d['y']
params=dict(depth=5,n_estimators=int(os.environ.get('EST','32')),
            learning_rate_weights=.001,learning_rate_index=.01,
            learning_rate_values=.05,learning_rate_leaf=.05,
            dropout=.1,selected_variables=.8,data_subset_fraction=1.,bootstrap=False,
            missing_values=False,optimizer='adam',cosine_decay_restarts=False,
            reduce_on_plateau_scheduler=False,label_smoothing=0.,use_class_weights=False,
            focal_loss=False,swa=False,es_metric=False,epochs=int(os.environ.get('EPOCHS','12')),
            batch_size=int(os.environ.get('BS','4096')),early_stopping_epochs=4,
            use_freq_enc=False,use_robust_scale_smoothing=False,problem_type='binary',
            use_category_embeddings=False,use_numeric_embeddings=False,
            attach_preprocessed_data=False,
            random_seed=42,verbose=1)
print('start',X.shape,'tr',tr.sum(),'va',va.sum(),flush=True)
t0=time.time(); m=GRANDE(params=params)
m.fit(X=X.loc[tr].reset_index(drop=True), y=pd.Series(y[tr]), X_val=X.loc[va].reset_index(drop=True), y_val=pd.Series(y[va]))
p=m.predict_proba(X.loc[va].reset_index(drop=True))[:,1]
print('score all',F.best_shift(p,y[va]),'regular',F.best_shift(p[~d['is_f'][va]],y[va][~d['is_f'][va]]),'mean',p.mean(),'std',p.std(),'sec',time.time()-t0,flush=True)
np.save('/root/grande_out/grande_official_vs%d.npy'%VS,p)
