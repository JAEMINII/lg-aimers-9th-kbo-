import os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/root')
import features44 as FF
from train_chan_3 import preprocess as PP

data = os.environ.get('AIMERS_DATA', '/root/data')
raw = PP.sort_by_row_id(pd.read_csv(Path(data)/'train.csv', encoding='utf-8-sig'))
season = raw.season.to_numpy()
for vs in (2022, 2024):
    f = FF.build(data, VS=vs)
    p44 = f['X44'][:, f['F44'].index('plat_dev')].astype(float)
    hist = PP.fit_history_tables(raw[season < vs])
    xc = PP.transform_features(raw, hist, train_mode=True)
    pc = xc['plat_dev'].to_numpy(float)
    m = season == vs
    print('VS', vs)
    for label, mm in [('all', np.ones(len(raw),bool)), ('train', season<vs), ('val',m)]:
        d = p44[mm]-pc[mm]
        print(label, 'p44', np.nanmean(p44[mm]),'pc',np.nanmean(pc[mm]),'diffmean',np.mean(d),'abs',np.mean(np.abs(d)),'max',np.max(np.abs(d)))
    print('val quantiles p44',np.quantile(p44[m],[0,.01,.1,.5,.9,.99,1]))
    print('val quantiles pc ',np.quantile(pc[m],[0,.01,.1,.5,.9,.99,1]))
