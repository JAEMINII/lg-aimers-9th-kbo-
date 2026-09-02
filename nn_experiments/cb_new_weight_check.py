"""Cheap validation of the risky 50-feature CatBoost replacement ratio."""
import importlib.util, os, sys
import numpy as np, pandas as pd

ROOT = os.environ.get('AIMERS_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, 'submit_41')
DATA = os.environ.get('AIMERS_DATA', os.path.join(ROOT, 'open (1)', 'data'))
sys.path.insert(0, os.path.join(ROOT, 'nn_experiments'))
import features44 as F

tr = pd.read_csv(os.path.join(DATA, 'train.csv'), encoding='utf-8-sig')
pool = tr[tr.season == 2024]
rng = np.random.default_rng(7)
N = min(int(os.environ.get('N', '50000')), len(pool))
sub = pool.iloc[np.sort(rng.choice(len(pool), N, replace=False))].copy().reset_index(drop=True)
y = sub.control_success.to_numpy(np.float64)
oldcwd = os.getcwd(); os.chdir(PKG); sys.path.insert(0, PKG)
spec = importlib.util.spec_from_file_location('p41cw', 'script.py')
S = importlib.util.module_from_spec(spec); S.__dict__['__name__']='p41cw'; spec.loader.exec_module(S)
history, z, shift = S.load_model()
X = S.build_features(sub, history)
pnew = S.predict_numpy(X.values.astype(np.float64), z)
pold = S.predict_numpy(X[history['_base_features']].values.astype(np.float64), history['_base_z'])
print(f'N={N} pnew={pnew.mean():.6f} pold={pold.mean():.6f} corr={np.corrcoef(pnew,pold)[0,1]:.5f}')
for w in np.arange(0, 1.01, .1):
    p = w*pnew + (1-w)*pold
    s, sh = F.best_shift(p, y)
    print(f'w_new={w:.1f} score={s:.2f} shift={sh:+.4f} fixed0={F.bss(p,y):.2f}')
os.chdir(oldcwd)
