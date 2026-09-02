"""Export the features44-style prior-season platoon deviation for deployment.

The hidden/test rows are 2025, so their legal platoon history is the target
history through 2024.  This reproduces features44's PCC calculation without
looking at any test labels or aggregating test rows.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/root')
import features44 as FF
from train_chan_3 import preprocess as PP


def prior_map(df, cutoff):
    tr = df[df.season < cutoff].copy()
    y = tr[FF.TGT].astype(float)
    gm = float(y.mean())
    lgph = tr.groupby(['pitcher_hand', 'batter_hand'])[FF.TGT].mean()
    lgp = tr.groupby('pitcher_hand')[FF.TGT].mean()
    ph = tr.groupby('pitcher_id').pitcher_hand.first()
    pc = tr.groupby(['pitcher_id', 'season', 'batter_hand'])[FF.TGT].agg(['size', 'sum']).unstack(fill_value=0)
    pc.columns = [f'{x}{int(h)}' for x, h in pc.columns]
    for h in (1, 2):
        for x in ('size', 'sum'):
            if f'{x}{h}' not in pc:
                pc[f'{x}{h}'] = 0.0
    pcc = pc.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    # Features44's target season is the next season, hence use cumulative
    # counts through the cutoff for each pitcher.
    totals = pc.groupby(level=0)[['size1', 'size2', 'sum1', 'sum2']].sum()
    out = {}
    for pid, row in totals.iterrows():
        p_hand = ph.get(pid, np.nan)
        if pd.isna(p_hand):
            continue
        pr = {h: float(lgph.get((p_hand, h), gm)) for h in (1, 2)}
        d = {}
        for h in (1, 2):
            d[h] = (float(row[f'sum{h}']) + 300.0 * pr[h]) / (float(row[f'size{h}']) + 300.0)
        pa = (float(row['sum1'] + row['sum2']) + 300.0 * float(lgp.get(p_hand, gm))) / (float(row['size1'] + row['size2']) + 300.0)
        out[int(pid)] = [float(d[1] - pa), float(d[2] - pa)]
    return out


def main():
    data = os.environ.get('AIMERS_DATA', '/root/data')
    df = PP.sort_by_row_id(pd.read_csv(Path(data) / 'train.csv', encoding='utf-8-sig'))
    for vs in (2022, 2024):
        mp = prior_map(df, vs)
        f = FF.build(data, VS=vs)
        p44 = f['X44'][:, f['F44'].index('plat_dev')].astype(float)
        # The next-season map is evaluated only for the validation pitcher's
        # hand; missing/first-season values should be exactly zero.
        vals = np.array([mp.get(int(pid), [0., 0.])[int(h)-1]
                         for pid, h in zip(df.pitcher_id, df.batter_hand)])
        m = df.season.to_numpy() == vs
        print(vs, 'val diff', np.mean(vals[m]-p44[m]), np.mean(np.abs(vals[m]-p44[m])),
              'max', np.max(np.abs(vals[m]-p44[m])), 'rows', len(mp))
    mp = prior_map(df, 2025)
    out = Path(os.environ.get('AIMERS_OUT', '/root/ms_export'))
    out.mkdir(parents=True, exist_ok=True)
    pids = np.array(sorted(mp), dtype=np.int64)
    vals = np.array([mp[int(p)] for p in pids], dtype=np.float32)
    np.savez_compressed(out / 'ms_plat_prior.npz', pitcher_id=pids, dev=vals)
    print(json.dumps({'out': str(out / 'ms_plat_prior.npz'), 'pitchers': len(pids)}))


if __name__ == '__main__':
    main()
