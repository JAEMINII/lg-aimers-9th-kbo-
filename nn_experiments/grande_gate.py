"""GRANDE gate experiment on the Aimers temporal folds.

This intentionally uses the official GRANDE_Module (gradient-trained hard
axis-aligned trees) but a leak-safe, compact preprocessing path.  The full
GRANDE wrapper pulls in AutoGluon and does expensive general-purpose encoding;
for this contest we already have the canonical F44 features, so we encode the
five small categorical fields one-hot and the four ID fields with leave-one-out
smoothed target encoding learned only from the pre-VS block.
"""
import os
import sys
import time
import types
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF

# GRANDE imports AutoGluon at module import time although the low-level module
# does not use it.  Avoid installing the very large AutoGluon distribution.
ag = types.ModuleType("autogluon")
agf = types.ModuleType("autogluon.features")
agg = types.ModuleType("autogluon.features.generators")
agg.LabelEncoderFeatureGenerator = object
sys.modules.update({"autogluon": ag, "autogluon.features": agf,
                    "autogluon.features.generators": agg})

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/root/GRANDE")
DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
VS = int(os.environ.get("VS", "2024"))
OUT = os.environ.get("AIMERS_OUT", "/root/grande_out")
os.makedirs(OUT, exist_ok=True)

import features44 as F  # noqa: E402
from GRANDE.GRANDE import GRANDE_Module  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score(p, y):
    return F.best_shift(np.asarray(p, np.float64), np.asarray(y, np.float64))[0]


def build_features(d):
    """Build numeric input with leak-safe TE and one-hot small categoricals."""
    X = d["X44"].astype(np.float32)
    y = d["y"].astype(np.float32)
    tr = d["m_tr"]
    cat = list(d["cat_idx"])
    # Raw F44 has these columns in this fixed order (features44.CAT_COLS).
    # Small categorical columns are first three plus both hand columns.
    small = [cat[i] for i in (0, 1, 2, 5, 6)]
    high = [cat[i] for i in (3, 4, 7, 8)]
    chunks = []
    # Numeric columns: standardize on the historical block only, impute median.
    ni = [j for j in range(X.shape[1]) if j not in cat]
    xn = X[:, ni].astype(np.float32)
    med = np.nanmedian(xn[tr], axis=0)
    xn = np.where(np.isfinite(xn), xn, med[None, :])
    mu = xn[tr].mean(0)
    sd = xn[tr].std(0) + 1e-5
    xn = np.clip((xn - mu) / sd, -12, 12).astype(np.float32)
    chunks.append(xn)
    # One-hot small fields. Unknown/missing gets an explicit extra bucket.
    for j in small:
        vals = X[tr, j]
        vals = vals[np.isfinite(vals)]
        uniq = np.unique(vals.astype(np.int64))
        z = np.where(np.isfinite(X[:, j]), X[:, j], -1).astype(np.int64)
        oh = np.zeros((len(X), len(uniq) + 1), np.float32)
        for k, v in enumerate(uniq):
            oh[:, k] = (z == int(v))
        oh[:, -1] = ~np.isin(z, uniq)
        chunks.append(oh)
    # Leave-one-out smoothed target encodings for IDs.  This prevents each
    # training row from seeing its own label; validation sees train-only sums.
    prior = float(y[tr].mean())
    alpha = float(os.environ.get("TE_ALPHA", "50"))
    for j in high:
        z = np.where(np.isfinite(X[:, j]), X[:, j], -1).astype(np.int64)
        uniq, inv = np.unique(z[tr], return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
        sm = np.bincount(inv, weights=y[tr], minlength=len(uniq)).astype(np.float64)
        pos = np.searchsorted(uniq, z)
        hit = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq)-1)] == z)
        out = np.full(len(X), prior, np.float32)
        ids = pos[hit]
        out[hit] = ((sm[ids] + alpha * prior) /
                    (cnt[ids] + alpha)).astype(np.float32)
        # LOO only on train rows.
        rr = np.flatnonzero(tr)
        out[rr] = ((sm[inv] - y[rr] + alpha * prior) /
                   (cnt[inv] - 1 + alpha)).astype(np.float32)
        chunks.append(((out - prior) * 8.0).reshape(-1, 1).astype(np.float32))
    Z = np.concatenate(chunks, axis=1)
    # A missingness channel is useful for differentiable splits.
    Z = np.concatenate([Z, (~np.isfinite(X)).astype(np.float32)], axis=1)
    return Z, y, tr


def make_model(nv, seed, depth, estimators, selected, dropout):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    params = dict(
        depth=depth, n_estimators=estimators,
        learning_rate_weights=0.001,
        learning_rate_index=0.01,
        learning_rate_values=0.05,
        learning_rate_leaf=0.05,
        dropout=dropout, selected_variables=selected,
        data_subset_fraction=1.0, bootstrap=False,
        missing_values=False,
        random_seed=seed, verbose=0, objective="binary",
        problem_type="binary", number_of_variables=nv,
        number_of_classes=1, device=DEV,
    )
    m = GRANDE_Module(params=params).to(DEV)
    if os.environ.get("COMPILE", "0") == "1":
        m = torch.compile(m, mode="reduce-overhead", fullgraph=False, dynamic=False)
    return m


def fit_predict(Z, y, tr, seed, depth, estimators, selected, dropout, epochs, batch):
    m = make_model(Z.shape[1], seed, depth, estimators, selected, dropout)
    ridx = np.flatnonzero(tr)
    frac = float(os.environ.get("TRAIN_FRAC", "1.0"))
    if frac < 1.0:
        rng = np.random.default_rng(seed + 9001)
        ridx = np.sort(rng.choice(ridx, size=max(1000, int(len(ridx) * frac)), replace=False))
    Xt = torch.from_numpy(Z[ridx]).to(DEV)
    yt = torch.from_numpy(y[ridx].astype(np.int64)).to(DEV)
    Xall = torch.from_numpy(Z).to(DEV)
    ls = float(os.environ.get("LR_SCALE", "1.0"))
    opt = torch.optim.Adam([
        {"params": m.split_values, "lr": 0.05 * ls},
        {"params": m.split_index_array, "lr": 0.01 * ls},
        {"params": m.estimator_weights, "lr": 0.001 * ls},
        {"params": m.leaf_classes_array, "lr": 0.05 * ls},
    ])
    bs = int(batch)
    m.train()
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(Xt.shape[0], device=DEV)
        tot = 0.0
        for s in range(0, Xt.shape[0], bs):
            ii = perm[s:s+bs]
            opt.zero_grad(set_to_none=True)
            logits = m(Xt[ii])
            loss = TF.cross_entropy(logits, yt[ii])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(ii)
        print(f"VS={VS} seed={seed} epoch={ep+1}/{epochs} loss={tot/len(Xt):.5f} sec={time.time()-t0:.0f}", flush=True)
    m.eval()
    out = np.empty(len(Z), np.float64)
    with torch.no_grad():
        for s in range(0, len(Z), 8192):
            lg = m(Xall[s:s+8192])
            out[s:s+8192] = lg.softmax(-1)[:, 1].detach().float().cpu().numpy()
    del m, Xt, yt, Xall, opt
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    t0 = time.time()
    d = F.build(DATA, VS=VS)
    Z, y, tr = build_features(d)
    va = d["m_va"]
    print(f"data X={d['X44'].shape} Z={Z.shape} train={tr.sum():,} val={va.sum():,} dev={DEV}", flush=True)
    # Fast-to-slow ladder.  Override GRANDE_PLANS with e.g. 'small,wide'.
    plans = {
        "small": dict(depth=5, estimators=64, selected=0.70, dropout=0.10, epochs=2, batch=8192),
        "wide": dict(depth=6, estimators=128, selected=0.80, dropout=0.10, epochs=3, batch=8192),
        "deep": dict(depth=5, estimators=256, selected=0.80, dropout=0.05, epochs=4, batch=8192),
        "wild": dict(depth=5, estimators=256, selected=0.70, dropout=0.00, epochs=8, batch=4096),
        "fair": dict(depth=5, estimators=64, selected=0.70, dropout=0.00, epochs=20, batch=4096),
    }
    chosen = [x for x in os.environ.get("GRANDE_PLANS", "small,wide").split(",") if x in plans]
    seeds = tuple(int(x) for x in os.environ.get("SEEDS", "42,1").split(","))
    for name in chosen:
        kw = plans[name]
        preds = []
        for seed in seeds:
            p = fit_predict(Z, y, tr, seed=seed, **kw)
            np.save(os.path.join(OUT, f"grande_vs{VS}_{name}_s{seed}.npy"), p)
            preds.append(p)
            print(f"{name} seed={seed} score_all={score(p[va],y[va]):.1f} score_r={score(p[va & ~d['is_f']],y[va & ~d['is_f']]):.1f}", flush=True)
        p = np.mean(preds, axis=0)
        np.save(os.path.join(OUT, f"grande_vs{VS}_{name}_mean.npy"), p)
        print(f"RESULT {name} seeds={seeds} all={score(p[va],y[va]):.1f} regular={score(p[va & ~d['is_f']],y[va & ~d['is_f']]):.1f} elapsed={time.time()-t0:.0f}s", flush=True)
