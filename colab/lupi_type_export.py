"""Train the audited pitch-type LUPI TabM on all official train rows.

Only the direct control-success head is exported.  The Trackman-derived
pitch-type labels are used while training and are deliberately absent from
the resulting NumPy models and submission-time feature path.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
for p in (SC, "/root/aimers", os.path.dirname(SC)):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
OUT = os.environ.get("LUPI_TYPE_OUT", "/root/aimers/_dl/lupi_type_final")
FINAL_SEASON = int(os.environ.get("FINAL_SEASON", "2025"))
ARM = os.environ.get("LUPI_TYPE_ARM", "type02")
SEEDS = tuple(int(x) for x in os.environ.get("LUPI_TYPE_SEEDS", "42,1,777").split(","))

import features44 as F  # noqa: E402
import tabm_gate_gpu as G  # noqa: E402
from export_ours import export, prep_stats  # noqa: E402
from lupi_type_aux import (  # noqa: E402
    EP2,
    LR1,
    OLD_F_MAX,
    OLD_W,
    TYPE_WEIGHTS,
    build_type,
    make_model,
    train_mt,
)
from train_chan_3 import preprocess as PP  # noqa: E402


def prepare():
    """Rebuild the exact deployable 45-column f2b feature matrix at 2025."""
    d = F.build(DATA, VS=FINAL_SEASON)
    season, is_f = d["season"].astype(np.int16), d["is_f"].astype(bool)
    y, f44 = d["y"].astype(np.float32), list(d["F44"])
    rid = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id"]).row_id.to_numpy()
    raw = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig"))
    pos_map = pd.Series(np.arange(len(raw)), index=raw.row_id).reindex(rid).to_numpy()
    if not np.isfinite(pos_map).all():
        raise ValueError("row-id alignment failed")

    hist = PP.fit_history_tables(raw[raw.season < FINAL_SEASON])
    xf = PP.transform_features(raw, hist, train_mode=True)
    cols = list(xf.columns)
    x = xf.to_numpy(np.float32)[pos_map.astype(np.int64)]
    # PP's training path has full-history platoon values.  The deployed
    # inference path is as-of, so keep training consistent with deployment.
    x[:, cols.index("plat_dev")] = d["X44"][:, f44.index("plat_dev")]

    old = season <= OLD_F_MAX
    c4 = np.where(old & is_f, 0., np.where(old & ~is_f, 1.,
                 np.where(is_f, 2., 3.))).astype(np.float32)[:, None]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    xin = np.c_[x, c4]
    xn, xc, cards, stats = prep_stats(xin, season < FINAL_SEASON, ci + [x.shape[1]])
    G.Xn, G.cards = xn, cards
    G.XN, G.XC = torch.from_numpy(xn), torch.from_numpy(xc)
    return d, rid, season, is_f, y, old, xn, xc, cards, stats, cols + ["abs_regime"]


def main():
    if ARM not in TYPE_WEIGHTS or TYPE_WEIGHTS[ARM] <= 0.0:
        raise ValueError(f"LUPI_TYPE_ARM must be a nonzero known arm, got {ARM!r}")
    if os.path.exists(OUT):
        raise FileExistsError(f"refusing to overwrite {OUT}")
    os.makedirs(OUT)
    t0 = time.time()
    d, rid, season, is_f, y, old, xn, xc, cards, stats, features = prepare()
    typ = build_type(rid, season, FINAL_SEASON)
    train = np.flatnonzero(season < FINAL_SEASON).astype(np.int64)
    if not (typ[train] >= 0).any():
        raise ValueError("no audited pitch-type labels in the training interval")
    branches = {
        "all": train,
        "regular": train[~is_f[train]],
        "futures": train[is_f[train]],
    }
    weights = {
        "all": np.where(is_f[train] & old[train], OLD_W, 1.).astype(np.float32),
        "regular": None,
        "futures": np.where(old[train[is_f[train]]], OLD_W, 1.).astype(np.float32),
    }
    cfg_base = dict(
        k=32, n_blocks=3, d_block=256, dropout=.1, d_emb=16,
        num_embeddings="linear_relu", backbone_kind="batch_ensemble",
        stage1_epochs=2, stage1_lr=LR1, stage2_lr=2e-4,
        regime="c4_f2b_platfix", plat_dev="asof", final_season=FINAL_SEASON,
        lupi="audited_current_pitch_type_auxiliary", lupi_weight=TYPE_WEIGHTS[ARM],
        lupi_inference="direct_head_only",
    )
    print(f"LUPI export arm={ARM} tw={TYPE_WEIGHTS[ARM]} rows={len(train):,} "
          f"labels={int((typ[train] >= 0).sum()):,} seeds={SEEDS}", flush=True)
    for seed in SEEDS:
        for branch, idx in branches.items():
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = make_model()
            train_mt(model, idx, 2, LR1, y, typ, TYPE_WEIGHTS[ARM], w=weights[branch],
                     seed=seed, tag=f"final {ARM} {branch} s{seed} S1")
            recent = idx[season[idx] == FINAL_SEASON - 1]
            if not len(recent):
                raise ValueError(f"{branch}: missing final-season stage-2 rows")
            params = G.stage2_params(model)
            recent_w = None if weights[branch] is None else weights[branch][season[idx] == FINAL_SEASON - 1]
            for epoch in range(EP2[branch]):
                train_mt(model, recent, 1, 2e-4, y, typ, TYPE_WEIGHTS[ARM], params=params,
                         w=recent_w, seed=seed + epoch,
                         tag=f"final {ARM} {branch} s{seed} S2e{epoch+1}")
            path = os.path.join(OUT, f"lupi_{ARM}_{branch}_s{seed}.npz")
            export(model, path, stats, cards, xn.shape[1],
                   dict(cfg_base, branch=branch, seed=seed, stage2_epochs=EP2[branch]), features)
            print(f"  saved {os.path.basename(path)} {os.path.getsize(path)/1e6:.2f} MB", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
    print(f"DONE elapsed={time.time()-t0:.0f}s out={OUT}", flush=True)


if __name__ == "__main__":
    main()
