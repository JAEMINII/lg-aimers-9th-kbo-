# -*- coding: utf-8 -*-
"""Train and export the rule-compliant pitcher x count12 CatBoost route.

This is intentionally separate from the incumbent submission.  It uses the
same 44 base features as submit_35 and appends three features whose target
tables are strictly prior-season tables.  The output is a NumPy-only
``trees.npz`` understood by the submission runner.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = os.environ.get("AIMERS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "open (1)", "data")
OUT = os.environ.get("C12_OUT", os.path.join(ROOT, "c12_out"))
os.makedirs(OUT, exist_ok=True)

TARGET = "control_success"
ALPHA_PC = 200.0
DECAY = 2.0
SEEDS = tuple(int(x) for x in os.environ.get(
    "CB_SEEDS", "1,42,777"
).split(",") if x)
BASE_FEATURES = None
INCLUDE_CMH = os.environ.get("INCLUDE_CMH", "0") == "1"


def log(msg: str):
    print(msg, flush=True)
    with open(os.path.join(OUT, "train.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def add_c12(frame: pd.DataFrame, *, return_tables=False):
    """Add strict prior-season pitcher x count12 features.

    For a row in season S, only seasons < S are used.  The final inference
    table is the all-training-season aggregate and is exported separately.
    """
    d = frame.copy()
    d["cnt12"] = (d["balls_before"].astype("int16") * 3
                   + d["strikes_before"].astype("int16")).astype("int16")
    d["_pid"] = d["pitcher_id"].astype("int64")

    b = d.groupby(["pitcher_id", "season", "cnt12"], sort=False)[TARGET] \
         .agg(["size", "sum"]).unstack(fill_value=0)
    b.columns = [f"{x}_{int(k)}" for x, k in b.columns]
    for k in range(12):
        for x in ("size", "sum"):
            col = f"{x}_{k}"
            if col not in b:
                b[col] = 0.0
    b = b.reindex(sorted(b.columns), axis=1)
    cc = b.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    keys = np.arange(12, dtype=np.int16)
    idx = pd.MultiIndex.from_arrays([d["pitcher_id"].to_numpy(),
                                     d["season"].to_numpy()])
    sz = cc[[f"size_{k}" for k in keys]].reindex(idx).fillna(0.0).to_numpy()
    sm = cc[[f"sum_{k}" for k in keys]].reindex(idx).fillna(0.0).to_numpy()
    kv = d["cnt12"].to_numpy(np.int16)
    rr = np.arange(len(d))
    n = sz[rr, kv]
    s = sm[rr, kv]
    n_all = sz.sum(axis=1)
    s_all = sm.sum(axis=1)
    g = float(d[TARGET].mean())
    prior_lut = d.groupby("cnt12")[TARGET].mean().reindex(keys).fillna(g).to_numpy()
    prior = prior_lut[kv]
    base = (s_all + ALPHA_PC * g) / (n_all + ALPHA_PC)
    rate = (s + ALPHA_PC * prior) / (n + ALPHA_PC)
    d["pc_c12_rate"] = rate.astype("float32")
    d["pc_c12_dev"] = (rate - base).astype("float32")
    d["pc_c12_n"] = np.log1p(n).astype("float32")

    if not return_tables:
        return d

    # All-season table used for 2025 test rows; no test rows participate.
    tot = d.groupby(["pitcher_id", "cnt12"], sort=False)[TARGET].agg(["size", "sum"])
    tot = tot.sort_index()
    ptotal = d.groupby("pitcher_id", sort=False)[TARGET].agg(["size", "sum"]).sort_index()
    pair_key = (tot.index.get_level_values(0).to_numpy(np.int64) * 16
                + tot.index.get_level_values(1).to_numpy(np.int64))
    tables = {
        "pc_key": pair_key.astype(np.int64),
        "pc_n": tot["size"].to_numpy(np.float64),
        "pc_s": tot["sum"].to_numpy(np.float64),
        "pc_pid": ptotal.index.to_numpy(np.int64),
        "pc_total_n": ptotal["size"].to_numpy(np.float64),
        "pc_total_s": ptotal["sum"].to_numpy(np.float64),
        "pc_prior": prior_lut.astype(np.float64),
        "pc_gmean": np.float64(g),
        "pc_alpha": np.float64(ALPHA_PC),
    }
    return d, tables


def add_cmh(frame: pd.DataFrame, *, return_tables=False):
    """Add strict prior-season pitcher x (count12 x batter-hand) features.

    The key is deliberately separate from the c12 table: batter_hand is coded
    as 1/2 in the source data, so count12*2+hand spans 1..24.  All exported
    rows are training/history rows; test rows never enter these tables.
    """
    d = frame.copy()
    if "cnt12" not in d:
        d["cnt12"] = (d["balls_before"].astype("int16") * 3
                       + d["strikes_before"].astype("int16")).astype("int16")
    d["pcmh"] = (d["cnt12"].astype("int16") * 2
                 + d["batter_hand"].astype("int16")).astype("int16")
    b = d.groupby(["pitcher_id", "season", "pcmh"], sort=False)[TARGET] \
         .agg(["size", "sum"]).unstack(fill_value=0)
    b.columns = [f"{x}_{int(k)}" for x, k in b.columns]
    keys = np.array(sorted({int(c.rsplit("_", 1)[1]) for c in b.columns}), dtype=np.int16)
    for k in keys:
        for x in ("size", "sum"):
            if f"{x}_{int(k)}" not in b:
                b[f"{x}_{int(k)}"] = 0.0
    b = b.reindex(sorted(b.columns), axis=1)
    cc = b.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    idx = pd.MultiIndex.from_arrays([d["pitcher_id"].to_numpy(),
                                     d["season"].to_numpy()])
    sz = cc[[f"size_{int(k)}" for k in keys]].reindex(idx).fillna(0.0).to_numpy()
    sm = cc[[f"sum_{int(k)}" for k in keys]].reindex(idx).fillna(0.0).to_numpy()
    kv = d["pcmh"].to_numpy(np.int16)
    pos = np.clip(np.searchsorted(keys, kv), 0, len(keys) - 1)
    hit = keys[pos] == kv
    rr = np.arange(len(d))
    n = np.where(hit, sz[rr, pos], 0.0)
    s = np.where(hit, sm[rr, pos], 0.0)
    g = float(d[TARGET].mean())
    prior_lut = d.groupby("pcmh")[TARGET].mean().reindex(keys).fillna(g).to_numpy()
    prior = prior_lut[pos]
    n_all, s_all = sz.sum(1), sm.sum(1)
    base = (s_all + ALPHA_PC * g) / (n_all + ALPHA_PC)
    rate = (s + ALPHA_PC * prior) / (n + ALPHA_PC)
    d["pc_cmh_rate"] = rate.astype("float32")
    d["pc_cmh_dev"] = (rate - base).astype("float32")
    d["pc_cmh_n"] = np.log1p(n).astype("float32")
    if not return_tables:
        return d
    tot = d.groupby(["pitcher_id", "pcmh"], sort=False)[TARGET].agg(["size", "sum"]).sort_index()
    ptotal = d.groupby("pitcher_id", sort=False)[TARGET].agg(["size", "sum"]).sort_index()
    pair_key = (tot.index.get_level_values(0).to_numpy(np.int64) * 32
                + tot.index.get_level_values(1).to_numpy(np.int64))
    tables = {
        "ph_key": pair_key.astype(np.int64),
        "ph_n": tot["size"].to_numpy(np.float64),
        "ph_s": tot["sum"].to_numpy(np.float64),
        "ph_pid": ptotal.index.to_numpy(np.int64),
        "ph_total_n": ptotal["size"].to_numpy(np.float64),
        "ph_total_s": ptotal["sum"].to_numpy(np.float64),
        "ph_prior_keys": keys.astype(np.int16),
        "ph_prior": prior_lut.astype(np.float64),
        "ph_gmean": np.float64(g),
        "ph_alpha": np.float64(ALPHA_PC),
    }
    return d, tables


def export_models(models_by_group, feature_names, history_tables, path):
    """Export CatBoost JSON oblivious trees into submit_35's NumPy format."""
    groups = [models_by_group["all"], models_by_group["regular"], models_by_group["futures"]]
    raw_models = [m for group in groups for m in group]
    n_per = len(groups[0])
    feat_rows, thr_rows, leaf_rows = [], [], []
    nan_left = None
    for model in raw_models:
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        model.save_model(tmp, format="json")
        with open(tmp, encoding="utf-8") as f:
            obj = json.load(f)
        os.remove(tmp)
        floats = obj["features_info"]["float_features"]
        fmap = {int(x["feature_index"]): x for x in floats}
        if nan_left is None:
            nf = max(int(x["flat_feature_index"]) for x in floats) + 1
            nan_left = np.ones(nf, np.uint8)
            for x in floats:
                if x.get("nan_value_treatment") == "Max":
                    nan_left[int(x["flat_feature_index"])] = 0
        sb = obj.get("scale_and_bias", [1.0, [0.0]])
        if abs(float(sb[0]) - 1.0) > 1e-10:
            raise RuntimeError(f"unexpected CatBoost scale/bias: {sb}")
        for tree in obj["oblivious_trees"]:
            splits = tree.get("splits") or []
            fi, th = [], []
            for split in splits:
                if split.get("split_type") != "FloatFeature":
                    raise RuntimeError(f"non-float split: {split}")
                ff = fmap[int(split["float_feature_index"])]
                fi.append(int(ff["flat_feature_index"]))
                if "border" in split:
                    th.append(float(split["border"]))
                else:
                    th.append(float(ff["borders"][int(split["border_idx"])]))
            feat_rows.append(fi)
            thr_rows.append(th)
            leaf_rows.append(tree["leaf_values"])
    depth = max(len(x) for x in feat_rows)
    full = 1 << depth
    for i in range(len(feat_rows)):
        feat_rows[i] += [0] * (depth - len(feat_rows[i]))
        thr_rows[i] += [np.inf] * (depth - len(thr_rows[i]))
        leaf_rows[i] = list(leaf_rows[i]) + [0.0] * (full - len(leaf_rows[i]))
    cb_feat = np.asarray(feat_rows, np.int32)
    cb_thr = np.asarray(thr_rows, np.float32)
    cb_leaf = np.asarray(leaf_rows, np.float64)
    pair = np.stack([cb_feat.ravel().astype(np.float64),
                     cb_thr.ravel().astype(np.float64)], axis=1)
    uniq, inv = np.unique(pair, axis=0, return_inverse=True)
    arrays = {
        "cb_feat": cb_feat,
        "cb_thr": cb_thr,
        "cb_leaf": cb_leaf,
        "cb_nan_left": nan_left,
        "cb_depth": np.int32(depth),
        "sp_feat": uniq[:, 0].astype(np.int32),
        "sp_thr": uniq[:, 1].astype(np.float32),
        "sp_idx": inv.reshape(cb_feat.shape).astype(np.int32),
        "n_models": np.int32(len(raw_models)),
        "n_per_group": np.int32(n_per),
        "blend_full": np.float64(0.6),
        "gt_col": np.int32(list(feature_names).index("game_type")),
        "features": np.asarray(feature_names, dtype="U64"),
        "calib_logit_shift": np.float64(-0.007),
    }
    arrays.update(history_tables)
    np.savez_compressed(path, **arrays)
    log(f"exported {path} ({os.path.getsize(path)/1024/1024:.2f} MB), "
        f"models={len(raw_models)} trees={len(cb_leaf)} depth={depth}")


def main():
    log(f"ROOT={ROOT} DATA={DATA} OUT={OUT}")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import features44
    built = features44.build(DATA, VS=2025, return_frame=True)
    frame = built["frame"]
    base = list(built["F44"])
    frame, tables = add_c12(frame, return_tables=True)
    features = base + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n"]
    if INCLUDE_CMH:
        frame, cmh_tables = add_cmh(frame, return_tables=True)
        tables.update(cmh_tables)
        features += ["pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
    X = frame[features].to_numpy(np.float32)
    y = frame[TARGET].to_numpy(np.float32)
    # features44 fixes game_type to R=0/F=1 before returning the frame.
    game_type = frame["game_type"].to_numpy()
    weights = (DECAY ** (frame["season"].to_numpy(np.float64) - 2019.0))
    log(f"rows={len(frame):,} features={len(features)} target={y.mean():.6f} include_cmh={INCLUDE_CMH}")
    log(f"groups all={len(y):,} regular={(game_type==0).sum():,} futures={(game_type==1).sum():,}")

    hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
              verbose=100, allow_writing_files=False)
    if os.environ.get("C12_CPU", "0") != "1":
        hp.update(task_type="GPU", devices="0")
    models = {"all": [], "regular": [], "futures": []}
    masks = {"all": np.ones(len(y), bool), "regular": game_type == 0, "futures": game_type == 1}
    for group in ("all", "regular", "futures"):
        mask = masks[group]
        for seed in SEEDS:
            t0 = time.time()
            log(f"fit group={group} seed={seed} n={int(mask.sum()):,} hp={hp}")
            model = CatBoostClassifier(random_seed=seed, **hp)
            model.fit(X[mask], y[mask], sample_weight=weights[mask])
            models[group].append(model)
            log(f"done group={group} seed={seed} sec={time.time()-t0:.1f}")
    export_models(models, features, tables, os.path.join(OUT, "trees_c12.npz"))
    np.savez_compressed(os.path.join(OUT, "train_features_sample.npz"),
                        y=y[:1000], X=X[:1000])
    with open(os.path.join(OUT, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
