# -*- coding: utf-8 -*-
"""Export the **regular** branch on the same f2b path (c4 4-level regime).

Why: c4_1gun.py measured the 1군 path with vs without the regime column.
     VS=2022  1군 +8.0 (t=6.32, 3/3)   VS=2024  1군 +22.9 (t=3.04, 3/3)
     The deployed 1군 path has no regime column at all. This closes that gap.
     all / futures are reused from f2b_platfix_final (same input matrix).

This is deliberately separate from the historical ``export_final.py`` path:

* 4-level regime code: old F/R=0/1, new F/R=2/3;
* old Futures rows have Stage-1 weight 0.1;
* training ``plat_dev`` is replaced with the as-of value from ``features44``;
* all gets one Stage-2 epoch and Futures gets four, both on 2024.

Those choices match ``fut_stage2b.py``.  The output is in the compact NumPy
format already understood by the submission inference code, so it introduces
no training-only dependency into the submitted package.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SC = Path(__file__).resolve().parent
ROOT = SC.parent
sys.path.insert(0, str(SC))
sys.path.insert(0, str(ROOT))

DATA = Path(os.environ.get("DATA", "/workspace/aimers/data"))
OUT = Path(os.environ.get("OUT", "/workspace/aimers/out/f2b_regular"))
FINAL_SEASON = int(os.environ.get("FINAL_SEASON", "2025"))
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "42").split(","))
OLD_F_MAX = 2022
OLD_W = 0.1
LR1 = 3e-3
LR2 = 2e-4
EPOCHS = {"all": 1, "futures": 4, "regular": 1}

# Keep VS out of the process environment.  tabm_gate_gpu uses VS only for its
# import-time gate arrays and would otherwise require non-existent cb_gate2025.
import features44 as F  # noqa: E402
import tabm_gate_gpu as G  # noqa: E402
from export_ours import export, prep_stats  # noqa: E402
from train_chan_3 import preprocess as PP  # noqa: E402


CFG_BASE = dict(
    k=32,
    n_blocks=3,
    d_block=256,
    dropout=0.1,
    d_emb=16,
    num_embeddings="linear_relu",
    backbone_kind="batch_ensemble",
    stage1_epochs=2,
    stage1_lr=LR1,
    stage2_lr=LR2,
    regime="c4_f2b_platfix",
    plat_dev="asof",
    final_season=FINAL_SEASON,
)


def _prepare():
    """Build exactly the train matrix used in fut_stage2b, at VS=2025."""
    d = F.build(str(DATA), VS=FINAL_SEASON)
    season = d["season"].astype(np.float64)
    is_f = d["is_f"].astype(bool)
    y = d["y"].astype(np.float32)
    f44 = list(d["F44"])

    raw = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    sorted_raw = PP.sort_by_row_id(raw)
    row_ids = raw["row_id"].to_numpy()
    if not np.array_equal(row_ids, sorted_raw["row_id"].to_numpy()):
        raise ValueError("train.csv row order is not row_id order; explicit alignment is required")
    if len(raw) != len(season):
        raise ValueError("features44 and raw train row counts differ")

    hist = PP.fit_history_tables(sorted_raw[sorted_raw.season < FINAL_SEASON])
    xdf = PP.transform_features(sorted_raw, hist, train_mode=True)
    cols = list(xdf.columns)
    x = xdf.to_numpy(dtype=np.float32)

    # This is the important platfix.  PP's training path otherwise derives a
    # full-history platoon table, while the final inference path is as-of.
    plat_col = cols.index("plat_dev")
    x[:, plat_col] = d["X44"][:, f44.index("plat_dev")].astype(np.float32)

    cat_idx = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    old = season <= OLD_F_MAX
    c4 = np.where(old & is_f, 0.0,
                  np.where(old & ~is_f, 1.0,
                           np.where(is_f, 2.0, 3.0))).astype(np.float32)[:, None]
    xin = np.concatenate([x, c4], axis=1)
    feats = cols + ["abs_regime"]
    return xin, cat_idx + [x.shape[1]], feats, season, is_f, y, old


def _train_export(branch, idx, weights, season, stats, cards, n_num, features, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = G.make_model()
    G.train(model, idx, 2, LR1, w=weights, seed=seed,
            tag=f"final {branch} s{seed} S1")
    s2 = idx[season[idx] == FINAL_SEASON - 1]
    if not len(s2):
        raise ValueError(f"{branch}: no Stage-2 rows for {FINAL_SEASON - 1}")
    params = G.stage2_params(model)
    for epoch in range(EPOCHS[branch]):
        G.train(model, s2, 1, LR2, params=params, seed=seed + epoch,
                tag=f"final {branch} s{seed} S2e{epoch + 1}")
    path = OUT / f"f2b_{branch}_s{seed}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    export(model, str(path), stats, cards, n_num,
           dict(CFG_BASE, branch=branch, seed=seed, stage2_epochs=EPOCHS[branch]),
           features)
    size_mb = path.stat().st_size / 1e6
    print(f"  saved {path.name:28s} {size_mb:6.2f} MB  "
          f"stage2={len(s2):,} x {EPOCHS[branch]}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    existing = list(OUT.glob("f2b_*.npz"))
    if existing:
        raise FileExistsError(f"output directory already has f2b artifacts: {existing}")

    xin, cat_idx, features, season, is_f, y, old = _prepare()
    train_mask = np.ones(len(xin), dtype=bool)
    xn, xc, cards, stats = prep_stats(xin, train_mask, cat_idx)
    G.Xn, G.cards = xn, cards
    G.XN, G.XC = torch.from_numpy(xn), torch.from_numpy(xc)
    G.YY = torch.from_numpy(y)

    all_idx = np.arange(len(xin), dtype=np.int64)
    fut_idx = all_idx[is_f]
    weights_all = np.where(is_f & old, OLD_W, 1.0).astype(np.float64)
    weights_fut = weights_all[fut_idx]
    print(f"f2b final: rows={len(xin):,}, futures={len(fut_idx):,}, "
          f"features={len(features)}, numeric={xn.shape[1]}, cats={cards.tolist()}",
          flush=True)
    print(f"seeds={SEEDS}; old Futures weight={OLD_W}; "
          f"S2 all={EPOCHS['all']} / futures={EPOCHS['futures']}", flush=True)

    # regular 브랜치만 굽는다. all / futures 는 f2b_platfix_final 에서 이미 나왔고
    # 같은 입력 행렬·같은 통계 위에서 만들어지므로 그대로 짝이 맞는다.
    reg_idx = all_idx[~is_f]
    weights_reg = np.ones(len(reg_idx), dtype=np.float64)
    print(f"regular rows={len(reg_idx):,}", flush=True)
    for seed in SEEDS:
        _train_export("regular", reg_idx, weights_reg, season, stats, cards,
                      xn.shape[1], features, seed)

    files = sorted(OUT.glob("f2b_*.npz"))
    total = sum(p.stat().st_size for p in files)
    print(f"DONE {len(files)} models, {total / 1e6:.2f} MB, "
          f"elapsed={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
