# -*- coding: utf-8 -*-
"""LUPI v2: predict current-pitch physics as masked auxiliary targets.

Unlike soft-label KD, the direct control-success target is never replaced.  A
TabM with nine outputs is trained on its normal binary loss at head 0 and, only
on audited train <-> Trackman matches, on eight standardized current-pitch
physical measurements at heads 1..8.  Inference and scoring use head 0 only.

The supplied Trackman file is used only while training this experiment.  This
script deliberately exports no inference artifact and no Trackman feature.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF


SC = os.path.dirname(os.path.abspath(__file__))
for p in (SC, "/root/aimers", os.path.dirname(SC)):
    if p not in sys.path:
        sys.path.insert(0, p)
DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
MATCH = os.environ.get("LUPI_MATCH", "lupi_match_v2.csv.gz")
FOLDS = tuple(int(x) for x in os.environ.get("LP_FOLDS", "2024,2022").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("LP_SEEDS", "42,1,777").split(","))
ARM_WEIGHTS = {"base": 0.0, "aux02": 0.02, "aux06": 0.06, "aux12": 0.12}
ARMS = tuple(x for x in os.environ.get("LP_ARMS", "base,aux02,aux06").split(",") if x)
OLD_F_MAX, OLD_W, LR1 = 2022, 0.1, 3e-3
EP2 = {"all": 1, "regular": 1, "futures": 4}
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
import rtdl_num_embeddings as rne                               # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def make_model(d_out: int = 9):
    return G.TabM.make(
        n_num_features=G.Xn.shape[1],
        cat_cardinalities=[int(c) for c in G.cards], d_out=d_out,
        num_embeddings=rne.LinearReLUEmbeddings(G.Xn.shape[1], d_embedding=16),
        arch_type="tabm", k=32, n_blocks=3, d_block=256, dropout=0.1,
    ).to(G.DEV)


def train_mt(model, idx, epochs, lr, y, aux, aux_weight, params=None, w=None,
             bs=2048, wd=3e-4, clip=5.0, seed=42, tag=""):
    """Primary binary loss plus a row-weighted, masked physics MSE."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ps = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ii = torch.from_numpy(idx.astype(np.int64))
    Y = torch.from_numpy(y.astype(np.float32))
    A = torch.from_numpy(aux.astype(np.float32))
    W = None if w is None else torch.from_numpy(w.astype(np.float32))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        total, main_total, aux_total = 0.0, 0.0, 0.0
        for s in range(0, len(idx), bs):
            local = perm[s:s + bs]
            b = ii[local]
            xn = G.XN[b].to(G.DEV, non_blocking=True)
            xc = G.XC[b].to(G.DEV, non_blocking=True)
            yb = Y[b].to(G.DEV, non_blocking=True)
            wb = None if W is None else W[local].to(G.DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xn, xc)
            direct = out[:, :, :1]
            target = yb[:, None, None].expand_as(direct)
            per = (0.5 * TF.binary_cross_entropy_with_logits(direct, target, reduction="none")
                   + 0.5 * (direct.sigmoid() - target).square()).mean(dim=(1, 2))
            main = per.mean() if wb is None else (per * wb).sum() / wb.sum()
            aux_loss = torch.zeros((), device=G.DEV)
            if aux_weight:
                ab = A[b].to(G.DEV, non_blocking=True)
                mask = torch.isfinite(ab)
                has = mask.any(dim=1)
                if has.any():
                    pred = out[:, :, 1:]
                    tgt = torch.nan_to_num(ab, nan=0.0)[:, None, :]
                    m3 = mask[:, None, :]
                    # Mean over both TabM members and available physical axes.
                    per_aux = ((pred - tgt).square() * m3).sum(dim=(1, 2)) / \
                              (m3.sum(dim=(1, 2)).clamp_min(1) * pred.shape[1])
                    aux_loss = (per_aux[has].mean() if wb is None else
                                (per_aux[has] * wb[has]).sum() / wb[has].sum())
            loss = main + aux_weight * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, clip)
            opt.step()
            n = len(b)
            total += float(loss.detach()) * n
            main_total += float(main.detach()) * n
            aux_total += float(aux_loss.detach()) * n
        sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} total={total/len(idx):.6f} "
              f"main={main_total/len(idx):.6f} phys_mse={aux_total/len(idx):.6f}")
    return model


@torch.no_grad()
def predict_direct(model, idx, bs=8192):
    model.eval()
    out = np.empty(len(idx), np.float64)
    ii = torch.from_numpy(idx.astype(np.int64))
    for s in range(0, len(idx), bs):
        b = ii[s:min(s + bs, len(idx))]
        z = model(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV))[:, :, 0]
        out[s:s + len(b)] = z.sigmoid().mean(dim=1).double().cpu().numpy()
    return out


@torch.no_grad()
def validation_aux_mse(model, idx, aux, bs=8192):
    """Sanity metric only: whether the physics heads learned their stated task."""
    vals = []
    ii = torch.from_numpy(idx.astype(np.int64))
    model.eval()
    for s in range(0, len(idx), bs):
        b = ii[s:min(s + bs, len(idx))]
        got = aux[b.cpu().numpy()]
        mask = np.isfinite(got)
        if not mask.any():
            continue
        pred = model(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV))[:, :, 1:].mean(1).cpu().numpy()
        vals.append(np.square(pred[mask] - got[mask]).mean())
    return float(np.mean(vals)) if vals else np.nan


def blend_members(vs: int):
    if vs == 2024:
        cb = (0.70 * np.load(os.path.join(DL, "pcgpu2024_c12_cmh_10.npy"))
              + 0.30 * np.load(os.path.join(DL, "pcgpu2024_base44_10.npy"))).astype(np.float64)
        ms = np.load(os.path.join(DL, "ms24_audit_direct.npy")).astype(np.float64)
    else:
        cb = np.load(os.path.join(DL, f"cb50fixed_{vs}.npy")).astype(np.float64)
        ms = np.load(os.path.join(DL, "ms22_audit_direct.npy")).astype(np.float64)
    return cb, np.load(os.path.join(DL, f"dg_{vs}_DIN.npy")).mean(0), ms


def build_aux_targets(rid, season, vs):
    m = pd.read_csv(os.path.join(DL, MATCH), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(rid)), index=rid)
    loc = pos.reindex(m.row_id).to_numpy()
    good = np.isfinite(loc)
    loc, m = loc[good].astype(np.int64), m.loc[good]
    out = np.full((len(rid), len(PHYS)), np.nan, np.float32)
    out[loc] = m[PHYS].to_numpy(np.float32)
    train = (season < vs)[:, None] & np.isfinite(out)
    mu = np.array([np.nanmean(out[:, j][train[:, j]]) for j in range(len(PHYS))], np.float32)
    sd = np.array([np.nanstd(out[:, j][train[:, j]]) + 1e-6 for j in range(len(PHYS))], np.float32)
    out = (out - mu) / sd
    print(f"  physics targets {int(np.isfinite(out[season < vs]).any(1).sum()):,} train / "
          f"{int(np.isfinite(out[season == vs]).any(1).sum()):,} gate; "
          f"mean={mu.round(3).tolist()} sd={sd.round(3).tolist()}", flush=True)
    return out


def prepare_fold(vs, d0, rid, tr_sorted, pos_map):
    season, isf, y, f44 = d0["season"], d0["is_f"], d0["y"], list(d0["F44"])
    hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < vs])
    xf = PP.transform_features(tr_sorted, hist, train_mode=True)
    cols = list(xf.columns)
    x = xf.to_numpy(dtype=np.float32)[pos_map]
    dv = F.build(DATA, VS=vs)
    x[:, cols.index("plat_dev")] = dv["X44"][:, f44.index("plat_dev")].astype(np.float32)
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0, np.where(old & ~isf, 1.0,
                  np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    xn, xc, cards = G.prep(np.concatenate([x, c4], 1), season < vs, ci + [x.shape[1]])
    G.Xn, G.cards = xn, cards
    G.XN, G.XC = torch.from_numpy(xn), torch.from_numpy(xc)
    gate = np.flatnonzero(season == vs).astype(np.int64)
    train = np.flatnonzero(season < vs).astype(np.int64)
    tif = isf[train]
    return gate, train, tif, old


def main():
    unknown = [a for a in ARMS if a not in ARM_WEIGHTS]
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}; choices={sorted(ARM_WEIGHTS)}")
    d0 = F.build(DATA, VS=2024)
    season = d0["season"].astype(np.int16)
    isf, y = d0["is_f"], d0["y"].astype(np.float64)
    rid = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id"])["row_id"].to_numpy()
    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos_map = pd.Series(np.arange(len(tr_sorted)), index=tr_sorted.row_id).reindex(rid).to_numpy()
    for vs in FOLDS:
        print("\n" + "=" * 100 + f"\nLUPI physics auxiliary VS={vs}; arms={ARMS}\n" + "=" * 100,
              flush=True)
        gate, train, tif, old = prepare_fold(vs, d0, rid, tr_sorted, pos_map)
        aux = build_aux_targets(rid, season, vs)
        isf_gate, yv = isf[gate], y[gate]
        indices = {
            "all": train,
            "regular": train[~tif],
            "futures": train[tif],
        }
        weights = {
            "all": np.where(tif & old[train], OLD_W, 1.0).astype(np.float32),
            "regular": None,
            "futures": np.where(old[train[tif]], OLD_W, 1.0).astype(np.float32),
        }
        all_preds = {}
        for arm in ARMS:
            aw = ARM_WEIGHTS[arm]
            ps = []
            t0 = time.time()
            for seed in SEEDS:
                branch = {}
                for name in ("all", "regular", "futures"):
                    idx = indices[name]
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    model = make_model()
                    train_mt(model, idx, 2, LR1, y, aux, aw, w=weights[name], seed=seed,
                             tag=f"VS{vs} {arm} {name} s{seed} S1")
                    recent = idx[season[idx] == vs - 1]
                    params = G.stage2_params(model)
                    for ep in range(EP2[name]):
                        train_mt(model, recent, 1, 2e-4, y, aux, aw, params=params,
                                 w=None if weights[name] is None else weights[name][season[idx] == vs - 1],
                                 seed=seed + ep, tag=f"VS{vs} {arm} {name} s{seed} S2e{ep + 1}")
                    branch[name] = predict_direct(model, gate)
                    if name != "futures":
                        print(f"  {arm} seed={seed} {name} val_phys_mse="
                              f"{validation_aux_mse(model, gate, aux):.4f}", flush=True)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                ps.append(np.where(isf_gate, 0.6 * branch["all"] + 0.4 * branch["futures"],
                                   0.6 * branch["all"] + 0.4 * branch["regular"]))
            all_preds[arm] = ps
            np.save(os.path.join(DL, f"lp_{vs}_{arm}.npy"), np.asarray(ps))
            print(f"  {arm} complete {time.time() - t0:.0f}s", flush=True)

        cb, din, ms = blend_members(vs)
        base = all_preds["base"]
        ref_mix = [F.best_shift(.31 * cb + .13 * q + .31 * din + .25 * ms, yv)[0] for q in base]
        print("\nsummary (best-shift diagnostics; primary output only)", flush=True)
        for arm, ps in all_preds.items():
            score = F.best_shift(np.mean(ps, 0), yv)[0]
            line = f"  {arm:6s} direct={score:8.1f}"
            if arm != "base":
                delta = [F.best_shift(a, yv)[0] - F.best_shift(b, yv)[0] for a, b in zip(ps, base)]
                mix = [F.best_shift(.31 * cb + .13 * q + .31 * din + .25 * ms, yv)[0] for q in ps]
                md = [a - b for a, b in zip(mix, ref_mix)]
                line += (f"  direct_delta={np.mean(delta):+.2f} {sum(v > 0 for v in delta)}/"
                         f"{len(delta)}  mix_delta={np.mean(md):+.2f} {sum(v > 0 for v in md)}/{len(md)}")
            print(line, flush=True)


if __name__ == "__main__":
    main()
