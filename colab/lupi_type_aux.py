# -*- coding: utf-8 -*-
"""LUPI v2: masked current pitch-type auxiliary classification.

The final prediction remains only head 0.  Current pitch type is a training
only auxiliary label on audited Trackman matches; no Trackman file or test-row
matching is needed at inference.
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
FOLDS = tuple(int(x) for x in os.environ.get("LT_FOLDS", "2024,2022").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("LT_SEEDS", "42,1,777").split(","))
TYPE_WEIGHTS = {"base": 0.0, "type005": 0.005, "type02": 0.02, "type05": 0.05}
ARMS = tuple(x for x in os.environ.get("LT_ARMS", "base,type005,type02").split(",") if x)
OLD_F_MAX, OLD_W, LR1 = 2022, 0.1, 3e-3
EP2 = {"all": 1, "regular": 1, "futures": 4}

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
import rtdl_num_embeddings as rne                               # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def make_model():
    return G.TabM.make(
        n_num_features=G.Xn.shape[1], cat_cardinalities=[int(c) for c in G.cards], d_out=5,
        num_embeddings=rne.LinearReLUEmbeddings(G.Xn.shape[1], d_embedding=16),
        arch_type="tabm", k=32, n_blocks=3, d_block=256, dropout=.1,
    ).to(G.DEV)


def train_mt(model, idx, epochs, lr, y, typ, tw, params=None, w=None, bs=2048,
             seed=42, tag=""):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    ps = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ii = torch.from_numpy(idx.astype(np.int64))
    Y, T = torch.from_numpy(y.astype(np.float32)), torch.from_numpy(typ.astype(np.int64))
    W = None if w is None else torch.from_numpy(w.astype(np.float32))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx)); all_loss = main_loss = type_loss = 0.0
        for s in range(0, len(idx), bs):
            local = perm[s:s + bs]; b = ii[local]
            xn, xc = G.XN[b].to(G.DEV, non_blocking=True), G.XC[b].to(G.DEV, non_blocking=True)
            yb, tb = Y[b].to(G.DEV, non_blocking=True), T[b].to(G.DEV, non_blocking=True)
            wb = None if W is None else W[local].to(G.DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True); z = model(xn, xc); direct = z[:, :, :1]
            yy = yb[:, None, None].expand_as(direct)
            per = (0.5 * TF.binary_cross_entropy_with_logits(direct, yy, reduction="none")
                   + 0.5 * (direct.sigmoid() - yy).square()).mean((1, 2))
            main = per.mean() if wb is None else (per * wb).sum() / wb.sum()
            tl = torch.zeros((), device=G.DEV)
            has = tb >= 0
            if tw and has.any():
                k = z.shape[1]
                logits = z[has, :, 1:].reshape(-1, 4)
                labels = tb[has, None].expand(-1, k).reshape(-1)
                per_t = TF.cross_entropy(logits, labels, reduction="none").reshape(-1, k).mean(1)
                tl = per_t.mean() if wb is None else (per_t * wb[has]).sum() / wb[has].sum()
            loss = main + tw * tl
            loss.backward(); torch.nn.utils.clip_grad_norm_(ps, 5.0); opt.step()
            n = len(b); all_loss += float(loss.detach()) * n; main_loss += float(main.detach()) * n
            type_loss += float(tl.detach()) * n
        sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} total={all_loss/len(idx):.6f} "
              f"main={main_loss/len(idx):.6f} type_ce={type_loss/len(idx):.6f}")


@torch.no_grad()
def predict(model, idx, typ=None, bs=8192):
    model.eval(); out = np.empty(len(idx), np.float64); correct = count = 0
    ii = torch.from_numpy(idx.astype(np.int64))
    for s in range(0, len(idx), bs):
        b = ii[s:s + bs]; z = model(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV))
        out[s:s + len(b)] = z[:, :, 0].sigmoid().mean(1).double().cpu().numpy()
        if typ is not None:
            got = typ[b.cpu().numpy()]; has = got >= 0
            if has.any():
                pred = z[:, :, 1:].mean(1).argmax(1).cpu().numpy()
                correct += int((pred[has] == got[has]).sum()); count += int(has.sum())
    return out, (correct / count if count else np.nan)


def blend_members(vs):
    if vs == 2024:
        cb = .7*np.load(os.path.join(DL, "pcgpu2024_c12_cmh_10.npy")) + .3*np.load(os.path.join(DL, "pcgpu2024_base44_10.npy"))
        ms = np.load(os.path.join(DL, "ms24_audit_direct.npy"))
    else:
        cb, ms = np.load(os.path.join(DL, f"cb50fixed_{vs}.npy")), np.load(os.path.join(DL, "ms22_audit_direct.npy"))
    return cb.astype(np.float64), np.load(os.path.join(DL, f"dg_{vs}_DIN.npy")).mean(0), ms.astype(np.float64)


def build_type(rid, season, vs):
    m = pd.read_csv(os.path.join(DL, MATCH), encoding="utf-8-sig")
    value = {"fastball": 0, "breaking": 1, "offspeed": 2}
    pos = pd.Series(np.arange(len(rid)), index=rid).reindex(m.row_id).to_numpy()
    ok = np.isfinite(pos); out = np.full(len(rid), -1, np.int64)
    out[pos[ok].astype(np.int64)] = m.loc[ok, "pitch_type_group"].map(value).fillna(3).astype(np.int64)
    print(f"  type labels train={int((out[season < vs] >= 0).sum()):,} gate={int((out[season == vs] >= 0).sum()):,} "
          f"dist={np.bincount(out[out >= 0], minlength=4).tolist()}", flush=True)
    return out


def prepare(vs, d0, rid, raw, pos_map):
    season, isf, f44 = d0["season"], d0["is_f"], list(d0["F44"])
    hist = PP.fit_history_tables(raw[raw.season < vs]); xf = PP.transform_features(raw, hist, train_mode=True)
    cols = list(xf.columns); x = xf.to_numpy(np.float32)[pos_map]
    dv = F.build(DATA, VS=vs); x[:, cols.index("plat_dev")] = dv["X44"][:, f44.index("plat_dev")]
    old = season <= OLD_F_MAX; c4 = np.where(old & isf, 0., np.where(old & ~isf, 1., np.where(isf, 2., 3.))).astype(np.float32)[:, None]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    xn, xc, cards = G.prep(np.c_[x, c4], season < vs, ci + [x.shape[1]])
    G.Xn, G.cards, G.XN, G.XC = xn, cards, torch.from_numpy(xn), torch.from_numpy(xc)
    train = np.flatnonzero(season < vs).astype(np.int64); tif = isf[train]
    return np.flatnonzero(season == vs).astype(np.int64), train, tif, old


def main():
    if "base" not in ARMS or any(a not in TYPE_WEIGHTS for a in ARMS):
        raise ValueError(f"arms must include base and be in {sorted(TYPE_WEIGHTS)}")
    d0 = F.build(DATA, VS=2024); season, isf, y = d0["season"].astype(np.int16), d0["is_f"], d0["y"].astype(np.float64)
    rid = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig", usecols=["row_id"]).row_id.to_numpy()
    raw = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig"))
    pos_map = pd.Series(np.arange(len(raw)), index=raw.row_id).reindex(rid).to_numpy()
    for vs in FOLDS:
        print("\n"+"="*94+f"\nLUPI pitch-type auxiliary VS={vs}; {ARMS}\n"+"="*94, flush=True)
        gate, train, tif, old = prepare(vs, d0, rid, raw, pos_map); typ = build_type(rid, season, vs)
        idxs = {"all": train, "regular": train[~tif], "futures": train[tif]}
        weights = {"all": np.where(tif & old[train], OLD_W, 1.).astype(np.float32),
                   "regular": None, "futures": np.where(old[train[tif]], OLD_W, 1.).astype(np.float32)}
        pred = {}
        for arm in ARMS:
            tw, allp, t0 = TYPE_WEIGHTS[arm], [], time.time()
            for seed in SEEDS:
                br = {}
                for name in ("all", "regular", "futures"):
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); model = make_model()
                    idx = idxs[name]
                    train_mt(model, idx, 2, LR1, y, typ, tw, w=weights[name], seed=seed,
                             tag=f"VS{vs} {arm} {name} s{seed} S1")
                    recent = idx[season[idx] == vs-1]; pars = G.stage2_params(model)
                    for e in range(EP2[name]):
                        ww = None if weights[name] is None else weights[name][season[idx] == vs-1]
                        train_mt(model, recent, 1, 2e-4, y, typ, tw, params=pars, w=ww, seed=seed+e,
                                 tag=f"VS{vs} {arm} {name} s{seed} S2e{e+1}")
                    br[name], acc = predict(model, gate, typ if name != "futures" else None)
                    if name != "futures": print(f"  {arm} seed={seed} {name} type_acc={acc:.4f}", flush=True)
                    del model; gc.collect(); torch.cuda.empty_cache()
                allp.append(np.where(isf[gate], .6*br["all"]+.4*br["futures"], .6*br["all"]+.4*br["regular"]))
            pred[arm] = allp; np.save(os.path.join(DL, f"lt_{vs}_{arm}.npy"), np.asarray(allp))
            print(f"  {arm} complete {time.time()-t0:.0f}s", flush=True)
        cb,din,ms = blend_members(vs); yv=y[gate]; ref=pred["base"]
        refmix=[F.best_shift(.31*cb+.13*q+.31*din+.25*ms,yv)[0] for q in ref]
        print("summary",flush=True)
        for arm, ps in pred.items():
            line=f"  {arm:8s} direct={F.best_shift(np.mean(ps,0),yv)[0]:8.1f}"
            if arm != "base":
                dd=[F.best_shift(a,yv)[0]-F.best_shift(b,yv)[0] for a,b in zip(ps,ref)]
                md=[F.best_shift(.31*cb+.13*q+.31*din+.25*ms,yv)[0]-r for q,r in zip(ps,refmix)]
                line+=f" direct_delta={np.mean(dd):+.2f} {sum(x>0 for x in dd)}/3 mix_delta={np.mean(md):+.2f} {sum(x>0 for x in md)}/3"
            print(line,flush=True)


if __name__ == "__main__":
    main()
