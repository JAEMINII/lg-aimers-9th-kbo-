# -*- coding: utf-8 -*-
"""Train/export the DeepFM candidate for the NumPy-only submission package."""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as Fnn

sys.path.insert(0, "/root/train_chan_3")
import preprocess as PP  # noqa: E402

DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
OUT = os.environ.get("DEEPFM_OUT", "/root/deepfm_full")
SEEDS = (42, 1, 777)
EPOCHS = int(os.environ.get("FM_EPOCHS", "3"))
BS = int(os.environ.get("FM_BS", "4096"))
EMB = int(os.environ.get("FM_EMB", "16"))
HID = int(os.environ.get("FM_HID", "128"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepFM(nn.Module):
    def __init__(self, n_num, cards, emb_dim=16, hidden=128):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(int(c), emb_dim) for c in cards])
        self.cat_linear = nn.ModuleList([nn.Embedding(int(c), 1) for c in cards])
        for e in self.emb:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)
        for e in self.cat_linear:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)
        ncat = len(cards)
        self.deep = nn.Sequential(
            nn.Linear(n_num + ncat * emb_dim, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.num_linear = nn.Linear(n_num, 1)

    def forward(self, xn, xc):
        e = torch.stack([q(xc[:, j]) for j, q in enumerate(self.emb)], dim=1)
        s = e.sum(dim=1)
        fm = 0.10 * 0.5 * (s.square() - e.square().sum(dim=1)).sum(dim=1, keepdim=True)
        first = self.num_linear(xn)
        first = first + sum(q(xc[:, j]) for j, q in enumerate(self.cat_linear))
        return first + fm + self.deep(torch.cat([xn, e.flatten(1)], dim=1))


def prep_train(df, cat_cols):
    cats = []
    cat_values = []
    for c in cat_cols:
        v = df[c].to_numpy()
        vals = np.unique(v)
        vals = vals[np.isfinite(vals)]
        cats.append(np.where(np.isin(v, vals), np.searchsorted(vals, v) + 1, 0))
        cat_values.append(vals.astype(np.float64))
    ci = [df.columns.get_loc(c) for c in cat_cols]
    # Numeric columns are all columns not in cat_cols.
    ni = np.asarray([j for j in range(df.shape[1]) if j not in set(ci)], dtype=np.int64)
    a = df.to_numpy(dtype=np.float64)
    raw = a[:, ni]
    miss = np.isnan(raw)
    med = np.nanmedian(raw, axis=0)
    raw = np.where(miss, med, raw)
    mu = raw.mean(0)
    sd = raw.std(0) + 1e-6
    xn = ((raw - mu) / sd).astype(np.float32)
    has_nan = miss.any(0)
    if has_nan.any():
        xn = np.concatenate([xn, miss[:, has_nan].astype(np.float32)], axis=1)
    xc = np.stack(cats, axis=1).astype(np.int64)
    return xn, xc, np.asarray([len(v) + 1 for v in cat_values]), {
        "cat_values": cat_values, "num_indices": ni, "median": med,
        "mu": mu, "sd": sd, "has_nan": has_nan, "cat_cols": list(cat_cols),
    }


def prep_apply(df, meta):
    a = df.to_numpy(dtype=np.float64)
    xc = []
    for j, vals in enumerate(meta["cat_values"]):
        v = a[:, df.columns.get_loc(meta["cat_cols"][j])]
        pos = np.clip(np.searchsorted(vals, v), 0, max(len(vals) - 1, 0))
        hit = (len(vals) > 0) & (vals[pos] == v)
        xc.append(np.where(hit, pos + 1, 0))
    raw = a[:, meta["num_indices"]]
    miss = np.isnan(raw)
    raw = np.where(miss, meta["median"], raw)
    xn = ((raw - meta["mu"]) / meta["sd"]).astype(np.float32)
    has_nan = meta["has_nan"]
    if has_nan.any():
        xn = np.concatenate([xn, miss[:, has_nan].astype(np.float32)], axis=1)
    return xn, np.stack(xc, 1).astype(np.int64)


def train_one(seed, xn, xc, y, sw, cards):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = DeepFM(xn.shape[1], cards, EMB, HID).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    XN, XC = torch.from_numpy(xn), torch.from_numpy(xc)
    YY, WW = torch.from_numpy(y.astype(np.float32)), torch.from_numpy(sw.astype(np.float32))
    ii = torch.arange(len(y), dtype=torch.long)
    m.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(y))
        total = 0.0
        for a in range(0, len(y), BS):
            b = perm[a:a + BS]
            opt.zero_grad(set_to_none=True)
            z = m(XN[ii[b]].to(DEV), XC[ii[b]].to(DEV))
            yt = YY[b].to(DEV)
            wt = WW[b].to(DEV)
            ce = Fnn.binary_cross_entropy_with_logits(z.squeeze(1), yt, reduction="none")
            br = (z.sigmoid().squeeze(1) - yt).square()
            loss = ((0.5 * ce + 0.5 * br) * wt).sum() / wt.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(b)
        sch.step()
        print(f"seed {seed} epoch {ep+1}/{EPOCHS} loss {total/len(y):.6f}", flush=True)
    return m


def save_model(m, path, meta):
    q = {"format": np.asarray("deepfm_numpy_v1"),
         "n_num": np.asarray(meta["n_num"], dtype=np.int64),
         "emb_dim": np.asarray(EMB, dtype=np.int64), "hidden": np.asarray(HID, dtype=np.int64),
         "num_indices": meta["num_indices"], "median": meta["median"],
         "mu": meta["mu"], "sd": meta["sd"], "has_nan": meta["has_nan"]}
    for j, v in enumerate(meta["cat_values"]):
        q[f"cat_values_{j}"] = v
    for k, v in m.state_dict().items():
        q[k.replace(".", "__")] = v.detach().cpu().numpy()
    np.savez(path, **q)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    train = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig"))
    test = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig"))
    hist = PP.fit_history_tables(train)
    tx = PP.transform_features(train, hist, train_mode=True)
    vx = PP.build_inference_features(test, hist)
    y = train[PP.TARGET].to_numpy(np.float32)
    cat_cols = list(PP.TABM_CATEGORICAL_FEATURES)
    xn, xc, cards, meta = prep_train(tx, cat_cols)
    # Recent-season emphasis and conservative old-futures shrinkage.
    season = train["season"].to_numpy(np.float64)
    game = train["game_type"].astype(str).to_numpy()
    sw = 2.0 ** (season - 2019.0)
    sw *= np.where((game == "F") & (season <= 2022), 0.1, 1.0)
    meta["n_num"] = xn.shape[1]
    vx_n, vx_c = prep_apply(vx, meta)
    np.savez(os.path.join(OUT, "prep.npz"), **{
        "num_indices": meta["num_indices"], "median": meta["median"],
        "mu": meta["mu"], "sd": meta["sd"], "has_nan": meta["has_nan"],
        **{f"cat_values_{j}": v for j, v in enumerate(meta["cat_values"])},
    })
    print(f"train={len(y):,} test={len(vx):,} features={tx.shape[1]} "
          f"numeric={xn.shape[1]} cards={cards.tolist()} elapsed={time.time()-t0:.0f}s", flush=True)
    for seed in SEEDS:
        m = train_one(seed, xn, xc, y, sw, cards)
        save_model(m, os.path.join(OUT, f"deepfm_s{seed}.npz"), meta)
        del m
        gc.collect(); torch.cuda.empty_cache()
    print(f"DONE elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
