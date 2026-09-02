# -*- coding: utf-8 -*-
"""Temporal DeepFM candidate for the current TabM/CatBoost gate.

This is deliberately a small, low-risk candidate rather than a replacement for
the production model.  It learns explicit pairwise interactions between the
categorical fields (pitcher/batter/team/hand/count) and is evaluated as a
logit-space blend with the existing gate baseline.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as Fnn

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/root")

import features44 as F  # noqa: E402
import tabm_gate_gpu as G  # noqa: E402

VS = int(os.environ.get("VS", "2024"))
SEEDS = (42, 1, 777)
EPOCHS = int(os.environ.get("FM_EPOCHS", "3"))
BS = int(os.environ.get("FM_BS", "4096"))
EMB = int(os.environ.get("FM_EMB", "16"))
HID = int(os.environ.get("FM_HID", "128"))
DECAY = float(os.environ.get("FM_DECAY", "2.0"))


def log(s):
    print(s, flush=True)
    with open(os.path.join(G.OUT, "deepfm_progress.txt"), "a", encoding="utf-8") as f:
        f.write(s + "\n")


class DeepFM(nn.Module):
    def __init__(self, n_num, cards, emb_dim=16, hidden=128):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(int(c), emb_dim) for c in cards])
        self.cat_linear = nn.ModuleList([nn.Embedding(int(c), 1) for c in cards])
        # nn.Embedding defaults to N(0, 1), which makes the FM interaction
        # explode at initialization when several fields are summed.
        for e in self.emb:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)
        for e in self.cat_linear:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)
        ncat = len(cards)
        d = n_num + ncat * emb_dim
        self.deep = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.num_linear = nn.Linear(n_num, 1)

    def forward(self, xn, xc):
        es = [e(xc[:, j]) for j, e in enumerate(self.emb)]
        e = torch.stack(es, dim=1)  # B,F,E
        summed = e.sum(dim=1)
        fm = 0.10 * 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1, keepdim=True)
        first = self.num_linear(xn)
        first = first + sum(m(xc[:, j]) for j, m in enumerate(self.cat_linear))
        deep = self.deep(torch.cat([xn, e.flatten(1)], dim=1))
        return first + fm + deep


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def loss(logits, y, w=None):
    yy = y[:, None]
    ce = Fnn.binary_cross_entropy_with_logits(logits, yy, reduction="none").squeeze(1)
    br = (logits.sigmoid().squeeze(1) - y).square()
    q = 0.5 * ce + 0.5 * br
    return q.mean() if w is None else (q * w).sum() / w.sum()


def train_one(seed, tr_idx, y, season, is_f, XN, XC, cards):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dev = G.DEV
    m = DeepFM(XN.shape[1], cards, EMB, HID).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ii = torch.from_numpy(tr_idx)
    yy = torch.from_numpy(y.astype(np.float32))
    # Match the production temporal emphasis, with conservative old-futures weight.
    sw = DECAY ** (season[tr_idx].astype(np.float64) - 2019.0)
    sw = sw * np.where(is_f[tr_idx] & (season[tr_idx] <= VS - 2), 0.1, 1.0)
    ww = torch.from_numpy(sw.astype(np.float32))
    m.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(tr_idx))
        tot = 0.0
        for a in range(0, len(tr_idx), BS):
            b = perm[a:a + BS]
            ix = ii[b]
            xn = XN[ix].to(dev, non_blocking=True)
            xc = XC[ix].to(dev, non_blocking=True)
            yt = yy[ix].to(dev, non_blocking=True)
            wt = ww[b].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            l = loss(m(xn, xc), yt, wt)
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            tot += float(l.detach()) * len(ix)
        sch.step()
        log(f"  deepfm s{seed} ep{ep+1}/{EPOCHS} loss {tot/len(tr_idx):.6f} "
            f"lr {opt.param_groups[0]['lr']:.6f}")
    return m


@torch.no_grad()
def predict(m, idx, XN, XC):
    m.eval()
    out = np.empty(len(idx), dtype=np.float64)
    ii = torch.from_numpy(idx)
    for a in range(0, len(idx), 8192):
        b = ii[a:a + 8192]
        out[a:a + len(b)] = m(
            XN[b].to(G.DEV, non_blocking=True), XC[b].to(G.DEV, non_blocking=True)
        ).squeeze(1).double().cpu().numpy()
    return out


def main():
    t0 = time.time()
    d = F.build(G.DATA, VS=VS)
    X = d["X44"]
    XN, XC, cards = G.prep(X, d["m_tr"], d["cat_idx"])
    G.Xn, G.cards = XN, cards
    G.XN, G.XC = torch.from_numpy(XN), torch.from_numpy(XC)
    tr_idx = np.where(d["m_tr"])[0]
    va_idx = np.where(d["m_va"])[0]
    gate = va_idx
    is_f = d["is_f"][gate]
    reg = ~is_f
    yv = d["y"][gate].astype(np.float64)
    # tabm_gate_gpu.CUR is already ordered over the VS validation gate.
    base = np.asarray(G.CUR, dtype=np.float64)
    base_l = logit(base)
    log(f"DeepFM VS={VS} X={X.shape} train={len(tr_idx):,} gate={len(gate):,} "
        f"cards={cards.tolist()} emb={EMB} hid={HID} ep={EPOCHS}")
    log(f"baseline 1gun={F.best_shift(base[reg], yv[reg])[0]:.1f} "
        f"overall={F.best_shift(base, yv)[0]:.1f}")

    rs = []
    for seed in SEEDS:
        m = train_one(seed, tr_idx, d["y"], d["season"], d["is_f"],
                      G.XN, G.XC, cards)
        rs.append(predict(m, gate, G.XN, G.XC))
        del m
        gc.collect()
        torch.cuda.empty_cache()
    fm_l = np.mean(np.stack(rs, 0), axis=0)
    np.save(os.path.join(G.OUT, f"deepfm_vs{VS}_logit.npy"), fm_l)
    log(f"DeepFM solo 1gun={F.best_shift(sigmoid(fm_l)[reg], yv[reg])[0]:.1f} "
        f"overall={F.best_shift(sigmoid(fm_l), yv)[0]:.1f}")

    best = (-1e18, None)
    for a in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40):
        p = sigmoid(base_l + a * (fm_l - base_l))
        s = F.best_shift(p[reg], yv[reg])[0]
        so = F.best_shift(p, yv)[0]
        log(f"  alpha={a:.2f}  1gun={s:.1f} ({s-F.best_shift(base[reg], yv[reg])[0]:+.1f}) "
            f"overall={so:.1f}")
        if s > best[0]:
            best = (s, a)
    log(f"BEST alpha={best[1]}  delta_1gun={best[0]-F.best_shift(base[reg], yv[reg])[0]:+.1f} "
        f"elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
