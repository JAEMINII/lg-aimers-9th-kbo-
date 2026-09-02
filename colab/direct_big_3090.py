"""Direct all-data TabM capacity control for the 24GB GPU.

This intentionally has no auxiliary state loss.  It is paired with
multistate_softmax.py to tell whether a gain comes from width/K or from the
multi-state target structure.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabm import TabM
from rtdl_num_embeddings import LinearReLUEmbeddings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("AIMERS_DATA", "/root/data"))
OUT = Path(os.environ.get("AIMERS_OUT", "/root/direct_big_out"))
OUT.mkdir(parents=True, exist_ok=True)
VS = int(os.environ.get("VS", "2024"))
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "42,1,777").split(","))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
K = int(os.environ.get("K", "64"))
D = int(os.environ.get("D", "512"))
BS = int(os.environ.get("BS", "2048"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda" and os.environ.get("AMP", "1") != "0"

import features44 as FF  # noqa: E402
import tabm_gate_gpu as G  # noqa: E402
from train_chan_3 import preprocess as PP  # noqa: E402


def score(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r)))


def shift_score(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1.0 - p))
    cs = np.linspace(-0.08, 0.08, 321)
    vals = [score(1.0 / (1.0 + np.exp(-(z + c))), y) for c in cs]
    i = int(np.argmax(vals))
    return float(vals[i]), float(cs[i])


def log(s):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    with (OUT / "progress.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def train(m, idx, xn, xc, y, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=float(os.environ.get("LR", "0.002")), weight_decay=3e-4)
    steps = EPOCHS * ((len(idx) + BS - 1) // BS)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    ii = torch.from_numpy(idx.astype(np.int64))
    yy = torch.from_numpy(y.astype(np.float32))
    m.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(idx))
        tot = 0.0
        for st in range(0, len(idx), BS):
            pos = perm[st:st + BS]
            b = ii[pos]
            xb = xn[b].to(DEVICE, non_blocking=True)
            cb = xc[b].to(DEVICE, non_blocking=True)
            yb = yy[b].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                lg = m(xb, cb).squeeze(-1)
                ym = yb[:, None].expand_as(lg)
                loss = 0.5 * F.binary_cross_entropy_with_logits(lg, ym)
                loss = loss + 0.5 * (lg.sigmoid() - ym).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step(); sch.step()
            tot += float(loss.detach()) * len(b)
        log(f"s{seed} ep{ep+1}/{EPOCHS} loss={tot/len(idx):.6f}")


@torch.no_grad()
def predict(m, xn, xc, idx):
    m.eval(); out = np.empty(len(idx), np.float64)
    ii = torch.from_numpy(idx.astype(np.int64))
    for st in range(0, len(idx), BS):
        b = ii[st:st + BS]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            p = m(xn[b].to(DEVICE), xc[b].to(DEVICE)).squeeze(-1).sigmoid().mean(1)
        out[st:st + len(b)] = p.float().cpu().numpy()
    return out


def main():
    b = FF.build(str(DATA), VS=VS)
    season, is_f, y = b["season"].astype(np.int16), b["is_f"].astype(bool), b["y"].astype(np.float32)
    old = season <= 2022
    c4 = np.where(old & is_f, 0.0, np.where(old & ~is_f, 1.0,
                 np.where(is_f, 2.0, 3.0))).astype(np.float32)[:, None]
    xin = np.concatenate([b["X44"].astype(np.float32), c4], axis=1)
    xn, xc, cards = G.prep(xin, b["m_tr"], list(b["cat_idx"]) + [44])
    xn, xc = torch.from_numpy(xn), torch.from_numpy(xc)
    tr, va = np.flatnonzero(b["m_tr"]), np.flatnonzero(b["m_va"])
    log(f"VS={VS} train={len(tr):,} val={len(va):,} xn={xn.shape} cards={cards.tolist()} K={K} D={D}")
    ps = []
    for seed in SEEDS:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        m = TabM.make(n_num_features=xn.shape[1], cat_cardinalities=[int(c) for c in cards],
                      d_out=1, num_embeddings=LinearReLUEmbeddings(xn.shape[1], d_embedding=16),
                      arch_type="tabm", k=K, n_blocks=3, d_block=D, dropout=0.1).to(DEVICE)
        train(m, tr, xn, xc, y, seed)
        ps.append(predict(m, xn, xc, va)); del m; gc.collect(); torch.cuda.empty_cache()
    p = np.mean(ps, axis=0); yv = y[va].astype(float); fv = is_f[va]
    out = {"overall": shift_score(p, yv), "regular": shift_score(p[~fv], yv[~fv]),
           "futures": shift_score(p[fv], yv[fv])}
    log(json.dumps(out)); (OUT / "report.json").write_text(json.dumps(out, indent=2))
    np.save(OUT / f"direct_vs{VS}.npy", p)


if __name__ == "__main__":
    main()
