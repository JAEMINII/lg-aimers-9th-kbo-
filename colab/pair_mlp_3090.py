"""Pair-interaction MLP probe for the 24GB GPU.

This is a validation-only experiment.  It keeps the legal F44 features and the
four-regime category, but adds explicit pitcher/batter and team interaction
vectors before a regular MLP.  Three branch heads are trained separately, so
the output can be compared with the incumbent all/regular/futures route.
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
import torch.nn as nn
import torch.nn.functional as TF

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("AIMERS_DATA", "/root/data"))
OUT = Path(os.environ.get("AIMERS_OUT", "/root/pair_mlp_out"))
OUT.mkdir(parents=True, exist_ok=True)
VS = int(os.environ.get("VS", "2024"))
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "42,1,777").split(","))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
K = int(os.environ.get("K", "16"))
D = int(os.environ.get("D", "512"))
DE = int(os.environ.get("DE", "32"))
BS = int(os.environ.get("BS", "8192"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda" and os.environ.get("AMP", "1") != "0"
AUG = os.environ.get("AUG", "0") == "1"

import features44 as F44  # noqa: E402


def log(s: str):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    with (OUT / "progress.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def score(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    r = float(y.mean())
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (r * (1.0 - r)))


def best_shift(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1.0 - p))
    grid = np.linspace(-0.08, 0.08, 321)
    vals = [score(1.0 / (1.0 + np.exp(-(z + c))), y) for c in grid]
    i = int(np.argmax(vals))
    return float(vals[i]), float(grid[i])


class PairMLP(nn.Module):
    def __init__(self, n_num, cards, k=K, d=D, de=DE):
        super().__init__()
        self.k = k
        self.num_emb = nn.Sequential(
            nn.Linear(n_num, n_num * 2), nn.GELU(),
            nn.Linear(n_num * 2, de),
        )
        self.cat = nn.ModuleList([nn.Embedding(int(c), de) for c in cards])
        # Explicit interactions for pitcher/batter and their team identities.
        d_in = de * (1 + len(cards) + 4)
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(d, d // 2), nn.GELU(),
        )
        self.head = nn.Linear(d // 2, k)

    def forward(self, xn, xc):
        ne = self.num_emb(xn)
        ce = [emb(xc[:, j]) for j, emb in enumerate(self.cat)]
        # F44 category order: top_bottom, game_type, base_state,
        # pitcher_id, batter_id, pitcher_hand, batter_hand,
        # pitcher_team_id, batter_team_id, regime.
        p, b = ce[3], ce[4]
        pt, bt = ce[7], ce[8]
        feats = ce + [p * b, (p - b).abs(), pt * bt, (pt - bt).abs()]
        h = self.trunk(torch.cat([ne] + feats, dim=1))
        return self.head(h)  # (B, K)


def prep(X, m_tr, cat_idx):
    n, p = X.shape
    ci = np.asarray(cat_idx, dtype=int)
    ni = np.asarray([j for j in range(p) if j not in set(ci)])
    xc = np.zeros((n, len(ci)), dtype=np.int64)
    cards = []
    for a, j in enumerate(ci):
        vals = np.unique(X[m_tr, j])
        vals = vals[~np.isnan(vals)]
        pos = np.clip(np.searchsorted(vals, X[:, j]), 0, max(len(vals) - 1, 0))
        hit = (vals[pos] == X[:, j]) if len(vals) else np.zeros(n, bool)
        xc[:, a] = np.where(hit, pos + 1, 0)
        cards.append(len(vals) + 1)
    xn = X[:, ni].astype(np.float64)
    miss = np.isnan(xn)
    med = np.nanmedian(xn[m_tr], axis=0)
    xn = np.where(miss, med, xn)
    mu = xn[m_tr].mean(0)
    sd = xn[m_tr].std(0) + 1e-6
    xn = ((xn - mu) / sd).astype(np.float32)
    hm = miss[m_tr].any(0)
    if hm.any():
        xn = np.concatenate([xn, miss[:, hm].astype(np.float32)], axis=1)
    return xn, xc, np.asarray(cards, dtype=np.int64)


def train(model, xn, xc, y, idx, epochs, lr, seed, tag, freeze=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=3e-4)
    steps = max(1, epochs * ((len(idx) + BS - 1) // BS))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ii = torch.from_numpy(idx.astype(np.int64))
    yy = torch.from_numpy(y.astype(np.float32))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        total = 0.0
        for st in range(0, len(idx), BS):
            pos = perm[st:st + BS]
            b = ii[pos]
            xnb = xn[b].to(DEVICE, non_blocking=True)
            xcb = xc[b].to(DEVICE, non_blocking=True)
            yb = yy[b].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                lg = model(xnb, xcb)
                ym = yb[:, None].expand_as(lg)
                loss = 0.5 * TF.binary_cross_entropy_with_logits(lg, ym)
                loss = loss + 0.5 * (lg.sigmoid() - ym).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sch.step()
            total += float(loss.detach()) * len(b)
        log(f"{tag} ep{ep + 1}/{epochs} loss={total / len(idx):.6f}")


@torch.no_grad()
def predict(model, xn, xc, idx):
    model.eval()
    out = np.empty(len(idx), np.float64)
    ii = torch.from_numpy(idx.astype(np.int64))
    for st in range(0, len(idx), BS):
        b = ii[st:st + BS]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            p = model(xn[b].to(DEVICE), xc[b].to(DEVICE)).sigmoid().mean(1)
        out[st:st + len(b)] = p.float().cpu().numpy()
    return out


def main():
    built = F44.build(str(DATA), VS=VS, return_frame=AUG)
    raw = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    season = built["season"].astype(np.int16)
    is_f = built["is_f"].astype(bool)
    y = built["y"].astype(np.float32)
    old = season <= 2022
    c4 = np.where(old & is_f, 0.0,
                  np.where(old & ~is_f, 1.0,
                           np.where(is_f, 2.0, 3.0))).astype(np.float32)[:, None]
    xin = np.concatenate([built["X44"].astype(np.float32), c4], axis=1)
    if AUG:
        fr = built["frame"]
        def col(name):
            return fr[name].fillna(0.0).to_numpy(np.float32)
        def logit(v):
            v = np.clip(v.astype(np.float32), 1e-4, 1.0 - 1e-4)
            return np.log(v / (1.0 - v)).astype(np.float32)
        ps, bs = col("p_is_succ"), col("b_is_succ")
        p_rate, b_rate = col("asof_pitcher_success_rate"), col("asof_batter_success_rate")
        pn, bn = col("asof_pitcher_n"), col("asof_batter_n")
        extra = np.column_stack([
            logit(ps), logit(bs), logit(p_rate), logit(b_rate),
            np.log1p(np.maximum(pn, 0.0)), np.log1p(np.maximum(bn, 0.0)),
            ps * bs, ps - bs, np.abs(ps - bs),
            ps * (1.0 - bs), (1.0 - ps) * bs,
        ]).astype(np.float32)
        xin = np.concatenate([xin, extra], axis=1)
    mtr = built["m_tr"]
    xn, xc, cards = prep(xin, mtr, list(built["cat_idx"]) + [44])
    xn = torch.from_numpy(xn)
    xc = torch.from_numpy(xc)
    tr_idx = np.flatnonzero(mtr)
    va_idx = np.flatnonzero(built["m_va"])
    log(f"VS={VS} X={xin.shape} Xn={xn.shape} Xc={xc.shape} cards={cards.tolist()} "
        f"train={len(tr_idx):,} val={len(va_idx):,} K={K} D={D} DE={DE} amp={AMP}")
    preds = {"all": [], "regular": [], "futures": []}
    masks = {"all": np.ones(len(tr_idx), bool),
             "regular": ~is_f[tr_idx], "futures": is_f[tr_idx]}
    for seed in SEEDS:
        for name in ("all", "regular", "futures"):
            idx = tr_idx[masks[name]]
            model = PairMLP(xn.shape[1], cards).to(DEVICE)
            train(model, xn, xc, y, idx, EPOCHS, 3e-3, seed,
                  f"s{seed} {name} S1")
            recent = idx[season[idx] == VS - 1]
            if len(recent):
                for p in model.parameters():
                    p.requires_grad_(False)
                for p in list(model.trunk[-2:].parameters()) + list(model.head.parameters()):
                    p.requires_grad_(True)
                train(model, xn, xc, y, recent, 1, 3e-4, seed,
                      f"s{seed} {name} S2")
            preds[name].append(predict(model, xn, xc, va_idx))
            del model
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    yv = y[va_idx].astype(float)
    fv = is_f[va_idx]
    out = {}
    for s in range(len(SEEDS)):
        p = 0.6 * preds["all"][s] + np.where(
            fv, 0.4 * preds["futures"][s], 0.4 * preds["regular"][s])
        out[str(SEEDS[s])] = {
            "route": {"overall": best_shift(p, yv)[0],
                      "regular": best_shift(p[~fv], yv[~fv])[0],
                      "futures": best_shift(p[fv], yv[fv])[0]},
            "all": best_shift(preds["all"][s], yv)[0],
        }
        np.save(OUT / f"pair_vs{VS}_s{SEEDS[s]}.npy", p)
    pmean = np.mean([0.6 * preds["all"][s] + np.where(
        fv, 0.4 * preds["futures"][s], 0.4 * preds["regular"][s])
        for s in range(len(SEEDS))], axis=0)
    out["mean"] = {"route": {"overall": best_shift(pmean, yv)[0],
                               "regular": best_shift(pmean[~fv], yv[~fv])[0],
                               "futures": best_shift(pmean[fv], yv[fv])[0]}}
    (OUT / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
