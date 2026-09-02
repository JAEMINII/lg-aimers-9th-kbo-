"""Aggressive multi-state TabM experiment.

Unlike aux_target.py, this uses one mutually-exclusive softmax head for
success/reverse/middle/other and reports that head directly as P(success).
The binary head is retained as a comparator, not as a residual correction.
Training uses the original 44 features and ID embeddings so this isolates the
target-structure hypothesis from the separate stat-only embedding hypothesis.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path("/root")
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("AIMERS_DATA", "/root/open (1)/data"))
OUT = Path(os.environ.get("AIMERS_OUT", "/root/multistate_out"))
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = tuple(int(x) for x in os.environ.get("FOLDS", "2022,2024").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "42,1,777").split(","))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BS = int(os.environ.get("BS", "1024"))
K = int(os.environ.get("K", "32"))
DBLOCK = int(os.environ.get("DBLOCK", "256"))
LR = float(os.environ.get("LR", "0.003"))
STATE_W = float(os.environ.get("STATE_W", "0.90"))
DIRECT_W = float(os.environ.get("DIRECT_W", "0.35"))
Brier_W = float(os.environ.get("BRIER_W", "0.25"))
MIN_SEASON = int(os.environ.get("MIN_SEASON", "-999"))
BRANCH = os.environ.get("BRANCH", "all")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda" and os.environ.get("AMP", "1") != "0"

import features44 as FF  # noqa: E402
import tabm_gate_gpu as G  # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings  # noqa: E402
from tabm import TabM  # noqa: E402
from train_chan_3 import preprocess as PP  # noqa: E402


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "progress.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def score(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r)))


def best_shift(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1.0 - p))
    cs = np.linspace(-0.08, 0.08, 321)
    vals = []
    for c in cs:
        q = 1.0 / (1.0 + np.exp(-(z + c)))
        vals.append(score(q, y))
    i = int(np.argmax(vals))
    return float(vals[i]), float(cs[i])


def recover_state(df):
    pid = df.pitcher_id.to_numpy()
    n = df.asof_pitcher_n.fillna(0.0).to_numpy(float)
    nxt = (pid[1:] == pid[:-1]) & np.isclose(np.diff(n), 1.0, atol=1e-8)
    src = np.flatnonzero(nxt) + 1
    dst = src - 1
    vals = {}
    for name, col in (("reverse", "asof_pitcher_reverse_rate"),
                      ("middle", "asof_pitcher_middle_rate")):
        cum = df[col].fillna(0.0).to_numpy(float) * n
        inc = cum[src] - cum[dst]
        lab = np.rint(inc)
        good = (np.abs(inc - lab) < 0.25) & ((lab == 0) | (lab == 1))
        out = np.full(len(df), np.nan)
        out[dst[good]] = lab[good]
        vals[name] = out
    y = df.control_success.to_numpy(float)
    state = np.full(len(df), -1, np.int64)
    state[y > 0.5] = 0
    known = np.isfinite(vals["reverse"]) & np.isfinite(vals["middle"])
    state[known & (y <= 0.5) & (vals["reverse"] > 0.5)] = 1
    state[known & (y <= 0.5) & (vals["reverse"] <= 0.5)
          & (vals["middle"] > 0.5)] = 2
    state[known & (y <= 0.5) & (vals["reverse"] <= 0.5)
          & (vals["middle"] <= 0.5)] = 3
    return state


def make_model(n_num, cards, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return TabM.make(
        n_num_features=n_num, cat_cardinalities=[int(c) for c in cards],
        d_out=5, num_embeddings=LinearReLUEmbeddings(n_num, d_embedding=16),
        arch_type="tabm", k=K, n_blocks=3, d_block=DBLOCK, dropout=0.1,
    ).to(DEVICE)


def forward_parts(model, xn, xc):
    raw = model(xn, xc)
    # TabM: (B, K, d_out).  Head 0 is direct binary; heads 1: are states.
    direct = raw[:, :, 0]
    states = raw[:, :, 1:]
    return direct, states


def train_model(model, idx, y, state, weights, seed, tag):
    torch.manual_seed(seed)
    ii = torch.from_numpy(idx.astype(np.int64))
    yy = torch.from_numpy(y.astype(np.float32))
    ss = torch.from_numpy(state.astype(np.int64))
    ww = torch.from_numpy(weights.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(idx))
        total = 0.0
        for start in range(0, len(idx), BS):
            pos = perm[start:start + BS]
            b = ii[pos]
            xn = G.XN[b].to(DEVICE, non_blocking=True)
            xc = G.XC[b].to(DEVICE, non_blocking=True)
            yb = yy[b].to(DEVICE, non_blocking=True)
            sb = ss[b].to(DEVICE, non_blocking=True)
            wb = ww[b].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                direct_lg, state_lg = forward_parts(model, xn, xc)
                # Use the mean logit for the autocast-safe training loss;
                # inference still averages member probabilities like TabM.
                direct_mean_lg = direct_lg.mean(dim=1)
                direct_p = torch.sigmoid(direct_mean_lg)
                bce = F.binary_cross_entropy_with_logits(
                    direct_mean_lg, yb, reduction="none"
                )
                brier = (direct_p - yb).square()
                m = sb >= 0
                if bool(m.any()):
                    ce_each = F.cross_entropy(
                        state_lg[m].reshape(-1, state_lg.shape[-1]),
                        sb[m, None].expand(-1, state_lg.shape[1]).reshape(-1),
                        reduction="none",
                    ).reshape(-1, state_lg.shape[1]).mean(dim=1)
                    ce = (ce_each * wb[m]).sum() / wb[m].sum()
                else:
                    ce = torch.zeros((), device=DEVICE)
                loss = ((DIRECT_W * bce + Brier_W * brier) * wb).sum() / wb.sum() + STATE_W * ce
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(b)
        log(f"  {tag} ep{ep+1}/{EPOCHS} loss={total/len(idx):.6f}")
    return model


@torch.no_grad()
def predict(model, idx):
    model.eval()
    ii = torch.from_numpy(idx.astype(np.int64))
    direct, state = [], []
    for start in range(0, len(idx), 8192):
        b = ii[start:start + 8192]
        xn = G.XN[b].to(DEVICE, non_blocking=True)
        xc = G.XC[b].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            dl, sl = forward_parts(model, xn, xc)
            direct.append(torch.sigmoid(dl).mean(dim=1).float().cpu().numpy())
            state.append(torch.softmax(sl, dim=2).mean(dim=1)[:, 0].float().cpu().numpy())
    return np.concatenate(direct), np.concatenate(state)


def main():
    raw = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    raw = PP.sort_by_row_id(raw)
    aux_state = recover_state(raw)
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    is_f = raw.game_type.astype(str).to_numpy() == "F"
    all_report = {}

    for vs in FOLDS:
        t0 = time.time()
        built = FF.build(str(DATA), VS=vs)
        X44 = built["X44"].astype(np.float32)
        Fnames = list(built["F44"])
        # The same four-regime categorical used by submit_41, treated as one
        # additional categorical input.  It is not a target-derived feature.
        old = season <= 2022
        c4 = np.where(old & is_f, 0.0,
                      np.where(old & ~is_f, 1.0,
                               np.where(is_f, 2.0, 3.0))).astype(np.float32)[:, None]
        ci = [Fnames.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        Xin = np.concatenate([X44, c4], axis=1)
        train_mask = (season < vs) & (season >= MIN_SEASON)
        if BRANCH == "regular":
            train_mask &= ~is_f
        elif BRANCH == "futures":
            train_mask &= is_f
        Xn, Xc, cards = G.prep(Xin, train_mask, ci + [X44.shape[1]])
        G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
        tr = train_mask
        va = season == vs
        tr_idx, va_idx = np.flatnonzero(tr), np.flatnonzero(va)
        # Preserve the established old-futures downweighting in the original
        # branch model while keeping the state hypothesis the only change.
        # Keep weights aligned to the global row indices because train_model
        # shuffles/indexes with tr_idx directly.  (A compact tr-only array
        # would be mis-indexed when MIN_SEASON excludes early rows.)
        w = np.ones(len(y), dtype=np.float32)
        w[tr_idx] = np.where(is_f[tr_idx] & old[tr_idx], 0.1, 1.0).astype(np.float32)
        log(f"VS={vs} X={Xin.shape} train={len(tr_idx):,} val={len(va_idx):,} "
            f"states={int(np.sum(aux_state[tr_idx]>=0)):,}/{len(tr_idx):,} "
            f"K={K} d={DBLOCK} amp={AMP}")
        pdirect, pstate = [], []
        for seed in SEEDS:
            m = make_model(Xn.shape[1], cards, seed)
            m = train_model(m, tr_idx, y, aux_state, w, seed, f"VS{vs} s{seed}")
            a, b = predict(m, va_idx)
            pdirect.append(a)
            pstate.append(b)
            del m
            torch.cuda.empty_cache()
        yv = y[va_idx].astype(float)
        fv = is_f[va_idx]
        report = {}
        for name, arrs in (("direct", pdirect), ("state", pstate),
                           ("mix25", [0.75*a + 0.25*b for a,b in zip(pdirect,pstate)]),
                           ("mix50", [0.50*a + 0.50*b for a,b in zip(pdirect,pstate)]),
                           ("mix75", [0.25*a + 0.75*b for a,b in zip(pdirect,pstate)])):
            p = np.mean(arrs, axis=0)
            a, sh = best_shift(p, yv)
            report[name] = {"overall": a, "shift": sh,
                            "regular": best_shift(p[~fv], yv[~fv])[0],
                            "futures": best_shift(p[fv], yv[fv])[0]}
            np.save(OUT / f"ms_vs{vs}_{name}.npy", p)
        all_report[str(vs)] = {"scores": report, "seconds": time.time() - t0}
        log(json.dumps({"VS": vs, **report}, ensure_ascii=False))
    (OUT / "report.json").write_text(json.dumps(all_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
