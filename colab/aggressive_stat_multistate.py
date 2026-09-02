"""Aggressive, stat-embedding multi-state experiment.

This is deliberately separate from the incumbent TabM submission.  It removes
raw pitcher/batter/team IDs from the model input and replaces them with
train-only, time-respecting smoothed statistics.  The shared network has a
binary control-success head and a four-state auxiliary head (success,
reverse, middle, other).  The auxiliary labels are recovered only where the
provided as-of cumulative rates make the transition unambiguous.

Run on the GPU host with VS=2022,2023,2024 (or a comma-separated FOLDS value).
The output is validation predictions and a compact progress log, not a
submission package.
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
import torch.nn as nn
import torch.nn.functional as F_t
from rtdl_num_embeddings import LinearReLUEmbeddings
from tabm import TabM

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REMOTE_ROOT = Path("/root")
if str(REMOTE_ROOT) not in sys.path:
    sys.path.insert(0, str(REMOTE_ROOT))

import features44 as F44  # noqa: E402


DATA = Path(os.environ.get("AIMERS_DATA", "/root/open (1)/data"))
OUT = Path(os.environ.get("AIMERS_OUT", "/root/aggressive_stat_out"))
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = tuple(int(x) for x in os.environ.get("FOLDS", "2022,2023,2024").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "42,1,777").split(","))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
BS = int(os.environ.get("BS", "8192"))
MODEL_KIND = os.environ.get("MODEL", "tabm")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TABM_K = int(os.environ.get("TABM_K", "8"))
TABM_BLOCK = int(os.environ.get("TABM_BLOCK", "128"))
STAT_DOUT = int(os.environ.get("STAT_DOUT", "5"))


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "progress.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, np.float64)
    r = float(y.mean())
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (r * (1.0 - r)))


def best_shift(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1.0 - p))
    # A small grid is sufficient for reporting and avoids scipy as a dependency.
    cs = np.linspace(-0.08, 0.08, 321)
    vals = []
    for c in cs:
        q = 1.0 / (1.0 + np.exp(-(z + c)))
        vals.append(score(q, y))
    i = int(np.argmax(vals))
    return float(vals[i]), float(cs[i])


def recover_aux(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recover only unambiguous transition labels from as-of cumulative rates."""
    cols = {
        "reverse": "asof_pitcher_reverse_rate",
        "middle": "asof_pitcher_middle_rate",
    }
    pid = df["pitcher_id"].to_numpy()
    n = df["asof_pitcher_n"].fillna(0.0).to_numpy(np.float64)
    out: dict[str, np.ndarray] = {}
    # The official data is row-id ordered.  Only adjacent n -> n+1 transitions
    # are used, matching the already-verified label reconstruction.
    nxt = (pid[1:] == pid[:-1]) & np.isclose(np.diff(n), 1.0, atol=1e-8)
    src = np.flatnonzero(nxt) + 1
    dst = src - 1
    for name, col in cols.items():
        cum = df[col].fillna(0.0).to_numpy(np.float64) * n
        inc = cum[src] - cum[dst]
        lab = np.rint(inc)
        good = (np.abs(inc - lab) < 0.25) & ((lab == 0.0) | (lab == 1.0))
        v = np.full(len(df), np.nan, np.float32)
        v[dst[good]] = lab[good].astype(np.float32)
        out[name] = v
    return out


def _codes(df: pd.DataFrame):
    p, pvals = pd.factorize(df["pitcher_id"], sort=False)
    b, bvals = pd.factorize(df["batter_id"], sort=False)
    pt, _ = pd.factorize(df["pitcher_team_id"], sort=False)
    bt, _ = pd.factorize(df["batter_team_id"], sort=False)
    ph = df["pitcher_hand"].fillna(-1).to_numpy(np.int32)
    bh = df["batter_hand"].fillna(-1).to_numpy(np.int32)
    # Compact pair keys; no pair embedding is used, only a smoothed statistic.
    pair = p.astype(np.int64) * (len(bvals) + 1) + b.astype(np.int64)
    pph = p.astype(np.int64) * 3 + np.maximum(ph, 0)
    pbh = p.astype(np.int64) * 3 + np.maximum(bh, 0)
    bph = b.astype(np.int64) * 3 + np.maximum(ph, 0)
    bbh = b.astype(np.int64) * 3 + np.maximum(bh, 0)
    return {
        "p": p.astype(np.int64), "b": b.astype(np.int64),
        "pt": pt.astype(np.int64), "bt": bt.astype(np.int64),
        "pair": pair, "pph": pph, "pbh": pbh, "bph": bph, "bbh": bbh,
        "ph": ph, "bh": bh,
    }


def _previous_train(codes: np.ndarray, y: np.ndarray, train_mask: np.ndarray):
    """Chronological previous count/sum for train rows, hist totals for val."""
    c = codes[train_mask]
    yy = y[train_mask].astype(np.float64)
    # cumcount and cumsum are both computed on rows preceding the current row.
    cs = pd.Series(yy).groupby(c, sort=False).cumsum().to_numpy() - yy
    cn = pd.Series(np.ones(len(c), np.float64)).groupby(c, sort=False).cumsum().to_numpy() - 1.0
    # Include categories that occur only in the validation season; those must
    # receive a zero-count prior rather than indexing past the history table.
    maxc = int(codes.max()) + 1 if len(codes) else 1
    hn = np.bincount(c, minlength=maxc).astype(np.float64)
    hs = np.bincount(c, weights=yy, minlength=maxc).astype(np.float64)
    return c, cs, cn, hs, hn


def _smooth_arrays(code_dict, y, train_mask, gmean):
    """Return per-row smoothed rates/count uncertainty for each entity."""
    n = len(y)
    groups = ["p", "b", "pair", "pt", "bt", "pph", "pbh", "bph", "bbh"]
    result = {}
    for name in groups:
        code = code_dict[name]
        _, prev_s, prev_n, hist_s, hist_n = _previous_train(code, y, train_mask)
        s = np.zeros(n, np.float64)
        c = np.zeros(n, np.float64)
        s[train_mask], c[train_mask] = prev_s, prev_n
        vmask = ~train_mask
        # Validation/future rows must see only the completed training history.
        s[vmask] = hist_s[code[vmask]]
        c[vmask] = hist_n[code[vmask]]
        alpha = {"p": 50.0, "b": 50.0, "pair": 80.0,
                 "pt": 120.0, "bt": 120.0, "pph": 140.0, "pbh": 140.0,
                 "bph": 140.0, "bbh": 140.0}[name]
        rate = (s + alpha * gmean) / (c + alpha)
        result[name] = np.column_stack([
            rate, rate - gmean, np.log1p(c), 1.0 / np.sqrt(c + alpha),
        ]).astype(np.float32)
    return result


def build_inputs(df: pd.DataFrame, vs: int):
    """Build base context plus stat groups, with no raw ID features."""
    y = df["control_success"].to_numpy(np.float32)
    season = df["season"].to_numpy(np.int16)
    train_mask = season < vs
    val_mask = season == vs

    built = F44.build(str(DATA), VS=vs, return_frame=False)
    base = built["X44"].astype(np.float32)
    names = list(built["F44"])
    # Remove all raw player/team IDs.  Hand/count/game context remain numeric.
    raw_id = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
    keep = [i for i, name in enumerate(names) if name not in raw_id]
    base = base[:, keep]
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    codes = _codes(df)
    gmean = float(y[train_mask].mean())
    stats = _smooth_arrays(codes, y, train_mask, gmean)
    # Explicit stat group order is stable and is saved with the checkpoint/report.
    stat_names = ["p", "b", "pair", "pt", "bt", "pph", "pbh", "bph", "bbh"]
    stat = np.concatenate([stats[k] for k in stat_names], axis=1)
    # A few legal context interactions expose the structure directly.
    balls = df["balls_before"].to_numpy(np.float32)
    strikes = df["strikes_before"].to_numpy(np.float32)
    cnt = (balls * 3.0 + strikes).astype(np.float32)
    li = df["li"].fillna(0.0).to_numpy(np.float32)
    context = np.column_stack([
        cnt / 10.0, (cnt == 0).astype(np.float32), (cnt >= 6).astype(np.float32),
        li, li * cnt / 10.0,
        (codes["ph"] == 1).astype(np.float32), (codes["bh"] == 1).astype(np.float32),
        ((codes["ph"] == 1) == (codes["bh"] == 1)).astype(np.float32),
    ])
    X = np.concatenate([base, stat, context], axis=1).astype(np.float32)
    # Fit normalization on the historical training side only.
    mu = np.nanmean(X[train_mask], axis=0)
    sd = np.nanstd(X[train_mask], axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-5)] = 1.0
    X = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    aux = recover_aux(df)
    # State 0 success; state 1 reverse; state 2 middle; state 3 other.
    # Rows for which the as-of transition is not known are masked in CE.
    state = np.full(len(df), -1, np.int64)
    state[y > 0.5] = 0
    rev, mid = aux["reverse"], aux["middle"]
    known = np.isfinite(rev) & np.isfinite(mid)
    state[known & (y <= 0.5) & (rev > 0.5)] = 1
    state[known & (y <= 0.5) & (rev <= 0.5) & (mid > 0.5)] = 2
    state[known & (y <= 0.5) & (rev <= 0.5) & (mid <= 0.5)] = 3
    return X, y, state, train_mask, val_mask, {
        "mu": mu.tolist(), "sd": sd.tolist(), "stat_groups": stat_names,
        "n_base": int(base.shape[1]), "n_features": int(X.shape[1]),
    }


class StatMultiStateNet(nn.Module):
    """Shared stat-token network, intentionally without ID embeddings."""

    def __init__(self, n_features: int, n_base: int, n_groups: int = 9, hidden: int = 512):
        super().__init__()
        # base/context before the stat columns; every group is four smoothed
        # statistics (posterior rate, deviation, log-count, uncertainty).
        self.n_base = n_base
        self.n_stat_start = n_base
        self.n_groups = n_groups
        self.base = nn.Sequential(nn.Linear(n_base, 160), nn.SiLU(), nn.LayerNorm(160))
        self.stat_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(4, 48), nn.SiLU(), nn.LayerNorm(48))
            for _ in range(n_groups)
        ])
        self.body = nn.Sequential(
            nn.Linear(160 + n_groups * 48, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Dropout(0.12), nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.LayerNorm(hidden // 2), nn.Dropout(0.08),
        )
        self.success = nn.Linear(hidden // 2, 1)
        self.states = nn.Linear(hidden // 2, 4)

    def forward(self, x):
        h0 = self.base(x[:, :self.n_base])
        parts = [h0]
        off = self.n_stat_start
        for proj in self.stat_proj:
            parts.append(proj(x[:, off:off + 4]))
            off += 4
        h = self.body(torch.cat(parts, dim=1))
        return self.success(h).squeeze(1), self.states(h)


class StatTabMNet(nn.Module):
    """Official TabM backbone over numeric smoothed-stat tokens only.

    d_out=5 gives one direct Bernoulli head plus four state logits.  There are
    no categorical cardinalities, so pitcher/batter IDs never enter the model.
    """

    def __init__(self, n_features: int, k: int = TABM_K):
        super().__init__()
        self.net = TabM.make(
            n_num_features=n_features, cat_cardinalities=None, d_out=STAT_DOUT,
            num_embeddings=LinearReLUEmbeddings(n_features, d_embedding=16),
            arch_type="tabm", k=k, n_blocks=3, d_block=TABM_BLOCK, dropout=0.1,
        )

    def forward(self, x):
        return self.net(x, None)


def train_one(X, y, state, tr_mask, va_mask, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n_base = int(X.shape[1] - 9 * 4 - 8)
    if MODEL_KIND == "tabm":
        model = StatTabMNet(X.shape[1]).to(DEVICE)
    else:
        model = StatMultiStateNet(X.shape[1], n_base=n_base).to(DEVICE)
    ti = np.flatnonzero(tr_mask)
    vi = np.flatnonzero(va_mask)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    st = torch.from_numpy(state)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=4e-4)
    steps = 0
    for ep in range(EPOCHS):
        rng = np.random.default_rng(seed + ep * 1009)
        order = ti[rng.permutation(len(ti))]
        model.train()
        loss_sum = 0.0
        for start in range(0, len(order), BS):
            b = order[start:start + BS]
            xb = Xt[b].to(DEVICE, non_blocking=True)
            yb = yt[b].to(DEVICE, non_blocking=True)
            sb = st[b].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            raw = model(xb)
            if MODEL_KIND == "tabm":
                lg = raw[:, :, 0]
                ls = raw[:, :, 1:] if STAT_DOUT > 1 else None
                # Direct head is averaged over the internal ensemble members.
                direct_p = torch.sigmoid(lg).mean(dim=1)
                bce = F_t.binary_cross_entropy(direct_p, yb)
                brier = (direct_p - yb).square().mean()
                if STAT_DOUT > 1:
                    m = sb >= 0
                    ce = (F_t.cross_entropy(
                        ls[m].reshape(-1, ls.shape[-1]),
                        sb[m, None].expand(-1, ls.shape[1]).reshape(-1),
                    ) if bool(m.any()) else torch.zeros((), device=DEVICE))
                else:
                    ce = torch.zeros((), device=DEVICE)
            else:
                lg, ls = raw
                bce = F_t.binary_cross_entropy_with_logits(lg, yb)
                brier = (torch.sigmoid(lg) - yb).square().mean()
                m = sb >= 0
                ce = (F_t.cross_entropy(ls[m], sb[m]) if bool(m.any())
                      else torch.zeros((), device=DEVICE))
            # Direct binary supervision remains present, but the four-state
            # head is deliberately strong enough to be an independent route.
            loss = 0.45 * bce + 0.35 * brier + 0.80 * ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(b)
            steps += 1
        model.eval()
        with torch.inference_mode():
            pp = []
            ps = []
            for start in range(0, len(vi), 65536):
                b = vi[start:start + 65536]
                raw = model(Xt[b].to(DEVICE, non_blocking=True))
                if MODEL_KIND == "tabm":
                    lg = raw[:, :, 0]
                    pp.append(torch.sigmoid(lg).mean(dim=1).cpu().numpy())
                    if STAT_DOUT > 1:
                        ps.append(torch.softmax(raw[:, :, 1:], 2).mean(dim=1)[:, 0].cpu().numpy())
                    else:
                        ps.append(torch.sigmoid(lg).mean(dim=1).cpu().numpy())
                else:
                    lg, ls = raw
                    pp.append(torch.sigmoid(lg).cpu().numpy())
                    ps.append(torch.softmax(ls, 1)[:, 0].cpu().numpy())
        pdirect = np.concatenate(pp)
        pstate = np.concatenate(ps)
        log(f"    seed={seed} ep={ep+1}/{EPOCHS} loss={loss_sum/len(order):.6f} "
            f"direct={score(pdirect, y[vi]):.2f} state={score(pstate, y[vi]):.2f} "
            f"mix50={score(0.5*pdirect + 0.5*pstate, y[vi]):.2f}")
    # Recompute final probabilities once; do not keep GPU checkpoints yet.
    model.eval()
    with torch.inference_mode():
        pp, ps = [], []
        for start in range(0, len(vi), 65536):
            b = vi[start:start + 65536]
            raw = model(Xt[b].to(DEVICE, non_blocking=True))
            if MODEL_KIND == "tabm":
                lg = raw[:, :, 0]
                pp.append(torch.sigmoid(lg).mean(dim=1).cpu().numpy())
                if STAT_DOUT > 1:
                    ps.append(torch.softmax(raw[:, :, 1:], 2).mean(dim=1)[:, 0].cpu().numpy())
                else:
                    ps.append(torch.sigmoid(lg).mean(dim=1).cpu().numpy())
            else:
                lg, ls = raw
                pp.append(torch.sigmoid(lg).cpu().numpy())
                ps.append(torch.softmax(ls, 1)[:, 0].cpu().numpy())
    return np.concatenate(pp), np.concatenate(ps), vi


def main():
    raw = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    suffix = raw["row_id"].astype(str).str.extract(r"(\d+)$", expand=False).astype(np.int64)
    df = raw.iloc[np.argsort(suffix.to_numpy(), kind="stable")].reset_index(drop=True)
    log(f"device={DEVICE} rows={len(df):,} folds={FOLDS} seeds={SEEDS} epochs={EPOCHS}")
    all_report = {}
    for vs in FOLDS:
        t0 = time.time()
        X, y, state, tr_mask, va_mask, meta = build_inputs(df, vs)
        vi = np.flatnonzero(va_mask)
        known = int(np.sum(state[tr_mask] >= 0))
        log(f"VS={vs} X={X.shape} train={int(tr_mask.sum()):,} val={len(vi):,} "
            f"state_known_train={known:,}/{int(tr_mask.sum()):,}")
        preds = []
        states = []
        for seed in SEEDS:
            pdirect, pstate, vi2 = train_one(X, y, state, tr_mask, va_mask, seed)
            assert np.array_equal(vi, vi2)
            preds.append(pdirect)
            states.append(pstate)
        yv = y[vi]
        isf = df.iloc[vi]["game_type"].astype(str).to_numpy() == "F"
        out = {}
        for name, arrs in (("direct", preds), ("state", states),
                           ("mix25", [0.75*a + 0.25*b for a,b in zip(preds,states)]),
                           ("mix50", [0.50*a + 0.50*b for a,b in zip(preds,states)]),
                           ("mix75", [0.25*a + 0.75*b for a,b in zip(preds,states)])):
            p = np.mean(arrs, axis=0)
            s_all, sh = best_shift(p, yv)
            out[name] = {"overall": s_all, "shift": sh,
                         "regular": best_shift(p[~isf], yv[~isf])[0],
                         "futures": best_shift(p[isf], yv[isf])[0]}
            np.save(OUT / f"statms_vs{vs}_{name}.npy", p)
        all_report[str(vs)] = {"meta": meta, "scores": out, "seconds": time.time()-t0}
        log(json.dumps({"VS": vs, **out}, ensure_ascii=False))
    (OUT / "report.json").write_text(json.dumps(all_report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
