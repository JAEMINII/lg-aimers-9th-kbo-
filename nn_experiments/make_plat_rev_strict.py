"""Build Kim-derived reverse/middle platoon features without fold leakage.

For every row, pitcher/context counts are from seasons before that row's
season.  The global pitcher-hand priors are also cumulative and stop before
the row's season (unlike the original exploratory plat_rev.py, which uses all
seasons and is therefore not a valid 2022/2024 walk-forward feature).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("AIMERS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "open (1)", "data")
OUT = os.environ.get("PLAT_OUT", os.path.join(ROOT, "colab", "_dl", "plat_rev_strict.csv.gz"))
ALPHA = 300.0
RATES = {"reverse": "asof_pitcher_reverse_rate", "middle": "asof_pitcher_middle_rate"}
USE = ["row_id", "pitcher_id", "batter_hand", "pitcher_hand", "season",
       "asof_pitcher_n", "control_success"] + list(RATES.values())


def recover(raw: pd.DataFrame, col: str) -> np.ndarray:
    n = raw["asof_pitcher_n"].to_numpy(np.float64)
    pid = raw["pitcher_id"].to_numpy()
    ok = np.r_[False, (pid[1:] == pid[:-1]) & (np.diff(n) == 1)]
    src, dst = np.where(ok)[0], np.where(ok)[0] - 1
    cum = raw[col].to_numpy(np.float64) * n
    inc = cum[src] - cum[dst]
    lab = np.round(inc)
    good = (np.abs(inc - lab) < 0.25) & ((lab == 0) | (lab == 1))
    out = np.full(len(raw), np.nan)
    out[dst[good]] = lab[good]
    return out


def one(raw: pd.DataFrame, lab: np.ndarray, name: str) -> dict[str, np.ndarray]:
    pid = raw["pitcher_id"].to_numpy()
    season = raw["season"].to_numpy(np.int16)
    bh = raw["batter_hand"].to_numpy(np.int16)
    ph = raw["pitcher_hand"].to_numpy(np.int16)
    have = np.isfinite(lab)
    df = pd.DataFrame({"pid": pid, "season": season, "bh": bh, "ph": ph,
                       "v": np.where(have, lab, 0.0),
                       "n": have.astype(np.float64)})

    # Per-pitcher, per-season, per-batter-hand totals; shift to prior seasons.
    g = df.groupby(["pid", "season", "bh"], sort=True)[["v", "n"]].sum()
    g = g.unstack("bh", fill_value=0.0).sort_index()
    for h in (1, 2):
        for x in ("v", "n"):
            if (x, h) not in g.columns:
                g[(x, h)] = 0.0
    g = g.sort_index(axis=1)
    prior = g.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    idx = pd.MultiIndex.from_arrays([pid, season])
    sub = prior.reindex(idx)
    vh = np.where(bh == 1, sub[("v", 1)].to_numpy(), sub[("v", 2)].to_numpy())
    nh = np.where(bh == 1, sub[("n", 1)].to_numpy(), sub[("n", 2)].to_numpy())
    vt = sub[("v", 1)].to_numpy() + sub[("v", 2)].to_numpy()
    nt = sub[("n", 1)].to_numpy() + sub[("n", 2)].to_numpy()

    # Global pitcher-hand x batter-hand priors, cumulative and shifted by season.
    gg = df.groupby(["season", "ph", "bh"], sort=True)[["v", "n"]].sum().sort_index()
    gp = gg.groupby(level=[1, 2]).cumsum().groupby(level=[1, 2]).shift(1).fillna(0.0)
    gt = df.groupby(["season", "ph"], sort=True)[["v", "n"]].sum().sort_index()
    gpt = gt.groupby(level=1).cumsum().groupby(level=1).shift(1).fillna(0.0)
    gh_idx = pd.MultiIndex.from_arrays([season, ph, bh])
    ga_idx = pd.MultiIndex.from_arrays([season, ph])
    gp2 = gp.reindex(gh_idx)
    gt2 = gpt.reindex(ga_idx)
    pr_h = gp2["v"].to_numpy() / np.maximum(gp2["n"].to_numpy(), 1.0)
    pr_a = gt2["v"].to_numpy() / np.maximum(gt2["n"].to_numpy(), 1.0)
    # No prior global observations: use neutral empirical defaults.
    pr_h = np.where(gp2["n"].to_numpy() > 0, pr_h, float(np.nanmean(lab)))
    pr_a = np.where(gt2["n"].to_numpy() > 0, pr_a, float(np.nanmean(lab)))
    p_h = (vh + ALPHA * pr_h) / (nh + ALPHA)
    p_a = (vt + ALPHA * pr_a) / (nt + ALPHA)
    return {f"{name}_plat": np.nan_to_num(p_h - p_a),
            f"{name}_rate_asof": np.nan_to_num(p_a)}


def main() -> None:
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=USE)
    out = {"row_id": raw["row_id"].to_numpy()}
    for name, col in RATES.items():
        out.update(one(raw, recover(raw, col), name))
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(OUT, pd.DataFrame(out).shape)
    print(pd.DataFrame(out).describe().to_string())


if __name__ == "__main__":
    main()
