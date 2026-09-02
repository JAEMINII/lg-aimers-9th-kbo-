"""Create leakage-free pitcher label-family residuals.

The as-of rate increments recover the previous pitch's label. Every aggregate
below is cumulative by pitcher/season (and global hand/season), so a row only
sees seasons strictly before its own season. This is an audit artifact, not a
change to any submission package.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "open (1)", "data")
OUT = os.path.join(ROOT, "colab", "_dl", "label_family_strict.csv.gz")
ALPHA = 300.0

use = ["row_id", "pitcher_id", "pitcher_hand", "batter_hand", "season",
       "base_state", "asof_pitcher_n", "control_success",
       "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
       "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
       "asof_pitcher_strike_rate"]
r = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig", usecols=use)
n = r["asof_pitcher_n"].to_numpy(np.float64)
pid = r["pitcher_id"].to_numpy()
src_ok = np.r_[False, (pid[1:] == pid[:-1]) & (np.diff(n) == 1)]
src = np.where(src_ok)[0]
dst = src - 1
rates = {
    "success": "asof_pitcher_success_rate", "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate", "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}
labels = {}
for name, col in rates.items():
    cum = r[col].to_numpy(np.float64) * n
    inc = cum[src] - cum[dst]
    lab = np.round(inc)
    good = (np.abs(inc - lab) < .25) & ((lab == 0) | (lab == 1))
    v = np.full(len(r), np.nan, dtype=np.float32)
    v[dst[good]] = lab[good]
    labels[name] = v
ok = np.isfinite(labels["success"])
assert np.mean(labels["success"][ok] == r.loc[ok, "control_success"].to_numpy()) > .9999
print("recovered", int(ok.sum()), "of", len(r), flush=True)

def prior_table(value, axis):
    z = pd.DataFrame({"pid": r.pitcher_id.to_numpy(), "season": r.season.to_numpy(),
                      "axis": axis, "v": np.where(np.isfinite(value), value, 0.),
                      "n": np.isfinite(value).astype(np.float64)})
    pa = z.groupby(["pid", "season", "axis"], sort=False)[["v", "n"]].sum().reset_index()
    pa = pa.sort_values(["pid", "axis", "season"])
    pa["pv"] = pa.groupby(["pid", "axis"], sort=False)["v"].cumsum() - pa["v"]
    pa["pn"] = pa.groupby(["pid", "axis"], sort=False)["n"].cumsum() - pa["n"]
    pt = z.groupby(["pid", "season"], sort=False)[["v", "n"]].sum().reset_index()
    pt = pt.sort_values(["pid", "season"])
    pt["tv"] = pt.groupby("pid", sort=False)["v"].cumsum() - pt["v"]
    pt["tn"] = pt.groupby("pid", sort=False)["n"].cumsum() - pt["n"]
    return pa[["pid", "season", "axis", "pv", "pn"]], pt[["pid", "season", "tv", "tn"]]

def global_prior(value, axis):
    z = pd.DataFrame({"hand": r.pitcher_hand.to_numpy(), "season": r.season.to_numpy(),
                      "axis": axis, "v": np.where(np.isfinite(value), value, 0.),
                      "n": np.isfinite(value).astype(np.float64)})
    ga = z.groupby(["hand", "season", "axis"], sort=False)[["v", "n"]].sum().reset_index()
    ga = ga.sort_values(["hand", "axis", "season"])
    ga["gv"] = ga.groupby(["hand", "axis"], sort=False)["v"].cumsum() - ga["v"]
    ga["gn"] = ga.groupby(["hand", "axis"], sort=False)["n"].cumsum() - ga["n"]
    gt = z.groupby(["hand", "season"], sort=False)[["v", "n"]].sum().reset_index()
    gt = gt.sort_values(["hand", "season"])
    gt["gtv"] = gt.groupby("hand", sort=False)["v"].cumsum() - gt["v"]
    gt["gtn"] = gt.groupby("hand", sort=False)["n"].cumsum() - gt["n"]
    return ga[["hand", "season", "axis", "gv", "gn"]], gt[["hand", "season", "gtv", "gtn"]]

out = {"row_id": r.row_id.to_numpy()}
axes = {"bh": r.batter_hand.to_numpy(), "base": r.base_state.astype(str).to_numpy()}
for lname, val in labels.items():
    for aname, axis in axes.items():
        pa, pt = prior_table(val, axis)
        ga, gt = global_prior(val, axis)
        key = pd.DataFrame({"pid": r.pitcher_id.to_numpy(), "season": r.season.to_numpy(),
                            "axis": axis, "hand": r.pitcher_hand.to_numpy()})
        q = key.merge(pa, on=["pid", "season", "axis"], how="left")
        q = q.merge(pt, on=["pid", "season"], how="left")
        q = q.merge(ga, on=["hand", "season", "axis"], how="left")
        q = q.merge(gt, on=["hand", "season"], how="left")
        g_axis = q.gv / q.gn.replace(0, np.nan)
        g_total = q.gtv / q.gtn.replace(0, np.nan)
        p_axis = (q.pv + ALPHA * g_axis.fillna(.5)) / (q.pn + ALPHA)
        p_total = (q.tv + ALPHA * g_total.fillna(.5)) / (q.tn + ALPHA)
        out[f"{lname}_{aname}"] = (p_axis - p_total).fillna(0.).astype(np.float32).to_numpy()
        print(lname, aname, "std", float(np.std(out[f"{lname}_{aname}"])), flush=True)
pd.DataFrame(out).to_csv(OUT, index=False)
print("wrote", OUT, len(out["row_id"]), list(out)[1:], flush=True)

