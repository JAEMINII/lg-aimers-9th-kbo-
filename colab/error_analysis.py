"""Temporal error audit for the 2022/2023/2024 gate predictions.

The aim is not to choose another model in-sample.  It finds residuals that
are large *and stable across held-out seasons*, separating a missing feature
or calibration issue from one-year noise.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root")
from train_chan_3 import preprocess as PP

DATA = Path(os.environ.get("AIMERS_DATA", "/root/data"))
OUT = Path(os.environ.get("AIMERS_OUT", "/root/error_audit"))
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    2022: {
        "ms": "/root/ms_k32d512_3090/ms_vs2022_direct.npy",
        "cb": "/root/aimers/cb_gate2022.npy",
        "tabm": "/root/arch3090_2022/ar_tabm_s{seed}.npy",
    },
    2023: {
        "ms": "/root/ms_2023_fix/ms_vs2023_direct.npy",
        "cb": "/root/aimers/cb_gate2023.npy",
        "tabm": "/root/arch3090_2023/ar_tabm_s{seed}.npy",
    },
    2024: {
        "ms": "/root/ms_k32d512_3090/ms_vs2024_direct.npy",
        "cb": "/root/aimers/cb_gate.npy",
        "tabm": "/root/arch3090_2024/ar_tabm_s{seed}.npy",
    },
}


def score(p, y):
    r = float(np.mean(y))
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r)))


def calibrate(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    grid = np.linspace(-.12, .12, 961)
    vals = np.array([score(1 / (1 + np.exp(-(z + shift))), y) for shift in grid])
    shift = float(grid[vals.argmax()])
    return 1 / (1 + np.exp(-(z + shift))), shift, float(vals.max())


def calibrate_temperature(p, y):
    """Find fold-local temperature/intercept; diagnostic only, not a proposal."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    best = (-np.inf, None, None)
    for temp in np.linspace(.70, 1.50, 81):
        b = 0.0
        for _ in range(10):
            q = 1 / (1 + np.exp(-np.clip(temp*z + b, -30, 30)))
            dq = q * (1 - q)
            g = 2 * np.mean((q-y) * dq)
            h = 2 * np.mean(dq*dq + (q-y)*dq*(1-2*q))
            if abs(h) < 1e-10:
                break
            b -= np.clip(g/h, -.1, .1)
        q = 1 / (1 + np.exp(-np.clip(temp*z + b, -30, 30)))
        s = score(q, y)
        if s > best[0]:
            best = (s, float(temp), float(b))
    return best


def add_axes(x):
    out = pd.DataFrame(index=x.index)
    out["game_type"] = x.game_type.astype(str)
    out["inning"] = pd.cut(x.inning, [0,1,2,3,4,5,6,7,8,9,99],
                            labels=["1","2","3","4","5","6","7","8","9","10+"]).astype(str)
    out["count"] = x.balls_before.astype(str) + "-" + x.strikes_before.astype(str)
    out["base"] = x.base_state.astype(str)
    out["hand"] = "P" + x.pitcher_hand.astype(str) + "-B" + x.batter_hand.astype(str)
    out["p_history"] = pd.cut(x.asof_pitcher_n, [-1,5,20,50,120,300,1e9],
                               labels=["0-5","6-20","21-50","51-120","121-300","300+"]).astype(str)
    out["b_history"] = pd.cut(x.asof_batter_n, [-1,5,20,50,120,300,1e9],
                               labels=["0-5","6-20","21-50","51-120","121-300","300+"]).astype(str)
    out["leverage"] = pd.cut(x.li.fillna(1), [-1,0.7,0.9,1.1,1.4,2,1e9],
                              labels=["<.7",".7-.9",".9-1.1","1.1-1.4","1.4-2","2+"]).astype(str)
    out["score_margin"] = pd.cut(np.abs(x.score_diff_pitcher_team), [-1,0,1,2,4,99],
                                  labels=["0","1","2","3-4","5+"]).astype(str)
    out["month"] = x.game_month.astype(str)
    return out


def group_rows(ax, y, p, year):
    rows = []
    for col in ax:
        tmp = pd.DataFrame({"group": ax[col].astype(str), "y": y, "p": p})
        for name, g in tmp.groupby("group", observed=True):
            n = len(g)
            if n < 750:
                continue
            rows.append({"axis": col, "group": str(name), "year": year, "n": n,
                         "actual": float(g.y.mean()), "pred": float(g.p.mean()),
                         "resid_pp": float(100 * (g.y.mean() - g.p.mean())),
                         "brier": float(np.mean((g.y-g.p)**2))})
    return rows


def main():
    raw = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
    all_rows, overall, reliability = [], [], []
    cached = {}
    for year, src in SOURCES.items():
        v = raw.loc[raw.season.eq(year)].reset_index(drop=True)
        y = v.control_success.to_numpy(float)
        ms = np.load(src["ms"])
        pms, sms, scms = calibrate(ms, y)
        ms_ts_score, ms_temp, ms_ts_shift = calibrate_temperature(ms, y)
        # Only MultiState is archived under a verified identical contract for
        # all three years.  Use it for temporal stability; historical CB
        # caches are from different old pipelines and cannot be combined.
        overall_row = {"year": year, "n": len(v), "rate": float(y.mean()),
                       "ms_score": scms, "ms_shift": sms,
                       "ms_brier": float(np.mean((pms-y)**2)), "ms_pred_sd": float(np.std(pms)),
                       "ms_temp_score": ms_ts_score, "ms_temp": ms_temp, "ms_temp_shift": ms_ts_shift}
        all_rows += group_rows(add_axes(v), y, pms, year)
        cached[year] = (add_axes(v), y, pms)
        if year == 2024:
            cb = np.load(src["cb"])
            tabm = np.mean([np.load(src["tabm"].format(seed=s)) for s in (42,1,777)], axis=0)
            if not (len(v) == len(ms) == len(cb) == len(tabm)):
                raise RuntimeError(f"length mismatch VS{year}: {len(v)}, {len(ms)}, {len(cb)}, {len(tabm)}")
            raw_blend = .70 * ms + .27 * cb + .03 * tabm
            p, shift, sc = calibrate(raw_blend, y)
            blend_ts_score, blend_temp, blend_ts_shift = calibrate_temperature(raw_blend, y)
            overall_row.update(blend_score=sc, blend_shift=shift,
                               blend_brier=float(np.mean((p-y)**2)), blend_pred_sd=float(np.std(p)),
                               blend_temp_score=blend_ts_score, blend_temp=blend_temp,
                               blend_temp_shift=blend_ts_shift)
            q = pd.qcut(pd.Series(p), 10, duplicates="drop")
            d = pd.DataFrame({"bin": q.astype(str), "y": y, "p": p})
            for name, g in d.groupby("bin", observed=True):
                reliability.append({"bin": str(name), "n": len(g), "actual": float(g.y.mean()),
                                    "pred": float(g.p.mean()), "resid_pp": float(100*(g.y.mean()-g.p.mean())),
                                    "brier": float(np.mean((g.y-g.p)**2))})
        overall.append(overall_row)
    gr = pd.DataFrame(all_rows)
    gr.to_csv(OUT / "all_group_residuals.csv", index=False)
    pivot = gr.pivot_table(index=["axis","group"], columns="year", values="resid_pp", aggfunc="first")
    counts = gr.pivot_table(index=["axis","group"], columns="year", values="n", aggfunc="first")
    stable = pivot.dropna().copy()
    stable["mean_abs_pp"] = stable.abs().mean(axis=1)
    stable["same_sign"] = ((stable.min(axis=1) > 0) | (stable.max(axis=1) < 0))
    stable["min_n"] = counts.reindex(stable.index).min(axis=1)
    stable = stable[(stable.min_n >= 1500) & stable.same_sign].sort_values("mean_abs_pp", ascending=False)
    # Largest 2024 errors are useful too, but label them as single-fold only.
    one = gr[gr.year.eq(2024) & (gr.n >= 3000)].copy()
    one["abs_pp"] = one.resid_pp.abs()
    one = one.sort_values("abs_pp", ascending=False).head(35)
    pd.DataFrame(overall).to_csv(OUT / "overall.csv", index=False)
    pd.DataFrame(reliability).to_csv(OUT / "reliability_2024.csv", index=False)
    stable.reset_index().to_csv(OUT / "stable_residuals.csv", index=False)
    one.to_csv(OUT / "largest_2024_residuals.csv", index=False)
    # Transfer test: fit one categorical residual correction on 2022/2023 and
    # apply it to 2024.  This is deliberately a weak one-axis-at-a-time test;
    # a gain here is evidence of a deployable missing interaction, not an
    # in-sample calibration trick.
    ax24, y24, p24 = cached[2024]
    transfer = []
    for axis in ax24.columns:
        parts = []
        for year in (2022, 2023):
            aa, yy, pp = cached[year]
            d = pd.DataFrame({"g": aa[axis].astype(str), "r": yy-pp})
            parts.append(d)
        for label, fit in (("2022", parts[0]), ("2023", parts[1]),
                           ("2022+2023", pd.concat(parts, ignore_index=True))):
            stats = fit.groupby("g", observed=True).r.agg(["mean", "count"])
            # Do not act on tiny group estimates; shrink weak groups to zero.
            corr = (stats["mean"] * stats["count"] / (stats["count"] + 5000.0)).to_dict()
            adj = ax24[axis].astype(str).map(corr).fillna(0.0).to_numpy(float)
            q = np.clip(p24 + adj, 1e-6, 1-1e-6)
            transfer.append({"axis": axis, "fit_years": label,
                             "gain": float(score(q, y24)-score(p24, y24)),
                             "rmse_adjust_pp": float(100*np.sqrt(np.mean(adj*adj)))} )
    trans = pd.DataFrame(transfer).sort_values("gain", ascending=False)
    trans.to_csv(OUT / "residual_transfer.csv", index=False)
    print("OVERALL")
    print(pd.DataFrame(overall).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nRELIABILITY_2024")
    print(pd.DataFrame(reliability).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSTABLE_RESIDUALS")
    print(stable.reset_index().head(30).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nLARGEST_2024")
    print(one.head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nRESIDUAL_TRANSFER_TO_2024")
    print(trans.head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(json.dumps({"out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
