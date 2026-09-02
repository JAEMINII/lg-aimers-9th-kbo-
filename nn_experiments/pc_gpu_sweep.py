# -*- coding: utf-8 -*-
"""GPU walk-forward sweep for rule-compliant pitcher interaction features.

This is a validation-only script.  Every target table is built from seasons
strictly before VS, and CatBoost is fit only on seasons < VS.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar

ROOT = os.environ.get("AIMERS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "open (1)", "data")
VS = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
SAVE_DIR = os.environ.get("SAVE_DIR", "")
MORE = os.environ.get("MORE", "0") == "1"
ONLY = {x for x in os.environ.get("ONLY", "").split(",") if x}
SEEDS = tuple(int(x) for x in os.environ.get("CB_SEEDS", "1,42,777").split(",") if x)
ALPHA = float(os.environ.get("PC_ALPHA", "200"))
TGT = "control_success"
TM_PATH = os.environ.get("TM_PATH", "")
LABEL_PATH = os.environ.get("LABEL_PATH", "")
CB_DEPTH = int(os.environ.get("CB_DEPTH", "4"))
CB_ITERS = int(os.environ.get("CB_ITERS", "400"))
CB_LR = float(os.environ.get("CB_LR", "0.05"))
CB_L2 = float(os.environ.get("CB_L2", "1.0"))
TM2S_SET = os.environ.get("TM2S_SET", "")


def score(p, y):
    r = y.mean()
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def cal(p, c):
    q = np.clip(p, 1e-6, 1.0 - 1e-6)
    z = np.log(q / (1.0 - q)) + c
    return 1.0 / (1.0 + np.exp(-z))


def best(p, y):
    r = minimize_scalar(lambda c: -score(cal(p, c), y), bounds=(-0.3, 0.3), method="bounded")
    return -r.fun, r.x


def trackman_matrix(path, tr_seasons, tr_pitchers, cols=None):
    """Prior-season, pitch-count-weighted TrackMan pitcher aggregates.

    For a row in season Y, only TrackMan seasons < Y contribute.  This is
    intentionally the same construction used by the historical TrackMan
    gate, but kept independent of the TabM code so CatBoost can be audited.
    """
    t = pd.read_csv(path)
    feat = [c for c in t.columns if c.startswith("tm_") or c.startswith("tm2s")]
    if cols is not None:
        feat = [c for c in feat if c in cols]
    t = t.sort_values(["pitcher_id", "season"])
    seasons = sorted(t.season.unique())
    w = t["tm_n"].to_numpy(dtype=np.float64)
    rows = {}
    for pid, g in t.groupby("pitcher_id", sort=False):
        cum_num = {c: 0.0 for c in feat}
        cum_den = {c: 0.0 for c in feat}
        for s, gs in g.groupby("season", sort=True):
            rows[(int(pid), int(s))] = {
                c: (cum_num[c] / cum_den[c] if cum_den[c] > 0 else np.nan)
                for c in feat
            }
            ww = gs["tm_n"].to_numpy(dtype=np.float64)
            for c in feat:
                vv = gs[c].to_numpy(dtype=np.float64)
                ok = np.isfinite(vv) & np.isfinite(ww)
                if ok.any():
                    cum_num[c] += float(np.sum(vv[ok] * ww[ok]))
                    cum_den[c] += float(np.sum(ww[ok]))
    M = np.full((len(tr_seasons), len(feat)), np.nan, dtype=np.float32)
    for i, (pid, s) in enumerate(zip(tr_pitchers, tr_seasons)):
        r = rows.get((int(pid), int(s)))
        if r is not None:
            M[i] = [r[c] for c in feat]
    return M, feat


def cell(frame, hist_mask, keycol, tag, alpha=ALPHA):
    """Per-pitcher prior-season smoothed cell encoding."""
    d = frame
    b = d.groupby(["pitcher_id", "season", keycol], sort=False)[TGT] \
         .agg(["size", "sum"]).unstack(fill_value=0)
    b.columns = [f"{x}_{int(k)}" for x, k in b.columns]
    keys = np.array(sorted({int(c.rsplit("_", 1)[1]) for c in b.columns}), dtype=np.int16)
    for k in keys:
        for x in ("size", "sum"):
            if f"{x}_{int(k)}" not in b:
                b[f"{x}_{int(k)}"] = 0.0
    cc = b.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    idx = pd.MultiIndex.from_arrays([d.pitcher_id.to_numpy(), d.season.to_numpy()])
    sz = cc[[f"size_{int(k)}" for k in keys]].reindex(idx).fillna(0.0).to_numpy()
    sm = cc[[f"sum_{int(k)}" for k in keys]].reindex(idx).fillna(0.0).to_numpy()
    kv = d[keycol].to_numpy(np.int16)
    pos = np.clip(np.searchsorted(keys, kv), 0, len(keys) - 1)
    hit = keys[pos] == kv
    row = np.arange(len(d))
    n = np.where(hit, sz[row, pos], 0.0)
    s = np.where(hit, sm[row, pos], 0.0)
    prior_lut = d.loc[hist_mask].groupby(keycol)[TGT].mean()
    g = float(d.loc[hist_mask, TGT].mean())
    prior = np.array([float(prior_lut.get(int(k), g)) for k in kv])
    base = (sm.sum(1) + alpha * g) / (sz.sum(1) + alpha)
    rate = (s + alpha * prior) / (n + alpha)
    d[f"pc_{tag}_rate"] = rate.astype("float32")
    d[f"pc_{tag}_dev"] = (rate - base).astype("float32")
    d[f"pc_{tag}_n"] = np.log1p(n).astype("float32")
    return [f"pc_{tag}_rate", f"pc_{tag}_dev", f"pc_{tag}_n"]


def main():
    # The remote scratch server keeps the shared feature builder at ROOT;
    # locally it also works when submit_35 is present because ROOT is first.
    sys.path.insert(0, ROOT)
    if not os.path.exists(os.path.join(ROOT, "features44.py")):
        sys.path.insert(0, os.path.join(ROOT, "submit_35"))
    import features44
    built = features44.build(DATA, VS=VS, return_frame=True)
    d = built["frame"]
    base = list(built["F44"])
    mtr, mva = built["m_tr"], built["m_va"]
    d["cnt12"] = (d.balls_before.astype("int16") * 3 + d.strikes_before.astype("int16")).astype("int16")
    d["cm"] = (d.cnt12 * 4 + d.pitcher_hand.astype("int16") * 2 + d.batter_hand.astype("int16")).astype("int16")
    d["pcmh"] = (d.cnt12 * 2 + d.batter_hand.astype("int16")).astype("int16")
    d["pco"] = (d.cnt12 * 3 + d.outs_before.astype("int16")).astype("int16")
    tb = d.top_bottom.map({"T": 0, "B": 1}) if d.top_bottom.dtype == object else d.top_bottom
    bs = d.base_state.map({"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                           "12_": 4, "1_3": 5, "_23": 6, "123": 7}) \
         if d.base_state.dtype == object else d.base_state
    d["pct"] = (d.cnt12 * 2 + tb.fillna(-1).astype("int16")).astype("int16")
    d["pcb"] = (d.cnt12 * 8 + bs.fillna(-1).astype("int16")).astype("int16")
    d["pcg"] = (d.cnt12 * 2 + d.game_type.astype("int16")).astype("int16")
    d["pci"] = (d.cnt12 * 10 + d.inning.astype("int16").clip(0, 9)).astype("int16")
    # Explicit count transforms of the as-of rates.  The base feature list
    # intentionally drops *_n, so expose both reliability and approximate
    # historical success counts to CatBoost as ordinary numeric features.
    pn0 = d["asof_pitcher_n"].fillna(0.0).astype("float32")
    bn0 = d["asof_batter_n"].fillna(0.0).astype("float32")
    d["hist_p_nlog"] = np.log1p(pn0).astype("float32")
    d["hist_b_nlog"] = np.log1p(bn0).astype("float32")
    d["hist_p_succ_n"] = (d["asof_pitcher_success_rate"].fillna(0.5) * pn0).astype("float32")
    d["hist_b_succ_n"] = (d["asof_batter_success_rate"].fillna(0.5) * bn0).astype("float32")
    d["hist_p_mid_n"] = (d["asof_pitcher_middle_rate"].fillna(0.15) * pn0).astype("float32")
    d["hist_p_rev_n"] = (d["asof_pitcher_reverse_rate"].fillna(0.25) * pn0).astype("float32")
    d["cur_p_succ_n"] = (d["p_is_succ"] * d["pn_cur"]).astype("float32")
    f_hist = ["hist_p_nlog", "hist_b_nlog", "hist_p_succ_n", "hist_b_succ_n",
              "hist_p_mid_n", "hist_p_rev_n", "cur_p_succ_n"]
    f_log = ["hist_p_nlog", "hist_b_nlog"]
    f_c12 = cell(d, mtr, "cnt12", "c12")
    f_cmh = cell(d, mtr, "pcmh", "cmh")
    f_cm = cell(d, mtr, "cm", "cm48")
    f_co = cell(d, mtr, "pco", "co36")
    f_ct = cell(d, mtr, "pct", "ct24")
    f_cb = cell(d, mtr, "pcb", "cb96")
    f_cg = cell(d, mtr, "pcg", "cg24")
    f_ci = cell(d, mtr, "pci", "ci120")
    cases = {
        "base44": base,
        "c12": base + f_c12,
        "c12_cmh": base + f_c12 + f_cmh,
        "c12_cm48": base + f_c12 + f_cm,
        "base_hist": base + f_hist,
        "c12_cmh_hist": base + f_c12 + f_cmh + f_hist,
        "c12_cmh_log": base + f_c12 + f_cmh + f_log,
    }
    if TM_PATH:
        raw_tm = pd.read_csv(TM_PATH, usecols=["tm_relh_sd", "tm_rels_sd", "tm_n"])
        if not {"tm_relh_sd", "tm_rels_sd", "tm_n"}.issubset(raw_tm.columns):
            raise ValueError("TM_PATH needs tm_relh_sd, tm_rels_sd, tm_n")
        # Read the key columns separately to preserve the exact train row order.
        keys = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                           usecols=["pitcher_id", "season"])
        tcols = ["tm_relh_sd", "tm_rels_sd"]
        tm2, names2 = trackman_matrix(TM_PATH, keys["season"].to_numpy(),
                                      keys["pitcher_id"].to_numpy(), cols=set(tcols))
        f_tm2 = [f"tm_{c}" for c in names2]
        for j, c in enumerate(f_tm2):
            d[c] = tm2[:, j]
        cases.update({
            "base_tm_sd2": base + f_tm2,
            "c12_cmh_tm_sd2": base + f_c12 + f_cmh + f_tm2,
        })
        if os.environ.get("TM_FULL", "0") == "1":
            tm_all, names_all = trackman_matrix(
                TM_PATH, keys["season"].to_numpy(), keys["pitcher_id"].to_numpy())
            f_tma = []
            for j, c in enumerate(names_all):
                name = "tmfull_" + c
                d[name] = tm_all[:, j]
                f_tma.append(name)
            cases.update({
                "base_tm_all": base + f_tma,
                "c12_cmh_tm_all": base + f_c12 + f_cmh + f_tma,
            })
        if os.environ.get("TM_2S", "0") == "1":
            tm2s, names_2s = trackman_matrix(
                TM_PATH, keys["season"].to_numpy(), keys["pitcher_id"].to_numpy(),
                cols={c for c in pd.read_csv(TM_PATH, nrows=0).columns if c.startswith("tm2s")})
            if TM2S_SET:
                wanted = {x.strip() for x in TM2S_SET.split(",") if x.strip()}
                keep = [j for j, c in enumerate(names_2s) if c in wanted]
                if not keep:
                    raise ValueError(f"TM2S_SET matched no columns: {TM2S_SET}")
                tm2s, names_2s = tm2s[:, keep], [names_2s[j] for j in keep]
            f_tm2s = []
            for j, c in enumerate(names_2s):
                name = "tm2s_" + c
                d[name] = tm2s[:, j]
                f_tm2s.append(name)
            cases.update({
                "base_tm_2s": base + f_tm2s,
                "c12_cmh_tm_2s": base + f_c12 + f_cmh + f_tm2s,
            })
    if LABEL_PATH:
        lf = pd.read_csv(LABEL_PATH).set_index("row_id").reindex(d["row_id"])
        if lf.isna().all(axis=1).any():
            raise ValueError("LABEL_PATH row_id alignment failed")
        lfcols = []
        for c in lf.columns:
            name = "lf_" + c
            d[name] = lf[c].fillna(0.0).to_numpy(np.float32)
            lfcols.append(name)
        # Evaluate each family member and a compact orthogonal bundle.  All
        # values come from the strict season-as-of generator.
        fmap = {c: "lf_" + c for c in lf.columns}
        for c in ("ball_bh", "ball_base", "reverse_bh", "middle_bh", "middle_base",
                  "strike_bh", "strike_base"):
            if c in fmap:
                cases["c12_cmh_" + c] = base + f_c12 + f_cmh + [fmap[c]]
        bundle = [fmap[c] for c in ("ball_bh", "ball_base", "reverse_bh") if c in fmap]
        if bundle:
            cases["c12_cmh_lf3"] = base + f_c12 + f_cmh + bundle
        cases["c12_cmh_lfall"] = base + f_c12 + f_cmh + lfcols
    # Optional Kim-derived label-family features.  The supplied file must be
    # generated with season-cumulative global priors (no validation-season
    # targets); this hook is deliberately opt-in for audit experiments.
    plat_path = os.environ.get("PLAT_PATH", "")
    if plat_path:
        pcd = pd.read_csv(plat_path).set_index("row_id").reindex(d["row_id"])
        if pcd.isna().any().any():
            raise ValueError("PLAT_PATH row_id alignment failed")
        for c in pcd.columns:
            d[f"kim_{c}"] = pcd[c].to_numpy(np.float32)
        f_kim = [f"kim_{c}" for c in pcd.columns]
        cases.update({
            "base_kim": base + f_kim,
            "c12_cmh_kim": base + f_c12 + f_cmh + f_kim,
            "c12_kim_rp": base + f_c12 + [f"kim_reverse_plat"],
        })
    if MORE:
        cases.update({
            "c12_co": base + f_c12 + f_co,
            "c12_ct": base + f_c12 + f_ct,
            "c12_cb": base + f_c12 + f_cb,
            "c12_cmh_co": base + f_c12 + f_cmh + f_co,
            "c12_cmh_cg": base + f_c12 + f_cmh + f_cg,
            "c12_cmh_ci": base + f_c12 + f_cmh + f_ci,
        })
    # The incumbent deliberately removes exact duplicate/context columns from
    # F44.  Keep an opt-in audit route for the hypothesis that CatBoost can
    # still exploit their different missingness or interaction patterns.  This
    # is validation-only; the normal submission feature list is untouched.
    if os.environ.get("RAW", "0") == "1":
        raw_n = [c for c in [PN, BN, MX] if c in d.columns]
        raw_context = [c for c in [
            "away_win_expectancy", "run_total_before", "score_diff_home",
            "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b",
        ] if c in d.columns]
        raw_all = raw_n + raw_context
        cases.update({
            "raw_n": base + raw_n,
            "raw_context": base + raw_context,
            "raw_all": base + raw_all,
            "c12_cmh_raw_n": base + f_c12 + f_cmh + raw_n,
            "c12_cmh_raw_all": base + f_c12 + f_cmh + raw_all,
        })
    if ONLY:
        cases = {k: v for k, v in cases.items() if k in ONLY}
    Xall = {n: d[f].to_numpy(np.float32) for n, f in cases.items()}
    ytr, yva = d.loc[mtr, TGT].to_numpy(np.float32), d.loc[mva, TGT].to_numpy(np.float64)
    isf_tr = d.loc[mtr, "game_type"].to_numpy() == 1
    isf_va = d.loc[mva, "game_type"].to_numpy() == 1
    w = 2.0 ** (d.loc[mtr, "season"].to_numpy(np.float64) - 2019.0)
    hp = dict(iterations=CB_ITERS, learning_rate=CB_LR, depth=CB_DEPTH, l2_leaf_reg=CB_L2,
              verbose=False, allow_writing_files=False, task_type="GPU", devices="0")
    print(f"VS={VS} train={mtr.sum():,} valid={mva.sum():,} regular={int((~isf_va).sum()):,}", flush=True)
    for name, xx in Xall.items():
        t0 = time.time()
        pred_all, pred_r, pred_f = [], [], []
        for sd in SEEDS:
            for which, mask in (("all", np.ones(len(ytr), bool)),
                                ("regular", ~isf_tr), ("futures", isf_tr)):
                model = CatBoostClassifier(random_seed=sd, **hp)
                model.fit(xx[mtr][mask], ytr[mask], sample_weight=w[mask])
                p = model.predict_proba(xx[mva])[:, 1]
                (pred_all if which == "all" else pred_r if which == "regular" else pred_f).append(p)
                del model
        pa = np.mean(pred_all, 0); pr = np.mean(pred_r, 0); pf = np.mean(pred_f, 0)
        route = .6 * pa + .4 * np.where(isf_va, pf, pr)
        s, sh = best(route[~isf_va], yva[~isf_va])
        sf = score(cal(route[~isf_va], -.007), yva[~isf_va])
        if SAVE_DIR:
            os.makedirs(SAVE_DIR, exist_ok=True)
            np.save(os.path.join(SAVE_DIR, f"pcgpu{VS}_{name}.npy"), route)
        print(json.dumps({"case": name, "features": xx.shape[1], "best": round(s, 3),
                          "best_shift": round(sh, 5), "fixed_-007": round(sf, 3),
                          "seconds": round(time.time()-t0, 1)}), flush=True)
        del xx, pred_all, pred_r, pred_f
        gc.collect()


if __name__ == "__main__":
    main()
