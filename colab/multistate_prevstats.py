"""Multi-state TabM with legal prior-season smoothed target statistics.

Adds three continuous features built only from seasons before the row's season:
pitcher overall, pitcher x batter-hand, and batter overall success rates.  For
the validation season they use the historical block; for training rows they
use only earlier seasons, so train/validation/test share the same temporal
contract.
"""
from multistate_softmax import *


def _prior_rates(raw, hist_mask, id_cols, hand_col=None, alpha=50.0):
    """Return a smoothed prior-season rate for each row in raw."""
    season = raw["season"].to_numpy(np.int16)
    hist = raw.loc[hist_mask]
    gm = float(hist["control_success"].mean())
    keys = id_cols + ([hand_col] if hand_col else [])
    tab = (hist.groupby(keys + ["season"], sort=False)["control_success"]
           .agg(["sum", "count"]).reset_index())
    tab = tab.sort_values(keys + ["season"], kind="mergesort")
    # One row per group-season: cumulative minus current is the prior-season
    # total, while a second groupby gives the all-history total for validation
    # and test seasons.
    tab["ps_prior"] = tab.groupby(keys, sort=False)["sum"].cumsum() - tab["sum"]
    tab["pn_prior"] = tab.groupby(keys, sort=False)["count"].cumsum() - tab["count"]
    totals = (tab.groupby(keys, sort=False)[["sum", "count"]].sum()
              .rename(columns={"sum": "ps_total", "count": "pn_total"})
              .reset_index())
    left = raw[keys + ["season"]].copy()
    left["_idx"] = np.arange(len(raw), dtype=np.int64)
    left = left.merge(tab[keys + ["season", "ps_prior", "pn_prior"]],
                      on=keys + ["season"], how="left", sort=False)
    left = left.merge(totals, on=keys, how="left", sort=False)
    left = left.sort_values("_idx", kind="mergesort")
    ps = np.where(hist_mask, left["ps_prior"].fillna(0.0), left["ps_total"].fillna(0.0))
    pn = np.where(hist_mask, left["pn_prior"].fillna(0.0), left["pn_total"].fillna(0.0))
    return np.asarray((ps + alpha * gm) / (pn + alpha), dtype=np.float32)


def main_prevstats():
    raw = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
    aux_state = recover_state(raw)
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    is_f = raw.game_type.astype(str).to_numpy() == "F"
    all_report = {}
    for vs in FOLDS:
        t0 = time.time()
        hist_mask = season < vs
        hist = PP.fit_history_tables(raw[hist_mask])
        Xbase = PP.transform_features(raw, hist, train_mode=True)
        # These are all strictly prior-season statistics.  For the current
        # validation season, hist_mask rows are the complete history.
        Xbase["p_prev_rate"] = _prior_rates(raw, hist_mask, ["pitcher_id"])
        Xbase["p_prev_hand_rate"] = _prior_rates(raw, hist_mask, ["pitcher_id"], "batter_hand")
        Xbase["b_prev_rate"] = _prior_rates(raw, hist_mask, ["batter_id"])
        names = list(Xbase.columns)
        old = season <= 2022
        c4 = np.where(old & is_f, 0., np.where(old & ~is_f, 1.,
                      np.where(is_f, 2., 3.))).astype(np.float32)[:, None]
        ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        Xin = np.concatenate([Xbase.to_numpy(np.float32), c4], 1)
        train_mask = hist_mask & (season >= MIN_SEASON)
        if BRANCH == "regular": train_mask &= ~is_f
        elif BRANCH == "futures": train_mask &= is_f
        Xn, Xc, cards = G.prep(Xin, train_mask, ci + [Xbase.shape[1]])
        G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
        tr_idx, va_idx = np.flatnonzero(train_mask), np.flatnonzero(season == vs)
        w = np.ones(len(y), np.float32); w[tr_idx] = np.where(is_f[tr_idx] & old[tr_idx], .1, 1.)
        log(f"PREVSTATS VS={vs} X={Xin.shape} train={len(tr_idx):,} val={len(va_idx):,} K={K} d={DBLOCK}")
        pdirect, pstate = [], []
        for seed in SEEDS:
            m = train_model(make_model(Xn.shape[1], cards, seed), tr_idx, y, aux_state, w,
                             seed, f"PREVSTATS VS{vs} s{seed}")
            a, b = predict(m, va_idx); pdirect.append(a); pstate.append(b)
            del m; torch.cuda.empty_cache()
        yv = y[va_idx].astype(float); fv = is_f[va_idx]; report = {}
        for name, arrs in (("direct", pdirect), ("state", pstate),
                           ("mix25", [.75*a+.25*b for a,b in zip(pdirect,pstate)])):
            p = np.mean(arrs, axis=0); a, sh = best_shift(p, yv)
            report[name] = {"overall": a, "shift": sh,
                            "regular": best_shift(p[~fv], yv[~fv])[0],
                            "futures": best_shift(p[fv], yv[fv])[0]}
            np.save(OUT / f"prevstats_vs{vs}_{name}.npy", p)
        all_report[str(vs)] = {"scores": report, "seconds": time.time()-t0}
        log(json.dumps({"PREVSTATS_VS": vs, **report}, ensure_ascii=False))
    (OUT / "report_prevstats.json").write_text(json.dumps(all_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main_prevstats()
