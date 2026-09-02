"""Canonical multi-state TabM probe with deployment-safe plat_dev handling.

The canonical inference path has a different pitcher-season ``plat_dev``
lookup from the season-local features44 probe.  This run removes that one
unstable channel from both train and validation, keeping the target-structure
experiment otherwise identical.
"""
from multistate_softmax import *


def main_nopdev():
    raw = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
    aux_state = recover_state(raw)
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    is_f = raw.game_type.astype(str).to_numpy() == "F"
    all_report = {}
    for vs in FOLDS:
        t0 = time.time()
        hist = PP.fit_history_tables(raw[season < vs])
        Xbase = PP.transform_features(raw, hist, train_mode=True)
        names = list(Xbase.columns)
        if "plat_dev" in names:
            Xbase["plat_dev"] = 0.0
        old = season <= 2022
        c4 = np.where(old & is_f, 0.0,
                      np.where(old & ~is_f, 1.0,
                               np.where(is_f, 2.0, 3.0))).astype(np.float32)[:, None]
        ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        Xin = np.concatenate([Xbase.to_numpy(np.float32), c4], 1)
        train_mask = (season < vs) & (season >= MIN_SEASON)
        if BRANCH == "regular":
            train_mask &= ~is_f
        elif BRANCH == "futures":
            train_mask &= is_f
        Xn, Xc, cards = G.prep(Xin, train_mask, ci + [Xbase.shape[1]])
        G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
        tr_idx, va_idx = np.flatnonzero(train_mask), np.flatnonzero(season == vs)
        w = np.ones(len(y), np.float32)
        w[tr_idx] = np.where(is_f[tr_idx] & old[tr_idx], .1, 1.)
        log(f"NOPDEV VS={vs} X={Xin.shape} train={len(tr_idx):,} val={len(va_idx):,} "
            f"states={int(np.sum(aux_state[tr_idx] >= 0)):,}/{len(tr_idx):,} K={K} d={DBLOCK}")
        pdirect, pstate = [], []
        for seed in SEEDS:
            m = train_model(make_model(Xn.shape[1], cards, seed), tr_idx, y,
                             aux_state, w, seed, f"NOPDEV VS{vs} s{seed}")
            a, b = predict(m, va_idx)
            pdirect.append(a); pstate.append(b)
            del m
            torch.cuda.empty_cache()
        yv = y[va_idx].astype(float); fv = is_f[va_idx]; report = {}
        for name, arrs in (("direct", pdirect), ("state", pstate),
                           ("mix25", [.75*a + .25*b for a, b in zip(pdirect, pstate)]),
                           ("mix50", [.50*a + .50*b for a, b in zip(pdirect, pstate)])):
            p = np.mean(arrs, axis=0)
            a, sh = best_shift(p, yv)
            report[name] = {"overall": a, "shift": sh,
                            "regular": best_shift(p[~fv], yv[~fv])[0],
                            "futures": best_shift(p[fv], yv[fv])[0]}
            np.save(OUT / f"nopdev_vs{vs}_{name}.npy", p)
        all_report[str(vs)] = {"scores": report, "seconds": time.time() - t0}
        log(json.dumps({"NOPDEV_VS": vs, **report}, ensure_ascii=False))
    (OUT / "report_nopdev.json").write_text(json.dumps(all_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main_nopdev()
