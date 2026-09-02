"""Ablate error-audit features on the deployable features44 MS contract.

AUDIT_MODE=counts: raw sample-size reliability only.
AUDIT_MODE=interactions: handedness and leverage interactions only.
AUDIT_MODE=handonly: one explicit 4-way handedness categorical only.
AUDIT_MODE=all: both sets.
"""
from multistate_softmax import *

AUDIT_MODE = os.environ.get("AUDIT_MODE", "counts")


def audit_features(raw, x44, names):
    """Return numeric extras and separately declared categorical extras."""
    p_n = raw.asof_pitcher_n.fillna(0.0).to_numpy(np.float32)
    b_n = raw.asof_batter_n.fillna(0.0).to_numpy(np.float32)
    mix_n = raw.asof_pitcher_pitchmix_n.fillna(0.0).to_numpy(np.float32)
    p_rate = raw.asof_pitcher_success_rate.fillna(.5).to_numpy(np.float32)
    b_rate = raw.asof_batter_success_rate.fillna(.5).to_numpy(np.float32)
    li = raw.li.fillna(1.0).to_numpy(np.float32)
    balls = raw.balls_before.to_numpy(np.float32)
    strikes = raw.strikes_before.to_numpy(np.float32)
    count12 = balls * 3.0 + strikes
    runners = raw.base_state.astype(str).str.count(r"[^_]").to_numpy(np.float32)
    p_is = x44[:, names.index("p_is_succ")]
    pieces, pnames, cats, cnames = [], [], [], []
    if AUDIT_MODE in ("counts", "all"):
        lp, lb, lm = np.log1p(p_n), np.log1p(b_n), np.log1p(mix_n)
        pieces += [lp[:, None], lb[:, None], lm[:, None],
                   (p_rate * lp)[:, None], (b_rate * lb)[:, None]]
        pnames += ["log_p_n", "log_b_n", "log_mix_n", "p_rate_x_log_pn", "b_rate_x_log_bn"]
    if AUDIT_MODE in ("interactions", "all", "handonly"):
        hp = raw.pitcher_hand.fillna(0).to_numpy(np.int64)
        hb = raw.batter_hand.fillna(0).to_numpy(np.int64)
        # A separate 4-value categorical gives TabM a direct matchup table,
        # instead of asking independent hand embeddings to synthesize it.
        hand_pair = np.where((hp >= 1) & (hp <= 2) & (hb >= 1) & (hb <= 2),
                             (hp - 1) * 2 + (hb - 1), -1).astype(np.float32)
        if AUDIT_MODE in ("interactions", "all"):
            pieces += [(li * count12)[:, None], (li * runners)[:, None],
                       (li * p_is)[:, None], (li * (hand_pair == 0))[:, None],
                       (li * (hand_pair == 1))[:, None], (li * (hand_pair == 2))[:, None],
                       (li * (hand_pair == 3))[:, None]]
            pnames += ["li_x_count12", "li_x_runners", "li_x_p_is",
                       "li_x_hand00", "li_x_hand01", "li_x_hand10", "li_x_hand11"]
        cats.append(hand_pair[:, None]); cnames.append("hand_pair")
    num = np.concatenate(pieces, axis=1).astype(np.float32) if pieces else np.empty((len(raw), 0), np.float32)
    cat = np.concatenate(cats, axis=1).astype(np.float32) if cats else np.empty((len(raw), 0), np.float32)
    return num, pnames, cat, cnames


def main_audit():
    raw = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
    aux_state = recover_state(raw)
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    is_f = raw.game_type.astype(str).to_numpy() == "F"
    report_all = {}
    for vs in FOLDS:
        t0 = time.time()
        built = FF.build(str(DATA), VS=vs)
        x44 = built["X44"].astype(np.float32)
        fnames = list(built["F44"])
        xnum, xnum_names, xcat, xcat_names = audit_features(raw, x44, fnames)
        old = season <= 2022
        c4 = np.where(old & is_f, 0.0, np.where(old & ~is_f, 1.0,
                      np.where(is_f, 2.0, 3.0))).astype(np.float32)[:, None]
        xin = np.concatenate([x44, xnum, c4, xcat], axis=1)
        ci = [fnames.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        c4_index = x44.shape[1] + xnum.shape[1]
        cat_idx = ci + [c4_index] + list(range(c4_index + 1, xin.shape[1]))
        train_mask = (season < vs) & (season >= MIN_SEASON)
        if BRANCH == "regular": train_mask &= ~is_f
        elif BRANCH == "futures": train_mask &= is_f
        xn, xc, cards = G.prep(xin, train_mask, cat_idx)
        G.XN, G.XC, G.cards = torch.from_numpy(xn), torch.from_numpy(xc), cards
        tr_idx, va_idx = np.flatnonzero(train_mask), np.flatnonzero(season == vs)
        w = np.ones(len(y), np.float32)
        w[tr_idx] = np.where(is_f[tr_idx] & old[tr_idx], .1, 1.)
        log(f"AUDIT={AUDIT_MODE} VS={vs} X={xin.shape} extras={xnum_names+xcat_names} "
            f"train={len(tr_idx):,} val={len(va_idx):,} K={K} d={DBLOCK}")
        direct, state = [], []
        for seed in SEEDS:
            m = train_model(make_model(xn.shape[1], cards, seed), tr_idx, y, aux_state, w,
                            seed, f"AUDIT-{AUDIT_MODE} VS{vs} s{seed}")
            a, b = predict(m, va_idx); direct.append(a); state.append(b)
            del m; torch.cuda.empty_cache()
        yv, fv = y[va_idx].astype(float), is_f[va_idx]
        out = {}
        for name, arrs in (("direct", direct), ("state", state),
                           ("mix25", [.75*a+.25*b for a,b in zip(direct,state)])):
            p = np.mean(arrs, 0); s, sh = best_shift(p, yv)
            out[name] = {"overall": s, "shift": sh,
                         "regular": best_shift(p[~fv], yv[~fv])[0],
                         "futures": best_shift(p[fv], yv[fv])[0]}
            np.save(OUT / f"audit_{AUDIT_MODE}_vs{vs}_{name}.npy", p)
        report_all[str(vs)] = {"scores": out, "seconds": time.time()-t0}
        log(json.dumps({"AUDIT": AUDIT_MODE, "VS": vs, **out}, ensure_ascii=False))
    (OUT / "report.json").write_text(json.dumps(report_all, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main_audit()
