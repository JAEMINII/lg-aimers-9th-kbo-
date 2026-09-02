# -*- coding: utf-8 -*-
"""GAM-lite — 저용량 가산 모델 (빈 lookup + 로지스틱 릿지). 다양성 후보.

의도적으로 기존 4계열과 반대 귀납편향: feature 별 소량 main effect + 상호작용 5개만.
원자료 직접 사용 (지인 전처리도 44열도 안 씀 — 파이프라인 다양성).
EB 수축 rate (r*=(nr+κμ)/(n+κ)) + reliability(n/(n+κ)) 포함.
판정: 54-정합 믹스에 λ 블렌드, 3폴드."""
import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
DL = os.path.join(SC, "_dl")
DATA = os.path.join(ROOT, "open (1)", "data")
W4 = np.array([0.29, 0.12, 0.29, 0.30])
CW = np.array([0.45, 0.10, 0.25, 0.20])
NB = 24
KAP = 300.0


def bss(p, t):
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


def sh(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))


def sc(p, t):
    r = minimize_scalar(lambda c: -bss(sh(p, c), t),
                        bounds=(-0.3, 0.3), method="bounded")
    return -r.fun


def ld(*names):
    a = [np.load(os.path.join(DL, n)).astype(np.float64) for n in names]
    a = [x.mean(0) if x.ndim == 2 else x for x in a]
    return np.mean(a, 0)


tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
rid = tr["row_id"].astype(str).str.extract(r"(\d+)$", expand=False).astype("int64")
tr = tr.assign(_rid=rid).sort_values("_rid", kind="mergesort").reset_index(drop=True)
y_all = tr["control_success"].to_numpy(np.float64)
season = tr["season"].to_numpy()
isf_all = tr["game_type"].astype(str).to_numpy() == "F"

lgt = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def build_cols(df, mu):
    n = df["asof_pitcher_n"].fillna(0.).to_numpy(float)
    r = df["asof_pitcher_success_rate"].fillna(mu).to_numpy(float)
    rs = (n * r + KAP * mu) / (n + KAP)
    nb = df["asof_batter_n"].fillna(0.).to_numpy(float)
    rb = df["asof_batter_success_rate"].fillna(mu).to_numpy(float)
    rbs = (nb * rb + KAP * mu) / (nb + KAP)
    num = {
        "li": df["li"].to_numpy(float),
        "hwe": df["home_win_expectancy"].to_numpy(float),
        "sdp": df["score_diff_pitcher_team"].to_numpy(float),
        "runt": df["run_total_before"].to_numpy(float),
        "p_rs": rs, "p_rel": n / (n + KAP),
        "b_rs": rbs, "b_rel": nb / (nb + KAP),
        "p_ball": df["asof_pitcher_ball_rate"].to_numpy(float),
        "p_strk": df["asof_pitcher_strike_rate"].to_numpy(float),
        "p_mid": df["asof_pitcher_middle_rate"].to_numpy(float),
        "p_rev": df["asof_pitcher_reverse_rate"].to_numpy(float),
        "b_mid": df["asof_batter_middle_rate"].to_numpy(float),
        "p_fb": df["asof_pitcher_fastball_rate"].to_numpy(float),
        "p_br": df["asof_pitcher_breaking_rate"].to_numpy(float),
        "trend3": lgt(df["asof_pitcher_prev3_game_success_rate"].to_numpy(float)) - lgt(rs),
        "trend1": lgt(df["asof_pitcher_prev1_game_success_rate"].to_numpy(float)) - lgt(rs),
        "logn": np.log1p(n), "lognb": np.log1p(nb),
    }
    cnt = (df["balls_before"].to_numpy(int) * 3
           + df["strikes_before"].to_numpy(int))
    hp = (df["pitcher_hand"].to_numpy(int) - 1) * 2 \
        + (df["batter_hand"].to_numpy(int) - 1)
    inn3 = np.clip((df["inning"].to_numpy(int) - 1) // 3, 0, 3)
    gt = (df["game_type"].astype(str).to_numpy() == "F").astype(int)
    cat = {
        "month": df["game_month"].to_numpy(int),
        "dow": df["game_dayofweek"].to_numpy(int),
        "inning": np.clip(df["inning"].to_numpy(int), 1, 12),
        "cnt": cnt, "outs": df["outs_before"].to_numpy(int),
        "base": df["runner_on_1b"].to_numpy(int) + 2 * df["runner_on_2b"].to_numpy(int) + 4 * df["runner_on_3b"].to_numpy(int),
        "tb": (df["top_bottom"].astype(str).to_numpy() == "T").astype(int),
        "gt": gt, "hp": hp,
        "nrun": df["num_runners_on"].to_numpy(int),
        "cnt_hp": cnt * 4 + hp,
        "cnt_out": cnt * 3 + df["outs_before"].to_numpy(int),
        "cnt_gt": cnt * 2 + gt,
        "cnt_inn3": cnt * 4 + inn3,
        "cnt_nrun": cnt * 4 + np.clip(df["num_runners_on"].to_numpy(int), 0, 3),
    }
    return num, cat


def onehot(num, cat, edges=None, fit_mask=None):
    cols, ncol = [], 0
    ed = {} if edges is None else edges
    for k, v in num.items():
        if edges is None:
            qs = np.quantile(v[fit_mask][np.isfinite(v[fit_mask])],
                             np.linspace(0, 1, NB + 1)[1:-1])
            ed[k] = np.unique(qs)
        b = np.digitize(np.where(np.isfinite(v), v, np.inf), ed[k])
        b = np.where(np.isfinite(v), b, len(ed[k]) + 1)
        cols.append(b + ncol)
        ncol += len(ed[k]) + 2
    for k, v in cat.items():
        if edges is None:
            ed["c_" + k] = int(v.max()) + 1
        m = ed["c_" + k]
        cols.append(np.clip(v, 0, m - 1) + ncol)
        ncol += m
    idx = np.column_stack(cols)
    n = idx.shape[0]
    X = sparse.csr_matrix(
        (np.ones(idx.size, np.float32),
         idx.ravel(), np.arange(0, idx.size + 1, idx.shape[1])),
        shape=(n, ncol))
    return X, ed


CFG = {
    2024: ("ta2024_base.npy", None, "dg_2024_DIN.npy", "msr_state6",
           "msp_2024_ms_s2"),
    2022: ("cb50fixed_2022.npy", "h2h_2022_friend_s42.npy", "dg_2022_DIN.npy",
           "msr22_state6", "msp_2022_ms_s2"),
    2023: ("cb50_2023.npy", "h2h_2023_friend_s42.npy", "dg_2023_DIN.npy",
           "msr23_state6", "msr23_ms_s2"),
}
for VS in (2024, 2022, 2023):
    m_tr = season < VS
    m_va = season == VS
    mu = y_all[m_tr].mean()
    num, cat = build_cols(tr, mu)
    X, ed = onehot(num, cat, edges=None, fit_mask=m_tr)
    w = np.where(isf_all & (season <= 2022), 0.1, 1.0) \
        * 1.5 ** (season.astype(float) - 2019.0)
    clf = LogisticRegression(C=0.5, solver="lbfgs", max_iter=300, tol=1e-4)
    clf.fit(X[m_tr], y_all[m_tr], sample_weight=w[m_tr])
    pg = clf.predict_proba(X[m_va])[:, 1]
    np.save(os.path.join(DL, f"gam_{VS}.npy"), pg)

    cbn, tbn, dnn, m6p, mop = CFG[VS]
    va = tr[m_va]
    trn = tr[m_tr]
    yv = y_all[m_va]
    isf = isf_all[m_va]
    pid = va["pitcher_id"].to_numpy()
    wp = set(trn["pitcher_id"])
    cb = ld(cbn)
    if VS == 2024:
        tab = (0.6 * np.load(f"{DL}/c4g_2024_c4_all_s42.npy")
               + 0.4 * np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
    else:
        tab = ld(tbn)
    din = ld(dnn)
    ms6 = ld(m6p + "_s42.npy", m6p + "_s1.npy")
    mso = ld(mop + "_s42.npy", mop + "_s1.npy")
    n_ = len(yv)
    cold = np.array([p not in wp for p in pid])
    F = np.column_stack([cb, tab, din, np.where(isf, mso, ms6)])
    Wm = np.tile(W4, (n_, 1))
    Wm[cold] = CW
    p54 = (F * Wm).sum(1)
    m = ~isf if VS == 2023 else np.ones(n_, bool)
    s54 = sc(p54[m], yv[m])
    out = (f"VS{VS}  GAM단독 {sc(pg[m], yv[m]):.1f}  "
           f"상관 {np.corrcoef(pg[m], p54[m])[0, 1]:.3f}  |")
    for lam in (0.02, 0.05, 0.08, 0.12):
        px = (1 - lam) * p54 + lam * pg
        out += f"  λ{lam:.2f} {sc(px[m], yv[m]) - s54:+.2f}"
    print(out, flush=True)
