# -*- coding: utf-8 -*-
"""k=32 멤버 덤프 분석 — 조언 1A(집계규칙), 1B(분산 감쇠), 2(보조헤드 센서).
전부 54-정합 믹스 위 1군 msa6 슬롯 대상. 3폴드."""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
DL = os.path.join(SC, "_dl")
DATA = os.path.join(ROOT, "open (1)", "data")
W4 = np.array([0.29, 0.12, 0.29, 0.30])
CW = np.array([0.45, 0.10, 0.25, 0.20])
W_NOMS = np.array([0.29, 0.12, 0.29, 0.0]) / 0.70


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


sig = lambda z: 1 / (1 + np.exp(-z))
lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["row_id", "season", "game_type", "pitcher_id",
                          "control_success"])
rid = tr["row_id"].astype(str).str.extract(r"(\d+)$", expand=False).astype("int64")
tr = tr.assign(_rid=rid).sort_values("_rid", kind="mergesort").reset_index(drop=True)
CFG = {
    2024: ("ta2024_base.npy", None, "dg_2024_DIN.npy", "msr_state6",
           "msp_2024_ms_s2"),
    2022: ("cb50fixed_2022.npy", "h2h_2022_friend_s42.npy", "dg_2022_DIN.npy",
           "msr22_state6", "msp_2022_ms_s2"),
    2023: ("cb50_2023.npy", "h2h_2023_friend_s42.npy", "dg_2023_DIN.npy",
           "msr23_state6", "msr23_ms_s2"),
}
for VS in (2024, 2022, 2023):
    cbn, tbn, dnn, m6p, mop = CFG[VS]
    va = tr[tr.season == VS]
    trn = tr[tr.season < VS]
    yv = va["control_success"].to_numpy(np.float64)
    isf = va["game_type"].astype(str).to_numpy() == "F"
    pid = va["pitcher_id"].to_numpy()
    wp = set(trn["pitcher_id"])
    cb = ld(cbn)
    if VS == 2024:
        tab = (0.6 * np.load(f"{DL}/c4g_2024_c4_all_s42.npy")
               + 0.4 * np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
    else:
        tab = ld(tbn)
    din = ld(dnn)
    mso = ld(mop + "_s42.npy", mop + "_s1.npy")
    Z42 = np.load(f"{DL}/msk_{VS}_s42_z.npy").astype(np.float32)
    Z1 = np.load(f"{DL}/msk_{VS}_s1_z.npy").astype(np.float32)
    Q = 0.5 * (np.load(f"{DL}/msk_{VS}_s42_q.npy")
               + np.load(f"{DL}/msk_{VS}_s1_q.npy"))
    n_ = len(yv)
    cold = np.array([p not in wp for p in pid])

    def mix(ms6_slot, wmask=None, wvec=None):
        F = np.column_stack([cb, tab, din, np.where(isf, mso, ms6_slot)])
        Wm = np.tile(W4, (n_, 1))
        Wm[cold] = CW
        if wmask is not None:
            Wm[wmask & ~cold] = wvec
        return (F * Wm).sum(1)

    m = ~isf if VS == 2023 else np.ones(n_, bool)
    P42 = sig(Z42)
    P1 = sig(Z1)
    p_mean = 0.5 * (P42.mean(1) + P1.mean(1))          # 현행 (검증용)
    ref = ld(m6p + "_s42.npy", m6p + "_s1.npy")
    print(f"VS{VS}  현행재현 최대차 {np.abs(p_mean-ref).max():.2e}")
    aggs = {
        "logit평균": 0.5 * (sig(Z42.mean(1)) + sig(Z1.mean(1))),
        "중앙값": 0.5 * (np.median(P42, 1) + np.median(P1, 1)),
        "절사10%": 0.5 * (np.sort(P42, 1)[:, 3:-3].mean(1)
                        + np.sort(P1, 1)[:, 3:-3].mean(1)),
        "64전체logit": sig(np.concatenate([Z42, Z1], 1).mean(1)),
    }
    base = sc(mix(p_mean)[m], yv[m])
    out = "  1A"
    for k, pm in aggs.items():
        out += f"  {k} {sc(mix(pm)[m], yv[m]) - base:+.2f}"
    print(out)

    # 1B: 분산 5분위 — MS 한계기여
    Zall = np.concatenate([Z42, Z1], 1).astype(np.float32)
    u = Zall.std(1)
    qs = np.quantile(u[m], [0.2, 0.4, 0.6, 0.8])
    qb = np.digitize(u, qs)
    pw = mix(p_mean)
    pn = mix(p_mean, wmask=np.ones(n_, bool), wvec=W_NOMS)
    out = "  1B분위(MS기여)"
    for q5 in range(5):
        mm = m & (qb == q5)
        d = sc(pw[mm], yv[mm]) - sc(pn[mm], yv[mm])
        out += f"  Q{q5+1} {d:+.1f}"
    print(out + f"  (u 중앙값 {np.median(u[m]):.3f})")
    z0 = np.median(lgt(p_mean))
    zz = lgt(p_mean)
    out = "  1B감쇠"
    for lam in (0.3, 0.7, 1.5):
        p_at = sig(z0 + (zz - z0) / (1 + lam * (u - np.median(u))
                                     .clip(0, None)))
        out += f"  λ{lam} {sc(mix(p_at)[m], yv[m]) - base:+.2f}"
    print(out)

    # 2: 보조헤드 센서
    D = np.abs(p_mean - Q[:, 0])
    H = -(Q * np.log(np.clip(Q, 1e-9, 1))).sum(1)
    e2 = (yv - pw) ** 2
    print(f"  2센서  corr(D,e2) {np.corrcoef(D[m], e2[m])[0,1]:+.4f}  "
          f"corr(H,e2) {np.corrcoef(H[m], e2[m])[0,1]:+.4f}")
    qsD = np.quantile(D[m], [0.8])
    hiD = D > qsD[0]
    mm = m & hiD
    d_hi = sc(pw[mm], yv[mm]) - sc(pn[mm], yv[mm])
    mm2 = m & ~hiD
    d_lo = sc(pw[mm2], yv[mm2]) - sc(pn[mm2], yv[mm2])
    print(f"  2센서 D상위20% MS기여 {d_hi:+.1f}  하위80% {d_lo:+.1f}")
