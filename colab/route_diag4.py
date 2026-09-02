# -*- coding: utf-8 -*-
"""if1g 채점 — 61 구성에서 msaif 엔진만 1군-특화판(msif1g)으로 교체한 믹스 델타."""
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


def bss(p, t):
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


def shift(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))


def sc(p, t):
    r = minimize_scalar(lambda c: -bss(shift(p, c), t),
                        bounds=(-0.3, 0.3), method="bounded")
    return -r.fun


def ld(*names):
    a = [np.load(os.path.join(DL, n)).astype(np.float64) for n in names]
    a = [x.mean(0) if x.ndim == 2 else x for x in a]
    return np.mean(a, 0)


tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["row_id", "season", "game_type", "pitcher_id",
                          "batter_id", "control_success"])
rid = tr["row_id"].astype(str).str.extract(r"(\d+)$", expand=False).astype("int64")
tr = tr.assign(_rid=rid).sort_values("_rid", kind="mergesort").reset_index(drop=True)

CFG = {
    2024: ("ta2024_base.npy", None, "dg_2024_DIN.npy", "msr_state6",
           "msp_2024_ms_s2", "msif_2024"),
    2022: ("cb50fixed_2022.npy", "h2h_2022_friend_s42.npy", "dg_2022_DIN.npy",
           "msr22_state6", "msp_2022_ms_s2", "msif_2022"),
    2023: ("cb50_2023.npy", "h2h_2023_friend_s42.npy", "dg_2023_DIN.npy",
           "msr23_state6", "msr23_ms_s2", "msif_2023"),
}
tot = []
for VS in (2024, 2022, 2023):
    cbn, tbn, dnn, m6p, mop, mifp = CFG[VS]
    va = tr[tr.season == VS]
    trn = tr[tr.season < VS]
    yv = va["control_success"].to_numpy(np.float64)
    isf = va["game_type"].astype(str).to_numpy() == "F"
    pid = va["pitcher_id"].to_numpy()
    bid = va["batter_id"].to_numpy()
    wp = set(trn["pitcher_id"])
    wb = set(trn["batter_id"])
    ti = trn["game_type"].astype(str).to_numpy() == "F"
    wpF = set(trn["pitcher_id"].to_numpy()[ti])
    wpR = set(trn["pitcher_id"].to_numpy()[~ti])
    wbF = set(trn["batter_id"].to_numpy()[ti])
    wbR = set(trn["batter_id"].to_numpy()[~ti])
    lp = trn.groupby("pitcher_id")["season"].max()
    cb = ld(cbn)
    if VS == 2024:
        tab = (0.6 * np.load(f"{DL}/c4g_2024_c4_all_s42.npy")
               + 0.4 * np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
    else:
        tab = ld(tbn)
    din = ld(dnn)
    ms6 = ld(m6p + "_s42.npy", m6p + "_s1.npy")
    mso = ld(mop + "_s42.npy", mop + "_s1.npy")
    mif = ld(mifp + "_s42.npy", mifp + "_s1.npy")
    g1 = ld(f"msif1g_{VS}_s42.npy", f"msif1g_{VS}_s1.npy")
    n = len(yv)
    cold = np.array([(p not in wp) or (b not in wb) for p, b in zip(pid, bid)])
    dc = ~cold & (np.array([(p not in (wpF if f else wpR))
                            for p, f in zip(pid, isf)])
                  | np.array([(b not in (wbF if f else wbR))
                              for b, f in zip(bid, isf)]))
    lsp = np.array([lp.get(v, -1) for v in pid])
    st = ~cold & ~dc & (lsp <= VS - 2) & ~isf
    idf = (cold & ~isf) | st

    def mix(eng):
        r = np.where(idf, eng, np.where(isf, mso, ms6))
        F = np.column_stack([cb, tab, din, r])
        Wm = np.tile(W4, (n, 1))
        Wm[cold] = CW
        return (F * Wm).sum(1)

    m = ~isf if VS == 2023 else np.ones(n, bool)
    tag = "1군만" if VS == 2023 else "전체"
    s61 = sc(mix(mif)[m], yv[m])
    d = sc(mix(g1)[m], yv[m]) - s61
    im = idf & m
    print(f"VS{VS} ({tag})  전체Δ {d:+.2f}  | idf부분 "
          f"{sc(mif[im], yv[im]):.1f} -> {sc(g1[im], yv[im]):.1f}  "
          f"단독(1군) {sc(mif[m & ~isf], yv[m & ~isf]):.1f} -> "
          f"{sc(g1[m & ~isf], yv[m & ~isf]):.1f}")
    tot.append(d)
print(f"합 {sum(tot):+.2f}")
