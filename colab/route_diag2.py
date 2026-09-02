# -*- coding: utf-8 -*-
"""futw0 절제 채점 — 60 구성(58+stale 1군한정) 위에서 MS 두 슬롯만
msw0 재학습본으로 교체했을 때의 믹스 델타. 3폴드."""
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
    2024: dict(cb=["ta2024_base.npy"], tab=None, din="dg_2024_DIN.npy",
               ms6=["msr_state6_s42.npy", "msr_state6_s1.npy"],
               mso=["msp_2024_ms_s2_s42.npy", "msp_2024_ms_s2_s1.npy"],
               mif=["msif_2024_s42.npy", "msif_2024_s1.npy"]),
    2022: dict(cb=["cb50fixed_2022.npy"], tab=["h2h_2022_friend_s42.npy"],
               din="dg_2022_DIN.npy",
               ms6=["msr22_state6_s42.npy", "msr22_state6_s1.npy"],
               mso=["msp_2022_ms_s2_s42.npy", "msp_2022_ms_s2_s1.npy"],
               mif=["msif_2022_s42.npy", "msif_2022_s1.npy"]),
    2023: dict(cb=["cb50_2023.npy"], tab=["h2h_2023_friend_s42.npy"],
               din="dg_2023_DIN.npy",
               ms6=["msr23_state6_s42.npy", "msr23_state6_s1.npy"],
               mso=["msr23_ms_s2_s42.npy", "msr23_ms_s2_s1.npy"],
               mif=["msif_2023_s42.npy", "msif_2023_s1.npy"]),
}

for VS in (2024, 2022, 2023):
    c = CFG[VS]
    va = tr[tr.season == VS]
    trn = tr[tr.season < VS]
    yv = va["control_success"].to_numpy(np.float64)
    isf = va["game_type"].astype(str).to_numpy() == "F"
    pid = va["pitcher_id"].to_numpy()
    bid = va["batter_id"].to_numpy()
    wp = set(trn["pitcher_id"].unique().tolist())
    wb = set(trn["batter_id"].unique().tolist())
    t_isf = trn["game_type"].astype(str).to_numpy() == "F"
    wpF = set(trn["pitcher_id"].to_numpy()[t_isf].tolist())
    wpR = set(trn["pitcher_id"].to_numpy()[~t_isf].tolist())
    wbF = set(trn["batter_id"].to_numpy()[t_isf].tolist())
    wbR = set(trn["batter_id"].to_numpy()[~t_isf].tolist())
    lp = trn.groupby("pitcher_id")["season"].max()

    cb = ld(*c["cb"])
    if VS == 2024:
        tab = 0.6 * np.load(os.path.join(DL, "c4g_2024_c4_all_s42.npy")) \
            + 0.4 * np.load(os.path.join(DL, "c4g_2024_c4_regular_s42.npy"))
        tab = tab.astype(np.float64)
    else:
        tab = ld(*c["tab"])
    din = ld(c["din"])
    ms6 = ld(*c["ms6"])
    mso = ld(*c["mso"])
    mif = ld(*c["mif"])
    w6 = ld(f"msw0_{VS}_s6_s42.npy", f"msw0_{VS}_s6_s1.npy")
    if VS != 2023:
        wo = ld(f"msw0_{VS}_ms_s42.npy", f"msw0_{VS}_ms_s1.npy")
    else:
        wo = mso
    n = len(yv)

    cold = np.array([(p not in wp) or (b not in wb) for p, b in zip(pid, bid)])
    dc = ~cold & (np.array([(p not in (wpF if f else wpR))
                            for p, f in zip(pid, isf)])
                  | np.array([(b not in (wbF if f else wbR))
                              for b, f in zip(bid, isf)]))
    lsp = np.array([lp.get(v, -1) for v in pid])
    st = ~cold & ~dc & (lsp <= VS - 2) & ~isf
    idf = cold | st

    def mix(m6, mo):
        msr = np.where(isf, mo, m6)
        msr = np.where(idf, mif, msr)
        F = np.column_stack([cb, tab, din, msr])
        Wm = np.tile(W4, (n, 1))
        Wm[cold] = CW
        return (F * Wm).sum(1)

    m = ~isf if VS == 2023 else np.ones(n, bool)
    tag = "1군만" if VS == 2023 else "전체"
    p59 = mix(ms6, mso)
    pw0 = mix(w6, wo)
    d_full = sc(pw0[m], yv[m]) - sc(p59[m], yv[m])
    dr = sc(pw0[m & ~isf], yv[m & ~isf]) - sc(p59[m & ~isf], yv[m & ~isf])
    out = f"VS{VS} ({tag})  전체Δ {d_full:+.2f}  1군Δ {dr:+.2f}"
    if VS != 2023:
        df_ = sc(pw0[isf], yv[isf]) - sc(p59[isf], yv[isf])
        out += f"  퓨처스Δ {df_:+.2f}"
    s6d = sc(w6[m], yv[m]) - sc(ms6[m], yv[m])
    out += f"  | state6단독Δ {s6d:+.1f}"
    print(out)
