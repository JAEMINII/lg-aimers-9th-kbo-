# -*- coding: utf-8 -*-
"""후속 무학습 진단 — 59(58+stale) 구성 위에서.
    a) stale 행에서 DIN / tab 제거 (id 임베딩 보유 계열이 stale 에서도 해로운가)
    b) 타자-stale 단독 행 MS->msif
    c) 저표본-warm (학습 투구수 <=K) 행 MS->msif
    d) msaif 센터 재점검 (cold|stale 행 기준 폴드 최적시프트)
    e) stale 리그별 조각 델타 (정보용)
"""
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
W_NODIN = np.array([0.29, 0.12, 0.0, 0.30]) / 0.71
W_NOTAB = np.array([0.29, 0.0, 0.29, 0.30]) / 0.88


def bss(p, t):
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


def shift(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))


def sc2(p, t):
    r = minimize_scalar(lambda c: -bss(shift(p, c), t),
                        bounds=(-0.3, 0.3), method="bounded")
    return -r.fun, r.x


def sc(p, t):
    return sc2(p, t)[0]


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
    lb = trn.groupby("batter_id")["season"].max()
    np_cnt = trn.groupby("pitcher_id").size()

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
    n = len(yv)

    cold = np.array([(p not in wp) or (b not in wb) for p, b in zip(pid, bid)])
    dc = ~cold & (np.array([(p not in (wpF if f else wpR))
                            for p, f in zip(pid, isf)])
                  | np.array([(b not in (wbF if f else wbR))
                              for b, f in zip(bid, isf)]))
    lsp = np.array([lp.get(v, -1) for v in pid])
    lsb = np.array([lb.get(v, -1) for v in bid])
    cnt = np.array([np_cnt.get(v, 0) for v in pid])
    st = ~cold & ~dc & (lsp <= VS - 2)
    stb = ~cold & ~dc & ~st & (lsb <= VS - 2)
    idf = cold | st

    ms_row = np.where(isf, mso, ms6)
    ms59 = np.where(idf, mif, ms_row)

    def mix(msr, extra_if=None, wmask=None, wvec=None):
        r = msr if extra_if is None else np.where(extra_if, mif, msr)
        F = np.column_stack([cb, tab, din, r])
        Wm = np.tile(W4, (n, 1))
        Wm[cold] = CW
        if wmask is not None:
            Wm[wmask] = wvec
        return (F * Wm).sum(1)

    m = ~isf if VS == 2023 else np.ones(n, bool)
    tag = "1군만" if VS == 2023 else "전체"
    p59 = mix(ms59)
    s59 = sc(p59[m], yv[m])
    print(f"\n=== VS{VS} ({tag})  base59 {s59:.1f}  st {st[m].mean():.4f}  "
          f"stb {stb[m].mean():.4f}")

    a1 = mix(ms59, wmask=st & ~cold, wvec=W_NODIN)
    a2 = mix(ms59, wmask=st & ~cold, wvec=W_NOTAB)
    print(f"  a) stale행 DIN제거 Δ {sc(a1[m], yv[m])-s59:+.2f}   "
          f"tab제거 Δ {sc(a2[m], yv[m])-s59:+.2f}")

    b1 = mix(ms59, extra_if=stb)
    mm = stb & m
    sub = (f"  (부분 ms {sc(ms59[mm], yv[mm]):.0f}->{sc(mif[mm], yv[mm]):.0f})"
           if mm.sum() > 300 else "")
    print(f"  b) 타자stale msif Δ {sc(b1[m], yv[m])-s59:+.2f}  n={mm.sum()}{sub}")

    for K in (50, 200):
        lw = ~cold & ~dc & ~st & ~stb & (cnt <= K)
        c1 = mix(ms59, extra_if=lw)
        print(f"  c) 저표본 warm(<= {K:3d}) msif Δ {sc(c1[m], yv[m])-s59:+.2f}  "
              f"n={(lw & m).sum()}")

    im = idf & m
    _, copt = sc2(mif[im], yv[im])
    print(f"  d) msaif 최적센터(cold|stale) {copt:+.4f}  (배치 -0.020)")

    for nm2, mk in (("stale-1군", st & ~isf), ("stale-퓨처스", st & isf)):
        mm = mk & m
        if mm.sum() > 300:
            print(f"  e) {nm2} n={mm.sum()}  부분 ms {sc(ms_row[mm], yv[mm]):.1f} "
                  f"-> msif {sc(mif[mm], yv[mm]):.1f}")
