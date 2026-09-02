# -*- coding: utf-8 -*-
"""라우팅 진단 3종 — 학습 없이 기존 폴드 배열만으로 3폴드 채점.

외부 검토자 제안의 사전관문(pre-gate). 전부 58 구성 위에서의 변경분.
    diag1  cold 행에서 DIN 의 한계 기여 (빼면 좋아지는가) -> id-free DIN 학습 여부 판정
    diag2  domain-cold (전역 warm 인데 현재 리그에서 미등장) 행의 MS -> id-free
    diag3  stale-ID (전역 warm 인데 직전시즌 미등장) 행의 MS -> id-free

정합성: 배열은 전부 row_id 정렬 train 의 season==VS 행 순서.
파리티 검사로 54->57->58 관문 델타를 재현해 재구성이 맞는지 먼저 확인한다.
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
CWX = np.array([0.45, 0.10, 0.0, 0.20]) / 0.75


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
    lb = trn.groupby("batter_id")["season"].max()

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
    for nm, a in (("cb", cb), ("tab", tab), ("din", din), ("ms6", ms6),
                  ("mso", mso), ("mif", mif)):
        assert len(a) == n, f"VS{VS} {nm} 길이 {len(a)} != {n}"

    cold_p = np.array([v not in wp for v in pid])
    cold_pb = cold_p | np.array([v not in wb for v in bid])
    seenL_p = np.array([(v in wpF) if f else (v in wpR)
                        for v, f in zip(pid, isf)])
    seenL_b = np.array([(v in wbF) if f else (v in wbR)
                        for v, f in zip(bid, isf)])
    dc = ~cold_pb & (~seenL_p | ~seenL_b)
    lsp = np.array([lp.get(v, -1) for v in pid])
    lsb = np.array([lb.get(v, -1) for v in bid])
    st_p = ~cold_pb & ~dc & (lsp <= VS - 2)
    st_pb = ~cold_pb & ~dc & ((lsp <= VS - 2) | (lsb <= VS - 2))

    ms_warm = np.where(isf, mso, ms6)
    fams = np.column_stack([cb, tab, din, ms_warm])
    fams_if = np.column_stack([cb, tab, din, mif])

    def mix(cold, use_if_on=None, cw_on=None, cold_w=CW):
        """cold: CW+msif 를 받는 행.  use_if_on: 추가로 MS만 msif 로 바꿀 행.
        cw_on: 추가로 가중만 CW 로 바꿀 행."""
        F = fams.copy()
        F[cold] = fams_if[cold]
        if use_if_on is not None:
            F[use_if_on] = fams_if[use_if_on]
        Wm = np.tile(W4, (n, 1))
        Wm[cold] = cold_w
        if cw_on is not None:
            Wm[cw_on] = CW
        return (F * Wm).sum(1)

    def mix_nomsif(cold):
        F = fams
        Wm = np.tile(W4, (n, 1))
        Wm[cold] = CW
        return (F * Wm).sum(1)

    m = ~isf if VS == 2023 else np.ones(n, bool)
    tag = "1군만" if VS == 2023 else "전체"
    p54 = mix_nomsif(cold_p)
    p57 = mix_nomsif(cold_pb)
    p58 = mix(cold_pb)
    s54, s57, s58 = sc(p54[m], yv[m]), sc(p57[m], yv[m]), sc(p58[m], yv[m])
    print(f"\n=== VS{VS} ({tag})  n={m.sum()}  cold_p {cold_p[m].mean():.3f}  "
          f"cold_pb {cold_pb[m].mean():.3f}  dc {dc[m].mean():.4f}  "
          f"st_p {st_p[m].mean():.4f}  st_pb {st_pb[m].mean():.4f}")
    print(f"  파리티  54 {s54:.1f}  57-54 {s57-s54:+.1f}  58-57 {s58-s57:+.1f}")

    cm = cold_pb & m
    if cm.sum() > 100:
        p58x = mix(cold_pb, cold_w=CWX)
        print(f"  diag1 cold부분(n={cm.sum()})  DIN있음 {sc(p58[cm], yv[cm]):.1f}"
              f"  DIN제거 {sc(p58x[cm], yv[cm]):.1f}"
              f"  Δ(제거-있음) {sc(p58x[cm], yv[cm])-sc(p58[cm], yv[cm]):+.1f}"
              f"  전체Δ {sc(p58x[m], yv[m])-s58:+.2f}")

    for nm2, mask in (("dc", dc), ("st_p", st_p), ("st_pb", st_pb)):
        mm = mask & m
        if mm.sum() < 100:
            print(f"  {nm2:5s} n={mm.sum()}  (표본 부족, 생략)")
            continue
        pa = mix(cold_pb, use_if_on=mask)
        pb_ = mix(cold_pb, use_if_on=mask, cw_on=mask)
        print(f"  {nm2:5s} n={mm.sum():6d}  ms부분 {sc(fams[mm][:, 3], yv[mm]):.1f}"
              f" -> msif부분 {sc(mif[mm], yv[mm]):.1f}"
              f"  | 전체Δ msif만 {sc(pa[m], yv[m])-s58:+.2f}"
              f"  msif+CW {sc(pb_[m], yv[m])-s58:+.2f}")
