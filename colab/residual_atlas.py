# -*- coding: utf-8 -*-
"""OOF 잔차 아틀라스 — 54-정합 믹스의 잔차를 저카디널리티 문맥별로 분해.

규율: 2022+2023 에서 후보 탐색(부호일치 & |δ|>기준 & 표본 기준) -> 동결 ->
2024 는 확정 검증에만 사용. 미씽 체제(동시결측 개수) 축도 같이 본다."""
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

MISSCOLS = [c for c in tr.columns if c.startswith("asof_")]
miss_cnt = tr[MISSCOLS].isna().sum(1).to_numpy()

CFG = {
    2024: ("ta2024_base.npy", None, "dg_2024_DIN.npy", "msr_state6",
           "msp_2024_ms_s2"),
    2022: ("cb50fixed_2022.npy", "h2h_2022_friend_s42.npy", "dg_2022_DIN.npy",
           "msr22_state6", "msp_2022_ms_s2"),
    2023: ("cb50_2023.npy", "h2h_2023_friend_s42.npy", "dg_2023_DIN.npy",
           "msr23_state6", "msr23_ms_s2"),
}


def contexts(df):
    cnt = df["balls_before"].to_numpy(int) * 3 + df["strikes_before"].to_numpy(int)
    hp = (df["pitcher_hand"].to_numpy(int) - 1) * 2 \
        + (df["batter_hand"].to_numpy(int) - 1)
    inn3 = np.clip((df["inning"].to_numpy(int) - 1) // 3, 0, 3)
    gt = (df["game_type"].astype(str).to_numpy() == "F").astype(int)
    mc = np.digitize(miss_cnt[df.index.to_numpy()], [1, 4, 8])
    return {
        "cnt": cnt, "hp": hp, "outs": df["outs_before"].to_numpy(int),
        "base": df["runner_on_1b"].to_numpy(int) + 2 * df["runner_on_2b"].to_numpy(int) + 4 * df["runner_on_3b"].to_numpy(int),
        "month": df["game_month"].to_numpy(int), "inn3": inn3,
        "nrun": np.clip(df["num_runners_on"].to_numpy(int), 0, 3),
        "misscnt": mc, "tb": (df["top_bottom"].astype(str).to_numpy() == "T").astype(int),
        "cnt_hp": cnt * 4 + hp, "cnt_outs": cnt * 3 + df["outs_before"].to_numpy(int),
        "cnt_gt": cnt * 2 + gt, "gt_misscnt": gt * 4 + mc,
    }


P, Y, C = {}, {}, {}
for VS in (2024, 2022, 2023):
    cbn, tbn, dnn, m6p, mop = CFG[VS]
    m_va = season == VS
    va = tr[m_va]
    trn = tr[season < VS]
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
    # 수준을 폴드 최적으로 맞춘 뒤의 잔차만 본다 (전역 수준차 제거)
    r = minimize_scalar(lambda c: -bss(sh(p54, c), yv),
                        bounds=(-0.3, 0.3), method="bounded")
    P[VS] = sh(p54, r.x)
    Y[VS] = yv
    C[VS] = contexts(va)
    C[VS]["_1gun"] = ~isf

# 탐색: 2022+2023(1군) 부호일치 & |δ|>0.004 & 각 폴드 n>=2000
cand = []
for key in C[2022]:
    if key == "_1gun":
        continue
    for g in np.unique(np.concatenate([np.unique(C[v][key]) for v in (2022, 2023)])):
        ds, ns = {}, {}
        for v in (2022, 2023):
            m = (C[v][key] == g) & C[v]["_1gun"]
            ns[v] = int(m.sum())
            ds[v] = float((Y[v][m] - P[v][m]).mean()) if m.sum() else 0.0
        if min(ns.values()) >= 2000 and abs(ds[2022]) > 0.004 \
                and abs(ds[2023]) > 0.004 and np.sign(ds[2022]) == np.sign(ds[2023]):
            cand.append((key, int(g), ds[2022], ds[2023], ns[2022], ns[2023]))
print(f"동결 후보 {len(cand)}개 (2022/2023 1군 기준)")
for key, g, d22, d23, n22, n23 in cand:
    m24 = (C[2024][key] == g) & C[2024]["_1gun"]
    d24 = float((Y[2024][m24] - P[2024][m24]).mean()) if m24.sum() else float("nan")
    ok = "확정" if m24.sum() and np.sign(d24) == np.sign(d22) \
        and abs(d24) > 0.002 else "기각"
    print(f"  {key:10s} g={g:3d}  δ22 {d22:+.4f}(n{n22})  δ23 {d23:+.4f}(n{n23})"
          f"  -> δ24 {d24:+.4f}(n{int(m24.sum())})  {ok}")

# 확정분만 수축 보정으로 2024 믹스 이득 추정
KAP = 20000.0
px = P[2024].copy()
lg = np.log(np.clip(px, 1e-6, 1 - 1e-6) / (1 - np.clip(px, 1e-6, 1 - 1e-6)))
applied = 0
for key, g, d22, d23, n22, n23 in cand:
    m24 = (C[2024][key] == g) & C[2024]["_1gun"]
    if not m24.sum():
        continue
    dref = 0.5 * (d22 + d23)
    ng = min(n22, n23)
    corr = (ng / (ng + KAP)) * dref / 0.25
    lg[m24] += corr
    applied += 1
pfix = 1 / (1 + np.exp(-lg))
m1 = C[2024]["_1gun"]
print(f"정직적용 {applied}개 셀 전부 (2024 필터 없음, 수축 κ={KAP:.0f})")
print(f"2024 전체 Δ {sc(pfix, Y[2024]) - sc(P[2024], Y[2024]):+.2f}  "
      f"1군 Δ {sc(pfix[m1], Y[2024][m1]) - sc(P[2024][m1], Y[2024][m1]):+.2f}")
