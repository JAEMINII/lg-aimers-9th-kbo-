# -*- coding: utf-8 -*-
"""reverse / middle 의 **플래툰 편차**를 만든다. plat_dev 의 형제다.

근거 (김광현.zip 산점도)
    유형별 산점도의 축이 (crossPlateX, 계산된 plate_z) 다. 즉 판정 범주는
    **홈플레이트 통과 위치**로 정의된다.
        middle    좌우 중앙(X~0)에 몰린 공
        reverse   좌우로 크게 벗어난 공
    "의도 반대쪽" 은 **타자 좌우에 따라 방향이 뒤집힌다**. 좌투수가 좌타자를
    상대할 때와 우타자를 상대할 때 반대쪽이 다르다.

빈틈
    44열에 있는 것    asof_pitcher_reverse_rate / middle_rate  (타자손 무관, 전체 비율)
                     plat_dev  = 투수의 **성공률** 플래툰 편차
    없는 것          reverse / middle 의 플래툰 편차

    투구 단위 라벨을 as-of 비율에서 99.94% 복원해뒀으므로 만들 수 있다.
    (같은 투수, asof_pitcher_n 이 정확히 1 증가한 쌍의 누적개수 차이)

만드는 법 — plat_dev 와 같은 규약
    (투수, 시즌, 타자손) 별 누적을 시즌 단위로 shift(1) 해서 as-of 로 만들고,
    ALPHA=300 으로 (투수손, 타자손)별 리그값 쪽으로 수축한다.
        rev_plat = P_reverse(그 타자손) - P_reverse(전체)
    2025 행에도 붙는다 — 투수 id 와 타자손만 있으면 되고 둘 다 test.csv 에 있다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
ALPHA = 300.0

import features44 as F                                          # noqa: E402

RATES = {"success": "asof_pitcher_success_rate",
         "reverse": "asof_pitcher_reverse_rate",
         "middle": "asof_pitcher_middle_rate"}
raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["row_id", "pitcher_id", "batter_hand", "pitcher_hand",
                           "season", "asof_pitcher_n", "control_success"]
                  + list(RATES.values()))
n = raw["asof_pitcher_n"].to_numpy(np.float64)
pid = raw["pitcher_id"].to_numpy()
ok = np.r_[False, (pid[1:] == pid[:-1]) & (np.diff(n) == 1)]
src = np.where(ok)[0]
dst = src - 1
LAB = {}
for nm, c in RATES.items():
    cum = raw[c].to_numpy(np.float64) * n
    inc = cum[src] - cum[dst]
    lab = np.round(inc)
    good = (np.abs(inc - lab) < 0.25) & ((lab == 0) | (lab == 1))
    v = np.full(len(raw), np.nan)
    v[dst[good]] = lab[good]
    LAB[nm] = v
y = raw["control_success"].to_numpy(np.float64)
m = np.isfinite(LAB["success"])
print(f"  라벨 복원 {int(m.sum()):,}행   success 일치 "
      f"{float((LAB['success'][m] == y[m]).mean()):.6f}")

d = F.build(DATA, VS=2025, return_frame=True)
assert (d["frame"]["row_id"].to_numpy() == raw["row_id"].to_numpy()).all()
X44, m_tr, F44 = d["X44"].astype(np.float64), d["m_tr"], list(d["F44"])
bh = raw["batter_hand"].to_numpy()
ph = raw["pitcher_hand"].to_numpy()
season = raw["season"].to_numpy()

C = {}
for nm in ("reverse", "middle"):
    v = LAB[nm]
    have = np.isfinite(v)
    df = pd.DataFrame({"pid": pid, "season": season, "bh": bh,
                       "v": np.where(have, v, 0.0), "n": have.astype(float)})
    # (투수, 시즌, 타자손) 누적 -> 시즌 단위 shift(1) = as-of
    g = df.groupby(["pid", "season", "bh"])[["v", "n"]].sum()
    g = g.unstack("bh", fill_value=0.0)
    cum = g.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    tot_v = sum(cum[("v", h)] for h in (1, 2))
    tot_n = sum(cum[("n", h)] for h in (1, 2))
    # 사전값 — (투수손, 타자손)별 리그값
    lg = df.assign(ph=ph).groupby(["ph", "bh"]).apply(
        lambda t: t["v"].sum() / max(t["n"].sum(), 1.0), include_groups=False)
    lgp = df.assign(ph=ph).groupby("ph").apply(
        lambda t: t["v"].sum() / max(t["n"].sum(), 1.0), include_groups=False)
    ph_of = pd.Series(ph, index=pid).groupby(level=0).first()
    idx = pd.MultiIndex.from_arrays([pid, season])
    sub = cum.reindex(idx)
    pho = ph_of.reindex(pid).to_numpy()
    pr_h = np.array([lg.get((p_, int(h_)), np.nan) for p_, h_ in zip(pho, bh)])
    pr_a = np.array([lgp.get(p_, np.nan) for p_ in pho])
    vh = np.where(bh == 1, sub[("v", 1)].to_numpy(), sub[("v", 2)].to_numpy())
    nh = np.where(bh == 1, sub[("n", 1)].to_numpy(), sub[("n", 2)].to_numpy())
    P_h = (vh + ALPHA * pr_h) / (nh + ALPHA)
    P_a = (tot_v.reindex(idx).to_numpy() + ALPHA * pr_a) / \
          (tot_n.reindex(idx).to_numpy() + ALPHA)
    C[f"{nm}_plat"] = np.nan_to_num(P_h - P_a)
    C[f"{nm}_rate_asof"] = np.nan_to_num(P_a)
    print(f"  {nm:8s} 플래툰 편차  SD {np.nanstd(C[f'{nm}_plat']):.5f}  "
          f"범위 [{np.nanmin(C[f'{nm}_plat']):+.4f}, "
          f"{np.nanmax(C[f'{nm}_plat']):+.4f}]")

med = np.nanmedian(X44[m_tr], 0)
Xc = np.where(np.isnan(X44), med, X44)[m_tr]
Xc = np.column_stack([Xc, np.ones(len(Xc))])


def resid(v):
    b, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return v - Xc @ b


ry = resid(y[m_tr])
sy = ry.std()
print(f"\n  44열 통제 편상관   임계 |r| > 0.005   (참고 plat_dev 는 +0.0148)\n")
rows = []
for c, v in C.items():
    vv = np.nan_to_num(v)[m_tr]
    if vv.std() < 1e-12:
        continue
    rv = resid(vv)
    rows.append((c, float((rv * ry).mean() / (rv.std() * sy + 1e-30))))
for c, r in sorted(rows, key=lambda t: -abs(t[1])):
    print(f"  {c:20s} {r:+.5f}  {'통과' if abs(r) > 0.005 else ''}")
print(f"\n  통과 {sum(1 for _, r in rows if abs(r) > 0.005)}개 / {len(rows)}개")
out = pd.DataFrame({c: np.nan_to_num(v) for c, v in C.items()})
out.insert(0, "row_id", raw["row_id"].to_numpy())
out.to_csv(os.path.join(SC, "_dl", "plat_rev.csv.gz"), index=False)
print("  저장  colab/_dl/plat_rev.csv.gz")
