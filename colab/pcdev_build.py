# -*- coding: utf-8 -*-
"""편차 피처를 **as-of** 로 만들고, 신호가 살아남는지 먼저 확인한다.

왜 다시 재나
    bat_prof2.py 는 트랙맨 전 구간(2019~2024)으로 집계했다. 관문은
    season < VS 로 학습하니 집계도 그래야 공정하다. 표본이 줄면 편차가
    잡음이 되므로 신호가 죽을 수 있다. 70 GPU분을 쓰기 전에 확인한다.

    배치(2025)는 트랙맨 전 구간이 그대로 과거라 문제없다.

만드는 것 — 폴드마다 한 벌
    VS=2022  트랙맨 2019~2021 로 집계   (관문)
    VS=2024  트랙맨 2019~2023 로 집계   (관문)
    VS=2025  트랙맨 2019~2024 로 집계   (배치)

키는 2025 행에도 있는 것만 쓴다 — (투수, 카운트) 와 (타자, 투수손).
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
MAPDIR = os.path.join(ROOT, "trackman_map")
DL = os.path.join(SC, "_dl")
A = 50.0
KEEP = [("pc", "br"), ("pc", "induced_vert_break"), ("pc", "extension"),
        ("pc", "rel_speed"), ("pc", "fb"), ("bh", "spin_rate")]
MET = sorted({m for _, m in KEEP})

import features44 as F                                          # noqa: E402

bm = pd.read_csv(os.path.join(MAPDIR, "batter_map.csv"), encoding="utf-8-sig")
pm = pd.read_csv(os.path.join(MAPDIR, "pitcher_map.csv"), encoding="utf-8-sig")
tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["season", "batter_trackman_id", "pitcher_trackman_id",
                          "pitch_type_group", "balls_before", "strikes_before",
                          "pitcher_hand", "rel_speed", "spin_rate",
                          "induced_vert_break", "horz_break", "extension"])
tm["bid"] = tm["batter_trackman_id"].map(dict(zip(bm.trackman_id, bm.batter_id)))
tm["pid"] = tm["pitcher_trackman_id"].map(dict(zip(pm.trackman_id, pm.pitcher_id)))
tm["fb"] = (tm["pitch_type_group"] == "fastball").astype(float)
tm["br"] = (tm["pitch_type_group"] == "breaking").astype(float)
tm["absmove"] = np.hypot(tm["induced_vert_break"], tm["horz_break"])
tm["ph"] = tm["pitcher_hand"].astype(str).str[:1].str.upper()
tm["cnt"] = tm["balls_before"].astype(str) + "-" + tm["strikes_before"].astype(str)

d = F.build(DATA, VS=2025, return_frame=True)
fr = d["frame"]
X44 = d["X44"].astype(np.float64)
y = d["y"].astype(np.float64)
season = d["season"].astype(int)
raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["row_id", "balls_before", "strikes_before",
                           "pitcher_hand"]).set_index("row_id")
raw = raw.reindex(fr["row_id"].to_numpy())
rcnt = (raw["balls_before"].astype(str) + "-" +
        raw["strikes_before"].astype(str)).to_numpy()
rph = raw["pitcher_hand"].map({1: "L", 2: "R"}).fillna("R").to_numpy()
K = {"pc": list(zip(fr["pitcher_id"], rcnt)),
     "bh": list(zip(fr["batter_id"], rph))}
WHO = {"pc": ("pid", "cnt"), "bh": ("bid", "ph")}


def devs(sub, who, extra):
    s = sub.dropna(subset=[who]).copy()
    s[who] = s[who].astype(int)
    base = s.groupby(who)[MET].mean()
    cell = s.groupby([who, extra])[MET].mean()
    n = s.groupby([who, extra]).size().to_numpy()
    b = base.reindex(cell.index.get_level_values(0)).to_numpy()
    sh = (n / (n + A))[:, None]
    return pd.DataFrame((cell.to_numpy() - b) * sh, index=cell.index,
                        columns=MET)


for VS in (2022, 2024, 2025):
    sub = tm[tm.season < VS]
    cols = {}
    for grp in ("pc", "bh"):
        t = devs(sub, *WHO[grp])
        r = t.reindex(pd.MultiIndex.from_tuples(K[grp]))
        for g2, m in KEEP:
            if g2 == grp:
                cols[f"{grp}_{m}"] = np.nan_to_num(r[m].to_numpy(np.float64))
    out = pd.DataFrame(cols)
    out.insert(0, "row_id", fr["row_id"].to_numpy())
    out.to_csv(os.path.join(DL, f"pcdev_{VS}.csv.gz"), index=False)
    cvg = float((out["pc_br"] != 0).mean())
    print(f"  VS={VS}  트랙맨 {len(sub):,}행  커버리지 {cvg*100:.1f}%  "
          f"-> pcdev_{VS}.csv.gz")

    # ---- 그 폴드의 학습 구간에서 편상관 재확인
    m = season < VS
    Xc = np.where(np.isnan(X44), np.nanmedian(X44[m], 0), X44)[m]
    Xc = np.column_stack([Xc, np.ones(len(Xc))])

    def resid(v):
        bb, *_ = np.linalg.lstsq(Xc, v, rcond=None)
        return v - Xc @ bb

    ry = resid(y[m])
    sy = ry.std()
    line = []
    for c in out.columns[1:]:
        v = out[c].to_numpy()[m]
        if v.std() < 1e-12:
            line.append(f"{c} 0")
            continue
        rv = resid(v)
        line.append(f"{c.replace('induced_vert_break','ivb'):14s}"
                    f"{(rv*ry).mean()/(rv.std()*sy+1e-30):+.5f}")
    print("      " + "  ".join(line))
