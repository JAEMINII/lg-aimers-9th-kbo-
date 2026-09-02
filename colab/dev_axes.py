# -*- coding: utf-8 -*-
"""통한 형태를 다른 상황 축으로 넓힌다.

근거
    f(투수, 카운트) - f(투수) 가 5개 통과했다. 형태가 맞으면 되는 것이므로
    상황 키를 바꿔서 더 있는지 본다.

축 (전부 2025 평가 행에 있는 키라 배치가 된다)
    pc   투수 x 카운트          <- 이미 통과. 대조군으로 같이 돌린다
    pr   투수 x 주자상황(base_state)
    pi   투수 x 이닝구간
    pb   투수 x 타자손          <- 44열의 plat_dev 는 '성공률' 편차고 이건 물리량이다
    bh   타자 x 투수손          <- 이미 1개 통과
    br   타자 x 주자상황

물리량
    fb / br(구종) / rel_speed / spin_rate / induced_vert_break / extension

집계는 트랙맨 전 구간(2019~2024)으로 스크린만 한다. 통과분은 관문 전에
as-of 로 다시 만든다 (pcdev_build.py 와 같은 절차).
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
THR, A = 0.005, 50.0
MET = ["fb", "brk", "rel_speed", "spin_rate", "induced_vert_break", "extension"]

import features44 as F                                          # noqa: E402

bm = pd.read_csv(os.path.join(MAPDIR, "batter_map.csv"), encoding="utf-8-sig")
pm = pd.read_csv(os.path.join(MAPDIR, "pitcher_map.csv"), encoding="utf-8-sig")
tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["batter_trackman_id", "pitcher_trackman_id", "inning",
                          "pitch_type_group", "balls_before", "strikes_before",
                          "pitcher_hand", "batter_hand", "rel_speed", "spin_rate",
                          "induced_vert_break", "extension"])
tm["bid"] = tm["batter_trackman_id"].map(dict(zip(bm.trackman_id, bm.batter_id)))
tm["pid"] = tm["pitcher_trackman_id"].map(dict(zip(pm.trackman_id, pm.pitcher_id)))
tm["fb"] = (tm["pitch_type_group"] == "fastball").astype(float)
tm["brk"] = (tm["pitch_type_group"] == "breaking").astype(float)
tm["cnt"] = tm["balls_before"].astype(str) + "-" + tm["strikes_before"].astype(str)
tm["ph"] = tm["pitcher_hand"].astype(str).str[:1].str.upper()
tm["bh"] = tm["batter_hand"].astype(str).str[:1].str.upper()
tm["inb"] = np.clip(tm["inning"].fillna(1).astype(int), 1, 9).astype(str)
# 트랙맨엔 주자 열이 없다 -> 주자 축은 train 쪽 키로만 만들 수 있어 제외한다
AX = [("pc", "pid", "cnt"), ("pi", "pid", "inb"), ("pb", "pid", "bh"),
      ("bh", "bid", "ph"), ("bc", "bid", "cnt"), ("bi", "bid", "inb")]
print(f"  트랙맨 {len(tm):,}행  (주자상황 축은 트랙맨에 주자 열이 없어 제외)")


def dev(who, extra):
    s = tm.dropna(subset=[who]).copy()
    s[who] = s[who].astype(int)
    base = s.groupby(who)[MET].mean()
    cell = s.groupby([who, extra])[MET].mean()
    n = s.groupby([who, extra]).size().to_numpy()
    b = base.reindex(cell.index.get_level_values(0)).to_numpy()
    return pd.DataFrame((cell.to_numpy() - b) * (n / (n + A))[:, None],
                        index=cell.index, columns=MET)


d = F.build(DATA, VS=2025, return_frame=True)
fr = d["frame"]
X44, y, m_tr = d["X44"].astype(np.float64), d["y"].astype(np.float64), d["m_tr"]
raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["row_id", "balls_before", "strikes_before", "inning",
                           "pitcher_hand", "batter_hand"]
                  ).set_index("row_id").reindex(fr["row_id"].to_numpy())
RK = {
    "cnt": (raw["balls_before"].astype(str) + "-"
            + raw["strikes_before"].astype(str)).to_numpy(),
    "inb": np.clip(raw["inning"].fillna(1).astype(int), 1, 9).astype(str).to_numpy(),
    "ph": raw["pitcher_hand"].map({1: "L", 2: "R"}).fillna("R").to_numpy(),
    "bh": raw["batter_hand"].map({1: "L", 2: "R"}).fillna("R").to_numpy(),
}
ID = {"pid": fr["pitcher_id"].to_numpy(), "bid": fr["batter_id"].to_numpy()}

Xc = np.where(np.isnan(X44), np.nanmedian(X44[m_tr], 0), X44)[m_tr]
Xc = np.column_stack([Xc, np.ones(len(Xc))])


def resid(v):
    bb, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return v - Xc @ bb


ry = resid(y[m_tr])
sy = ry.std()
print(f"\n{'='*76}\n  44열 통제 편상관   임계 |r| > {THR}\n{'='*76}")
allr = []
for tag, who, ex in AX:
    t = dev(who, ex)
    sub = t.reindex(pd.MultiIndex.from_arrays([ID[who], RK[ex]]))
    line = []
    for m in MET:
        v = np.nan_to_num(sub[m].to_numpy(np.float64))[m_tr]
        if v.std() < 1e-12:
            continue
        rv = resid(v)
        r = float((rv * ry).mean() / (rv.std() * sy + 1e-30))
        allr.append((f"{tag}_{m}", r))
        line.append(f"{m[:8]:>8s} {r:+.5f}{'*' if abs(r) > THR else ' '}")
    print(f"  {tag} ({who} x {ex}, 셀 {len(t):,})")
    print("     " + "  ".join(line))
allr.sort(key=lambda t: -abs(t[1]))
ok = [c for c, r in allr if abs(r) > THR]
print(f"\n  통과 {len(ok)}개 / {len(allr)}개")
for c, r in allr[:10]:
    print(f"    {c:26s} {r:+.5f}  {'통과' if abs(r) > THR else ''}")
print("\n  * 표시가 임계 초과. 새 축에서 통과가 나오면 as-of 로 다시 만들어")
print("  관문에 올린다. pc 는 이미 대기 중이라 여기선 대조군이다.")
