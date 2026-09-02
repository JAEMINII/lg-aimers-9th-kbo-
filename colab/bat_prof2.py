# -*- coding: utf-8 -*-
"""선수 '안에서 변하는' 트랙맨 피처만 만든다. 상수형은 정의상 무효라서다.

앞선 판(bat_prof.py)이 왜 0/21 이었나
    TABM_CATEGORICAL_FEATURES 에 pitcher_id 와 batter_id 가 **둘 다 있다**.
    즉 투수 792명·타자 830명이 각자 임베딩을 갖는다. 타자당 1,776행으로
    표적에 대고 직접 학습된다.

    그러면 **타자당 상수 피처는 batter_id 원핫의 span 안에 통째로 들어간다.**
    편상관을 제대로(=임베딩까지 통제해서) 재면 0.0035 가 아니라 정확히 0 이다.
    앞선 트랙맨 5전 5패도 같은 이유다 — 전부 투수당 상수였다.

그래서 이번엔 뺀다
    dev = f(선수, 상황) - f(선수)

    선수 성분이 소거되므로 임베딩이 흉내낼 수 없다. 남는 건 **이 선수만의
    상황별 편차** — 상호작용이다. 44열에 이런 형태는 없다.

    train.csv 확인 결과
        구종 비율   asof_pitcher_{fastball,breaking,offspeed}_rate  <- 투수쪽·무조건부
        물리량      **없음** (구속·회전·무브먼트·릴리스 전부)
        타자쪽 배합 **없음**
    조건부 배합과 물리량은 둘 다 새 정보다.

수축
    셀 표본이 적으면 편차가 잡음이다. A=50 으로 선수 평균 쪽으로 당긴다.
        dev = n / (n + A) * (v_cell - v_player)
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

import features44 as F                                          # noqa: E402

MET = ["fb", "br", "rel_speed", "spin_rate", "absmove",
       "induced_vert_break", "extension"]

bm = pd.read_csv(os.path.join(MAPDIR, "batter_map.csv"), encoding="utf-8-sig")
pm = pd.read_csv(os.path.join(MAPDIR, "pitcher_map.csv"), encoding="utf-8-sig")
t2b = dict(zip(bm["trackman_id"], bm["batter_id"]))
t2p = dict(zip(pm["trackman_id"], pm["pitcher_id"]))
print(f"  매핑  타자 {len(t2b)}명 / 투수 {len(t2p)}명")

tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["batter_trackman_id", "pitcher_trackman_id",
                          "pitch_type_group", "balls_before", "strikes_before",
                          "pitcher_hand", "rel_speed", "spin_rate",
                          "induced_vert_break", "horz_break", "extension"])
tm["bid"] = tm["batter_trackman_id"].map(t2b)
tm["pid"] = tm["pitcher_trackman_id"].map(t2p)
tm["fb"] = (tm["pitch_type_group"] == "fastball").astype(float)
tm["br"] = (tm["pitch_type_group"] == "breaking").astype(float)
tm["absmove"] = np.hypot(tm["induced_vert_break"], tm["horz_break"])
tm["ph"] = tm["pitcher_hand"].astype(str).str[:1].str.upper()
tm["cnt"] = tm["balls_before"].astype(str) + "-" + tm["strikes_before"].astype(str)
print(f"  트랙맨 {len(tm):,}행   타자매핑 {tm.bid.notna().mean()*100:.1f}%"
      f"  투수매핑 {tm.pid.notna().mean()*100:.1f}%")


def dev_table(df, who, extra):
    """f(선수, 상황) - f(선수) 를 수축해서 낸다."""
    base = df.groupby(who)[MET].mean()
    cell = df.groupby([who, extra])[MET].mean()
    n = df.groupby([who, extra]).size().rename("n")
    b = base.reindex(cell.index.get_level_values(0)).to_numpy()
    sh = (n.to_numpy() / (n.to_numpy() + A))[:, None]
    out = pd.DataFrame((cell.to_numpy() - b) * sh, index=cell.index,
                       columns=MET)
    return out


tb = tm.dropna(subset=["bid"]).copy()
tb["bid"] = tb["bid"].astype(int)
tp = tm.dropna(subset=["pid"]).copy()
tp["pid"] = tp["pid"].astype(int)
D = {
    "bc": dev_table(tb, "bid", "cnt"),      # 타자 x 카운트
    "bh": dev_table(tb, "bid", "ph"),       # 타자 x 투수손
    "pc": dev_table(tp, "pid", "cnt"),      # 투수 x 카운트
}
for k, v in D.items():
    print(f"  {k} 셀 {len(v):,}개")

# ---------------------------------------------------------------- 조인
d = F.build(DATA, VS=2025, return_frame=True)
fr = d["frame"]
X44 = d["X44"].astype(np.float64)
y = d["y"].astype(np.float64)
m_tr = d["m_tr"]
raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["row_id", "balls_before", "strikes_before",
                           "pitcher_hand"])
raw = raw.set_index("row_id").reindex(fr["row_id"].to_numpy()).reset_index()
cnt = (raw["balls_before"].astype(str) + "-" + raw["strikes_before"].astype(str))
ph = raw["pitcher_hand"].map({1: "L", 2: "R"}).fillna("R")
KEYS = {"bc": list(zip(fr["batter_id"], cnt)),
        "bh": list(zip(fr["batter_id"], ph)),
        "pc": list(zip(fr["pitcher_id"], cnt))}

C = {}
for k, tab in D.items():
    idx = pd.MultiIndex.from_tuples(KEYS[k])
    sub = tab.reindex(idx)
    for m in MET:
        C[f"{k}_{m}"] = sub[m].to_numpy(np.float64)
print(f"  후보 {len(C)}개   커버리지 "
      f"bc {np.isfinite(C['bc_fb']).mean()*100:.1f}%  "
      f"pc {np.isfinite(C['pc_fb']).mean()*100:.1f}%")

# ---------------------------------------------------------------- 편상관
med = np.nanmedian(X44[m_tr], 0)
Xc = np.where(np.isnan(X44), med, X44)[m_tr]
Xc = np.column_stack([Xc, np.ones(len(Xc))])


def resid(v):
    b, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return v - Xc @ b


ry = resid(y[m_tr])
sy = ry.std()
print(f"\n{'='*72}\n  44열 통제 편상관  |r| > {THR}   "
      f"(참고 lg_f_share -0.0174, 귀무 0.0009)\n{'='*72}")
rows = []
for c in sorted(C):
    v = C[c][m_tr]
    v = np.where(np.isfinite(v), v, 0.0)          # 없으면 0 = 편차 없음
    if v.std() < 1e-12:
        continue
    rv = resid(v)
    rows.append((c, float((rv * ry).mean() / (rv.std() * sy + 1e-30))))
rows.sort(key=lambda t: -abs(t[1]))
for c, r in rows:
    print(f"  {c:26s} {r:+.5f}  {'통과' if abs(r) > THR else ''}")
ok = [c for c, r in rows if abs(r) > THR]
print(f"\n  통과 {len(ok)}개 / {len(rows)}개")
if ok:
    keep = ok
    print(f"  최강 {rows[0][0]} ({rows[0][1]:+.5f})  -> 관문엔 하나씩 넣는다")
else:
    keep = [r[0] for r in rows[:3]]
    print("  전멸. 선수 안에서 변하는 형태로도 신호가 없다.")
out = pd.DataFrame({c: np.where(np.isfinite(C[c]), C[c], 0.0) for c in keep})
out.insert(0, "row_id", fr["row_id"].to_numpy())
out.to_csv(os.path.join(SC, "_dl", "bat_prof2.csv.gz"), index=False)
print(f"  저장  colab/_dl/bat_prof2.csv.gz ({len(keep)}열)")
