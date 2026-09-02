# -*- coding: utf-8 -*-
"""타자 트랙맨 프로필을 만들고 44열 통제 편상관으로 거른다.

왜 이 축인가
    44열의 타자 정보는 3개다 (asof_batter_n / success_rate / middle_rate).
    투수는 16개다. 그리고 그 3개 중 **물리량은 0개**다.

    앞선 트랙맨 시도 5번이 전부 0 이었던 이유는 44열의 asof_pitcher_* 16개가
    투수 물리량을 이미 담고 있어서였다. 타자 쪽은 담은 열 자체가 없다.
    같은 벽이 아니다.

누출이 없다 (구조적으로)
    trackman_history.csv 에는 control_success 가 **없다**. 물리량뿐이다.
    그래서 타자별로 집계해도 결과가 새어 들어올 경로가 없다.
    plat_dev 때와 성격이 완전히 다르다.

    배치도 깨끗하다 — 트랙맨은 2019~2024, 평가는 2025 다. 엄격히 과거다.

무엇을 집계하나
    수준     이 타자가 보는 공의 평균 구속·회전·무브먼트·릴리스
    배합     패스트볼 / 브레이킹 / 오프스피드 비율
    조건부   카운트별 배합 차이  <- 투수 쪽에서 유일하게 통과한 fb_rate_gap 과
             같은 형태다. '승부를 피하는 타자' 가 여기서 드러난다.

스크린 기준 (screen-features-by-partial-correlation 규율)
    44열 전부를 통제한 뒤 |r| > 0.005 만 통과.
    참고값 — lg_f_share 가 -0.0174 로 통과했고, 귀무는 0.00093 수준이다.
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
THR = 0.005

import features44 as F                                          # noqa: E402

PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]

# ---------------------------------------------------------------- 1  집계
bm = pd.read_csv(os.path.join(MAPDIR, "batter_map.csv"), encoding="utf-8-sig")
t2b = dict(zip(bm["trackman_id"], bm["batter_id"]))
print(f"  타자 매핑 {len(t2b)}명")

tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["batter_trackman_id", "pitch_type_group",
                          "balls_before", "strikes_before"] + PHYS)
tm["bid"] = tm["batter_trackman_id"].map(t2b)
tm = tm.dropna(subset=["bid"]).copy()
tm["bid"] = tm["bid"].astype(int)
print(f"  매핑된 타자의 투구 {len(tm):,}행 / 전체 1,793,078")

g = tm.groupby("bid")
P = pd.DataFrame(index=g.size().index)
P["b_n"] = g.size()
for c in PHYS:
    P["b_" + c] = g[c].mean()
P["b_speed_sd"] = g["rel_speed"].std()
P["b_spin_sd"] = g["spin_rate"].std()
P["b_speed_drop"] = P["b_rel_speed"] - P["b_zone_speed"]
tm["_mv"] = np.hypot(tm["induced_vert_break"], tm["horz_break"])
P["b_absmove"] = tm.groupby("bid")["_mv"].mean()

for k in ("fastball", "breaking", "offspeed"):
    P["b_" + k[:2] + "_rate"] = g["pitch_type_group"].apply(
        lambda s, k=k: float((s == k).mean()))

# 조건부 — 투수가 앞선 카운트 vs 몰린 카운트
ah = tm["strikes_before"] > tm["balls_before"]
bh = tm["balls_before"] > tm["strikes_before"]
for nm, m in (("ahead", ah), ("behind", bh)):
    sub = tm[m]
    P["b_fb_" + nm] = sub.groupby("bid")["pitch_type_group"].apply(
        lambda s: float((s == "fastball").mean()))
    P["b_sp_" + nm] = sub.groupby("bid")["rel_speed"].mean()
P["b_fb_gap"] = P["b_fb_behind"] - P["b_fb_ahead"]      # 승부를 피하는가
P["b_sp_gap"] = P["b_sp_behind"] - P["b_sp_ahead"]
P = P.drop(columns=["b_n"])
print(f"  프로필 {P.shape[1]}개 x 타자 {len(P)}명")

# ---------------------------------------------------------------- 2  조인
d = F.build(DATA, VS=2025, return_frame=True)
fr = d["frame"]
X44 = d["X44"].astype(np.float64)
y = d["y"].astype(np.float64)
m_tr = d["m_tr"]
bid = fr["batter_id"].to_numpy()
print(f"\n  train {len(fr):,}행   학습구간 {int(m_tr.sum()):,}행")

C = {}
for c in P.columns:
    v = pd.Series(bid).map(P[c]).to_numpy(np.float64)
    C[c] = v
cov = np.isfinite(C["b_rel_speed"]).mean()
print(f"  커버리지 {cov*100:.1f}%  (매핑 없는 타자는 리그평균 = 중립)")

# ---------------------------------------------------------------- 3  편상관
med = np.nanmedian(X44[m_tr], 0)
Xc = np.where(np.isnan(X44), med, X44)[m_tr]
Xc = np.column_stack([Xc, np.ones(len(Xc))])
yc = y[m_tr]


def resid(v):
    b, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return v - Xc @ b


ry = resid(yc)
sy = ry.std()
print(f"\n{'='*74}\n  44열 통제 편상관   임계 |r| > {THR}   "
      f"(참고 lg_f_share -0.0174, 귀무 0.0009)\n{'='*74}")
rows = []
for c in sorted(C):
    v = C[c][m_tr]
    if not np.isfinite(v).any():
        continue
    v = np.where(np.isfinite(v), v, np.nanmean(v))
    if v.std() < 1e-12:
        continue
    rv = resid(v)
    r = float((rv * ry).mean() / (rv.std() * sy + 1e-30))
    rows.append((c, r))
rows.sort(key=lambda t: -abs(t[1]))
for c, r in rows:
    print(f"  {c:20s} {r:+.5f}   {'통과' if abs(r) > THR else ''}")

ok = [c for c, r in rows if abs(r) > THR]
print(f"\n  통과 {len(ok)}개 / {len(rows)}개")
if ok:
    print(f"  최강 {rows[0][0]} ({rows[0][1]:+.5f})")
    print("  규율: 통과분은 **하나씩** 관문에 넣는다. 5개 묶으면 3폴드 전부")
    print("  음수, 최강 1개만 넣으면 전부 양수였던 전례가 있다.")
out = pd.DataFrame({c: C[c] for c in (ok if ok else [r[0] for r in rows[:3]])})
out.insert(0, "row_id", fr["row_id"].to_numpy())
out.to_csv(os.path.join(SC, "_dl", "bat_prof.csv.gz"), index=False)
P.to_csv(os.path.join(MAPDIR, "batter_profile.csv"), encoding="utf-8-sig")
print(f"\n  저장  colab/_dl/bat_prof.csv.gz  ({out.shape[1]-1}열)")
