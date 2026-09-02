# -*- coding: utf-8 -*-
"""진짜 출루율을 복원한다. 앞선 판이 틀렸다.

앞선 판의 결함
    볼넷 15.0% (실제 9~10%), 삼진 23.5% (실제 18~19%) 로 둘 다 과다계상이었다.
    '아웃이 안 늘었다' 만 보고 볼넷이라 했는데 3볼에서 안타로 나간 것도 그렇다.

제대로 하는 법 — 몸 수 항등식
    타석 전 주자 R, 아웃 O, 득점 S. 타석 후 R', O', S'.
    타석에 관여한 몸은 R + 1 (주자들 + 타자) 이고, 그들은 셋 중 하나가 된다.
        R + 1 = R' + (S' - S) + (O' - O)
    이 항등식이 맞는 타석만 쓴다. 안 맞으면 하프이닝 경계를 잘못 잡았거나
    도루/견제사 같은 타석 외 사건이 낀 것이다.

    출루 = 아웃이 안 늘었고 몸이 늘어난 타석 (안타/볼넷/사구/실책).
    야수선택은 아웃이 늘어 자동으로 빠진다. 이게 OBP 정의에 가깝다.

검증
    복원한 리그 출루율이 KBO 실제(0.35~0.37)와 맞는지 본다.
    안 맞으면 복원이 틀린 것이고 그 위에 쌓은 결론도 못 믿는다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(SC, "_dl")
ALPHA = 150.0
GK = ["season", "game_month", "game_dayofweek", "pitcher_team_id",
      "batter_team_id"]

import features44 as F                                          # noqa: E402

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["row_id", "batter_id", "balls_before", "strikes_before",
                          "outs_before", "inning", "top_bottom", "season",
                          "runner_on_1b", "runner_on_2b", "runner_on_3b",
                          "run_total_before"] + GK)
n = len(tr)
b = tr["balls_before"].to_numpy(np.int16)
s = tr["strikes_before"].to_numpy(np.int16)
o = tr["outs_before"].to_numpy(np.int16)
R = (tr["runner_on_1b"].fillna(0).to_numpy(np.int16)
     + tr["runner_on_2b"].fillna(0).to_numpy(np.int16)
     + tr["runner_on_3b"].fillna(0).to_numpy(np.int16))
S = tr["run_total_before"].fillna(0).to_numpy(np.float64)
bid = tr["batter_id"].to_numpy()

half = pd.util.hash_pandas_object(
    tr[GK + ["inning", "top_bottom"]].astype(str).agg("|".join, axis=1),
    index=False).to_numpy()
new_pa = np.r_[True, (bid[1:] != bid[:-1]) | (b[1:] < b[:-1]) | (s[1:] < s[:-1])]
last = np.r_[new_pa[1:], True]
li = np.where(last)[0]                       # 각 타석의 마지막 투구
fi = np.where(new_pa)[0]                     # 각 타석의 첫 투구
npa = len(li)
print(f"  타석 {npa:,}개  투구/타석 {n/npa:.2f}")

nxt = np.minimum(li + 1, n - 1)
same_half = (half[nxt] == half[li]) & (li + 1 < n)
dO = np.where(same_half, o[nxt] - o[li], 0).astype(np.int32)
dS = np.where(same_half, S[nxt] - S[li], 0.0)
Ra = np.where(same_half, R[nxt], 0).astype(np.int32)
Rb = R[li].astype(np.int32)

ident = (Rb + 1) == (Ra + dS + dO)
print(f"  몸 수 항등식이 맞는 타석 {ident.sum():,} / {npa:,} "
      f"({ident.mean()*100:.1f}%)   하프이닝 끝은 {(~same_half).sum():,}개")

# 하프이닝을 끝낸 타석은 항등식을 못 쓰지만 거의 전부 3아웃이다.
# 이걸 분모에서 빼면 아웃만 사라져 출루율이 0.45 로 부풀었다. 아웃으로 센다.
reached = ident & same_half & (dO == 0) & ((Ra + dS) > Rb)
use = (ident & same_half) | (~same_half)
print(f"  판정 가능한 타석 {use.sum():,} ({use.mean()*100:.1f}%)")
print(f"  복원 출루율 {reached[use].mean():.4f}   "
      f"(KBO 실제 0.35~0.37)")
for yr in sorted(tr["season"].unique()):
    m = use & (tr["season"].to_numpy()[li] == yr)
    if m.sum() > 5000:
        print(f"    {yr}  {reached[m].mean():.4f}  ({m.sum():,}타석)")

# ---- as-of 타자 출루율 (그 타석보다 앞선 타석만)
pab = pd.DataFrame({"bid": bid[li], "ok": reached.astype(float),
                    "use": use.astype(float)})
cn = pab.groupby("bid")["ok"].cumsum() - pab["ok"]
cd = pab.groupby("bid")["use"].cumsum() - pab["use"]
pri = reached[use].mean()
obp_pa = ((cn + ALPHA * pri) / (cd + ALPHA)).to_numpy()
# 타석 값을 그 타석의 모든 투구로 펼친다
obp = np.repeat(obp_pa, np.diff(np.r_[fi, n]))
nobs = np.repeat(cd.to_numpy(), np.diff(np.r_[fi, n]))
print(f"  as-of 표본 중앙 {np.median(nobs):,.0f}타석   "
      f"출루율 분포 {np.percentile(obp,[5,50,95]).round(4).tolist()}")

# ---- 44열 통제 편상관 + 44열의 타자 열들과 얼마나 다른가
d = F.build(DATA, VS=2025, return_frame=True)
assert (d["frame"]["row_id"].to_numpy() == tr["row_id"].to_numpy()).all()
X44, y, m_tr = d["X44"].astype(np.float64), d["y"].astype(np.float64), d["m_tr"]
F44 = list(d["F44"])
for c in ("asof_batter_success_rate", "asof_batter_n", "asof_batter_middle_rate"):
    if c in F44:
        v = X44[:, F44.index(c)]
        ok = np.isfinite(v) & m_tr
        print(f"  obp vs {c:28s} 상관 {np.corrcoef(obp[ok], v[ok])[0,1]:+.4f}")

Xc = np.where(np.isnan(X44), np.nanmedian(X44[m_tr], 0), X44)[m_tr]
Xc = np.column_stack([Xc, np.ones(len(Xc))])


def resid(v):
    bb, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return v - Xc @ bb


ry = resid(y[m_tr])
rv = resid(obp[m_tr])
r = float((rv * ry).mean() / (rv.std() * ry.std() + 1e-30))
print(f"\n  44열 통제 편상관  obp {r:+.5f}   "
      f"{'통과' if abs(r) > 0.005 else '미달 (임계 0.005)'}")
np.save(os.path.join(DL, "obp_asof.npy"), obp.astype(np.float32))
print("  저장  colab/_dl/obp_asof.npy")
