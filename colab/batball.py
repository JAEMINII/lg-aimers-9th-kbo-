# -*- coding: utf-8 -*-
"""타자별 볼/볼넷/삼진 비율을 투구 순서에서 복원하고 스크린한다.

왜 이게 OPS 보다 낫나
    OPS 는 장타율 때문에 주자 진루를 추정해야 하고 야수선택·병살·희생타에서
    오차가 남는다. 반면 볼/스트라이크는 **타석 안 카운트 변화로 정확히 복원**된다.

    그리고 44열을 보면
        asof_pitcher_ball_rate / strike_rate   있다
        타자쪽 대응물                          **없다**
    타자 열은 n / success_rate / middle_rate 셋뿐이다.

    볼넷을 많이 얻는 타자는 투수가 존을 못 잡는 타자다. 재민님이 말한
    '뛰어난 타자' 를 제구 관점에서 직접 잰 값이고 표적과 가깝다.

복원 규칙
    타석 경계   타자가 바뀌거나 볼/스트라이크가 되감기면 새 타석
    투구 결과   같은 타석 다음 투구에서 balls 가 오르면 볼, strikes 가 오르면 스트라이크
    볼넷        타석 마지막 투구가 balls_before==3 이고 아웃이 안 늘었다
    삼진        타석 마지막 투구가 strikes_before==2 이고 아웃이 늘었다
    마지막 투구 자체의 볼/스트라이크는 모른다 -> 비율 계산에서 뺀다

누출 방지
    전부 as-of 누적이다. 그 행보다 **앞선 행**만 쓴다 (cumsum 후 shift(1)).
    44열의 asof_* 와 같은 규약이다.
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
THR, ALPHA = 0.005, 200.0

import features44 as F                                          # noqa: E402

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["row_id", "batter_id", "balls_before",
                          "strikes_before", "outs_before", "inning",
                          "top_bottom", "season"])
n = len(tr)
b = tr["balls_before"].to_numpy(np.int16)
s = tr["strikes_before"].to_numpy(np.int16)
o = tr["outs_before"].to_numpy(np.int16)
bid = tr["batter_id"].to_numpy()
new_pa = np.r_[True, (bid[1:] != bid[:-1]) | (b[1:] < b[:-1]) | (s[1:] < s[:-1])]
pa = np.cumsum(new_pa)
last = np.r_[new_pa[1:], True]                 # 타석의 마지막 투구
print(f"  {n:,}투구  타석 {pa.max():,}개  투구/타석 {n/pa.max():.2f}")

# ---- 투구 결과 (마지막 투구 제외)
ball = np.zeros(n, bool)
strike = np.zeros(n, bool)
nxt = np.r_[np.arange(1, n), n - 1]
same = ~last
ball[same] = b[nxt[same]] > b[same]
strike[same] = s[nxt[same]] > s[same]
known = same & (ball | strike)
print(f"  결과를 아는 투구 {known.sum():,} ({known.mean()*100:.1f}%)  "
      f"볼 {ball[known].mean()*100:.1f}%")

# ---- 타석 결과
o_nxt = o[nxt]
out_made = last & (o_nxt > o)                  # 같은 하프이닝이면 아웃 증가
walk = last & (b == 3) & ~out_made
punch = last & (s == 2) & out_made
print(f"  볼넷 {walk.sum():,} ({walk.sum()/pa.max()*100:.1f}% of 타석)  "
      f"삼진 {punch.sum():,} ({punch.sum()/pa.max()*100:.1f}%)  "
      f"(KBO 실제 볼넷 9~10%, 삼진 18~19%)")

# ---- as-of 누적 (그 행보다 앞선 행만)
def asof(num, den, key):
    N = pd.Series(num.astype(np.float64)).groupby(key).cumsum() \
        - num.astype(np.float64)
    D = pd.Series(den.astype(np.float64)).groupby(key).cumsum() \
        - den.astype(np.float64)
    pri = float(num.sum()) / max(float(den.sum()), 1.0)
    return ((N + ALPHA * pri) / (D + ALPHA)).to_numpy(), D.to_numpy()


key = pd.Series(bid)
bal, nb = asof(ball & known, known, key)
wk, npa_ = asof(walk, last, key)
pk, _ = asof(punch, last, key)
print(f"  as-of 표본 중앙 {np.median(nb):,.0f}투구")

C = {"bat_ball_rate": bal, "bat_bb_rate": wk, "bat_k_rate": pk}

# ---- 편차형도 만든다 (오늘 통한 형태). 타자 x 카운트
cnt = pd.Series(b.astype(str)).str.cat(pd.Series(s.astype(str)), sep="-")
gb = pd.DataFrame({"bid": bid, "cnt": cnt.to_numpy(),
                   "ball": (ball & known).astype(float), "kn": known.astype(float)})
cell = gb.groupby(["bid", "cnt"])[["ball", "kn"]].sum()
base = gb.groupby("bid")[["ball", "kn"]].sum()
cv = cell["ball"] / cell["kn"].clip(lower=1)
bv = (base["ball"] / base["kn"].clip(lower=1)).reindex(
    cell.index.get_level_values(0)).to_numpy()
nn = cell["kn"].to_numpy()
dev = pd.Series((cv.to_numpy() - bv) * (nn / (nn + 50.0)), index=cell.index)
C["bc_ball_dev"] = dev.reindex(
    pd.MultiIndex.from_arrays([bid, cnt.to_numpy()])).to_numpy()
C["bc_ball_dev"] = np.nan_to_num(C["bc_ball_dev"])

# ---- 스크린
d = F.build(DATA, VS=2025, return_frame=True)
assert (d["frame"]["row_id"].to_numpy() == tr["row_id"].to_numpy()).all()
X44, y, m_tr = d["X44"].astype(np.float64), d["y"].astype(np.float64), d["m_tr"]
Xc = np.where(np.isnan(X44), np.nanmedian(X44[m_tr], 0), X44)[m_tr]
Xc = np.column_stack([Xc, np.ones(len(Xc))])


def resid(v):
    bb, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return v - Xc @ bb


ry = resid(y[m_tr])
sy = ry.std()
print(f"\n{'='*66}\n  44열 통제 편상관  |r| > {THR}\n{'='*66}")
res = []
for c in C:
    v = np.nan_to_num(C[c])[m_tr]
    rv = resid(v)
    res.append((c, float((rv * ry).mean() / (rv.std() * sy + 1e-30))))
res.sort(key=lambda t: -abs(t[1]))
for c, r in res:
    print(f"  {c:18s} {r:+.5f}  {'통과' if abs(r) > THR else ''}")
ok = [c for c, r in res if abs(r) > THR]
print(f"\n  통과 {len(ok)}개 / {len(res)}개")
if ok:
    out = pd.DataFrame({c: np.nan_to_num(C[c]) for c in ok})
    out.insert(0, "row_id", tr["row_id"].to_numpy())
    out.to_csv(os.path.join(DL, "batball.csv.gz"), index=False)
    print(f"  저장 batball.csv.gz  최강 {res[0][0]} ({res[0][1]:+.5f})")
else:
    print("  타자 볼넷/볼 비율도 44열 위에 얹을 게 없다.")
