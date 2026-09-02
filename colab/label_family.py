# -*- coding: utf-8 -*-
"""복원한 범주 라벨 x 상황 축 = 계열 전체를 스크린한다.

발견의 흐름
    김광현 산점도의 축이 (crossPlateX, 계산된 plate_z) 다 — 판정 범주는 홈플레이트
    통과 **위치**로 정의된다. middle 은 좌우 중앙에 몰린 공, reverse 는 크게 벗어난 공.
    위치는 트랙맨 물리량(릴리스 + 무브먼트 + 속도)으로 유도할 수 있지만 **2025 는 없다.**

    그래서 위치 자체가 아니라 **그 위치가 만든 범주 라벨**을 쓴다. 라벨은 as-of
    누적 비율에서 99.94% 복원된다 (같은 투수, asof_pitcher_n 이 1 증가한 쌍).
    그리고 라벨 기반 프로필은 **조회표로 2025 행에 붙는다** — 투수 id 와 상황 키만
    있으면 되고 둘 다 test.csv 에 있다.

계열
    plat_dev = P(success | 투수, 타자손) - P(success | 투수)
    이게 이 프로젝트에서 점수를 올린 **유일한 피처**다 (편상관 +0.0148, LB +11).
    그게 이 계열의 한 칸이었다는 걸 오늘 알았다. 나머지 칸은 아무도 안 봤다.

        라벨   success / reverse / middle / ball / strike
        축     타자손 / 카운트 / 이닝 / 주자상황 / 아웃
    축은 전부 test.csv 에 있는 열이어야 한다 (조회표로 붙이려면).

만드는 법 — plat_dev 규약 그대로
    (투수, 시즌, 축값) 누적 -> 시즌 단위 shift(1) 로 as-of
    ALPHA 로 (투수손, 축값)별 리그값 쪽으로 수축
    편차 = P(라벨 | 투수, 축값) - P(라벨 | 투수)

주의
    스크린은 거르는 도구지 맞히는 도구가 아니다. 오늘 pcdev 가 -0.0077 로 통과했다가
    관문에서 졌다. 통과분은 관문으로 다시 재야 한다.
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
THR = 0.005

import features44 as F                                          # noqa: E402

RATES = {"success": "asof_pitcher_success_rate",
         "reverse": "asof_pitcher_reverse_rate",
         "middle": "asof_pitcher_middle_rate",
         "ball": "asof_pitcher_ball_rate",
         "strike": "asof_pitcher_strike_rate"}
raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["row_id", "pitcher_id", "batter_hand", "pitcher_hand",
                           "season", "asof_pitcher_n", "control_success",
                           "balls_before", "strikes_before", "inning",
                           "outs_before", "base_state"] + list(RATES.values()))
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
agree = float((LAB["success"][m] == y[m]).mean())
print(f"  라벨 복원 {int(m.sum()):,}행 ({m.mean()*100:.1f}%)   "
      f"success 일치 {agree:.6f}")
if agree < 0.9999:
    raise ValueError("복원 검증 실패")
for nm in RATES:
    print(f"    {nm:8s} 평균 {np.nanmean(LAB[nm]):.4f}")

season = raw["season"].to_numpy()
ph = raw["pitcher_hand"].to_numpy()
AX = {
    "bh": raw["batter_hand"].to_numpy(),
    "cnt": (raw["balls_before"].astype(str) + "-"
            + raw["strikes_before"].astype(str)).to_numpy(),
    "inn": np.clip(raw["inning"].fillna(1).astype(int), 1, 9).to_numpy(),
    "base": raw["base_state"].astype(str).to_numpy(),
    "out": raw["outs_before"].astype(int).to_numpy(),
}
print(f"\n  축별 수준 수  " + "  ".join(
    f"{k}:{len(np.unique(v))}" for k, v in AX.items()))

d = F.build(DATA, VS=2025, return_frame=True)
assert (d["frame"]["row_id"].to_numpy() == raw["row_id"].to_numpy()).all()
X44, m_tr = d["X44"].astype(np.float64), d["m_tr"]


def build(lab, axname):
    """P(라벨 | 투수, 축값) - P(라벨 | 투수).  전부 as-of."""
    v = LAB[lab]
    have = np.isfinite(v)
    a = AX[axname]
    df = pd.DataFrame({"pid": pid, "season": season, "a": a,
                       "v": np.where(have, v, 0.0), "n": have.astype(float)})
    g = df.groupby(["pid", "season", "a"])[["v", "n"]].sum().unstack("a",
                                                                     fill_value=0.0)
    cum = g.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    levels = list(cum["v"].columns)
    tot_v = cum["v"].sum(axis=1)
    tot_n = cum["n"].sum(axis=1)
    idx = pd.MultiIndex.from_arrays([pid, season])
    sub_v = cum["v"].reindex(idx)
    sub_n = cum["n"].reindex(idx)
    pos = {L: i for i, L in enumerate(levels)}
    col = np.array([pos.get(x, -1) for x in a])
    take_v = np.full(len(a), np.nan)
    take_n = np.full(len(a), np.nan)
    Vm, Nm = sub_v.to_numpy(), sub_n.to_numpy()
    okc = col >= 0
    take_v[okc] = Vm[np.arange(len(a))[okc], col[okc]]
    take_n[okc] = Nm[np.arange(len(a))[okc], col[okc]]
    # 사전값 — (투수손, 축값)별 리그값 / (투수손)별 리그값
    lg = df.assign(ph=ph).groupby(["ph", "a"]).apply(
        lambda t: t["v"].sum() / max(t["n"].sum(), 1.0), include_groups=False)
    lgp = df.assign(ph=ph).groupby("ph").apply(
        lambda t: t["v"].sum() / max(t["n"].sum(), 1.0), include_groups=False)
    pho = pd.Series(ph, index=pid).groupby(level=0).first().reindex(pid).to_numpy()
    pr_h = np.array([lg.get((p_, x_), np.nan) for p_, x_ in zip(pho, a)])
    pr_a = np.array([lgp.get(p_, np.nan) for p_ in pho])
    P_h = (take_v + ALPHA * pr_h) / (take_n + ALPHA)
    P_a = (tot_v.reindex(idx).to_numpy() + ALPHA * pr_a) / \
          (tot_n.reindex(idx).to_numpy() + ALPHA)
    return np.nan_to_num(P_h - P_a)


med = np.nanmedian(X44[m_tr], 0)
Xc = np.where(np.isnan(X44), med, X44)[m_tr]
Xc = np.column_stack([Xc, np.ones(len(Xc))])


def resid(v):
    b, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return v - Xc @ b


ry = resid(y[m_tr])
sy = ry.std()
print(f"\n{'='*74}\n  44열 통제 편상관   임계 |r| > {THR}"
      f"   (참고 plat_dev +0.0148 = success x bh)\n{'='*74}")
print(f"  {'라벨':10s}" + "".join(f"{a:>11s}" for a in AX))
C, rows = {}, []
for lab in RATES:
    line = f"  {lab:10s}"
    for axn in AX:
        v = build(lab, axn)
        C[f"{lab}_{axn}"] = v
        vv = v[m_tr]
        if vv.std() < 1e-12:
            line += f"{'-':>11s}"
            continue
        rv = resid(vv)
        r = float((rv * ry).mean() / (rv.std() * sy + 1e-30))
        rows.append((f"{lab}_{axn}", r))
        line += f"{r:>+10.5f}" + ("*" if abs(r) > THR else " ")
    print(line)
rows.sort(key=lambda t: -abs(t[1]))
okl = [c for c, r in rows if abs(r) > THR]
print(f"\n  통과 {len(okl)}개 / {len(rows)}개   (* 표시)")
for c, r in rows[:8]:
    print(f"    {c:20s} {r:+.5f}")
if okl:
    out = pd.DataFrame({c: C[c] for c in okl})
    out.insert(0, "row_id", raw["row_id"].to_numpy())
    out.to_csv(os.path.join(SC, "_dl", "label_family.csv.gz"), index=False)
    print(f"\n  저장  colab/_dl/label_family.csv.gz  ({len(okl)}열)")
print("\n  스크린은 거르는 도구다. 통과분은 관문으로 다시 재야 한다 —")
print("  오늘 pcdev 가 -0.0077 로 통과했다가 관문에서 졌다.")
