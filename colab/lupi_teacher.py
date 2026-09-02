# -*- coding: utf-8 -*-
"""LUPI 교사 상한 측정 — 현재 투구의 물리량이 관문 점수를 얼마나 올리나.

이 숫자가 증류로 학생에게 옮길 수 있는 지식의 상한이다.
    작다 (+0~5)   -> 특권 정보가 별 게 없다. 축을 싸게 닫는다
    크다 (+30~)   -> LUPI 가 실체를 갖는다. 증류 단계로 간다

설계
    행: lupi_match 로 현재 투구 물리량이 붙는 1군 행만 (83.6만)
    팔: base      44열 (features44)
        phys      44열 + 물리량 8 + 구종군 코드
        physdev   위 + 투수별 as-of 평균과의 편차 8  (이 공이 평소와 달랐나)
    관문: VS=2024 (학습 <2024, 채점 2024 매칭행), CatBoost 3시드
    주의: physdev 의 투수 평균은 시즌 cumsum-shift 로 as-of (누출 방지).
          교사는 학습 전용이다. 추론에 물리량이 없다는 건 알고 있다 — 그래서 증류다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "colab"))
DATA = os.environ.get("AIMERS_DATA", os.path.join(ROOT, "open (1)", "data"))
DL = os.environ.get("AIMERS_DL", os.path.join(ROOT, "colab", "_dl"))
MATCH_FILE = os.environ.get("LUPI_MATCH", "lupi_match.csv.gz")
TAG = os.environ.get("LUPI_TAG", "")
SEEDS = (1, 42, 777)
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]

import features44 as F                                          # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402

d = F.build(DATA, VS=2024, return_frame=True)
fr = d["frame"]
X44 = d["X44"].astype(np.float32)
names = list(d["F44"])
season = fr["season"].to_numpy()
y = fr["control_success"].to_numpy(np.float64)

M = pd.read_csv(os.path.join(DL, MATCH_FILE), encoding="utf-8-sig")
M["ptg"] = M["pitch_type_group"].map(
    {"fastball": 0, "breaking": 1, "offspeed": 2}).fillna(3)
pos = pd.Series(np.arange(len(fr)), index=fr["row_id"].to_numpy())
mrow = pos.reindex(M["row_id"].to_numpy()).to_numpy()
ok = np.isfinite(mrow)
M, mrow = M[ok], mrow[ok].astype(int)
print(f"  매칭행 {len(M):,}")

P = np.full((len(fr), len(PHYS) + 1), np.nan, np.float32)
P[mrow, :len(PHYS)] = M[PHYS].to_numpy(np.float32)
P[mrow, len(PHYS)] = M["ptg"].to_numpy(np.float32)

# 투수별 as-of 물리량 평균 (시즌 cumsum-shift) -> 이 공의 편차
pid = fr["pitcher_id"].to_numpy()
DEV = np.full((len(fr), len(PHYS)), np.nan, np.float32)
z = pd.DataFrame({"pid": pid, "season": season})
for j, c in enumerate(PHYS):
    z["v"] = P[:, j]
    z["n"] = np.isfinite(P[:, j]).astype(float)
    z["vf"] = np.where(np.isfinite(P[:, j]), P[:, j], 0.0)
    g = z.groupby(["pid", "season"])[["vf", "n"]].sum()
    cc = g.groupby(level=0).cumsum().groupby(level=0).shift(1)
    idx = pd.MultiIndex.from_arrays([pid, season])
    sv = cc["vf"].reindex(idx).to_numpy()
    sn = cc["n"].reindex(idx).to_numpy()
    mu = np.where(np.nan_to_num(sn) >= 50, sv / np.maximum(sn, 1), np.nan)
    DEV[:, j] = P[:, j] - mu
print("  편차 계산 완료  유효률 "
      f"{np.isfinite(DEV[mrow]).all(1).mean()*100:.1f}% (매칭행 기준)")

has = np.zeros(len(fr), bool)
has[mrow] = True
m_tr = (season < 2024) & has
m_te = (season == 2024) & has
yv = y[m_te]
print(f"  학습 {int(m_tr.sum()):,}  관문 {int(m_te.sum()):,}  "
      f"관문 성공률 {yv.mean():.4f}")

ARMS = {
    "base": X44,
    "phys": np.concatenate([X44, P], 1),
    "physdev": np.concatenate([X44, P, DEV], 1),
}
hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False)
sc = lambda p: F.best_shift(p, yv)[0]
R = {}
for nm, X in ARMS.items():
    t0 = time.time()
    ps = []
    for sd in SEEDS:
        mo = CatBoostClassifier(random_seed=sd, **hp)
        mo.fit(X[m_tr], y[m_tr])
        ps.append(mo.predict_proba(X[m_te])[:, 1])
    R[nm] = ps
    stem = f"lupi24_{TAG + '_' if TAG else ''}{nm}.npy"
    np.save(os.path.join(DL, stem), np.asarray(ps))
    print(f"  {nm:8s} {sc(np.mean(ps,0)):8.1f}   {time.time()-t0:5.0f}s", flush=True)
ref = R["base"]
print(f"\n{'='*72}\n  교사 상한 (VS=2024, 매칭행, CatBoost 44열 기준)\n{'='*72}")
for nm in ARMS:
    p = np.mean(R[nm], 0)
    dd = [sc(a) - sc(b) for a, b in zip(R[nm], ref)]
    print(f"  {nm:8s} {sc(p):8.1f}   기준 대비 "
          + " ".join(f"{v:+7.1f}" for v in dd)
          + f"   {sum(1 for v in dd if v > 0)}/3")
print("\n  +30 이상이면 증류 단계로 간다. +5 아래면 축을 닫는다.")
