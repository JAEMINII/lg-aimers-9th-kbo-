# -*- coding: utf-8 -*-
"""CatBoost 쪽 plat_dev 누출도 재본다. CPU 전용이라 GPU 학습과 안 다툰다.

배경
    TabM 쪽 plat_dev 누출을 고쳐서 리더보드 1058 -> 1069 (+11) 였다.
    그런데 CatBoost 도 같은 룩업을 쓴다.

        trees.npz   plat_a (792)  plat_l (792)  plat_r (792)   투수 단위
    배치된 CatBoost 는 이 누출 테이블로 학습됐다.

    그리고 우리 관문 CatBoost(cb_gate.npy)는 features44 를 쓰는데 거기 plat_dev 는
    **이미 as-of** 다. 즉 **재는 것과 싣는 것이 다르다.** 지금까지 CatBoost 를
    관문에서 평가한 값은 배치본보다 좋은 판본이었던 셈이다.

무엇을 재나 (VS=2024, 시즌가중 2.0, 400라운드, depth 4, 3시드)
    asof    features44 의 plat_dev (as-of)          <- 관문에서 쓰던 것
    leaky   지인 preprocess 의 plat_dev (누출)       <- 배치된 것
    나머지 43열은 두 경우 완전히 동일하다 (상관 1.00000 확인함).

판정
    단독 점수와 **혼합 기여**를 둘 다 본다. 총 GBDT 비중 0.30 을 고정하고
    TabM 과 섞었을 때의 차이가 배치에서 실제로 얻을 값이다.
    HistGB 를 +14.7 인 줄 알았다가 총비중 고정하니 +0.1 이었던 전례가 있다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"
VS = 2024
DECAY = 2.0
SEEDS = (42, 1234, 2025)

import features44 as F                                          # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

d = F.build(DATA, VS=VS)
X = d["X44"].astype(np.float64)
y = d["y"]
m_tr, m_va = d["m_tr"], d["m_va"]
F44 = list(d["F44"])
season = d["season"].astype(np.float64)
isf = d["is_f"]
gate = np.where(m_va)[0]
yv = y[gate].astype(np.float64)
isf_g = isf[gate]
R = ~isf_g

# 지인 판본 plat_dev 를 같은 창에서 만든다
tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                          encoding="utf-8-sig"))
hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id"])
pos = pd.Series(np.arange(len(tr_sorted)), index=tr_sorted["row_id"].to_numpy())
leaky = Xs["plat_dev"].to_numpy(np.float64)[
    pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]

j = F44.index("plat_dev")
asof = X[:, j].copy()
print(f"\n  두 판본 상관 {np.corrcoef(asof, leaky)[0,1]:.4f}   "
      f"최대차 {np.abs(asof-leaky).max():.4f}")
print(f"  표적상관   as-of {np.corrcoef(asof[m_tr], y[m_tr])[0,1]:+.5f}   "
      f"leaky {np.corrcoef(leaky[m_tr], y[m_tr])[0,1]:+.5f}\n")

med = np.nanmedian(X[m_tr], 0)
w = DECAY ** (season[m_tr] - 2019)


def run(vals):
    Xv = X.copy()
    Xv[:, j] = vals
    Xv = np.where(np.isnan(Xv), med, Xv)
    ps = []
    for s in SEEDS:
        m = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4,
                               l2_leaf_reg=1.0, verbose=0, random_seed=s,
                               allow_writing_files=False, thread_count=6)
        m.fit(Xv[m_tr], y[m_tr].astype(int), sample_weight=w)
        ps.append(m.predict_proba(Xv[gate])[:, 1].astype(np.float64))
    return ps


def sc(p, msk=None):
    msk = np.ones(len(yv), bool) if msk is None else msk
    return F.best_shift(p[msk], yv[msk])[0]


P = {}
for nm, vals in (("asof", asof), ("leaky", leaky)):
    t0 = time.time()
    P[nm] = run(vals)
    p = np.mean(P[nm], 0)
    print(f"  {nm:6s} 단독 {sc(p):7.1f}  1군 {sc(p, R):7.1f}  "
          f"퓨처스 {sc(p, isf_g):7.1f}   {time.time()-t0:.0f}s")

dd = [sc(a) - sc(b) for a, b in zip(P["asof"], P["leaky"])]
mu = float(np.mean(dd)); se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
print(f"\n  단독 짝차이  {mu:+.1f} +- {se:.1f}  t={mu/max(se,1e-9):.2f}  "
      f"{sum(1 for x in dd if x>0)}/{len(dd)}   개별 [{', '.join(f'{v:+.0f}' for v in dd)}]")

# ---- 혼합 기여: GBDT 총량 0.30 고정
OUT = os.path.join(SC, "_dl")
tm = None
for cand in ("sc_s42.npy",):
    p = os.path.join(OUT, cand)
    if os.path.exists(p):
        tm = np.load(p)
if tm is None:
    print("\n  TabM 관문 예측이 로컬에 없어 혼합 기여는 서버에서 잰다.")
else:
    r0 = sc(0.30 * np.mean(P["leaky"], 0) + 0.70 * tm)
    v = sc(0.30 * np.mean(P["asof"], 0) + 0.70 * tm)
    print(f"\n  혼합 (CB 0.30 / TabM 0.70)   leaky {r0:.1f}  asof {v:.1f}  "
          f"차이 {v-r0:+.1f}")

print("\n  단독이 올라도 총비중 고정 혼합에서 안 오르면 배치 이득이 없다.")
