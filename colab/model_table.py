# -*- coding: utf-8 -*-
"""지금까지 재본 모델을 **하나의 잣대**로 다시 세운다.

문제
    기록된 숫자들이 공정하지 않다. 시기·설정·시드·폴드가 제각각이고 일부는
    스케줄러 결함 시절(코사인을 배치마다 돌려 2에폭 중 후반을 버리던 때) 값이다.

통일한 것
    피처   features44 의 44열 (TabM 이 쓰는 것과 같다)
    학습   season < 2024      채점   season == 2024 (253,507행)
    점수   최적 시프트에서의 값 (수준 보정과 판별력이 섞이지 않게)
    시드   3개, 짝비교 가능하도록 같은 시드 집합

통일하지 못한 것 (표에 명시한다)
    GPU 계열(TabM/MNCA/FT-T/ResNet/TabR/TabPFN)은 지금 GPU 가 야간 스윕으로
    차 있어 재학습을 못 한다. **저장된 예측을 같은 잣대로 채점만** 했다.
    그래서 폴드·채점은 공정해졌지만 튜닝 노력과 학습 조건은 여전히 다르다.

    CPU 계열(CatBoost/HistGB/XGBoost/LightGBM/로지스틱)은 **여기서 새로 학습**한다.
    같은 예산(400라운드·depth 4·lr 0.05)으로 맞추고 아무것도 튜닝하지 않는다.
    '튜닝 안 한 같은 예산' 이 계열 간 비교로는 가장 공정한 축이다.
"""
import os
import sys
import time

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(SC, "_dl")
VS = 2024
SEEDS = (42, 1, 777)
DECAY = 2.0
N_EST, DEPTH, LR, L2 = 400, 4, 0.05, 1.0

import features44 as F                                          # noqa: E402

d = F.build(DATA, VS=VS)
X = d["X44"].astype(np.float64)
y = d["y"].astype(np.float64)
season = d["season"].astype(int)
isf = d["is_f"]
m_tr = d["m_tr"]
gate = np.where(season == VS)[0]
yv, isf_g = y[gate], isf[gate]
g0 = int(gate.min())
med = np.nanmedian(X[m_tr], 0)
Xf = np.where(np.isnan(X), med, X)
w = DECAY ** (season[m_tr].astype(np.float64) - 2019)
print(f"  학습 {int(m_tr.sum()):,}  채점 {len(gate):,}  피처 {X.shape[1]}\n")


def score(p, m=None):
    m = np.ones(len(p), bool) if m is None else m
    return F.best_shift(np.asarray(p, np.float64)[m], yv[m])[0]


RES = []


def add(nm, ps, note=""):
    p = np.mean(ps, 0) if isinstance(ps, list) else ps
    RES.append((nm, len(ps) if isinstance(ps, list) else 1,
                score(p), score(p, ~isf_g), score(p, isf_g), p, note))
    print(f"  {nm:26s} {score(p):8.1f}  {note}")


# ------------------------------------------------------- 저장된 GPU 계열 채점
def load(stem):
    ps = [np.load(f"{DL}/{stem}_s{s}.npy") for s in SEEDS
          if os.path.exists(f"{DL}/{stem}_s{s}.npy")]
    if not ps and os.path.exists(f"{DL}/{stem}.npy"):
        ps = [np.load(f"{DL}/{stem}.npy")]
    return ps


print("  [저장된 예측 — 채점만 통일]")
for nm, stem, note in (
        ("TabM (배치 계열)", "sn_2024_base", "현행"),
        ("MNCA (c=16384)", "mnca2_c16384", ""),
        ("FT-Transformer", "ftt_ftt", ""),
        ("ResNet", "ftt_resnet", ""),
        ("TabR (linear_relu)", "tabr_linear_relu_c32768", "시드 1개"),
        ("TabR (PLR)", "tabr_plr_c32768", "예측평균 0.550 — 망가진 실행")):
    ps = load(stem)
    if ps:
        add(nm, ps, note)

# ------------------------------------------------------- CPU 계열 재학습
print("\n  [CPU 계열 — 같은 예산으로 새로 학습]")
from catboost import CatBoostClassifier                         # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier     # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
import lightgbm as lgb                                          # noqa: E402
import xgboost as xgb                                           # noqa: E402

yi = y.astype(int)


def cat(seed, decay):
    ww = decay ** (season[m_tr].astype(np.float64) - 2019)
    m = CatBoostClassifier(iterations=N_EST, learning_rate=LR, depth=DEPTH,
                           l2_leaf_reg=L2, verbose=0, random_seed=seed,
                           allow_writing_files=False, thread_count=8)
    m.fit(Xf[m_tr], yi[m_tr], sample_weight=ww)
    return m.predict_proba(Xf[gate])[:, 1]


t0 = time.time()
add("CatBoost (decay 2.0)", [cat(s, 2.0) for s in SEEDS], "배치 설정")
print(f"      {time.time()-t0:.0f}s")
add("CatBoost (decay 1.0)", [cat(s, 1.0) for s in SEEDS], "시즌가중 없음")


def hgb(seed):
    m = HistGradientBoostingClassifier(max_iter=N_EST, learning_rate=LR,
                                       max_depth=DEPTH, l2_regularization=L2,
                                       random_state=seed, early_stopping=False)
    m.fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    return m.predict_proba(Xf[gate])[:, 1]


add("HistGB (decay 2.0)", [hgb(s) for s in SEEDS])


def xg(seed):
    m = xgb.XGBClassifier(n_estimators=N_EST, learning_rate=LR,
                          max_depth=DEPTH, reg_lambda=L2, tree_method="hist",
                          random_state=seed, n_jobs=8, eval_metric="logloss")
    m.fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    return m.predict_proba(Xf[gate])[:, 1]


add("XGBoost (decay 2.0)", [xg(s) for s in SEEDS])


def lg(seed):
    m = lgb.LGBMClassifier(n_estimators=N_EST, learning_rate=LR,
                           max_depth=DEPTH, reg_lambda=L2, random_state=seed,
                           n_jobs=8, verbose=-1)
    m.fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    return m.predict_proba(Xf[gate])[:, 1]


add("LightGBM (decay 2.0)", [lg(s) for s in SEEDS])

mu, sd = Xf[m_tr].mean(0), Xf[m_tr].std(0) + 1e-9
lr_m = LogisticRegression(max_iter=200, n_jobs=8)
lr_m.fit((Xf[m_tr] - mu) / sd, yi[m_tr], sample_weight=w)
add("로지스틱 회귀", lr_m.predict_proba((Xf[gate] - mu) / sd)[:, 1], "선형 기준선")

j = list(d["F44"]).index("asof_pitcher_success_rate")
add("투수 성공률 한 열", np.nan_to_num(X[gate, j], nan=float(np.nanmean(X[m_tr, j]))),
    "가장 단순한 기준선")

# ------------------------------------------------------- TabPFN (부분표본)
if os.path.exists(f"{DL}/tabpfn_smoke.npy"):
    ii = np.load(f"{DL}/tabpfn_smoke_idx.npy") - g0     # 전체학습 -> 관문 상대
    pp = np.load(f"{DL}/tabpfn_smoke.npy")
    ok = (ii >= 0) & (ii < len(yv))
    tb = RES[0][5]
    print(f"\n  [TabPFN — {int(ok.sum()):,}행 부분표본이라 따로]")
    print(f"    TabPFN {F.best_shift(pp[ok], yv[ii[ok]])[0]:8.1f}"
          f"    같은 행에서 TabM {F.best_shift(tb[ii[ok]], yv[ii[ok]])[0]:8.1f}")

# ------------------------------------------------------- 표
tabm = RES[0][5]
print("\n" + "=" * 88)
print(f"  모델 비교 — 학습 season<{VS} / 채점 {VS} / 최적 시프트 / 44열 공통")
print("=" * 88)
print(f"  {'모델':26s} {'시드':>4s} {'전체':>9s} {'1군':>9s} {'퓨처스':>9s} "
      f"{'TabM상관':>9s}  비고")
for nm, ns, a, b, c, p, note in sorted(RES, key=lambda t: -t[2]):
    r = float(np.corrcoef(p, tabm)[0, 1])
    print(f"  {nm:26s} {ns:>4d} {a:9.1f} {b:9.1f} {c:9.1f} {r:9.4f}  {note}")
print("\n  CPU 계열은 여기서 같은 예산(400라운드·depth4·lr0.05·튜닝없음)으로 학습했다.")
print("  GPU 계열은 저장된 예측을 같은 잣대로 채점만 했다 — 학습 조건은 여전히 다르다.")
np.save(os.path.join(DL, "table_preds.npy"),
        np.array([r[5] for r in RES], dtype=np.float32))
