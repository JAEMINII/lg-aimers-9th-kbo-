# -*- coding: utf-8 -*-
"""GBDT 계열을 각자 **제 설정**으로 다시 재운다. 앞선 표가 불공정했다.

무엇이 불공정했나
    model_table.py 에서 넷을 같은 예산(400라운드·max_depth=4·lr 0.05)으로 맞췄다.
    그런데 max_depth=4 는 CatBoost 의 대칭 트리에는 자연스럽지만
        LightGBM  leaf-wise 성장인데 깊이를 4로 묶으면 잎이 16개로 갇힌다
        XGBoost   depth-wise 라 6~8 이 통상 기본이다
    라 나머지 셋을 목 조른 셈이다. 로지스틱 회귀(1군 819.6)가 LightGBM·HistGB·
    XGBoost 를 전부 이긴 게 그 증거다. **같은 예산이 같은 수렴은 아니다.**

이번 기준
    계열마다 통상적인 설정을 주고 라운드를 넉넉히 준다. 튜닝은 여전히 안 한다.
    "각자 상식적인 기본값" 이 계열 비교로는 '같은 예산' 보다 공정하다.

공통
    피처 44열 / 학습 season<2024 / 채점 2024 / 최적 시프트 / 시즌가중 2.0 / 시드 2개
"""
import os
import sys
import time

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
VS, SEEDS, DECAY = 2024, (42, 1), 2.0

import features44 as F                                          # noqa: E402

d = F.build(DATA, VS=VS)
X = d["X44"].astype(np.float64)
y = d["y"].astype(np.float64)
season = d["season"].astype(int)
isf, m_tr = d["is_f"], d["m_tr"]
gate = np.where(season == VS)[0]
yv, isf_g = y[gate], isf[gate]
med = np.nanmedian(X[m_tr], 0)
Xf = np.where(np.isnan(X), med, X)
yi = y.astype(int)
w = DECAY ** (season[m_tr].astype(np.float64) - 2019)
print(f"  학습 {int(m_tr.sum()):,}  채점 {len(gate):,}  시드 {len(SEEDS)}개\n")

import lightgbm as lgb                                          # noqa: E402
import xgboost as xgb                                           # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier     # noqa: E402

RES = []


def run(nm, fn, note=""):
    t0 = time.time()
    ps = [fn(s) for s in SEEDS]
    p = np.mean(ps, 0)
    a = F.best_shift(p, yv)[0]
    b = F.best_shift(p[~isf_g], yv[~isf_g])[0]
    c = F.best_shift(p[isf_g], yv[isf_g])[0]
    RES.append((nm, a, b, c, note))
    print(f"  {nm:30s} {a:8.1f}  1군 {b:8.1f}  퓨처스 {c:8.1f}  "
          f"{time.time()-t0:5.0f}s  {note}")


run("CatBoost d4 it400 (배치)",
    lambda s: CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4,
                                 l2_leaf_reg=1.0, verbose=0, random_seed=s,
                                 allow_writing_files=False, thread_count=8)
    .fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    .predict_proba(Xf[gate])[:, 1], "현행 설정")

run("CatBoost d6 it1500",
    lambda s: CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6,
                                 l2_leaf_reg=3.0, verbose=0, random_seed=s,
                                 allow_writing_files=False, thread_count=8)
    .fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    .predict_proba(Xf[gate])[:, 1], "깊고 길게")

run("LightGBM leaves63 it1500",
    lambda s: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05,
                                 num_leaves=63, max_depth=-1, reg_lambda=3.0,
                                 min_child_samples=100, colsample_bytree=0.8,
                                 subsample=0.8, subsample_freq=1,
                                 random_state=s, n_jobs=8, verbose=-1)
    .fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    .predict_proba(Xf[gate])[:, 1], "leaf-wise 해제")

run("LightGBM leaves127 it1500",
    lambda s: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05,
                                 num_leaves=127, max_depth=-1, reg_lambda=5.0,
                                 min_child_samples=200, colsample_bytree=0.8,
                                 subsample=0.8, subsample_freq=1,
                                 random_state=s, n_jobs=8, verbose=-1)
    .fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    .predict_proba(Xf[gate])[:, 1], "더 넓게")

run("XGBoost d6 it1500",
    lambda s: xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05,
                                max_depth=6, reg_lambda=3.0, min_child_weight=20,
                                subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", random_state=s, n_jobs=8,
                                eval_metric="logloss")
    .fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    .predict_proba(Xf[gate])[:, 1], "통상 깊이")

run("XGBoost d8 it1000",
    lambda s: xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05,
                                max_depth=8, reg_lambda=5.0, min_child_weight=50,
                                subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", random_state=s, n_jobs=8,
                                eval_metric="logloss")
    .fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    .predict_proba(Xf[gate])[:, 1], "더 깊게")

run("HistGB leaves63 it1500",
    lambda s: HistGradientBoostingClassifier(max_iter=1500, learning_rate=0.05,
                                             max_leaf_nodes=63, max_depth=None,
                                             l2_regularization=3.0,
                                             min_samples_leaf=100,
                                             random_state=s, early_stopping=False)
    .fit(Xf[m_tr], yi[m_tr], sample_weight=w)
    .predict_proba(Xf[gate])[:, 1], "잎 제한 해제")

print("\n" + "=" * 84)
print(f"  GBDT 재측정 — 계열마다 제 설정. 학습 season<{VS} / 채점 {VS}")
print("=" * 84)
print(f"  {'모델':30s} {'전체':>8s} {'1군':>9s} {'퓨처스':>9s}  비고")
for nm, a, b, c, note in sorted(RES, key=lambda t: -t[1]):
    print(f"  {nm:30s} {a:8.1f} {b:9.1f} {c:9.1f}  {note}")
print("\n  참고 — 같은 폴드에서 TabM 930.4 / MNCA 897.5 / 로지스틱 744.2 였다.")
print("  앞선 표의 LightGBM 773.6 / XGBoost 756.0 / HistGB 760.3 은 max_depth=4 에")
print("  묶여 과소평가였다. 여기 숫자가 그 계열들의 실제 실력에 가깝다.")
