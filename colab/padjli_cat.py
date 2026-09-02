# -*- coding: utf-8 -*-
"""p_adj_li 를 CatBoost 에서 다년 백테스트로 재본다.

배경
    제안서가 제시한 피처다. p_adj_cm 과 계산 절차가 같고 절단 축만 li 10분위다.
        li_rel   = 그 구간에서 (control_success ~ p_is_succ) 회귀 기울기 / 전체 기울기
        p_adj_li = gmean + (p_is_succ - gmean) x li_rel
    근거는 CatBoost 3시드에서 BASE 726.70 -> 749.08 (+22.4), permutation 3위(87.9).

내가 처음에 편상관으로 버렸는데 그 도구를 못 믿게 됐다
    편상관은 선형·가법이라 비선형 모델과 안 맞는다. 그리고 오늘 한 폴드 관문으로
    내린 판정이 리더보드에서 세 번 뒤집혔다. 그래서 제안서가 검증한 모델
    (CatBoost)에서 직접, 다년으로 다시 잰다.

    다만 구조적 의심은 남는다 — 학습 구간에서 잰 li_rel 이 전부 1.0 근처였다.
        [0.979 0.989 0.948 1.028 1.000 1.014 1.006 0.983 0.975 1.040]
    li_rel ~ 1 이면 p_adj_li ~ p_is_succ 라 이미 있는 피처의 복제다.
    복제 피처는 permutation importance 가 높게 나온다(하나를 섞어도 나머지가
    자동으로 메워지지 않는다). 그러니 87.9 는 근거가 못 된다.
    이 실험은 그 의심이 맞는지 확인하는 것이다.

규약 (오늘 정한 것)
    폴드 셋 2022 / 2023 / 2024, 같은 시드끼리 짝비교
    최적 시프트와 고정 시프트를 함께 찍는다
    전체(R+F) 채점
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
SEEDS = (42, 1234, 2025)
N_BINS = 10
ALPHA = 50.0
FIXED_SHIFT = -0.0145
HP = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, thread_count=6)

import features44 as F                                          # noqa: E402


def li_tables(li, p_is, y):
    """학습 구간에서만 만든다. 경계도 학습 구간 분위수다(규칙 4)."""
    edges = np.quantile(li, np.linspace(0, 1, N_BINS + 1))[1:-1]
    b = np.digitize(li, edges)
    g = y.mean()
    slope_all = np.cov(p_is, y, bias=True)[0, 1] / p_is.var()
    rel = np.ones(N_BINS)
    for k in range(N_BINS):
        m = b == k
        v = p_is[m].var()
        if m.sum() >= 5000 and v > 1e-8:
            rel[k] = (np.cov(p_is[m], y[m], bias=True)[0, 1] / v) / slope_all
    return edges, np.clip(rel, 0.0, 2.0), g


def run_fold(VS):
    d = F.build(DATA, VS=VS)
    X, y, F44 = d["X44"], d["y"].astype(np.float64), list(d["F44"])
    m_tr, m_va = d["m_tr"], d["m_va"]
    li = X[:, F44.index("li")].astype(np.float64)
    p_is = X[:, F44.index("p_is_succ")].astype(np.float64)
    edges, rel, g = li_tables(li[m_tr], p_is[m_tr], y[m_tr])
    p_adj_li = g + (p_is - g) * rel[np.digitize(li, edges)]

    yv = y[m_va]
    # 우리 실제 CatBoost(submit_12/train_model.py)도 cat_features 를 안 쓴다.
    # encode_categoricals 로 정수 코드화한 값을 수치로 넣는다. 그 조건을 맞춘다.
    res = {}
    for name, Xa in (("base", X),
                     ("+p_adj_li", np.concatenate([X, p_adj_li[:, None].astype(np.float32)], 1))):
        ps = []
        for sd in SEEDS:
            m = CatBoostClassifier(random_seed=sd, **HP)
            m.fit(Xa[m_tr], y[m_tr].astype(int))
            ps.append(m.predict_proba(Xa[m_va])[:, 1])
        res[name] = ps
    return res, yv, rel


if __name__ == "__main__":
    print(f"CatBoost {HP}  시드 {SEEDS}")
    allres = {}
    for VS in (2022, 2023, 2024):
        t0 = time.time()
        res, yv, rel = run_fold(VS)
        allres[VS] = (res, yv)
        print(f"\n===== VS={VS}  검증 {len(yv):,}행  성공률 {yv.mean():.4f} =====")
        print("  li_rel", np.round(rel, 3))
        for name, ps in res.items():
            r = np.mean(ps, 0)
            o = F.best_shift(r, yv)[0]
            fx = F.bss(F.shift(r, FIXED_SHIFT), yv)
            print(f"  {name:11s} 최적 {o:9.1f}   고정 {fx:9.1f}   "
                  f"[{', '.join(f'{F.best_shift(p, yv)[0]:.0f}' for p in ps)}]")
        for tag, f in (("최적", lambda p: F.best_shift(p, yv)[0]),
                       ("고정", lambda p: F.bss(F.shift(p, FIXED_SHIFT), yv))):
            dif = [f(a) - f(b) for a, b in zip(res["+p_adj_li"], res["base"])]
            mu = float(np.mean(dif))
            se = float(np.std(dif, ddof=1)) / np.sqrt(len(dif))
            print(f"    짝차이 {tag}  {mu:+8.1f} ± {se:5.1f}  "
                  f"{sum(1 for x in dif if x > 0)}/{len(dif)}")
        print(f"  ({time.time()-t0:.0f}s)")

    print("\n세 폴드에서 부호가 일관되고 크기가 의미 있어야 채택한다.")
    print("li_rel 이 전부 1.0 근처면 p_adj_li 는 p_is_succ 의 복제다.")
