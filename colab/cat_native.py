# -*- coding: utf-8 -*-
"""CatBoost 에 cat_features 를 켠다. 우리 CatBoost 는 투수를 못 보고 있다.

문제
    지금 pitcher_id(793종) / batter_id(831종) 를 **숫자 열**로 넣는다.
    트리에게 '투수 452번 vs 453번' 은 의미 없는 정수 순서라 사실상 못 쓴다.

    추측이 아니다. lg_f_share(투수 커리어 퓨처스 비중)가 CatBoost 3폴드에서
    전부 양수(+12.2 / +77.8 / +3.5)였는데 TabM 에서는 -27.5 였다.
    TabM 은 793칸 임베딩으로 투수 정체성을 이미 알고, CatBoost 만 몰랐던 것이다.

    cat_features 를 켜면 ordered target statistics 가 그 열들을 누출 통제된
    타깃 인코딩으로 바꾼다. 기전이 명확하고 한 번도 안 해봤다.

함정 — 단독 점수만 보면 안 된다
    CatBoost 가 기여하는 이유는 단독 점수가 아니라 **TabM 과 다른 실수를 하기**
    때문이다. 투수 정체성을 쓰게 만들면 TabM 이 하는 일을 따라하게 되고,
    상관이 오르면서 앙상블 값어치가 떨어질 수 있다.
    전처리를 통일했다가 CatBoost~TabM 상관이 0.894 -> 0.924 로 오르며
    손해 본 전례가 있다.

    그래서 셋을 같이 찍는다.
        단독 점수
        TabM 과의 상관
        **같은 GBDT 총량**에서의 혼합 기여   <- 이게 판정 기준이다

무엇을 재나 (VS=2024, 전체 R+F 채점, 시드 3개)
    base       현행. cat_features 없음, depth 4
    native     cat_features 켬 (범주 9열), 나머지 동일
    native_d7  cat_features + depth 7 (basic 문서의 설정)
    native_sub cat_features + depth 7 + MVS bootstrap subsample 0.8

    범주 열은 features44 의 cat_idx 를 그대로 쓴다.
    CatBoost 는 범주 열을 정수/문자열로 받아야 하므로 float -> int 로 캐스팅한다.
    결측이 있는 범주 열은 별도 코드(-1)로 채운다.
"""
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier, Pool

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
DL = os.path.join(SC, "_dl")
VS = 2024
DECAY = 2.0
SEEDS = (42, 1234, 2025)
BASE = dict(iterations=400, learning_rate=0.05, l2_leaf_reg=1.0, verbose=0,
            allow_writing_files=False, thread_count=6)

import features44 as F                                          # noqa: E402


def build_pool(X, cat_idx, y=None, w=None):
    """범주 열을 정수로 바꿔 Pool 을 만든다. 나머지는 그대로 float."""
    Xo = X.astype(object)
    for j in cat_idx:
        col = X[:, j]
        col = np.where(np.isnan(col), -1.0, col).astype(np.int64)
        Xo[:, j] = col
    return Pool(Xo, label=y, weight=w, cat_features=list(cat_idx))


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    X, y = d["X44"].astype(np.float64), d["y"]
    m_tr, m_va = d["m_tr"], d["m_va"]
    ci = list(d["cat_idx"])
    gate = np.where(m_va)[0]
    yv = y[gate].astype(np.float64)
    isf = d["is_f"][gate]
    w = DECAY ** (d["season"][m_tr].astype(np.float64) - 2019)
    F44 = list(d["F44"])
    print(f"  학습 {m_tr.sum():,}  관문 {len(gate):,}  범주 지정 {len(ci)}열")
    print(f"    {', '.join(F44[j] for j in ci)}\n")

    plans = [("base", dict(depth=4), False),
             ("native", dict(depth=4), True),
             ("native_d7", dict(depth=7), True),
             ("native_sub", dict(depth=7, bootstrap_type="MVS",
                                 subsample=0.8), True)]
    P = {}
    for nm, extra, native in plans:
        t0 = time.time()
        ps = []
        for s in SEEDS:
            hp = dict(BASE, random_seed=s, **extra)
            m = CatBoostClassifier(**hp)
            if native:
                m.fit(build_pool(X[m_tr], ci, y[m_tr].astype(int), w))
                ps.append(m.predict_proba(build_pool(X[gate], ci))[:, 1])
            else:
                m.fit(X[m_tr], y[m_tr].astype(int), sample_weight=w)
                ps.append(m.predict_proba(X[gate])[:, 1])
        P[nm] = np.mean(ps, 0).astype(np.float64)
        np.save(os.path.join(DL, f"cn_{nm}.npy"), P[nm])
        print(f"  {nm:11s} {time.time()-t0:6.0f}s")

    # TabM 관문 예측 — 고친 스케줄러 산출물이 있으면 그걸, 없으면 구판
    tm = None
    for cand in ("rf_all_reg4_s42.npy", "br_all_reg_s42.npy", "fb_base_s42.npy"):
        p = os.path.join(DL, cand)
        if os.path.exists(p):
            tm = np.load(p)
            print(f"\n  TabM 기준: {cand}")
            break
    HG = np.load(os.path.join(DL, "hg_route_d2.0.npy"))

    print(f"\n  {'구성':11s} {'단독':>9s} {'1군':>9s} {'퓨처스':>9s} "
          f"{'TabM상관':>9s} {'HistGB상관':>11s}")
    for nm, *_ in plans:
        p = P[nm]
        print(f"  {nm:11s} {F.best_shift(p, yv)[0]:9.1f} "
              f"{F.best_shift(p[~isf], yv[~isf])[0]:9.1f} "
              f"{F.best_shift(p[isf], yv[isf])[0]:9.1f} "
              f"{np.corrcoef(p, tm)[0,1]:9.4f} "
              f"{np.corrcoef(p, HG)[0,1]:11.4f}")

    if tm is not None:
        print("\n  같은 GBDT 총량에서의 혼합 기여 — 이게 판정 기준이다")
        print(f"  {'총량':>6s}  {'구성':22s} {'전체':>9s} {'base대비':>9s}")
        for tot in (0.14, 0.20, 0.29):
            ref = F.best_shift((1 - tot) * tm + tot * P["base"], yv)[0]
            print(f"  {tot:6.2f}  {'base (현행)':22s} {ref:9.1f} {0.0:+9.1f}")
            for nm, *_ in plans[1:]:
                v = F.best_shift((1 - tot) * tm + tot * P[nm], yv)[0]
                print(f"  {'':6s}  {nm:22s} {v:9.1f} {v-ref:+9.1f}")
            # HistGB 자리를 유지한 3원도 본다
            for nm in ("base", "native"):
                v = F.best_shift((1 - tot) * tm + tot * 0.5 * P[nm]
                                 + tot * 0.5 * HG, yv)[0]
                print(f"  {'':6s}  {nm + ' + HistGB 반반':22s} {v:9.1f} "
                      f"{v-ref:+9.1f}")
            print()

    print("  단독이 올라도 TabM 상관이 같이 오르면 앙상블 이득은 준다.")
    print("  GBDT 는 전체의 0.20 이라 단독 +40 이 혼합으로는 +4 쯤이다.")
