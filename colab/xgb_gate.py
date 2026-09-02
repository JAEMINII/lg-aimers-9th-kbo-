# -*- coding: utf-8 -*-
"""XGBoost 를 GBDT 칸에 넣어본다. 한 번도 안 해본 항목이다. CPU 라 GPU 와 안 다툰다.

질문
    지금 GBDT 칸은 0.30 을 CatBoost 혼자 쓴다 (submit_27, LB 1058).
    그 칸을 XGBoost 와 나눠 쓰는 게 나은가?

    단독 점수로 판정하지 않는다. HistGB 를 +14.7 인 줄 알았다가 **총비중을
    고정하니 +0.1** 이었던 전례가 있다. 같은 총비중 0.30 안에서 CatBoost 와
    XGBoost 의 배합만 바꿔가며 혼합 점수를 본다.

배경
    XGBoost 는 코드·문서·서버 어디에도 흔적이 없다. 기각된 게 아니라 미실시다.
    기각된 건 LightGBM 인데 그것도 TabM 도입 전 베이스라인 단계(608~692) 얘기라
    현행 구성에서 잰 적이 없다.

    찬성 근거 — 같은 GBDT 라도 분할 탐색과 정규화가 달라 실수 패턴이 다르다.
        실제로 GBDT 안에서도 TabM 상관이 벌어진다:
        HistGB 0.9031 < CatBoost 0.9492
    반대 근거 — GBDT 다양성 가설은 이미 리더보드에서 한 번 졌다.
        HistGB 를 빼고 CatBoost 를 0.10 -> 0.30 으로 키운 게 1057 -> 1058.

설정
    CatBoost 와 같은 조건에 맞춘다 — 시즌가중 2.0^(season-2019), 400라운드,
    lr 0.05, 그리고 44열 그대로. 깊이만 XGBoost 관례에 맞춰 6 을 기본으로
    두고 4 도 같이 본다(CatBoost 는 4 를 쓴다).

    범주 열은 CatBoost 와 마찬가지로 숫자로 넣는다. pitcher_id 를 트리가
    정수 순서로 밖에 못 보는 문제는 이미 확인했고(cat_features 실험, 혼합 기여 0),
    XGBoost 도 같은 처지다. 여기서 그걸 다시 열지는 않는다.
"""
import os
import sys
import time

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
DATA = "/workspace/aimers/data"
OUT = "/workspace/aimers/out"
VS = 2024
DECAY = 2.0
SEEDS = (42, 1234, 2025)

import features44 as F                                          # noqa: E402
import xgboost as xgb                                           # noqa: E402


def sc(p, y):
    return F.best_shift(p, y)[0]


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    X, y = d["X44"].astype(np.float32), d["y"]
    m_tr = d["m_tr"]
    gate = np.where(d["m_va"])[0]
    yv = y[gate].astype(np.float64)
    isf = d["is_f"][gate]
    R = ~isf
    FULL = np.ones(len(gate), bool)
    w = DECAY ** (d["season"][m_tr].astype(np.float64) - 2019)
    print(f"  학습 {m_tr.sum():,}  관문 {len(gate):,}  "
          f"xgboost {xgb.__version__}\n")

    dtr = xgb.DMatrix(X[m_tr], label=y[m_tr], weight=w)
    dva = xgb.DMatrix(X[gate])

    PLANS = (("xgb_d6", dict(max_depth=6, eta=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=10)),
             ("xgb_d4", dict(max_depth=4, eta=0.05, subsample=1.0,
                             colsample_bytree=1.0, min_child_weight=1)))
    P = {}
    for nm, hp in PLANS:
        t0 = time.time()
        ps = []
        for s in SEEDS:
            par = dict(hp, objective="binary:logistic", eval_metric="logloss",
                       seed=s, nthread=6, tree_method="hist")
            bst = xgb.train(par, dtr, num_boost_round=400)
            ps.append(bst.predict(dva).astype(np.float64))
        P[nm] = np.mean(ps, 0)
        np.save(os.path.join(OUT, f"xgb_{nm}.npy"), P[nm])
        print(f"  {nm:8s} {time.time()-t0:6.0f}s  단독 {sc(P[nm], yv):7.1f}")

    CB = G_CB = np.load(os.path.join(SC, "cb_gate.npy"))
    TMs = [np.where(isf, np.load(f"{OUT}/emb2_linear_relu_f_s{s}.npy"),
                    np.load(f"{OUT}/emb2_linear_relu_r_s{s}.npy"))
           for s in (42, 1, 777)]
    TM = np.mean(TMs, 0)

    print(f"\n  {'':10s} {'단독':>8s} {'1군':>8s} {'퓨처스':>8s} "
          f"{'TabM상관':>9s} {'CB상관':>8s}")
    for nm in ("CatBoost",) + tuple(n for n, _ in PLANS):
        p = CB if nm == "CatBoost" else P[nm]
        print(f"  {nm:10s} {sc(p, yv):8.1f} {sc(p[R], yv[R]):8.1f} "
              f"{sc(p[isf], yv[isf]):8.1f} {np.corrcoef(p, TM)[0,1]:9.4f} "
              f"{np.corrcoef(p, CB)[0,1]:8.4f}")

    # ---- 판정: GBDT 총비중 0.30 고정, 그 안에서 배합만 바꾼다
    TOT = 0.30
    print(f"\n  GBDT 총비중 {TOT:.2f} 고정.  나머지 {1-TOT:.2f} 는 TabM")
    print(f"  {'배합':28s} {'전체':>9s} {'CatBoost 단독 대비':>18s}")
    print("  " + "-" * 60)
    ref = sc((1 - TOT) * TM + TOT * CB, yv)
    print(f"  {'CatBoost 1.00 (현행)':28s} {ref:9.1f} {0.0:+18.1f}")
    for nm, _ in PLANS:
        for a in (0.25, 0.50, 0.75, 1.00):
            g = (1 - a) * CB + a * P[nm]
            v = sc((1 - TOT) * TM + TOT * g, yv)
            print(f"  {f'CB {1-a:.2f} / {nm} {a:.2f}':28s} {v:9.1f} "
                  f"{v - ref:+18.1f}")

    print("\n  판정선은 +5. 그 미만이면 GBDT 칸은 CatBoost 혼자 쓴다.")
    print("  단독이 높아도 CB 상관이 높으면 혼합 기여는 0 이다 —")
    print("  HistGB 가 총비중 고정에서 +0.1 이었던 것과 같은 함정이다.")
