# -*- coding: utf-8 -*-
"""등판 강도 2열을 8시드 짝비교로 잰다.

무엇을 더하나
    load_rate  log1p(pn_cur) - log1p(경과 개월)
    load_lin   pn_cur / 경과 개월
    pn_cur 은 인시즌 투구수, 경과 개월은 game_month - 2 (3월 개막 기준).

왜 이것만 남았나
    행마다 값이 변하는 후보 여덟 개를 편상관으로 걸렀다. 통제는 71개 —
    44열의 강한 것들 + 카운트 더미 + p_is_succ + b_is_succ + pn_cur + bn_cur
    + 카운트x매치업 더미.
```
        부하_로그        단순 +0.0242   편상관 +0.0111   <- 살아남음
        부하_월평균투구   단순 +0.0323   편상관 +0.0104   <- 살아남음
        타자 인시즌 3종   전부 0   (b_is_succ 와 같은 것이었다)
        폼 추세 4종       전부 0   (prev1/3/5 가 이미 있다)
```
    비교로, 기존 피처 asof_batter_middle_rate 가 같은 통제에서 -0.008 이다.

    말이 되는 신호이기도 하다. pn_cur 과 game_month 는 둘 다 44열에 있지만
    그 '비율' 은 모델이 스스로 만들어야 한다. 나눗셈은 신경망이 잘 못 배우는
    형태라 명시적으로 주는 게 도움이 될 수 있다.

기준선은 시즌가중 3.5 다. 학습량이 같은 변형이라 관문을 믿을 수 있는 축이다.
혼합 기준으로 판정한다 — 오늘 단독과 혼합이 반대로 움직이는 걸 여러 번 봤다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777, 2, 3, 5, 11, 23)
DECAY = 3.5
VS = 2024
ALPHA = 50.0

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


def load_cols(F44, X):
    """pn_cur 과 game_month 로 등판 강도를 만든다. 둘 다 이미 44열에 있다."""
    pn = X[:, F44.index("pn_cur")].astype(np.float64)
    month = X[:, F44.index("game_month")].astype(np.float64)
    elapsed = np.clip(month - 2.0, 1.0, None)      # 3월 개막
    return np.stack([np.log1p(pn) - np.log1p(elapsed), pn / elapsed],
                    1).astype(np.float32)


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    F44 = list(d["F44"])
    base = d["X44"]
    add = load_cols(F44, base)
    X2 = np.concatenate([base, add], 1)
    G.log(f"  추가 2열  범위 {add[:, 0].min():.2f}~{add[:, 0].max():.2f} / "
          f"{add[:, 1].min():.1f}~{add[:, 1].max():.1f}   "
          f"결측 {np.isnan(add).mean()*100:.2f}%")

    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    M = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    def blend(r):
        return sc(0.10 * CB + 0.30 * M + 0.60 * r)

    def load(X):
        Xn, Xc, cards = G.prep(X, G.m_tr, d["cat_idx"])
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    P0, P1 = {}, {}
    G.log("")
    G.log("  시드    기준단독  부하단독   기준혼합  부하혼합   혼합차이")
    for sd in SEEDS:
        t0 = time.time()
        load(base)
        P0[sd] = one(sd, decay=DECAY)
        load(X2)
        P1[sd] = one(sd, decay=DECAY)
        G.log(f"  {sd:4d}  {sc(P0[sd]):9.1f} {sc(P1[sd]):9.1f}  "
              f"{blend(P0[sd]):9.1f} {blend(P1[sd]):9.1f}  "
              f"{blend(P1[sd]) - blend(P0[sd]):+9.1f}   {time.time()-t0:.0f}s")

    for tag, f in (("단독", sc), ("혼합", blend)):
        dif = [f(P1[s]) - f(P0[s]) for s in SEEDS]
        m, sd_ = float(np.mean(dif)), float(np.std(dif, ddof=1))
        se = sd_ / np.sqrt(len(dif))
        G.log(f"\n  {tag} 짝차이  평균 {m:+.1f}  표준편차 {sd_:.1f}  표준오차 {se:.1f}  "
              f"양수 {sum(1 for x in dif if x > 0)}/{len(dif)}  t={m/se:.2f}  "
              f"{'확정' if abs(m) > 2 * se else '미확정'}")

    r0 = np.mean([P0[s] for s in SEEDS], 0)
    r1 = np.mean([P1[s] for s in SEEDS], 0)
    np.save(os.path.join(OUT, "load_tabm.npy"), r1)
    G.log(f"\n  8시드평균  기준 단독 {sc(r0):7.1f} 혼합 {blend(r0):7.1f}   "
          f"부하 단독 {sc(r1):7.1f} 혼합 {blend(r1):7.1f}")
    G.log(f"  상관  기준↔부하 {np.corrcoef(r0, r1)[0,1]:.4f}   "
          f"부하↔CatBoost {np.corrcoef(r1, CB)[0,1]:.4f}")
