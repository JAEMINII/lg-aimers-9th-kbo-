# -*- coding: utf-8 -*-
"""다년 백테스트. 오늘 '확정' 한 것들이 2024 에서만 좋았던 건지 확인한다.

왜 바꾸나
    오늘 내린 판정이 전부 VS=2024 한 폴드에서 나왔다. 시드를 8개로 늘리고
    짝비교를 하고 t=4.86 을 받아도, 그건 '그 해 안에서' 안정적이라는 뜻이지
    '다른 해에도 통한다' 는 뜻이 아니다. 연도는 하나였다.

    그리고 리더보드가 세 번 연속 반대로 갔다.
        관문   ~930 -> 935.0 -> 939.5
        LB     1046 -> 1044  -> 1041
    2024 과적합이 유력한 설명이다.

무엇을 바꾸나
    1. 폴드 셋 (VS=2022, 2023, 2024). 세 폴드에서 일관돼야 채택한다.
    2. 고정 시프트로도 채점한다. 실제 제출은 고정값을 쓰는데 최적 시프트로 재면
       '보정만 해주면 좋아지는' 구성이 과대평가된다. 둘 다 찍어 비교한다.
    3. 전체(R+F) 채점. 1군만 보면 퓨처스 붕괴를 놓친다.

무엇을 재나
    base        시즌가중 없음, 44열
    sw3.5       시즌가중 3.5
    sw3.5+load  + 등판 강도 2열
    sw3.5+low   + 저카디널리티 범주화
    sw3.5+둘다  (제출본 17 구성)

    폴드마다 시드 4개. 폴드 간 결과가 갈리면 그게 답이다.

사용법
    VS=2022 python3 multifold.py     (폴드마다 따로 돌린다 — 모듈이 import 시점에
                                      VS 로 데이터를 만들기 때문)
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777, 2)
LOWCARD = ["game_month", "game_dayofweek", "inning",
           "balls_before", "strikes_before", "outs_before"]
FIXED_SHIFT = -0.0145        # 제출본이 실제로 쓰는 값

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc_opt(p):
    """최적 시프트에서. 지금까지 쓰던 방식."""
    return F.best_shift(p[FULL], yv[FULL])[0]


def sc_fix(p):
    """고정 시프트에서. 실제 제출과 같은 조건."""
    return F.bss(F.shift(p[FULL], FIXED_SHIFT), yv[FULL])


def load_cols(F44, X):
    pn = X[:, F44.index("pn_cur")].astype(np.float64)
    mo = X[:, F44.index("game_month")].astype(np.float64)
    el = np.clip(mo - 2.0, 1.0, None)
    return np.stack([np.log1p(pn) - np.log1p(el), pn / el], 1).astype(np.float32)


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    F44 = list(d["F44"])
    base = d["X44"]
    withload = np.concatenate([base, load_cols(F44, base)], 1)
    names_l = F44 + ["load_rate", "load_lin"]
    ci_b = list(d["cat_idx"])
    ci_c = sorted(set(ci_b) | {F44.index(c) for c in LOWCARD})
    ci_cl = sorted(set(ci_b) | {names_l.index(c) for c in LOWCARD})

    G.log(f"\n===== VS={VS}  학습 {G.m_tr.sum():,}행  검증 {len(gate):,}행 "
          f"(성공률 {yv.mean():.4f}) =====")

    plans = [
        ("base",        base,     ci_b,  None),
        ("sw3.5",       base,     ci_b,  3.5),
        ("sw3.5+load",  withload, ci_b,  3.5),
        ("sw3.5+low",   base,     ci_c,  3.5),
        ("sw3.5+둘다",   withload, ci_cl, 3.5),
    ]
    res = {}
    G.log("")
    G.log("  구성          최적시프트   고정시프트   개별시드(최적)")
    for name, X, ci, decay in plans:
        Xn, Xc, cards = G.prep(X, G.m_tr, ci)
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        t0 = time.time()
        kw = {} if decay is None else dict(decay=decay)
        ps = [one(sd, **kw) for sd in SEEDS]
        res[name] = ps
        r = np.mean(ps, 0)
        np.save(os.path.join(OUT, f"mf{VS}_{name.replace('+','_')}.npy"), r)
        G.log(f"  {name:12s} {sc_opt(r):10.1f} {sc_fix(r):12.1f}   "
              f"[{', '.join(f'{sc_opt(p):.0f}' for p in ps)}]   {time.time()-t0:.0f}s")

    G.log("")
    G.log("  base 대비 짝차이 (같은 시드끼리)")
    for name in ("sw3.5", "sw3.5+load", "sw3.5+low", "sw3.5+둘다"):
        for tag, f in (("최적", sc_opt), ("고정", sc_fix)):
            dif = [f(a) - f(b) for a, b in zip(res[name], res["base"])]
            m = float(np.mean(dif))
            se = float(np.std(dif, ddof=1)) / np.sqrt(len(dif))
            G.log(f"    {name:12s} {tag}  {m:+7.1f} ± {se:4.1f}  "
                  f"{sum(1 for x in dif if x > 0)}/{len(dif)}")
    G.log("")
    G.log("  세 폴드(2022/2023/2024)에서 일관돼야 채택한다.")
    G.log("  한 폴드에서만 크면 그 해 과적합이다.")
