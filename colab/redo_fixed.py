# -*- coding: utf-8 -*-
"""스케줄러를 고친 뒤 브랜치 구성을 처음부터 다시 잰다.

왜 다시 재나
    codegap 에서 우리 학습 코드의 결함을 찾았다. 코사인을 배치마다 돌려
    lr 을 0 까지 완전히 감쇠시키고 있었고, 에폭이 2개뿐이라 후반부가 통째로
    버려졌다. 에폭 단위로 바꾸니 1군 행 869.5 -> 888.8 (+19.2, 3/3, t=2.86)
    이고 지인 코드(874.6)를 넘는다.

    문제는 **지금까지의 관문 수치가 전부 그 결함 위에서 나왔다는 것**이다.
    reg4 +11.3, 브랜치 비중, 체제 처리 — 전부 덜 학습된 모델끼리의 비교였다.
    학습이 제대로 되면 순위가 바뀔 수 있다. 특히 체제 코드처럼 '모델이
    구분을 배울 수 있느냐' 에 달린 장치는 학습량에 민감하다.

무엇을 만드나 (지인 전처리, VS=2024, 시드 4개)
    44열        all_base, regular_base
    45열 2단계   all_reg2, futures_reg2      0 = 옛 퓨처스 / 1 = 나머지
    45열 4단계   all_reg4, futures_reg4      0 옛F / 1 옛1군 / 2 새F / 3 새1군
    옛 체제 퓨처스 학습가중 0.1 은 45열 판본 둘 다에 건다.

무엇을 재나
    브랜치별 예측을 시드별로 저장하고 사후에 조합을 훑는다. 재학습 없이
    1군 조합과 퓨처스 조합을 따로 최적화하고, CatBoost/HistGB 를 얹은
    **전체 혼합**으로도 찍는다. 배치가 그 형태이기 때문이다.

    시드마다 따로 점수를 내고 짝비교한다. 배치가 1시드다.
    비교는 최적 시프트에서 한다 — 제출 때마다 시프트를 다시 잡으므로
    고정 시프트로 재면 수준 보정과 판별력이 섞인다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777, 2)
OLD_F_MAX = 2022
OLD_W = 0.1

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(yv), bool)


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def fit(seed, idx, w, season, tag):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, 2e-3, w=w, seed=seed, tag=f"{tag} S1")
    s2 = season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
            tag=f"{tag} S2")
    p = G.predict(m, gate)
    del m
    torch.cuda.empty_cache()
    return p


if __name__ == "__main__":
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c2 = ((isf & ~old) | (~isf)).astype(np.float32)[:, None]
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]

    m_tr = G.m_tr
    tr_idx = np.where(m_tr)[0]
    t_isf = isf[tr_idx]
    t_old = t_isf & old[tr_idx]
    w10 = np.where(t_old, OLD_W, 1.0).astype(np.float64)
    G.log(f"\n  학습 {len(tr_idx):,}  퓨처스 {t_isf.sum():,} "
          f"(옛 체제 {t_old.sum():,})  관문 {len(gate):,} (1군 {R.sum():,})")

    packs = [("base", Xf, ci,
              [("all_base", np.ones(len(tr_idx), bool), None),
               ("regular_base", ~t_isf, None)]),
             ("reg2", np.concatenate([Xf, c2], 1), ci + [Xf.shape[1]],
              [("all_reg2", np.ones(len(tr_idx), bool), w10),
               ("futures_reg2", t_isf, w10[t_isf])]),
             ("reg4", np.concatenate([Xf, c4], 1), ci + [Xf.shape[1]],
              [("all_reg4", np.ones(len(tr_idx), bool), w10),
               ("futures_reg4", t_isf, w10[t_isf])])]

    P = {}
    for tag, X, c, plans in packs:
        Xn, Xc, cards = G.prep(X, m_tr, c)
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        G.log(f"  [{tag}] 수치 {Xn.shape[1]}열 범주 {Xc.shape[1]}열 "
              f"카디널리티 {cards.tolist()[-1]}")
        for nm, sel, w in plans:
            t0 = time.time()
            for s in SEEDS:
                P[(nm, s)] = fit(s, tr_idx[sel], w, season, nm)
                np.save(os.path.join(OUT, f"rf_{nm}_s{s}.npy"), P[(nm, s)])
            G.log(f"    {nm:14s} {time.time()-t0:.0f}s")

    names = ["all_base", "regular_base", "all_reg2", "futures_reg2",
             "all_reg4", "futures_reg4"]
    G.log("\n  브랜치 단독, 최적 시프트, 시드평균")
    G.log(f"  {'브랜치':14s} {'1군 행':>9s} {'퓨처스 행':>10s}")
    for n in names:
        a = np.mean([P[(n, s)] for s in SEEDS], 0)
        G.log(f"  {n:14s} {sc(a, R):9.1f} {sc(a, is_f):10.1f}")

    CB = np.load(os.path.join(SC, "cb_gate.npy")).astype(np.float64)
    hg = os.path.join(SC, "_dl", "hg_route_d2.0.npy")
    HG = np.load(hg) if os.path.exists(hg) else CB.copy()
    if not os.path.exists(hg):
        G.log("\n  주의: hg_route_d2.0.npy 가 없어 HistGB 자리에 CatBoost 를 넣었다.")

    def blend(rf, ff, s):
        return F.best_shift(0.1 * CB + 0.1 * HG
                            + 0.8 * np.where(is_f, ff(s), rf(s)), yv)[0]

    cands = {
        "submit_20 (base 0.6/0.4 + reg2)":
            (lambda s: 0.6 * P[("all_base", s)] + 0.4 * P[("regular_base", s)],
             lambda s: 0.6 * P[("all_reg2", s)] + 0.4 * P[("futures_reg2", s)]),
        "1군 all_base 단독 / F reg2":
            (lambda s: P[("all_base", s)],
             lambda s: 0.6 * P[("all_reg2", s)] + 0.4 * P[("futures_reg2", s)]),
        "1군 reg4 단독 / F reg4 0.4":
            (lambda s: P[("all_reg4", s)],
             lambda s: 0.6 * P[("all_reg4", s)] + 0.4 * P[("futures_reg4", s)]),
        "1군 reg4 단독 / F reg2 0.4":
            (lambda s: P[("all_reg4", s)],
             lambda s: 0.6 * P[("all_reg2", s)] + 0.4 * P[("futures_reg2", s)]),
        "1군 reg2 단독 / F reg2 0.4":
            (lambda s: P[("all_reg2", s)],
             lambda s: 0.6 * P[("all_reg2", s)] + 0.4 * P[("futures_reg2", s)]),
        "1군 base+0.2reg / F reg4 0.4":
            (lambda s: 0.8 * P[("all_base", s)] + 0.2 * P[("regular_base", s)],
             lambda s: 0.6 * P[("all_reg4", s)] + 0.4 * P[("futures_reg4", s)]),
    }
    G.log("\n  전체 혼합 (0.1 CB + 0.1 HistGB + 0.8 TabM), 최적 절편")
    G.log(f"  {'구성':34s} {'평균':>8s} {'20대비':>8s}  시드별")
    base = None
    for nm, (rf, ff) in cands.items():
        v = [blend(rf, ff, s) for s in SEEDS]
        if base is None:
            base = v
        d = [a - b for a, b in zip(v, base)]
        G.log(f"  {nm:34s} {np.mean(v):8.1f} {np.mean(d):+8.1f}  "
              f"[{', '.join(f'{x:+.1f}' for x in d)}] "
              f"{sum(1 for x in d if x > 0)}/{len(d)}")

    G.log("\n  스케줄러를 고치기 전 같은 표에서는 'reg4 단독 / reg4 0.4' 가 +11.3 이었다.")
    G.log("  학습이 제대로 되면 그 이득이 줄어들 수 있다 — 체제 코드는 모델이")
    G.log("  구분을 배울 여력이 있을 때 덜 필요하기 때문이다. 여기서 확인한다.")
