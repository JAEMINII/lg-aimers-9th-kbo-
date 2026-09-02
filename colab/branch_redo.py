# -*- coding: utf-8 -*-
"""퓨처스/레귤러 브랜치를 처음부터 다시 짠다. 60/40 도 다시 잡는다.

왜
    submit_20 이 1057 로 신기록을 냈고, 그 이득의 대부분이 '퓨처스 행에만
    regime 을 쓴 것' 이었다. 그런데 지금 구성에는 손 안 댄 가정이 둘 남아 있다.

    1. 퓨처스 브랜치의 65.4%가 폐지된 규칙으로 학습돼 있다.
       퓨처스 학습 161,004행 중 옛 체제(<=2022)가 105,308행이다.
       2023 ABS 도입으로 성공률이 70.87% -> 47.29% 로 무너졌으니 그 표본은
       다른 게임이다. 지금은 가중 0.1 로 사후에 누르고 있을 뿐이다
       (유효 표본 66,227행). 아예 새 체제만으로 학습하는 판본과 맞대봐야 한다.

    2. 60/40 은 퓨처스 브랜치가 오염돼 있을 때 정한 값이다.
       그때 퓨처스 브랜치 관문 점수가 481.4 였다. 지금은 630.1 이다.
       브랜치가 좋아졌으면 더 실을 수 있다. 비중을 다시 훑는다.

    또 하나. 체제 처리는 1군을 -18.8 내린다. 양 리그에 다 나오는 투수가
    453명이라(퓨처스 투수 633명 중) 퓨처스 표본을 누르면 그 투수들의
    임베딩 학습량이 같이 준다는 설명이 유력하다. 그래서 1군 쪽 all 은
    누르지 않는 판본도 같이 만든다.

무엇을 만드나 (시드마다 7개, 지인 전처리, VS=2024)
    44열
      all_base       전부
      regular_base   1군만
      futures_base   퓨처스 전부 (옛 체제 포함)
      futures_new    퓨처스 새 체제만 (2023~, 55,696행)
      all_drop       전부에서 옛 체제 퓨처스만 뺌
    45열 (+ abs_regime)
      all_reg        전부, 옛 체제 퓨처스 가중 0.1
      futures_reg    퓨처스 전부, 옛 체제 가중 0.1      <- submit_20 이 쓰는 것

무엇을 재나
    브랜치별 예측을 시드별로 저장하고 사후에 비중을 훑는다. 재학습 없이
    1군 조합과 퓨처스 조합을 따로 최적화한다.

    채점은 구간별로 분모를 따로 잡는다. 전체 분모로 퓨처스를 재면
    슬라이스 점수가 왜곡된다 — 전에 '2스트라이크 약점' 을 그렇게 잘못 읽었다.

    시드마다 따로 점수를 내고 짝비교한다. 배치가 1시드이므로 1시드에서 잰다.
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
FIXED = -0.004

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv


def sc(p, m):
    """구간 안에서 최적 시프트. 분모도 그 구간 것을 쓴다."""
    return F.best_shift(p[m], yv[m])[0]


def scf(p, m):
    return F.bss(F.shift(p[m], FIXED), yv[m])


def friend_matrix():
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xdf = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xdf.columns)
    Xf = Xdf.to_numpy(dtype=np.float32)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(Xf)), index=tr["row_id"].to_numpy())
    return Xf[pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()], cols


def train_one(seed, idx, w, tag):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, 2e-3, w=w, seed=seed, tag=f"{tag} S1")
    s2 = G.season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
            tag=f"{tag} S2")
    p = G.predict(m, gate)
    del m
    torch.cuda.empty_cache()
    return p


def set_matrix(X, ci):
    Xn, Xc, cards = G.prep(X, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)


if __name__ == "__main__":
    Xf, cols = friend_matrix()
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    season, isf = G.season.astype(np.float64), G.is_f
    flag = ((isf & (season > OLD_F_MAX)) | (~isf)).astype(np.float32)[:, None]
    Xr = np.concatenate([Xf, flag], 1)
    ci_r = ci + [Xf.shape[1]]

    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    t_old = t_isf & (season[tr_idx] <= OLD_F_MAX)
    G.log(f"\n  학습 {len(tr_idx):,}행   퓨처스 {t_isf.sum():,} "
          f"(옛 체제 {t_old.sum():,} = {t_old.sum()/max(t_isf.sum(),1)*100:.1f}%)")
    G.log(f"  관문 {len(gate):,}행   퓨처스 {is_f.sum():,}\n")

    P = {}
    for seed in SEEDS:
        t0 = time.time()
        set_matrix(Xf, ci)                                   # 44열
        for name, sel, w in (
                ("all_base",     np.ones(len(tr_idx), bool), None),
                ("regular_base", ~t_isf,                     None),
                ("futures_base", t_isf,                      None),
                ("futures_new",  t_isf & ~t_old,             None),
                ("all_drop",     ~t_old,                     None)):
            P[(name, seed)] = train_one(seed, tr_idx[sel], w, name)
        set_matrix(Xr, ci_r)                                 # 45열
        for name, sel in (("all_reg", np.ones(len(tr_idx), bool)),
                          ("futures_reg", t_isf)):
            w = np.where(t_old[sel], OLD_W, 1.0).astype(np.float64)
            P[(name, seed)] = train_one(seed, tr_idx[sel], w, name)
        G.log(f"  seed {seed}  {time.time()-t0:.0f}s")
    for (n, s), v in P.items():
        np.save(os.path.join(OUT, f"br_{n}_s{s}.npy"), v)

    names = sorted({n for n, _ in P})
    G.log("\n  브랜치 단독 (해당 구간에서, 시드평균)")
    G.log(f"  {'브랜치':14s} {'1군행':>9s} {'퓨처스행':>9s}")
    for n in names:
        a = np.mean([P[(n, s)] for s in SEEDS], 0)
        G.log(f"  {n:14s} {sc(a, ~is_f):9.1f} {sc(a, is_f):9.1f}")

    # ---------------------------------------------------------- 퓨처스 조합
    G.log("\n  퓨처스 행 조합 스윕   p = (1-w) x A + w x B   (고정시프트, 시드평균)")
    G.log(f"  {'A':14s} {'B':14s} " + " ".join(f"w={w:.1f}" for w in
                                               np.arange(0.0, 1.01, 0.2)))
    bestF = (-1e9, None)
    for A in ("all_base", "all_reg", "all_drop"):
        for B in ("futures_base", "futures_reg", "futures_new"):
            row = []
            for w in np.arange(0.0, 1.01, 0.2):
                v = [scf((1 - w) * P[(A, s)] + w * P[(B, s)], is_f)
                     for s in SEEDS]
                mu = float(np.mean(v))
                row.append(f"{mu:7.1f}")
                if mu > bestF[0]:
                    bestF = (mu, (A, B, round(float(w), 2)))
            G.log(f"  {A:14s} {B:14s} " + " ".join(row))
    G.log(f"  -> 최선 {bestF[1]}  {bestF[0]:.1f}")
    cur = [scf(0.6 * P[("all_reg", s)] + 0.4 * P[("futures_reg", s)], is_f)
           for s in SEEDS]
    G.log(f"  현행(submit_20: all_reg 0.6 + futures_reg 0.4)  {np.mean(cur):.1f}")

    # ---------------------------------------------------------- 1군 조합
    G.log("\n  1군 행 조합 스윕   p = (1-w) x A + w x regular_base")
    G.log(f"  {'A':14s} " + " ".join(f"w={w:.1f}" for w in
                                     np.arange(0.0, 1.01, 0.2)))
    bestR = (-1e9, None)
    for A in ("all_base", "all_reg", "all_drop"):
        row = []
        for w in np.arange(0.0, 1.01, 0.2):
            v = [scf((1 - w) * P[(A, s)] + w * P[("regular_base", s)], ~is_f)
                 for s in SEEDS]
            mu = float(np.mean(v))
            row.append(f"{mu:7.1f}")
            if mu > bestR[0]:
                bestR = (mu, (A, round(float(w), 2)))
        G.log(f"  {A:14s} " + " ".join(row))
    G.log(f"  -> 최선 {bestR[1]}  {bestR[0]:.1f}")
    cur2 = [scf(0.6 * P[("all_base", s)] + 0.4 * P[("regular_base", s)], ~is_f)
            for s in SEEDS]
    G.log(f"  현행(submit_20: all_base 0.6 + regular_base 0.4)  {np.mean(cur2):.1f}")

    # ---------------------------------------------------------- 짝비교
    G.log("\n  현행 대비 짝차이 (시드끼리 짝. 배치가 1시드이므로 1시드에서 잰다)")
    for tag, m, cand, base in (
            ("퓨처스", is_f,
             lambda s: (1 - bestF[1][2]) * P[(bestF[1][0], s)]
             + bestF[1][2] * P[(bestF[1][1], s)],
             lambda s: 0.6 * P[("all_reg", s)] + 0.4 * P[("futures_reg", s)]),
            ("1군", ~is_f,
             lambda s: (1 - bestR[1][1]) * P[(bestR[1][0], s)]
             + bestR[1][1] * P[("regular_base", s)],
             lambda s: 0.6 * P[("all_base", s)] + 0.4 * P[("regular_base", s)])):
        d = [scf(cand(s), m) - scf(base(s), m) for s in SEEDS]
        mu, se = float(np.mean(d)), float(np.std(d, ddof=1)) / 2.0
        G.log(f"    {tag:6s} {mu:+7.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/4  [{', '.join(f'{x:+.1f}' for x in d)}]")

    G.log("\n  주의: 퓨처스 구간 점수는 전체 점수에 11.8% 비율로 들어간다.")
    G.log("  퓨처스 +100 은 전체로 약 +12 다. 1군 이득과 직접 비교하지 말 것.")
