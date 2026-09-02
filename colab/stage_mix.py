# -*- coding: utf-8 -*-
"""Stage1/Stage2 를 섞고, Stage2 에폭을 훑는다. 한 학습에서 둘 다 나온다.

왜 이 축인가
    지금은 Stage2 결과만 쓴다. Stage1(전체 시즌)로 배운 뒤 마지막 시즌(2024)
    하나로 1에폭 미세조정하고 끝이다.

    퓨처스 브랜치의 Stage2 는 30,010행 / 배치 2048 = **약 15 스텝**이다.
    15번의 경사하강으로 2024 에 적응시키는 거라, 과적합도 과소적합도 가능하다.
    한쪽이면 Stage1 을 섞는 게 낫고, 다른 쪽이면 에폭을 늘리는 게 낫다.
    지금까지 둘 다 안 해봤다.

설계 — 한 번 학습으로 둘을 동시에 얻는다
    Stage2 를 1에폭씩 끊어 돌리며 매 시점 예측을 저장한다.
        e0 = Stage1 직후            (= Stage2 를 안 한 것)
        e1 = Stage2 1에폭 후        (= 현행)
        e2, e3
    그러면
        에폭 스윕      e0 / e1 / e2 / e3 비교
        혼합 스윕      w x e1 + (1-w) x e0  를 사후에 훑는다 (재학습 0)
    가 같은 산출물에서 나온다.

    브랜치는 all 과 futures 둘 다 만든다. 퓨처스 라우팅이 0.6 all + 0.4 futures
    라서, 어느 쪽 Stage2 를 건드리는 게 효과적인지 따로 봐야 한다.

채점
    1군 행은 all 브랜치로, 퓨처스 행은 라우팅 후로 잰다. 구간별 분모.
    시드 짝비교, 최적 시프트. 배치가 1시드이므로 시드별로 점수를 낸다.
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
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
MAX_E2 = 3

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def fit_stages(seed, idx, w, season, tag):
    """Stage1 후, 그리고 Stage2 매 에폭 후의 예측을 전부 돌려준다."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, LR1, w=w, seed=seed, tag=f"{tag} S1")
    outs = [G.predict(m, gate)]                       # e0 = Stage1 직후
    s2 = idx[season[idx] == VS - 1]
    ps = G.stage2_params(m)
    for e in range(MAX_E2):
        # 1에폭씩 끊어 돌린다. 매번 새 옵티마이저라 3에폭 연속과 정확히
        # 같지는 않지만, 코사인이 에폭 단위라 lr 궤적은 사실상 같다.
        G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e, tag=f"{tag} S2e{e+1}")
        outs.append(G.predict(m, gate))
    del m
    torch.cuda.empty_cache()
    return outs                                        # [e0, e1, e2, e3]


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
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xn, Xc, cards = G.prep(np.concatenate([Xf, c4], 1), G.m_tr,
                           ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
    n_s2_f = int((t_isf & (season[tr_idx] == VS - 1)).sum())
    G.log(f"\n  퓨처스 Stage2 표본 {n_s2_f:,}행 / 배치 2048 = "
          f"{int(np.ceil(n_s2_f/2048))} 스텝")

    A, Fb = {}, {}
    for s in SEEDS:
        t0 = time.time()
        A[s] = fit_stages(s, tr_idx, w10, season, "all")
        Fb[s] = fit_stages(s, tr_idx[t_isf], w10[t_isf], season, "fut")
        for e in range(MAX_E2 + 1):
            np.save(os.path.join(OUT, f"sm_all_e{e}_s{s}.npy"), A[s][e])
            np.save(os.path.join(OUT, f"sm_fut_e{e}_s{s}.npy"), Fb[s][e])
        G.log(f"  seed {s}  {time.time()-t0:.0f}s")

    G.log("\n  [A] Stage2 에폭 — 1군 행 (all 브랜치 단독)")
    G.log(f"  {'에폭':>6s} {'1군 행':>9s}   e1(현행) 대비 짝차이")
    for e in range(MAX_E2 + 1):
        v = [sc(A[s][e], R) for s in SEEDS]
        d = [sc(A[s][e], R) - sc(A[s][1], R) for s in SEEDS]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d)) if e != 1 else 0.0
        tail = "" if e == 1 else (f"{mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
                                  f"{sum(1 for x in d if x > 0)}/4")
        G.log(f"  {('e'+str(e)):>6s} {np.mean(v):9.1f}   {tail}")

    G.log("\n  [B] Stage2 에폭 — 퓨처스 행 (0.6 all_e1 + 0.4 fut_eX)")
    for e in range(MAX_E2 + 1):
        v = [sc(0.6 * A[s][1] + 0.4 * Fb[s][e], is_f) for s in SEEDS]
        d = [x - y for x, y in zip(v, [sc(0.6 * A[s][1] + 0.4 * Fb[s][1], is_f)
                                       for s in SEEDS])]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d)) if e != 1 else 0.0
        tail = "" if e == 1 else (f"{mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
                                  f"{sum(1 for x in d if x > 0)}/4")
        G.log(f"  {('e'+str(e)):>6s} {np.mean(v):9.1f}   {tail}")

    G.log("\n  [C] Stage1/Stage2 혼합  w x e1 + (1-w) x e0")
    G.log(f"  {'w':>6s} {'1군 행':>9s} {'퓨처스 행':>10s}")
    for w in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0):
        r = [sc(w * A[s][1] + (1 - w) * A[s][0], R) for s in SEEDS]
        f_ = [sc(0.6 * (w * A[s][1] + (1 - w) * A[s][0])
                 + 0.4 * (w * Fb[s][1] + (1 - w) * Fb[s][0]), is_f)
              for s in SEEDS]
        mark = "  <- 현행" if w == 1.0 else ""
        G.log(f"  {w:6.1f} {np.mean(r):9.1f} {np.mean(f_):10.1f}{mark}")

    G.log("\n  [D] 퓨처스 브랜치만 혼합 (all 은 e1 고정)")
    G.log(f"  {'w':>6s} {'퓨처스 행':>10s}   현행 대비")
    base = [sc(0.6 * A[s][1] + 0.4 * Fb[s][1], is_f) for s in SEEDS]
    for w in (1.0, 0.8, 0.6, 0.4, 0.0):
        v = [sc(0.6 * A[s][1] + 0.4 * (w * Fb[s][1] + (1 - w) * Fb[s][0]), is_f)
             for s in SEEDS]
        d = [x - y for x, y in zip(v, base)]
        G.log(f"  {w:6.1f} {np.mean(v):10.1f}   {np.mean(d):+7.1f}  "
              f"{sum(1 for x in d if x > 0)}/4")

    G.log("\n  퓨처스 구간 +100 은 전체로 약 +12 다 (퓨처스 11.8%).")
    G.log("  조합 부류라 관문 부호도 못 믿는다. 크게 이기는 게 아니면 안 옮긴다.")
