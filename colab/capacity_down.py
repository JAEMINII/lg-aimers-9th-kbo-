# -*- coding: utf-8 -*-
"""용량을 **아래로** 내려 본다. 지금까지 위로만 재봤고 전부 나빴다.

왜
    위로 올린 네 판본이 전부 졌다 (VS=2024, all 브랜치, 1군 행, 4시드 짝비교)
        k=64          -4.1   2/4
        d_block 512  -14.1   0/4
        n_blocks 4    -5.1   2/4
        tabm-packed -117.7   0/4
    위가 다 나쁘면 지금이 이미 필요 이상일 수 있다. 아래는 한 번도 안 가봤다.

    그리고 속도가 지금 실질 제약이다. submit_25 가 평가 서버에서 9분 26초였고
    그 92%가 TabM 순전파다. 한 패스 210초짜리를 1군 행에 6번 돌린다.
    순전파 비용은 대략 k x n_blocks x d_block^2 에 비례하므로
        k 32->16      2배 빠름
        d 256->192   1.8배
        blocks 3->2  1.5배
    작아져도 점수가 같으면 그 예산으로 시드를 늘리거나 Stage2 를 더 돌릴 수 있다.
    지금 t 값이 계속 애매한 게 시드 4개 제약 때문인데 그게 풀린다.

무엇을 재나 (지인 전처리, VS=2024, all 브랜치, 1군 행, 시드 4개)
    base      k32 d256 b3   현행
    k16       k16 d256 b3
    k8        k8  d256 b3
    d192      k32 d192 b3
    d128      k32 d128 b3
    k16d192   k16 d192 b3
    b2        k32 d256 b2

    Stage1 2에폭 lr 3e-3, Stage2 1에폭 2e-4 output+마지막블록으로 고정.
    바뀌는 건 용량뿐이다.

    학습 시간과 추론 시간을 같이 찍는다. 점수가 같다면 그게 판단 근거다.

판정
    시드 짝비교, 최적 시프트, 부호 일관성. 최적화 부류라 관문으로 정할 수 있다
    (스케줄러 +19.2 -> LB +21.6 으로 이 부류는 맞았다).
    다만 크기는 못 맞힌다 — lr 은 관문 +6.1 인데 LB +1 이었다.
    그래서 '이기는 것' 보다 '지지 않으면서 빠른 것' 을 찾는 게 목적이다.
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

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(yv), bool)


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def fit(seed, idx, season, **kw):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model(**kw)
    t_tr = time.time()
    G.train(m, idx, 2, LR1, seed=seed, tag="S1")
    s2 = season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed, tag="S2")
    t_tr = time.time() - t_tr
    t_pr = time.time()
    p = G.predict(m, gate)
    t_pr = time.time() - t_pr
    npar = sum(x.numel() for x in m.parameters())
    del m
    torch.cuda.empty_cache()
    return p, npar, t_tr, t_pr


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

    Xn, Xc, cards = G.prep(Xf, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    season = G.season.astype(np.float64)
    tr_idx = np.where(G.m_tr)[0]

    plans = [("base", {}),
             ("k16", dict(k=16)),
             ("k8", dict(k=8)),
             ("d192", dict(d_block=192)),
             ("d128", dict(d_block=128)),
             ("k16d192", dict(k=16, d_block=192)),
             ("b2", dict(n_blocks=2))]
    P, INFO = {}, {}
    for nm, kw in plans:
        t0 = time.time()
        tt = pp = 0.0
        for s in SEEDS:
            P[(nm, s)], npar, a, b = fit(s, tr_idx, season, **kw)
            tt += a
            pp += b
            np.save(os.path.join(OUT, f"cd_{nm}_s{s}.npy"), P[(nm, s)])
        INFO[nm] = (npar, tt / len(SEEDS), pp / len(SEEDS))
        G.log(f"  {nm:9s} 파라미터 {npar:>9,}  학습 {tt/len(SEEDS):5.0f}s/시드  "
              f"추론 {pp/len(SEEDS):5.1f}s/시드  총 {time.time()-t0:.0f}s")

    G.log("\n  all 브랜치 단독, 최적 시프트, 시드별 계산 후 평균")
    G.log(f"  {'구성':9s} {'1군 행':>9s} {'전체':>9s} {'퓨처스':>9s} "
          f"{'추론':>7s} {'속도비':>7s}   시드별(1군)")
    t_base = INFO["base"][2]
    for nm, _ in plans:
        v = [sc(P[(nm, s)], R) for s in SEEDS]
        G.log(f"  {nm:9s} {np.mean(v):9.1f} "
              f"{np.mean([sc(P[(nm,s)], FULL) for s in SEEDS]):9.1f} "
              f"{np.mean([sc(P[(nm,s)], is_f) for s in SEEDS]):9.1f} "
              f"{INFO[nm][2]:7.1f} {t_base/max(INFO[nm][2],1e-9):6.2f}x   "
              f"[{', '.join(f'{x:.0f}' for x in v)}]")

    G.log("\n  base 대비 짝차이 (1군 행)")
    for nm, _ in plans[1:]:
        d = [sc(P[(nm, s)], R) - sc(P[("base", s)], R) for s in SEEDS]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
        G.log(f"    {nm:9s} {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/4  "
              f"[{', '.join(f'{x:+.1f}' for x in d)}]")

    G.log("\n  목표는 '이기는 것' 이 아니라 '지지 않으면서 빠른 것' 이다.")
    G.log("  현재 배치가 9분 26초이고 그 92%가 TabM 순전파다. 2배 빨라지면")
    G.log("  시드를 6~8개로 늘리거나 Stage2 를 더 돌릴 예산이 생긴다.")
