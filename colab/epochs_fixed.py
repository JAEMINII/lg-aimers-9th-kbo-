# -*- coding: utf-8 -*-
"""스케줄러를 고친 뒤 에폭 수를 다시 잰다.

왜 다시 재나
    ep4 는 예전에 관문 +5.5 였는데 리더보드 -61 로 기각됐다. 그런데 그 판정은
    **망가진 스케줄러 위에서** 나온 것이다. 코사인을 전체 스텝에 걸면 에폭을
    늘려도 lr 이 그만큼 더 길게 감쇠할 뿐이라 후반이 계속 버려진다.

    codegap 에서 확인했다 — 에폭 단위 코사인으로 바꾸니 2에폭에서만 +19.2 였다.
    lr 궤적이 달라졌으니 '몇 에폭이 최적인가' 도 다시 물어야 한다.

    지인 설정의 epochs=2 도 그쪽 스케줄러(에폭 단위)에 맞춰 고른 값이다.
    같은 스케줄러를 쓰게 됐으니 그 2 가 우리한테도 최적인지는 별개 문제다.

무엇을 재나 (지인 전처리, VS=2024, all 브랜치, 1군 행 채점, 시드 4개)
    ep2   현행
    ep3
    ep4
    ep2_lr3   에폭은 2 인데 lr 을 3e-3 으로 (같은 총 학습량을 lr 로 밀어보기)

    Stage2 는 전부 1에폭 2e-4, output + 마지막 블록으로 고정한다.
    바뀌는 건 Stage1 뿐이다.

주의
    관문 한 폴드다. ep4 는 예전에 관문에서 이겼는데 리더보드에서 -61 이었다.
    여기서 이겨도 그것만으로 배치하지 않는다 — 리더보드 이력이 반대였던
    항목이라 특히 조심한다.
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

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(yv), bool)


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def fit(seed, idx, season, ep1, lr1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, ep1, lr1, seed=seed, tag=f"S1(ep{ep1},lr{lr1})")
    s2 = season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed, tag="S2")
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

    Xn, Xc, cards = G.prep(Xf, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    season = G.season.astype(np.float64)
    tr_idx = np.where(G.m_tr)[0]

    plans = [("ep2", 2, 2e-3), ("ep3", 3, 2e-3), ("ep4", 4, 2e-3),
             ("ep2_lr3", 2, 3e-3)]
    P = {}
    for nm, ep, lr in plans:
        t0 = time.time()
        for s in SEEDS:
            P[(nm, s)] = fit(s, tr_idx, season, ep, lr)
            np.save(os.path.join(OUT, f"ep_{nm}_s{s}.npy"), P[(nm, s)])
        G.log(f"  {nm:8s} {time.time()-t0:.0f}s")

    G.log("\n  all 브랜치 단독, 최적 시프트, 시드별 계산 후 평균")
    G.log(f"  {'구성':10s} {'1군 행':>9s} {'전체':>9s} {'퓨처스':>9s}   시드별(1군)")
    for nm, *_ in plans:
        v = [sc(P[(nm, s)], R) for s in SEEDS]
        G.log(f"  {nm:10s} {np.mean(v):9.1f} "
              f"{np.mean([sc(P[(nm,s)], FULL) for s in SEEDS]):9.1f} "
              f"{np.mean([sc(P[(nm,s)], is_f) for s in SEEDS]):9.1f}   "
              f"[{', '.join(f'{x:.0f}' for x in v)}]")

    G.log("\n  ep2 대비 짝차이 (1군 행)")
    for nm, *_ in plans[1:]:
        d = [sc(P[(nm, s)], R) - sc(P[("ep2", s)], R) for s in SEEDS]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
        G.log(f"    {nm:10s} {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/{len(d)}  "
              f"[{', '.join(f'{x:+.1f}' for x in d)}]")

    G.log("\n  ep4 는 예전에 관문 +5.5 인데 리더보드 -61 이었다(망가진 스케줄러 위).")
    G.log("  여기서 이겨도 그것만으로 배치하지 않는다. 다년 폴드로 한 번 더 본다.")
