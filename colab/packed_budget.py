# -*- coding: utf-8 -*-
"""tabm-packed 가 학습 부족으로 무너진 건지 본다.

관측
    arch_sweep 에서 tabm-packed 가 1군 -117.7 (0/4), 퓨처스 -121.7 (0/4) 였다.
    파라미터가 23,217,792 로 tabm(874,048)의 26.6배다.

왜 이상한가
    packed 는 k개 완전독립 MLP 를 묶은 것, 즉 딥 앙상블이다. 딥 앙상블이
    단일 모델보다 **나쁠** 구조적 이유가 없다. -117 은 '비싸지만 비슷하다' 가
    아니라 학습이 안 됐다는 신호에 가깝다.

가설
    2에폭 / lr 3e-3 은 874k 짜리에 맞춰 고른 값이다. 26배 큰 모델에 같은
    예산을 주면 부족할 수 있다. 오늘 이 모델 계열이 학습 예산에 민감하다는
    걸 두 번 확인했다 — 스케줄러 고쳐 +19.2, lr 올려 +6.1.

무엇을 재나 (지인 전처리, VS=2024, all 브랜치, 1군 행, 시드 3개)
    tabm          현행 기준선 (2에폭 lr3e-3)
    packed        같은 예산
    packed_lr5    lr 5e-3
    packed_ep4    4에폭
    packed_ep4lr5 4에폭 + lr 5e-3

    학습이 부족한 거라면 예산을 늘릴수록 가파르게 올라야 한다.
    그래도 -100 근처면 구조가 이 데이터에 안 맞는 것이고 그대로 접는다.

시드가 3개인 이유
    packed 는 시드당 68초라 tabm(87초)보다 오히려 싸다. 그런데 ep4 판본이
    두 배가 되므로 3개로 줄여 전체를 20분 안에 맞춘다. 부호 일관성은
    3/3 으로 본다.

주의
    이겨도 곧바로 배치 못 한다. 파라미터 23M 이면 npz 가 90MB 쯤 되고
    시드 3개면 270MB 다. 추론 시간과 제출 용량을 먼저 확인해야 한다.
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
SEEDS = (42, 1, 777)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(yv), bool)


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def fit(seed, idx, season, arch, ep, lr):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model(arch=arch)
    G.train(m, idx, ep, lr, seed=seed, tag=f"{arch} ep{ep} lr{lr:g}")
    s2 = season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed, tag="S2")
    p = G.predict(m, gate)
    n = sum(x.numel() for x in m.parameters())
    del m
    torch.cuda.empty_cache()
    return p, n


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

    plans = [("tabm", "tabm", 2, 3e-3),
             ("packed", "tabm-packed", 2, 3e-3),
             ("packed_lr5", "tabm-packed", 2, 5e-3),
             ("packed_ep4", "tabm-packed", 4, 3e-3),
             ("packed_ep4lr5", "tabm-packed", 4, 5e-3)]
    P, NP = {}, {}
    for nm, arch, ep, lr in plans:
        t0 = time.time()
        for s in SEEDS:
            P[(nm, s)], NP[nm] = fit(s, tr_idx, season, arch, ep, lr)
            np.save(os.path.join(OUT, f"pk_{nm}_s{s}.npy"), P[(nm, s)])
        G.log(f"  {nm:14s} 파라미터 {NP[nm]:>11,}  {time.time()-t0:.0f}s")

    G.log("\n  all 브랜치 단독, 최적 시프트, 시드별 계산 후 평균")
    G.log(f"  {'구성':14s} {'1군 행':>9s} {'전체':>9s} {'퓨처스':>9s}   시드별(1군)")
    for nm, *_ in plans:
        v = [sc(P[(nm, s)], R) for s in SEEDS]
        G.log(f"  {nm:14s} {np.mean(v):9.1f} "
              f"{np.mean([sc(P[(nm,s)], FULL) for s in SEEDS]):9.1f} "
              f"{np.mean([sc(P[(nm,s)], is_f) for s in SEEDS]):9.1f}   "
              f"[{', '.join(f'{x:.0f}' for x in v)}]")

    G.log("\n  tabm 대비 짝차이 (1군 행)")
    for nm, *_ in plans[1:]:
        d = [sc(P[(nm, s)], R) - sc(P[("tabm", s)], R) for s in SEEDS]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
        G.log(f"    {nm:14s} {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/3  "
              f"[{', '.join(f'{x:+.0f}' for x in d)}]")

    G.log("\n  packed 계열이 예산을 늘릴수록 가파르게 오르면 학습 부족이 맞다.")
    G.log("  -100 근처에 머물면 구조가 이 데이터에 안 맞는 것이고 접는다.")
    G.log("  이겨도 파라미터 23M 이라 제출 용량·추론 시간부터 확인해야 한다.")
