# -*- coding: utf-8 -*-
"""Stage1 학습률을 훑는다. 지금까지 상수 하나로 얻은 것 중 신호가 제일 단단하다.

경위
    codegap 에서 코사인을 배치마다 돌려 lr 을 0 까지 감쇠시키던 걸 찾아
    에폭 단위로 고쳤다. 1군 행 869.5 -> 888.8 (+19.2).
    그건 실효 lr 을 올린 조치였다.

    epochs_fixed 에서 방향이 확인됐다 (all 브랜치, 1군 행, 4시드 짝비교)
        ep3       -6.5 +- 5.7  t=-1.15  1/4     에폭은 늘리면 안 된다
        ep4      -37.6 +- 6.3  t=-5.98  0/4     리더보드 -61 과 방향 일치
        ep2_lr3   +6.1 +- 1.3  t= 4.61  4/4     lr 을 올리는 건 통한다
    퓨처스도 576.2 -> 592.9 로 같이 올랐다.

    두 결과가 같은 방향을 가리킨다. 이 모델은 lr 측면에서 계속 덜 학습돼 있었다.
    그러면 3e-3 이 끝인지, 더 갈 수 있는지를 봐야 한다.

무엇을 재나 (지인 전처리, VS=2024, all 브랜치, 1군 행 채점, 시드 4개)
    lr 2e-3(현행) / 3e-3 / 4e-3 / 5e-3, Stage1 2에폭 고정
    Stage2 는 전부 1에폭 2e-4, output + 마지막 블록으로 고정한다.

    고원 위에서는 덜 극단적인 쪽을 고른다. 시즌가중을 3~8 고원에서 3.5 로
    고른 것과 같은 규율이다. 최댓값이 경계에 붙으면 더 훑는다.

주의
    한 폴드다. 다만 lr 은 피처나 라우팅과 달리 연도별 구조에 덜 민감한
    최적화 하이퍼파라미터라, 폴드 간 뒤집힘 위험이 상대적으로 낮다.
    그래도 배치 전에 다년으로 한 번 더 본다.
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
LRS = (2e-3, 3e-3, 4e-3, 5e-3)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(yv), bool)


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def fit(seed, idx, season, lr1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, lr1, seed=seed, tag=f"S1 lr{lr1:g}")
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

    P = {}
    for lr in LRS:
        t0 = time.time()
        for s in SEEDS:
            P[(lr, s)] = fit(s, tr_idx, season, lr)
            np.save(os.path.join(OUT, f"lr_{lr:g}_s{s}.npy"), P[(lr, s)])
        G.log(f"  lr {lr:g}  {time.time()-t0:.0f}s")

    G.log("\n  all 브랜치 단독, 최적 시프트, 시드별 계산 후 평균")
    G.log(f"  {'lr':>8s} {'1군 행':>9s} {'전체':>9s} {'퓨처스':>9s}   시드별(1군)")
    for lr in LRS:
        v = [sc(P[(lr, s)], R) for s in SEEDS]
        G.log(f"  {lr:8g} {np.mean(v):9.1f} "
              f"{np.mean([sc(P[(lr,s)], FULL) for s in SEEDS]):9.1f} "
              f"{np.mean([sc(P[(lr,s)], is_f) for s in SEEDS]):9.1f}   "
              f"[{', '.join(f'{x:.0f}' for x in v)}]")

    G.log("\n  2e-3 대비 짝차이")
    for lr in LRS[1:]:
        for tag, m in (("1군", R), ("퓨처스", is_f), ("전체", FULL)):
            d = [sc(P[(lr, s)], m) - sc(P[(2e-3, s)], m) for s in SEEDS]
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            G.log(f"    lr {lr:g}  {tag:6s} {mu:+7.1f}+-{se:5.1f} "
                  f"t={mu/max(se,1e-9):5.2f} {sum(1 for x in d if x > 0)}/4  "
                  f"[{', '.join(f'{x:+.1f}' for x in d)}]")

    best = max(LRS, key=lambda l: np.mean([sc(P[(l, s)], R) for s in SEEDS]))
    G.log(f"\n  1군 기준 최선 lr = {best:g}")
    if best == LRS[-1]:
        G.log("  최댓값이 경계다. 더 큰 lr 을 훑어야 한다.")
    else:
        G.log("  고원이면 덜 극단적인 쪽을 고른다. 시즌가중 3.5 를 고른 것과 같은 규율이다.")
