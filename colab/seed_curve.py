# -*- coding: utf-8 -*-
"""시드를 늘리면 몇 점 버는지 곡선을 그린다. 예산 확보의 값어치를 먼저 잰다.

왜 이걸 먼저 하나
    평가서버에 L4 GPU 22.4GB 가 있고 제출 10GB 중 29.5MB 만 쓰고 있다.
    추론을 GPU 로 옮기면 예산이 남고 그걸 시드로 채울 수 있다.

    그런데 '몇 점짜리냐' 를 모르는 채로 포팅에 몇 시간을 쓰는 건 순서가 틀렸다.
    torch 를 싣는 데는 실질적 위험이 둘 있다 — 설치 10분 안에 되는지, 규칙 4
    감사가 GPU 경로에서도 최대차 0 인지. 제출 슬롯이 하루 5개고 실패하면
    한 장을 버린다. 그러니 보상부터 재고 판단한다.

무엇을 재나
    배치 구성 그대로 8시드를 학습하고, k개를 평균했을 때의 관문 점수를 본다.
    k마다 가능한 조합을 여러 개 뽑아 평균해서 곡선을 매끄럽게 만든다
    (k=1 은 8개 전부, k=4 는 무작위 조합 20개 식).

    그리고 score(k) = S_inf - c/k 로 맞춰 **무한 시드 천장**을 추정한다.
    지금 배치가 시드 1개이므로 k=1 -> k=8 의 차이가 곧 확보 가능한 이득이다.

읽는 법
    곡선이 k=4 쯤에서 평평해지면 굳이 GPU 포팅까지 갈 필요 없이 CPU 예산 안에서
    시드를 조금 늘리는 것으로 끝날 수도 있다 (지금 9분 26초 / 10분이라 여유가
    거의 없지만, HistGB 를 뺀 submit_27 은 트리 12,000개 순회를 아낀다).
    k=8 에서도 계속 오르면 GPU 포팅의 값어치가 그만큼 커진다.
"""
import itertools
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
SEEDS = (42, 1, 777, 2, 7, 13, 99, 2024)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
MAX_COMB = 20

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def one(seed, tr_idx, t_isf, w, season):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"{br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e,
                    tag=f"{br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(is_f, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


if __name__ == "__main__":
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])
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

    P = []
    for sd in SEEDS:
        t0 = time.time()
        p = one(sd, tr_idx, t_isf, w10, season)
        np.save(os.path.join(OUT, f"sc_s{sd}.npy"), p)
        P.append(p)
        G.log(f"  seed {sd:5d}  단독 {sc(p):7.1f}  {time.time()-t0:.0f}s")

    rng = np.random.default_rng(0)
    print("\n" + "=" * 76)
    print(f"  {'시드수':>6s} {'전체':>9s} {'표준편차':>9s} {'1군':>9s} "
          f"{'퓨처스':>9s}   k=1 대비")
    print("=" * 76)
    curve = []
    for k in range(1, len(SEEDS) + 1):
        combs = list(itertools.combinations(range(len(SEEDS)), k))
        if len(combs) > MAX_COMB:
            combs = [tuple(rng.choice(len(SEEDS), k, replace=False))
                     for _ in range(MAX_COMB)]
        vals, v1, v2 = [], [], []
        for c in combs:
            q = np.mean([P[i] for i in c], 0)
            vals.append(sc(q))
            v1.append(sc(q, R))
            v2.append(sc(q, is_f))
        mu = float(np.mean(vals))
        curve.append((k, mu))
        base = curve[0][1]
        print(f"  {k:6d} {mu:9.1f} {float(np.std(vals)):9.2f} "
              f"{float(np.mean(v1)):9.1f} {float(np.mean(v2)):9.1f}   "
              f"{mu - base:+9.1f}   (조합 {len(combs)}개)")

    # score(k) = S_inf - c/k 로 맞춰 천장을 본다
    ks = np.array([c[0] for c in curve], float)
    ys = np.array([c[1] for c in curve], float)
    A = np.c_[np.ones(len(ks)), 1.0 / ks]
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    print(f"\n  score(k) = {coef[0]:.1f} - {-coef[1]:.1f}/k  로 맞춤")
    print(f"  무한 시드 천장 추정 {coef[0]:.1f}   "
          f"(k=1 {ys[0]:.1f} -> k=8 {ys[-1]:.1f} -> 무한 {coef[0]:.1f})")
    print(f"\n  지금 배치는 시드 1개다. k=1 -> k=8 이 확보 가능한 이득이고,")
    print(f"  천장까지의 나머지는 시드를 더 늘려도 안 오는 몫이다.")
    print(f"  이 값이 작으면 GPU 포팅의 우선순위를 낮춘다.")
