# -*- coding: utf-8 -*-
"""학습곡선. n 을 바꿔가며 관문 점수를 잰다. 두 가지를 동시에 답한다.

① TabPFN 의 상한선
    TabPFN v2 는 학습 1만 행 / 500피처가 설계 목표다. 우리는 1,221,585행이라
    122배다. 부분표본 배깅으로 돌릴 수는 있지만 각 하위모델이 데이터의 0.8%만
    본다. 그게 얼마나 손해인지 문헌이 아니라 **우리 데이터**에서 재야 한다.

    1만 행으로 학습한 TabM 의 점수가 곧 그 구간의 대략적 천장이다. TabPFN 이
    같은 1만 행에서 튜닝된 모델보다 나아봐야 그 근처이지, 122만 행 모델을
    넘을 수는 없다.

② 우리가 정보 포화 상태인가
    곡선이 122만에서 아직 오르고 있으면 데이터/피처를 더 넣을 여지가 있다.
    평평하면 이미 뽑을 걸 다 뽑았다는 뜻이고, 남은 4% 는 다른 데서 와야 한다.

    오늘 "상황 x 투수 조건부 축 여덟 중 여덟이 신호 0%" 를 확인했다.
    곡선이 평평하면 같은 결론을 다른 각도에서 확인하는 셈이다.

설계
    배치 구성 그대로 (지인 전처리 + abs_regime + 옛퓨처스 가중 0.1,
    Stage1 2에폭 lr 3e-3, Stage2). all 브랜치 하나만 — 곡선 모양만 보면 된다.

    부분표본은 **시즌 비율을 유지**해서 뽑는다. 무작위로 뽑으면 작은 n 에서
    시즌 구성이 흔들려 곡선에 잡음이 낀다.

    작은 n 에서 에폭 2회는 스텝이 너무 적다 (1만 행 = 5스텝). 그래서 스텝 수를
    맞춘다 — n 이 작으면 에폭을 늘려 총 스텝을 최소 200 으로 채운다.
    안 그러면 '데이터가 적어서' 가 아니라 '학습을 덜 해서' 진 게 된다.
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
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
MIN_STEPS = 200
SIZES = (10_000, 30_000, 100_000, 300_000, 600_000, 1_221_585)

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


def subsample(tr_idx, season, n, seed):
    """시즌 비율을 유지해서 n 개를 뽑는다."""
    if n >= len(tr_idx):
        return tr_idx
    rng = np.random.default_rng(seed)
    out = []
    s = season[tr_idx]
    for v in np.unique(s):
        pool = tr_idx[s == v]
        k = int(round(n * len(pool) / len(tr_idx)))
        k = min(max(k, 1), len(pool))
        out.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(out))


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
    w_all = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
    wmap = dict(zip(tr_idx.tolist(), w_all.tolist()))
    G.log(f"  전체 학습 {len(tr_idx):,}   크기 {len(SIZES)}종 x 시드 {len(SEEDS)}개")

    print("\n" + "=" * 84)
    print(f"  {'학습행수':>10s} {'비율':>7s} {'에폭':>5s} {'스텝':>6s} "
          f"{'전체':>8s} {'1군':>8s} {'퓨처스':>8s}   전체 대비")
    print("=" * 84)
    ref = None
    for n in SIZES:
        vals = []
        t0 = time.time()
        for sd in SEEDS:
            idx = subsample(tr_idx, season, n, sd)
            w = np.array([wmap[i] for i in idx.tolist()])
            steps_1ep = int(np.ceil(len(idx) / 2048))
            ep = max(2, int(np.ceil(MIN_STEPS / steps_1ep)))
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            m = G.make_model()
            G.train(m, idx, ep, LR1, w=w, seed=sd, tag=f"n{n} s{sd} S1")
            s2 = idx[season[idx] == VS - 1]
            if len(s2) >= 256:
                G.train(m, s2, 1, 2e-4, params=G.stage2_params(m), seed=sd,
                        tag=f"n{n} s{sd} S2")
            vals.append(G.predict(m, gate))
            del m
            torch.cuda.empty_cache()
        p = np.mean(vals, 0)
        np.save(os.path.join(OUT, f"lc_n{n}.npy"), p)
        a, b, c = sc(p), sc(p, R), sc(p, is_f)
        if ref is None:
            ref = a
        steps_1ep = int(np.ceil(min(n, len(tr_idx)) / 2048))
        ep = max(2, int(np.ceil(MIN_STEPS / steps_1ep)))
        print(f"  {min(n, len(tr_idx)):10,d} {min(n,len(tr_idx))/len(tr_idx)*100:6.1f}% "
              f"{ep:5d} {ep*steps_1ep:6d} {a:8.1f} {b:8.1f} {c:8.1f}   "
              f"{a-ref:+8.1f}   {time.time()-t0:.0f}s")

    print("\n  읽는 법")
    print("    1만 행 점수 = TabPFN 이 놓일 자리의 대략적 천장이다.")
    print("    곡선이 122만에서 평평하면 정보 포화 — 데이터를 더 줘도 안 오른다.")
    print("    아직 오르면 피처/데이터 쪽에 여지가 남아 있다는 뜻이다.")
