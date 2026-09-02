# -*- coding: utf-8 -*-
"""같은 추론 비용에서 '시드를 늘리는 것' 과 '변형을 섞는 것' 중 무엇이 나은가.

발견
    저카디널리티 범주화   단독 -4.2  혼합 +2.0
    Stage1 을 섞기        단독  -0.5  혼합 +1.3 (0.5:0.5, 아직 오르는 중)
    둘 다 '단독은 그대로거나 낮은데 혼합이 오른다'.

    우리 앙상블의 병목은 상관 0.9 였다. TabM 을 더 좋게 만드는 것보다
    다르게 만드는 쪽이 혼합에 이롭다는 뜻이다. 그동안 단독 점수만 보고
    설정을 골라왔는데 그 기준이 틀렸을 수 있다.

공정한 비교
    Stage1 을 섞으려면 추론에서 모델이 하나 더 필요하다. 시드를 하나 더 쓰는 것과
    같은 비용이다. 그래서 '모델 2개' 예산에서 배치만 바꿔 비교한다.

        A  시드2개 x Stage2            현재 제출본
        B  시드1개 x (Stage2 + Stage1)
        C  시드1개 Stage2 + 시드1개 범주화Stage2
        D  시드1개 Stage2 + 시드1개 범주화Stage1

    전부 브랜치당 모델 2개다. 추론 시간이 같다.
    비교는 혼합 기준으로 한다. 단독 점수는 참고로만 찍는다.
"""
import itertools
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
DECAY = 3.5
VS = 2024
LOWCARD = ["game_month", "game_dayofweek", "inning",
           "balls_before", "strikes_before", "outs_before"]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


def run(seed, ci):
    """Stage1 예측과 Stage2 예측을 둘 다 돌려준다. 학습 비용은 한 번이다."""
    Xn, Xc, cards = G.prep(BIGX, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    tr_idx = np.where(G.m_tr)[0]
    W = DECAY ** (G.season[tr_idx].astype(np.float64) - 2019)
    P1, P2 = {}, {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[sel]
        w = W[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, 2e-3, w=w, seed=seed, tag=f"{br} S1")
        P1[br] = G.predict(m, gate)
        s2 = G.season[idx] == VS - 1
        G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), w=w[s2],
                seed=seed, tag=f"{br} S2")
        P2[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()

    def route(P):
        return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                        0.4 * P["regular"] + 0.6 * P["all"])
    return route(P1), route(P2)


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    BIGX = d["X44"]
    F44 = list(d["F44"])
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    M = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))
    ci_b = list(d["cat_idx"])
    ci_l = sorted(set(ci_b) | {F44.index(c) for c in LOWCARD})

    def blend(r):
        return sc(0.10 * CB + 0.30 * M + 0.60 * r)

    B1, B2, L1, L2 = {}, {}, {}, {}
    for sd in SEEDS:
        t0 = time.time()
        B1[sd], B2[sd] = run(sd, ci_b)
        L1[sd], L2[sd] = run(sd, ci_l)
        G.log(f"  seed {sd} 완료 {time.time()-t0:.0f}s")

    G.log("")
    G.log("  구성                            단독      혼합    (모델 2개 예산)")
    combos = {
        "A 시드2 x Stage2": [B2[SEEDS[0]], B2[SEEDS[1]]],
        "B 시드1 x (S2+S1)": [B2[SEEDS[0]], B1[SEEDS[0]]],
        "C S2 + 범주화S2": [B2[SEEDS[0]], L2[SEEDS[0]]],
        "D S2 + 범주화S1": [B2[SEEDS[0]], L1[SEEDS[0]]],
        "(참고) 시드1 x S2": [B2[SEEDS[0]]],
        "(참고) 시드4 x S2": [B2[s] for s in SEEDS],
    }
    for name, parts in combos.items():
        r = np.mean(parts, 0)
        G.log(f"  {name:30s} {sc(r):8.1f} {blend(r):9.1f}")

    G.log("")
    G.log("  시드 전체를 써서 배치별로 (시드 평균 후 섞기)")
    b2 = np.mean([B2[s] for s in SEEDS], 0)
    b1 = np.mean([B1[s] for s in SEEDS], 0)
    l2 = np.mean([L2[s] for s in SEEDS], 0)
    l1 = np.mean([L1[s] for s in SEEDS], 0)
    for name, r in (("Stage2만", b2), ("S2+S1", (b2 + b1) / 2),
                    ("S2+범주S2", (b2 + l2) / 2),
                    ("S2+S1+범주S2+범주S1", (b2 + b1 + l2 + l1) / 4)):
        G.log(f"    {name:24s} 단독 {sc(r):8.1f}   혼합 {blend(r):9.1f}")

    G.log("")
    G.log("  상관 (다양성 확인)")
    for n1, n2, r1, r2 in (("S2", "S1", b2, b1), ("S2", "범주S2", b2, l2),
                           ("S2", "범주S1", b2, l1)):
        G.log(f"    {n1} ↔ {n2:8s} {np.corrcoef(r1, r2)[0,1]:.4f}")
    G.log(f"    S2 ↔ CatBoost  {np.corrcoef(b2, CB)[0,1]:.4f}")
    for name, r in (("S2", b2), ("범주S2", l2)):
        G.log(f"    {name:8s} ↔ CatBoost {np.corrcoef(r, CB)[0,1]:.4f}  "
              f"↔ flatMLP {np.corrcoef(r, M)[0,1]:.4f}")
