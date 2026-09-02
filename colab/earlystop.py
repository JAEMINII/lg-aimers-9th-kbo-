# -*- coding: utf-8 -*-
"""브랜치마다 에폭을 홀드아웃으로 정하게 한다. 고정 에폭을 벗어난다.

문제
    train_conditional.py:179 가 model.fit 에 X_val 을 안 넘긴다. 그래서
    official_tabm.py:338 의 조기종료 분기가 죽어 있고, 무조건 Stage1 2에폭 /
    Stage2 1에폭을 돌고 끝난다. 우리 관문 코드도 같은 구조다.

    그런데 브랜치별로 배치 수가 열 배 넘게 차이난다.
        all       약 122만행 / 2048 = 596배치
        regular   약 109만행        = 533배치
        futures   약 16만행         =  79배치
    Stage2 는 마지막 시즌만 쓰므로 더 심하다. 퓨처스는 15배치뿐이다.
    하나의 에폭 수가 셋 모두에 맞을 이유가 없다.

방법
    학습 구간에서 5%를 무작위로 떼어 홀드아웃으로 쓴다. 에폭마다 홀드아웃
    Brier 를 재고 가장 좋은 에폭의 가중치를 되돌린다.
    시즌 단위로 떼지 않고 무작위로 떼는 이유 — 마지막 시즌은 Stage2 가 써야 한다.
    5%면 학습 데이터 손실이 작고, 관문(2024)과는 완전히 분리돼 있다.

    Stage1 은 최대 8에폭, Stage2 는 최대 6에폭까지 보고 알아서 멈추게 한다.
    고른 에폭을 브랜치별로 찍어 둔다 — 그 자체가 정보다.

주의
    이건 '학습량' 축이라 관문이 순위를 틀린 전례가 있다(ep1=4 를 관문은 +5.5 로
    쳤는데 리더보드는 -61). 관문에서 좋아도 그것만으로 제출을 정하지 않는다.
    다만 고정 에폭을 사람이 고르는 것보다는 데이터가 고르는 쪽이 낫다.
"""
import copy
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
HOLD = 0.05

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


@torch.no_grad()
def brier(model, idx, bs=8192):
    model.eval()
    tot, n = 0.0, 0
    ii = torch.from_numpy(idx)
    for s in range(0, len(idx), bs):
        b = ii[s:s + bs]
        lg = model(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV))
        p = lg.sigmoid().mean(dim=1).squeeze(-1)
        t = G.YY[b].to(G.DEV)
        tot += float(((p - t) ** 2).sum())
        n += len(b)
    return tot / n


def train_es(model, idx, val_idx, max_ep, lr, params=None, w=None, seed=42,
             tag=""):
    """에폭마다 홀드아웃을 재고 최고 지점의 가중치를 되돌린다."""
    best, best_ep, best_state = 1e9, 0, None
    for ep in range(1, max_ep + 1):
        G.train(model, idx, 1, lr, params=params, w=w, seed=seed + ep,
                tag=f"{tag} ep{ep}")
        v = brier(model, val_idx)
        if v < best - 1e-6:
            best, best_ep = v, ep
            best_state = copy.deepcopy(model.state_dict())
        G.log(f"      {tag} ep{ep}  holdout brier {v:.6f}"
              f"{'  <- best' if best_ep == ep else ''}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_ep


def run(seed, max1=8, max2=6):
    tr_idx = np.where(G.m_tr)[0]
    rng = np.random.default_rng(12345)          # 홀드아웃은 시드와 무관하게 고정
    hold = rng.random(len(tr_idx)) < HOLD
    W = DECAY ** (G.season[tr_idx].astype(np.float64) - 2019)
    P, chosen = {}, {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        fit = tr_idx[sel & ~hold]
        val = tr_idx[sel & hold]
        w = W[sel & ~hold]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        e1 = train_es(m, fit, val, max1, 2e-3, w=w, seed=seed, tag=f"{br} S1")
        s2 = G.season[fit] == VS - 1
        s2v = G.season[val] == VS - 1
        e2 = train_es(m, fit[s2], val[s2v] if s2v.any() else val, max2, 2e-4,
                      params=G.stage2_params(m), w=w[s2], seed=seed, tag=f"{br} S2")
        chosen[br] = (e1, e2)
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    r = np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                 0.4 * P["regular"] + 0.6 * P["all"])
    return r, chosen


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    Xn, Xc, cards = G.prep(d["X44"], G.m_tr, d["cat_idx"])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    tr_idx = np.where(G.m_tr)[0]
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        n = int(sel.sum())
        n2 = int((sel & (G.season[tr_idx] == VS - 1)).sum())
        G.log(f"  {br:8s} Stage1 {n:9,}행 {n//2048:4d}배치   "
              f"Stage2 {n2:8,}행 {n2//2048:4d}배치")

    ps = []
    for sd in SEEDS:
        t0 = time.time()
        r, ch = run(sd)
        ps.append(r)
        G.log(f"\n  seed {sd}  고른 에폭 " +
              "  ".join(f"{b} S1={e1} S2={e2}" for b, (e1, e2) in ch.items()) +
              f"   단독 {sc(r):.1f}   {time.time()-t0:.0f}s\n")
    r = np.mean(ps, 0)
    G.log(f"  조기종료 {len(SEEDS)}시드평균  단독 {sc(r):7.1f}   "
          f"제출비중 {sc(0.10*CB + 0.30*MLPF + 0.60*r):7.1f}")
    np.save(os.path.join(OUT, "es_tabm.npy"), r)
    G.log("  (비교) 고정 2/1 4시드평균 단독 908.3  제출비중 934.6")
