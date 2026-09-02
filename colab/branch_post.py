# -*- coding: utf-8 -*-
"""브랜치별 예측을 저장하고, 라우팅 방식을 사후에 전부 훑는다.

지금까지 스크립트는 합쳐진 예측만 남겨서 라우팅을 바꿔 보려면 매번 재학습했다.
브랜치별로 남기면 한 번 학습으로 아래를 전부 잰다 — 재학습 없이 몇 초다.

    A 확률 혼합 비중        0.0 ~ 1.0  (0.4 가 현행, 1.0 이 순수 라우팅)
    B logit 잔차 라우팅      logit(p_all) + alpha x (logit(p_br) - logit(p_all))
                            F/R 에 다른 alpha 를 줄 수 있다
    C 브랜치별 보정          F 와 R 에 다른 시프트

왜 다시 재나
    분기 비중은 예전에 훑었고 1.0(순수 라우팅)이 833.9 로 최악, 0.2 가 879.1 이었다.
    그런데 그때 퓨처스 브랜치는 옛 체제 데이터로 오염돼 있었다(퓨처스 481.4).
    체제 처리로 619.4 까지 올라왔으니 최적 비중이 달라졌을 수 있다.
    브랜치가 좋아졌으면 더 실을 만하다.

학습은 체제 처리를 건 판본 하나만 한다 (flag + futures 브랜치 가중 0.1).
regime2 에서 브랜치별로 나눠 거는 것과 전체에 거는 것이 거의 같게 나왔으므로
구현이 간단한 쪽을 쓴다.
"""
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
OLD_F_MAX = 2022
EPS = 1e-6
FIXED = -0.005      # submit_18(1048)이 실제로 쓰는 시프트

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    """최적 시프트. 구성 간 판별력 비교용."""
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def scf(p, m=None):
    """고정 시프트. 배치에서 실제로 받는 점수다.

    최적 시프트로만 재면 미보정 구성이 과대평가된다. flatMLP 에서 직접 봤다 —
    관문은 빼면 -36 이라 했는데 리더보드는 +2 였다. 그 차이의 절반이
    '관문이 후보마다 최적 시프트를 공짜로 주는' 데서 나왔다.
    """
    m = FULL if m is None else m
    return F.bss(F.shift(p[m], FIXED), yv[m])


def lg(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def train_branches(seed, X, ci, fw):
    Xn, Xc, cards = G.prep(X, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    tr_idx = np.where(G.m_tr)[0]
    old_f = G.is_f[tr_idx] & (G.season[tr_idx] <= OLD_F_MAX)
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[sel]
        ow = fw.get(br, 1.0)
        w = None if ow == 1.0 else np.where(old_f[sel], ow, 1.0).astype(np.float64)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, 2e-3, w=w, seed=seed, tag=f"{br} S1")
        s2 = G.season[idx] == VS - 1
        G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
                tag=f"{br} S2")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return P


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    base = d["X44"]
    ci = list(d["cat_idx"])
    season, isf = d["season"].astype(np.float64), d["is_f"]
    flag = ((isf & (season > OLD_F_MAX)) | (~isf)).astype(np.float32)[:, None]
    X = np.concatenate([base, flag], 1)
    ci_f = ci + [base.shape[1]]

    Ps = []
    for sd in SEEDS:
        t0 = time.time()
        Ps.append(train_branches(sd, X, ci_f, {"all": 0.1, "futures": 0.1,
                                               "regular": 0.1}))
        G.log(f"  seed {sd} {time.time()-t0:.0f}s")
    A = {b: np.mean([p[b] for p in Ps], 0) for b in ("all", "futures", "regular")}
    for b, v in A.items():
        np.save(os.path.join(OUT, f"bp_{b}.npy"), v)
    # 시드별로도 남긴다. 평균만 저장하면 나중에 시드 불확실성을 못 본다 —
    # 실제로 alpha 선택을 확인하려다 시드별 예측이 없어 못 했다.
    for i, sd in enumerate(SEEDS):
        for b in ("all", "futures", "regular"):
            np.save(os.path.join(OUT, f"bp_{b}_s{sd}.npy"), Ps[i][b])
    G.log("")
    G.log(f"  브랜치 단독   all {sc(A['all']):7.1f}   "
          f"futures(F행) {sc(A['futures'], is_f):7.1f}   "
          f"regular(R행) {sc(A['regular'], ~is_f):7.1f}")

    def route_prob(w):
        return np.where(is_f, w * A["futures"] + (1 - w) * A["all"],
                        w * A["regular"] + (1 - w) * A["all"])

    G.log("")
    G.log("  A. 확률 혼합 비중   (최적시프트 / 고정시프트)")
    best = (-1e9, None)
    bestf = (-1e9, None)
    for w in np.arange(0.0, 1.01, 0.1):
        r = route_prob(w)
        s, sf = sc(r), scf(r)
        best = max(best, (s, w))
        bestf = max(bestf, (sf, w))
        G.log(f"    분기 {w:.1f}   전체 {s:8.1f} / {sf:8.1f}   "
              f"1군 {sc(r, ~is_f):8.1f}   퓨처스 {sc(r, is_f):8.1f}")
    G.log(f"    -> 최적시프트 기준 {best[1]:.1f} ({best[0]:.1f})   "
          f"고정시프트 기준 {bestf[1]:.1f} ({bestf[0]:.1f})")

    G.log("")
    G.log("  B. logit 잔차 라우팅  logit(all) + alpha x (logit(분기) - logit(all))")
    G.log("     F/R 에 다른 alpha 를 준다. 점수는 고정 시프트 기준")
    la = lg(A["all"])
    lf, lr = lg(A["futures"]), lg(A["regular"])
    bb = (-1e9, None, None)
    for af in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
        row = []
        for ar in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            z = np.where(is_f, la + af * (lf - la), la + ar * (lr - la))
            s = scf(1 / (1 + np.exp(-z)))      # 고정 시프트로 판정한다
            bb = max(bb, (s, af, ar))
            row.append(f"{s:7.1f}")
        G.log(f"    aF={af:.1f}  " + " ".join(row))
    G.log(f"     (aR = 0.0 0.2 0.4 0.6 0.8 1.0)")
    G.log(f"    -> 최적 aF={bb[1]:.1f} aR={bb[2]:.1f}  ({bb[0]:.1f})")

    G.log("")
    G.log("  C. 브랜치별 보정")
    G.log("     주의: 검증 데이터에서 최적 시프트를 구하면 배치에서 못 받는 값이다.")
    G.log("     그래서 F/R 시프트 차이가 '얼마나 필요한가' 만 본다. 그 값을 쓰려면")
    G.log("     학습 구간에서 따로 추정해야 한다.")
    r = route_prob(bestf[1])
    sF = F.best_shift(r[is_f], yv[is_f])[1]
    sR = F.best_shift(r[~is_f], yv[~is_f])[1]
    G.log(f"    검증에서 본 최적  F {sF:+.4f}   R {sR:+.4f}   차이 {sF-sR:+.4f}")
    G.log(f"    예측평균  F {r[is_f].mean():.4f} (실제 {yv[is_f].mean():.4f})   "
          f"R {r[~is_f].mean():.4f} (실제 {yv[~is_f].mean():.4f})")
    G.log(f"    고정 공통시프트 {scf(r):8.1f}")
    G.log("")
    G.log("  주의: 여기서 고른 비중·alpha·시프트는 검증 연도에 맞춘 값이다.")
    G.log("  한 폴드에서 고른 하이퍼파라미터는 리더보드에서 여러 번 뒤집혔다.")
