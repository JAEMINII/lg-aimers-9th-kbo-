# -*- coding: utf-8 -*-
"""pitcher_id 마스킹을 **통하는 구간에만** 건다. 1차 실험의 후속.

1차에서 본 것 (3시드)
    팔          전체     1군행    퓨처스행     1군 짝차이      퓨처스 짝차이
    base      901.7    892.5    660.1
    keep1     886.5    895.1    496.4      +2.5 (2/3)    -163.8 (0/3)
    keep2     901.7    898.7    612.7      +6.2 (2/3)     -47.5 (0/3)
    keep1_pb  877.8    894.3    434.9      +1.8 (2/3)    -225.2 (0/3)
    rand50    902.0    895.9    653.8      +3.4 (2/3)      -6.3 (1/3)

    1군에서는 네 팔 전부 양수, 퓨처스에서는 전부 재앙이다. 퓨처스 투수는
    행이 적어 마스킹하면 임베딩이 통째로 죽는다. 전체가 -0.0 인 건 그 상쇄다.

    체제처리가 퓨처스 +157.6 / 1군 -18.8 이었던 것의 정확한 거울상이다.
    그때 배운 규율을 그대로 적용한다 — 처리는 통하는 구간에만 건다.

이번에 재는 것
    구간별로 다른 모델을 태운다. 제출 10GB 예산에 지금 29.5MB 를 쓰므로
    모델을 나눠 싣는 비용은 사실상 0 이다.

        base       1군 = 0.6 all_u  + 0.4 reg_u    퓨 = 0.6 all_u  + 0.4 fut_u
        k2_both    1군 = 0.6 all_k2 + 0.4 reg_k2   퓨 = 0.6 all_k2 + 0.4 fut_u
        k2_1gun    1군 = 0.6 all_k2 + 0.4 reg_k2   퓨 = 0.6 all_u  + 0.4 fut_u  <-핵심
        r50_1gun   1군 = 0.6 all_r  + 0.4 reg_r    퓨 = 0.6 all_u  + 0.4 fut_u
        k3_1gun    keep3 판본. keep2 > keep1 이었으니 더 완만한 쪽도 본다

    한 시드에 브랜치 모델을 7개 학습하고 사후에 조합한다. 팔마다 따로
    학습하면 같은 모델을 여러 번 만드는 낭비가 된다.

시드 4개인 이유
    1차의 1군 +6.2 는 SE 3.9 로 t=1.6, 부호일치 2/3 이었다. 신호는 보이는데
    유의하지 않다. 판정하려면 시드를 늘려야 한다.

무엇이 원인인지도 같이 가른다
    rand50 도 1군 +3.4 였다. 효과의 절반쯤은 최신성이 아니라 임베딩 과의존을
    푼 정규화다. k2_1gun 과 r50_1gun 의 차이가 최신성 순수 몫이다.
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
EP2 = {"all": 1, "regular": 1, "futures": 4}

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


def fit(br, Xc, seed, idx, w, season):
    G.XC = torch.from_numpy(Xc)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, LR1, w=w, seed=seed, tag=f"{br} s{seed} S1")
    s2 = idx[season[idx] == VS - 1]
    ps = G.stage2_params(m)
    for e in range(EP2[br]):
        G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e,
                tag=f"{br} s{seed} S2e{e+1}")
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
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    catnames = list(PP.TABM_CATEGORICAL_FEATURES)
    ci = [cols.index(c) for c in catnames]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xn, Xc_u, cards = G.prep(np.concatenate([Xf, c4], 1), G.m_tr,
                             ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN = torch.from_numpy(Xn)

    p_col = catnames.index("pitcher_id")
    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)

    def masked(keep=None, rnd=None, seed=0):
        Xc = Xc_u.copy()
        if keep is not None:
            m = tr_idx[season[tr_idx] < VS - keep]
        else:
            r = np.random.default_rng(seed)
            m = tr_idx[r.random(len(tr_idx)) < rnd]
        Xc[m, p_col] = 0
        return Xc

    XC = {"u": Xc_u, "k2": masked(keep=2), "k3": masked(keep=3),
          "r50": masked(rnd=0.5, seed=0)}
    for k, v in XC.items():
        n = len(np.unique(v[tr_idx, p_col])) - 1
        G.log(f"  {k:4s} 살아남은 투수 {n:4d} / {cards[p_col]-1}   "
              f"가린 행 {int((v[tr_idx, p_col] == 0).sum()) - int((Xc_u[tr_idx, p_col] == 0).sum()):,}")

    P = {}
    for sd in SEEDS:
        t0 = time.time()
        for key in ("u", "k2", "k3", "r50"):
            P[("all", key, sd)] = fit("all", XC[key], sd, tr_idx, w10, season)
            P[("reg", key, sd)] = fit("regular", XC[key], sd, tr_idx[~t_isf],
                                      w10[~t_isf], season)
        P[("fut", "u", sd)] = fit("futures", XC["u"], sd, tr_idx[t_isf],
                                  w10[t_isf], season)
        G.log(f"  seed {sd} 끝 {time.time()-t0:.0f}s")

    def compose(a_key, r_key, f_all_key, sd):
        r = 0.6 * P[("all", a_key, sd)] + 0.4 * P[("reg", r_key, sd)]
        f = 0.6 * P[("all", f_all_key, sd)] + 0.4 * P[("fut", "u", sd)]
        return r, f

    ARMS = (("base", "u", "u", "u"), ("k2_both", "k2", "k2", "k2"),
            ("k2_1gun", "k2", "k2", "u"), ("k3_1gun", "k3", "k3", "u"),
            ("r50_1gun", "r50", "r50", "u"))

    def full(x):
        return np.where(is_f, x[1], x[0])

    G.log("\n" + "=" * 92)
    G.log(f"  {'팔':10s} {'전체':>8s} {'1군행':>8s} {'퓨처스행':>9s}"
          f"   기준 대비 (짝차이, 부호일치)")
    G.log("=" * 92)
    RES = {nm: [compose(a, r, f, sd) for sd in SEEDS]
           for nm, a, r, f in ARMS}
    base = RES["base"]
    for nm, *_ in ARMS:
        r = RES[nm]
        v = [float(np.mean([sc(full(x)) for x in r])),
             float(np.mean([sc(x[0], R) for x in r])),
             float(np.mean([sc(x[1], is_f) for x in r]))]
        if nm == "base":
            G.log(f"  {nm:10s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   <- 기준")
            continue
        tail = []
        for lab, fn, msk in (("전체", full, FULL),
                             ("1군", lambda x: x[0], R),
                             ("퓨처스", lambda x: x[1], is_f)):
            d = [sc(fn(a), msk) - sc(fn(b), msk) for a, b in zip(r, base)]
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            tail.append(f"{lab} {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f} "
                        f"{sum(1 for x in d if x > 0)}/{len(d)}")
        G.log(f"  {nm:10s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   "
              + "  ".join(tail))

    G.log("\n  읽는 법")
    G.log("    k2_1gun 이 전체에서 +5 이상 & 4/4 면 배치로 옮긴다.")
    G.log("    k2_1gun - r50_1gun 이 최신성의 순수 몫이다. 그 차이가 작으면")
    G.log("      연도를 자를 게 아니라 드롭아웃으로 다뤄야 한다.")
    G.log("    k3_1gun > k2_1gun 이면 아직 표본이 모자란 것이다.")
