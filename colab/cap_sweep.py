# -*- coding: utf-8 -*-
"""k 와 d_block 을 **같이** 훑는다. 앞선 k=64 기각은 조건을 빼고 잰 것이다.

왜 다시 여는가
    TabM 공식 문서에 이렇게 적혀 있다.
        "increasing k up to a certain threshold can improve performance"
        "When increasing k, also increase d_block or n_blocks"
    두 번째 문장이 조건이다. 우리 기록은 "k=32 -> 64  -7.4 이고 시간은 2배" 인데,
    그때 d_block 은 256 그대로였다. **문서가 요구하는 조건을 빼고 잰 셈이다.**

    "기각 근거가 모델·설정에 묶여 있으면 그게 바뀔 때 다시 재라" — CatBoost -539 가
    설정 실수였고 제대로 재니 +20.3 이었던 전례가 있다.

팔 (d_block 범위는 문서상 64~1024)
    base       k=32  d=256    현행
    k64d256    k=64  d=256    이미 진 구성. 재현 대조군
    k64d384    k=64  d=384    조건을 지킨 구성
    k64d512    k=64  d=512
    k32d384    k=32  d=384    d_block 단독 효과를 가른다

    k64d256 이 다시 -7 근처면 앞선 실험과 이어진 것이다.
    k64d384/512 가 base 를 넘으면 조건이 빠졌던 게 맞다.

주의
    k=64 는 파라미터가 두 배라 배치당 시간이 늘어난다. 추론 예산(10분)은
    GPU forward 가 2.18초라 여유가 크므로 제약이 아니다.
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
FOLDS = (2022, 2024)
OLD_F_MAX, OLD_W, LR1 = 2022, 0.1, 3e-3
EP2 = {"all": 1, "regular": 1, "futures": 4}
# (이름, k, d_block)
ARMS = [("base", 32, 256), ("k64d256", 64, 256), ("k64d384", 64, 384),
        ("k64d512", 64, 512), ("k32d384", 32, 384)]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

K, DB = 32, 256


def one(seed, tr_idx, t_isf, w, season, VS, gate, isf_g):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model(k=K, d_block=DB)
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"VS{VS} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        pr = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=pr, seed=seed + e,
                    tag=f"VS{VS} {br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(isf_g, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


if __name__ == "__main__":
    d0 = F.build(DATA, VS=2024)
    season = d0["season"].astype(np.float64)
    isf, y = d0["is_f"], d0["y"].astype(np.float64)
    F44 = list(d0["F44"])
    rid = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id"])["row_id"].to_numpy()
    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(rid).to_numpy()

    ALL = {}
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        G.gate, G.yv = gate, yv
        m_tr = season < VS
        hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
        Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
        cols = list(Xs.columns)
        Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
        ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        dv = F.build(DATA, VS=VS)
        Xfr[:, cols.index("plat_dev")] = \
            dv["X44"][:, F44.index("plat_dev")].astype(np.float32)
        old = season <= OLD_F_MAX
        c4 = np.where(old & isf, 0.0,
                      np.where(old & ~isf, 1.0,
                               np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
        Xin = np.concatenate([Xfr, c4], 1)
        Xn, Xc, cards = G.prep(Xin, m_tr, ci + [Xfr.shape[1]])
        G.Xn, G.cards = Xn, cards
        G.m_tr = m_tr
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}")

        for nm, kk, db in ARMS:
            K, DB = kk, db
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"cap_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:9s} k={kk} d_block={db} 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 96)
    print("  k x d_block 동시 스윕 — 문서가 요구하는 조건을 지킨 재측정")
    print("=" * 96)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm, kk, db in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:9s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)
    print("\n  k64d256 이 -7 근처면 앞선 기각(k=32->64 -7.4)과 이어진 것이다.")
    print("  k64d384/512 가 base 를 넘으면 조건(d_block 동반 증가)이 빠졌던 게 맞다.")
    print("  판정선 — 두 폴드 같은 부호 & 3/3.")
