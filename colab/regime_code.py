# -*- coding: utf-8 -*-
"""abs_regime 코드를 몇 단계로 줄지 정한다. branch_redo 와 짝비교한다.

지금 (submit_20, 1057)
    0 = 옛 퓨처스(<=2022)
    1 = 나머지 전부 = { 새 퓨처스(2023~) + 1군 전 연도 }
    새 퓨처스가 '옛 1군' 과 한 코드를 쓴다.

재민님 제안 (reg3) — 시대로 묶는다
    0 = 옛 퓨처스   1 = 옛 1군   2 = { 새 퓨처스 + 새 1군 }

전체 교차 (reg4) — 모델이 알아서 고르게 둔다
    0 = 옛 퓨처스   1 = 옛 1군   2 = 새 퓨처스   3 = 새 1군

예상과 근거
    1군에는 2023 단절이 없다. 성공률이 50.37 -> 50.31 -> 48.97 로 완만히
    내려갈 뿐이고 그 추세는 season 이 이미 담는다(CatBoost 중요도 3위).
    그래서 '옛 1군' 을 따로 떼는 것 자체는 새 정보가 아닐 수 있다.
    반면 새 퓨처스를 옛 1군이 아니라 새 1군과 묶는 변화는 실질적이다.

    퓨처스는 2023 에 70.87% -> 47.29% 로 무너졌다(-23.6%p). 1군은 -0.06%p.

규약
    branch_redo 와 **같은 시드, 같은 전처리, 같은 학습 절차**를 쓴다.
    그래야 br_all_reg_s{seed}.npy / br_futures_reg_s{seed}.npy 와 짝비교가 된다.
    가중 0.1 도 그대로 건다 — 코드 단계만 바꾸는 실험이다.

    채점은 퓨처스 구간에서 한다. 지금 체제 처리는 퓨처스 행에만 쓰기 때문이다.
    1군 구간도 같이 찍어 부작용을 본다.
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
OLD_F_MAX = 2022
OLD_W = 0.1
FIXED = -0.004

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv


def scf(p, m):
    return F.bss(F.shift(p[m], FIXED), yv[m])


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def friend_matrix():
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xdf = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xdf.columns)
    Xf = Xdf.to_numpy(dtype=np.float32)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(Xf)), index=tr["row_id"].to_numpy())
    return Xf[pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()], cols


def codes(kind, isf, season):
    old = season <= OLD_F_MAX
    if kind == "reg3":
        return np.where(old & isf, 0.0, np.where(old & ~isf, 1.0, 2.0))
    if kind == "reg4":
        return np.where(old & isf, 0.0,
                        np.where(old & ~isf, 1.0,
                                 np.where(isf, 2.0, 3.0)))
    raise ValueError(kind)


def train_one(seed, idx, w, tag):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, 2e-3, w=w, seed=seed, tag=f"{tag} S1")
    s2 = G.season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed, tag=f"{tag} S2")
    p = G.predict(m, gate)
    del m
    torch.cuda.empty_cache()
    return p


if __name__ == "__main__":
    Xf, cols = friend_matrix()
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    season, isf = G.season.astype(np.float64), G.is_f
    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    t_old = t_isf & (season[tr_idx] <= OLD_F_MAX)

    base = {}
    for n in ("all_reg", "futures_reg"):
        for s in SEEDS:
            f = os.path.join(OUT, f"br_{n}_s{s}.npy")
            if not os.path.exists(f):
                raise SystemExit(f"없음: {f} — branch_redo 를 먼저 돌려라")
            base[(n, s)] = np.load(f)

    P = {}
    for kind in ("reg3", "reg4"):
        cd = codes(kind, isf, season).astype(np.float32)[:, None]
        X = np.concatenate([Xf, cd], 1)
        ci_k = ci + [Xf.shape[1]]
        Xn, Xc, cards = G.prep(X, G.m_tr, ci_k)
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        G.log(f"\n  {kind}  코드 분포 "
              f"{dict(zip(*np.unique(cd[G.m_tr].ravel(), return_counts=True)))}"
              f"  카디널리티 {cards.tolist()[-1]}")
        for br, sel in (("all", np.ones(len(tr_idx), bool)), ("futures", t_isf)):
            for s in SEEDS:
                t0 = time.time()
                w = np.where(t_old[sel], OLD_W, 1.0).astype(np.float64)
                P[(kind, br, s)] = train_one(s, tr_idx[sel], w, f"{kind} {br}")
                np.save(os.path.join(OUT, f"rc_{kind}_{br}_s{s}.npy"),
                        P[(kind, br, s)])
                G.log(f"    {kind} {br:8s} seed {s:3d}  {time.time()-t0:.0f}s")

    G.log("\n  퓨처스 행: 0.6 x all + 0.4 x futures  (고정시프트, 시드평균)")
    cur = [scf(0.6 * base[("all_reg", s)] + 0.4 * base[("futures_reg", s)], is_f)
           for s in SEEDS]
    G.log(f"    reg2 (현행, submit_20)  {np.mean(cur):8.1f}   "
          f"[{', '.join(f'{v:.0f}' for v in cur)}]")
    for kind in ("reg3", "reg4"):
        v = [scf(0.6 * P[(kind, "all", s)] + 0.4 * P[(kind, "futures", s)], is_f)
             for s in SEEDS]
        d = [a - b for a, b in zip(v, cur)]
        mu, se = float(np.mean(d)), float(np.std(d, ddof=1)) / 2.0
        G.log(f"    {kind:22s} {np.mean(v):8.1f}   "
              f"짝차이 {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/4  [{', '.join(f'{x:+.0f}' for x in d)}]")

    G.log("\n  1군 행에서 all 브랜치만 (부작용 확인)")
    curR = [scf(base[("all_reg", s)], ~is_f) for s in SEEDS]
    G.log(f"    reg2 (현행)             {np.mean(curR):8.1f}")
    for kind in ("reg3", "reg4"):
        v = [scf(P[(kind, "all", s)], ~is_f) for s in SEEDS]
        d = [a - b for a, b in zip(v, curR)]
        G.log(f"    {kind:22s} {np.mean(v):8.1f}   짝차이 {np.mean(d):+7.1f}  "
              f"{sum(1 for x in d if x > 0)}/4")

    G.log("\n  퓨처스 구간 +100 은 전체로 약 +12 다 (퓨처스 11.8%).")
    G.log("  1군은 all 브랜치를 regime 판본으로 안 쓰므로 위 부작용은 참고용이다.")
