# -*- coding: utf-8 -*-
"""1군 행의 regular 브랜치 비중을 0.0~1.0 으로 훑는다. 재학습은 시드당 한 번뿐.

질문
    "1군은 그냥 regular 로만 학습해서 예측하면 어떤가" (w=1.0)
    그리고 그 반대 끝 "regular 를 아예 빼면" (w=0.0) 도 같이 본다.
    현행은 w=0.4 다.
        1군 행 = (1-w) x all + w x regular

왜 재학습이 필요 없나
    브랜치 예측을 따로 저장해두면 비중은 사후 조합이다. 그래서 시드당
    all / regular / futures 세 번만 학습하면 비중 전체를 훑을 수 있다.

같이 답하는 것
    regular 브랜치가 얼마나 얇은가. all 은 122만 행, regular 는 108만 행으로
    **거의 같은 데이터**를 본다. 두 예측의 상관이 0.99 근처면 regular 는
    all 위에 얹을 게 별로 없다는 뜻이고, 그러면 앵커 수축(residual expert)을
    1군에 얹어봐야 의미가 없다. 그 전제를 여기서 확인한다.

    퓨처스 브랜치는 배치 현행(0.6/0.4, e1/last_block)으로 고정한다.
    바뀌는 것은 1군 행의 비중 하나뿐이다.

채점
    전체를 주 잣대로 하고 1군 구간도 같이 본다. 시드 3개 짝비교.
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
WS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

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

    RES = {}
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
        G.Xn, G.cards, G.m_tr = Xn, cards, m_tr
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        fut_idx, reg_idx = tr_idx[t_isf], tr_idx[~t_isf]
        w_all = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        w_f = np.where(old[fut_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  all {len(tr_idx):,}  regular {len(reg_idx):,}  "
              f"futures {len(fut_idx):,}   검증 1군 {int((~isf_g).sum()):,}")

        P = {"all": {}, "regular": {}, "futures": {}}
        for sd in SEEDS:
            for nmb, idxb, wb in (("all", tr_idx, w_all),
                                  ("regular", reg_idx, None),
                                  ("futures", fut_idx, w_f)):
                torch.manual_seed(sd)
                torch.cuda.manual_seed_all(sd)
                m = G.make_model()
                G.train(m, idxb, 2, LR1, w=wb, seed=sd, tag=f"VS{VS} {nmb} s{sd}")
                s2 = idxb[season[idxb] == VS - 1]
                G.train(m, s2, 1, 2e-4, params=G.stage2_params(m), seed=sd,
                        tag=f"VS{VS} {nmb} s{sd} S2")
                P[nmb][sd] = G.predict(m, gate)
                np.save(os.path.join(OUT, f"rw_{VS}_{nmb}_s{sd}.npy"), P[nmb][sd])
                del m
                torch.cuda.empty_cache()
        G.log("    세 브랜치 학습 완료. 이제 비중은 사후 조합이다.")
        RES[VS] = P

    def sc(p, yv, m):
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 92)
    print("  1군 행의 regular 비중 스윕 — 재학습 없음, 사후 조합")
    print("=" * 92)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        allm = np.ones(len(yv), bool)
        P = RES[VS]
        ra = np.mean([np.corrcoef(P["all"][s][~isf_g],
                                  P["regular"][s][~isf_g])[0, 1] for s in SEEDS])
        print(f"\n  VS={VS}   1군 {int((~isf_g).sum()):,}행")
        print(f"    all 과 regular 의 1군 예측 상관  {ra:.4f}")
        sa = np.mean([sc(P["all"][s], yv, ~isf_g) for s in SEEDS])
        sr = np.mean([sc(P["regular"][s], yv, ~isf_g) for s in SEEDS])
        print(f"    단독 1군 점수   all {sa:8.1f}   regular {sr:8.1f}")
        print(f"    {'w':>5s} {'전체':>9s} {'1군':>9s}   {'현행(0.4) 대비':>24s}")
        base = [np.where(isf_g, 0.6 * P["all"][s] + 0.4 * P["futures"][s],
                         0.6 * P["all"][s] + 0.4 * P["regular"][s])
                for s in SEEDS]
        for w in WS:
            ps = [np.where(isf_g, 0.6 * P["all"][s] + 0.4 * P["futures"][s],
                           (1 - w) * P["all"][s] + w * P["regular"][s])
                  for s in SEEDS]
            p = np.mean(ps, 0)
            line = (f"    {w:5.2f} {sc(p, yv, allm):9.1f} {sc(p, yv, ~isf_g):9.1f}")
            dd = [sc(a, yv, allm) - sc(b, yv, allm) for a, b in zip(ps, base)]
            mu = float(np.mean(dd))
            se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
            mark = "  <- 현행" if abs(w - 0.4) < 1e-9 else ""
            line += (f"   {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):6.2f} "
                     f"{sum(1 for v in dd if v>0)}/{len(dd)}{mark}")
            print(line)
    print("\n  w=1.0 이 '1군을 regular 로만 예측' 이고 w=0.0 이 'regular 를 뺀 것' 이다.")
    print("  상관이 0.99 근처면 regular 는 all 위에 얹을 게 별로 없다는 뜻이다.")
