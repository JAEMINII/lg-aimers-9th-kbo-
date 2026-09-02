# -*- coding: utf-8 -*-
"""퓨처스 브랜치의 Stage2 를 축별로 가른다. scope 는 한 번도 스윕한 적이 없다.

배경 — 배치본과 관문이 어긋나 있다
    배치 (export_fbregime.py)   퓨처스 Stage2 = **1에폭**, last_block
    관문 스크립트                퓨처스 Stage2 = 4에폭, last_block
    그런데 stage_ext 가 이미 이렇게 쟀다 (퓨처스 행, 0.6 all_e1 + 0.4 fut_eX, 4시드)
        e0 642.8  -7.5 t=-8.16 0/4     e2 655.8  +5.6 t=11.52 4/4
        e1 650.2  기준                  e3 659.1  +8.9 t= 5.97 4/4
    **늘릴수록 좋아진다.** 배치가 e1 에 머물러 있는 게 방치된 손해다.

무엇이 진짜 미탐색인가
    scope   `train_conditional.py` 의 선택지는 head / last_block 인데 우리는
            늘 last_block 만 썼다. 브랜치별로 다르게 준 적도 없다.
    자료범위 Stage2 를 VS-1 한 시즌만 쓴다. 퓨처스는 2023 부터 이미 새 체제라
            2023+2024 를 쓰면 표본이 3만 -> 6만이 된다.
            **용량이 아니라 자료를 늘리는 것**이라 신호 0.93% 벽에 안 걸린다.

이미 기각된 것은 넣지 않는다
    Stage1 만 쓰기            e0 = -7.5, 0/4
    Stage1/Stage2 예측 혼합    w 내릴수록 단조 하락, 0/4

팔 (퓨처스 브랜치만 바꾼다. all 은 e1/last_block 로 고정하고 시드마다 한 번만 학습)
    e1_lb     1에폭 last_block   <- 배치 현행
    e4_lb     4에폭 last_block   <- 우리 관문 현행
    e1_hd     1에폭 head
    e4_hd     4에폭 head
    e4_2y     4에폭 last_block, Stage2 자료 = 2023+2024 (2024 가중 2배)
    e8_lb     8에폭 last_block   <- e3 까지 단조였으니 어디서 꺾이는지 본다

채점
    퓨처스 행만 따로 본다. 전체 평균으로 보면 11.8% 라 묻힌다.
    1군 행은 어느 팔에서도 안 변하므로(퓨처스 브랜치만 건드린다) 참고로만 찍는다.
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
# (이름, 에폭, scope, 2년치 여부)
ARMS = [("e1_lb", 1, "lb", False), ("e4_lb", 4, "lb", False),
        ("e1_hd", 1, "hd", False), ("e4_hd", 4, "hd", False),
        ("e4_2y", 4, "lb", True), ("e8_lb", 8, "lb", False)]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def head_params(model):
    """출력층만 연다. last_block 보다 좁은 scope."""
    for p in model.parameters():
        p.requires_grad_(False)
    ps = list(model.output.parameters())
    for p in ps:
        p.requires_grad_(True)
    return ps


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
        G.Xn, G.cards, G.m_tr = Xn, cards, m_tr
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        fut_idx = tr_idx[t_isf]
        w_f = np.where(old[fut_idx], OLD_W, 1.0).astype(np.float64)
        n1 = int((season[fut_idx] == VS - 1).sum())
        n2 = int(((season[fut_idx] == VS - 1) | (season[fut_idx] == VS - 2)).sum())
        G.log(f"\n  VS={VS}  퓨처스 학습 {len(fut_idx):,}  "
              f"Stage2 1년 {n1:,} / 2년 {n2:,}  검증퓨처스 {int(isf_g.sum()):,}")

        # ---- all 브랜치는 팔마다 안 변하므로 시드당 한 번만 학습한다
        P_all = {}
        w_all = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        for sd in SEEDS:
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            m = G.make_model()
            G.train(m, tr_idx, 2, LR1, w=w_all, seed=sd, tag=f"VS{VS} all s{sd}")
            s2 = tr_idx[season[tr_idx] == VS - 1]
            G.train(m, s2, 1, 2e-4, params=G.stage2_params(m), seed=sd,
                    tag=f"VS{VS} all s{sd} S2")
            P_all[sd] = G.predict(m, gate)
            del m
            torch.cuda.empty_cache()
        G.log("    all 브랜치 완료 (팔 공통)")

        for nm, ep, scope, two in ARMS:
            t0 = time.time()
            ps = []
            for sd in SEEDS:
                torch.manual_seed(sd)
                torch.cuda.manual_seed_all(sd)
                m = G.make_model()
                G.train(m, fut_idx, 2, LR1, w=w_f, seed=sd,
                        tag=f"VS{VS} fut {nm} s{sd} S1")
                sy = season[fut_idx]
                if two:
                    sel = fut_idx[(sy == VS - 1) | (sy == VS - 2)]
                    ww = np.where(season[sel] == VS - 1, 2.0, 1.0).astype(np.float64)
                else:
                    sel, ww = fut_idx[sy == VS - 1], None
                pr = head_params(m) if scope == "hd" else G.stage2_params(m)
                for e in range(ep):
                    G.train(m, sel, 1, 2e-4, params=pr, w=ww, seed=sd + e,
                            tag=f"VS{VS} fut {nm} s{sd} S2e{e+1}")
                pf = G.predict(m, gate)
                ps.append(0.6 * P_all[sd] + 0.4 * pf)
                del m
                torch.cuda.empty_cache()
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"fs_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:6s} ep={ep} scope={scope} 2년={two} "
                  f"끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m):
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 92)
    print("  퓨처스 Stage2 — scope x 에폭 x 자료범위. **퓨처스 행 기준**")
    print("=" * 92)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "e1_lb")]
        print(f"\n  VS={VS}   (기준 = e1_lb, 배치 현행)")
        for nm, ep, scope, two in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:6s} 퓨처스 {sc(p, yv, isf_g):8.1f}   "
                    f"전체 {sc(p, yv, np.ones(len(yv), bool)):8.1f}")
            if nm != "e1_lb":
                dd = [sc(a, yv, isf_g) - sc(b, yv, isf_g)
                      for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   퓨처스차 {mu:+7.1f}+-{se:4.1f} "
                         f"t={mu/max(se,1e-9):5.2f} {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)
    print("\n  stage_ext 는 e2 +5.6 (t=11.52 4/4) / e3 +8.9 (t=5.97 4/4) 였다.")
    print("  퓨처스는 11.8% 라 전체로는 이 값의 0.118 배만 남는다.")
    print("  e4_2y 가 이기면 '자료를 늘리는' 축이 열린 것이다 — 용량 축과 다르다.")
