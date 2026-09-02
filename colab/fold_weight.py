# -*- coding: utf-8 -*-
"""비중 최적점이 연도마다 안정적인가. 이 축을 영구히 닫거나 확정한다.

문제
    지금 CatBoost 비중 0.30 은 리더보드 두 점(1057/1058)으로 고른 값이다.
    관문은 0.50 이 최적이라고 하는데, 관문이 이 축에서 12배 과대평가한다는 걸
    실측했다 (w_CB 0.10 -> 0.30 이 관문 +12.1 / 리더보드 +1.0).

    그래서 관문을 믿을 수도 안 믿을 수도 없는 상태로 매번 흔들리고 있다.
    scipy 로 더 정밀하게 최적화하는 건 답이 아니다 — **편향된 목적함수를
    정밀하게 최적화하면 정밀하게 틀린다.**

    진짜 문제는 폴드가 하나라는 것이다. 기억에도 남아 있다:
    '한 폴드 관문은 그 해에 과적합한다. 다년 백테스트로 판정할 것.'

설계
    VS = 2022 / 2023 / 2024 세 폴드. 각각 학습 = season < VS, 검증 = VS.
    확장창(expanding)이라 배치 구조(2019~2024 학습 -> 2025)와 같은 모양이다.

    VS=2021 은 뺐다 — 학습 시즌이 2개뿐이라 배치(6개)와 너무 다르다.

    각 폴드에서 배치 구성 그대로 만든다.
        TabM     3시드 x 3브랜치, 라우팅 0.6/0.4, Stage1 2에폭 lr 3e-3, Stage2
        CatBoost 3시드, 시즌가중 2.0, 400라운드, depth 4

    그리고 w_CB 를 훑어 폴드별 최적점을 찾는다.

읽는 법
    세 폴드의 최적점이 0.25~0.35 로 모이면  -> 0.30 이 맞다. 축을 확정하고 닫는다.
    0.1~0.6 으로 흩어지면                  -> 비중 최적화 자체가 잡음이다.
                                             현행 유지하고 이 축을 영구히 닫는다.
    어느 쪽이든 결론이 나온다. 지금처럼 흔들리는 상태가 끝난다.

주의
    퓨처스는 2023 체제 전환 때문에 폴드마다 성격이 다르다 (VS=2022 는 학습에
    새 체제 퓨처스가 0개, VS=2023 은 검증연도가 전환년). 그래서 **1군 행 기준**
    최적점을 주 지표로 삼고 전체는 참고로만 본다. 1군이 88.2% 이기도 하다.
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
CB_SEEDS = (42, 1234, 2025)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
DECAY = 2.0
EP2 = {"all": 1, "regular": 1, "futures": 4}
FOLDS = (2022, 2023, 2024)
WS = np.round(np.arange(0.0, 0.65, 0.05), 2)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def fold(VS, raw44, tr_sorted, pos_map):
    """한 폴드를 통째로 만든다. 반환: (yv, is_f, p_tabm, p_cb)"""
    season = raw44["season"].astype(np.float64)
    isf = raw44["is_f"]
    m_tr = season < VS
    m_va = season == VS
    gate = np.where(m_va)[0]
    yv = raw44["y"][gate].astype(np.float64)

    # ---- 지인 전처리, 이력은 학습 구간에서만
    hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
    Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
    cols = list(Xs.columns)
    Xf = Xs.to_numpy(dtype=np.float32)[pos_map]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xn, Xc, cards = G.prep(np.concatenate([Xf, c4], 1), m_tr,
                           ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    G.gate = gate

    tr_idx = np.where(m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)

    ps = []
    for sd in SEEDS:
        P = {}
        for br, sel in (("all", np.ones(len(tr_idx), bool)),
                        ("regular", ~t_isf), ("futures", t_isf)):
            idx, ww = tr_idx[sel], w10[sel]
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            m = G.make_model()
            G.train(m, idx, 2, LR1, w=ww, seed=sd, tag=f"VS{VS} {br} s{sd} S1")
            s2 = idx[season[idx] == VS - 1]
            pr = G.stage2_params(m)
            for e in range(EP2[br]):
                G.train(m, s2, 1, 2e-4, params=pr, seed=sd + e,
                        tag=f"VS{VS} {br} s{sd} S2e{e+1}")
            P[br] = G.predict(m, gate)
            del m
            torch.cuda.empty_cache()
        ps.append(np.where(isf[gate], 0.6 * P["all"] + 0.4 * P["futures"],
                           0.6 * P["all"] + 0.4 * P["regular"]))
    p_tabm = np.mean(ps, 0)

    # ---- CatBoost, 같은 폴드에서
    X44 = raw44["X44"].astype(np.float64)
    med = np.nanmedian(X44[m_tr], 0)
    Xc44 = np.where(np.isnan(X44), med, X44)
    wcb = DECAY ** (season[m_tr] - 2019)
    cps = []
    for s in CB_SEEDS:
        mdl = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4,
                                 l2_leaf_reg=1.0, verbose=0, random_seed=s,
                                 allow_writing_files=False, thread_count=6)
        mdl.fit(Xc44[m_tr], raw44["y"][m_tr].astype(int), sample_weight=wcb)
        cps.append(mdl.predict_proba(Xc44[gate])[:, 1])
    p_cb = np.mean(cps, 0).astype(np.float64)
    return yv, isf[gate], p_tabm, p_cb


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)          # 마스크·라벨용. VS 는 여기서 안 쓴다
    raw44 = dict(X44=d["X44"], y=d["y"], season=d["season"], is_f=d["is_f"])
    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()

    res = {}
    for VS in FOLDS:
        t0 = time.time()
        yv, isf_g, p_tm, p_cb = fold(VS, raw44, tr_sorted, pos_map)
        np.savez(os.path.join(OUT, f"fw_{VS}.npz"), yv=yv, isf=isf_g,
                 tm=p_tm, cb=p_cb)
        res[VS] = (yv, isf_g, p_tm, p_cb)
        G.log(f"  VS={VS} 끝 {time.time()-t0:.0f}s  "
              f"검증 {len(yv):,}행 (1군 {int((~isf_g).sum()):,})")

    def sc(p, y):
        return F.best_shift(p, y)[0]

    for lab, pick in (("1군 행", lambda i: ~i), ("전체", lambda i: np.ones(len(i), bool))):
        print("\n" + "=" * 78)
        print(f"  [{lab}] w_CB 별 점수와 폴드별 최적점")
        print("=" * 78)
        print(f"  {'w_CB':>6s} " + " ".join(f"{f'VS{v}':>10s}" for v in FOLDS))
        curves = {}
        for w in WS:
            row = []
            for VS in FOLDS:
                yv, isf_g, p_tm, p_cb = res[VS]
                m = pick(isf_g)
                row.append(sc((1 - w) * p_tm[m] + w * p_cb[m], yv[m]))
            curves[w] = row
            mark = "  <- 현행" if abs(w - 0.30) < 1e-9 else ""
            print(f"  {w:6.2f} " + " ".join(f"{v:10.1f}" for v in row) + mark)
        print(f"\n  {'폴드':>8s} {'최적 w_CB':>10s} {'최적 점수':>10s} "
              f"{'w=0.30 점수':>12s} {'차이':>8s}")
        opts = []
        for i, VS in enumerate(FOLDS):
            col = [(curves[w][i], w) for w in WS]
            best = max(col)
            at30 = curves[0.30][i]
            opts.append(best[1])
            print(f"  {VS:8d} {best[1]:10.2f} {best[0]:10.1f} {at30:12.1f} "
                  f"{best[0]-at30:+8.1f}")
        print(f"\n  최적점 {opts}   폭 {max(opts)-min(opts):.2f}")
        if max(opts) - min(opts) <= 0.10:
            print("  -> 모였다. 이 값을 믿고 축을 확정한다.")
        else:
            print("  -> 흩어졌다. 비중 최적화가 잡음이다. 현행 0.30 유지하고 축을 닫는다.")

    print("\n  퓨처스는 2023 체제 전환으로 폴드 성격이 달라 1군 행을 주 지표로 본다.")
