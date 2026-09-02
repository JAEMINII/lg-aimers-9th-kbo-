# -*- coding: utf-8 -*-
"""퓨처스 브랜치를 전이학습으로 다시 짠다. 표본 부족을 다른 표본에서 빌린다.

문제
    퓨처스 학습 161,004행 중 105,308행(65.4%)이 2023 체제 붕괴 이전이다.
    성공률이 70.87% -> 47.29% 로 무너졌으니 그 표본은 다른 게임이다.
    새 체제는 55,696행뿐이다.

지금까지 시도한 두 가지와 그 결과 (관문 VS=2024, 퓨처스 행, 최적 시프트)
    가중 0.1 로 누르기   futures_reg   587.5
    새 체제만 학습        futures_new   286.7   <- 참패
    표본 부족이 체제 오염보다 해로웠다. 그래서 '버리기' 는 답이 아니다.

여기서 재는 제3의 길 — 옛 표본을 버리지도 누르지도 않고 **사전학습에만** 쓴다
    w0.1      현행. Stage1 전체 퓨처스(옛것 가중 0.1) -> Stage2 마지막 시즌
    preft     Stage1 전체 퓨처스(가중 없음) -> Stage2 **새 체제 퓨처스 전체**
    preft_w   Stage1 전체 퓨처스(가중 0.1)  -> Stage2 새 체제 퓨처스 전체
    xleague   Stage1 **전 리그 전체**       -> Stage2 새 체제 퓨처스 전체

    xleague 의 근거는 겸업 투수다. 두 리그를 오간 투수가 453명으로 퓨처스
    투수 633명의 71.6% 다. 1군 1,314,088행에서 배운 투수 표현을 가지고
    퓨처스로 들어간다.

    Stage2 범위는 지금과 같이 output + 마지막 블록이다. 사전학습이 바뀌는
    실험이지 미세조정 범위를 바꾸는 실험이 아니다.

채점
    퓨처스 행에서, 배치와 같은 라우팅으로 잰다: 0.6 x all + 0.4 x 퓨처스브랜치.
    all 은 시드마다 한 번만 학습해 네 팔이 공유한다 — 비교 대상은 퓨처스
    브랜치뿐이므로 all 을 팔마다 새로 뽑으면 잡음만 는다.

    체제 코드는 reg4(4단계 교차)를 쓴다. 어제 그게 2단계를 이겼다.
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

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def fit(seed, s1_idx, s1_w, s2_idx, tag):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, s1_idx, 2, 2e-3, w=s1_w, seed=seed, tag=f"{tag} S1")
    if len(s2_idx):
        G.train(m, s2_idx, 1, 2e-4, params=G.stage2_params(m), seed=seed,
                tag=f"{tag} S2")
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
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    code = np.where(old & isf, 0.0,
                    np.where(old & ~isf, 1.0,
                             np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    X = np.concatenate([Xf, code], 1)
    Xn, Xc, cards = G.prep(X, G.m_tr, ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    m_tr = G.m_tr
    tr_idx = np.where(m_tr)[0]
    t_isf = isf[tr_idx]
    t_old = t_isf & old[tr_idx]
    f_idx = tr_idx[t_isf]                       # 퓨처스 전체
    f_new = tr_idx[t_isf & ~t_old]              # 새 체제 퓨처스
    last = tr_idx[t_isf & (season[tr_idx] == VS - 1)]
    wf = np.where(t_old[t_isf], OLD_W, 1.0).astype(np.float64)
    w_all = np.where(t_old, OLD_W, 1.0).astype(np.float64)
    G.log(f"\n  퓨처스 전체 {len(f_idx):,}  새 체제 {len(f_new):,}  "
          f"마지막 시즌 {len(last):,}  전 리그 {len(tr_idx):,}")

    plans = [("w0.1",    f_idx,  wf,    last),
             ("preft",   f_idx,  None,  f_new),
             ("preft_w", f_idx,  wf,    f_new),
             ("xleague", tr_idx, w_all, f_new)]

    ALL, P = {}, {}
    for s in SEEDS:
        t0 = time.time()
        ALL[s] = fit(s, tr_idx, w_all, tr_idx[season[tr_idx] == VS - 1], "all")
        np.save(os.path.join(OUT, f"ft_all_s{s}.npy"), ALL[s])
        for nm, i1, w1, i2 in plans:
            P[(nm, s)] = fit(s, i1, w1, i2, nm)
            np.save(os.path.join(OUT, f"ft_{nm}_s{s}.npy"), P[(nm, s)])
        G.log(f"  seed {s}  {time.time()-t0:.0f}s")

    G.log("\n  퓨처스 행, 라우팅 0.6 x all + 0.4 x 브랜치, 최적 시프트")
    G.log(f"  {'구성':10s} {'브랜치단독':>11s} {'라우팅후':>10s}   시드별(라우팅)")
    base = None
    for nm, *_ in plans:
        solo = np.mean([sc(P[(nm, s)], is_f) for s in SEEDS])
        rt = [sc(0.6 * ALL[s] + 0.4 * P[(nm, s)], is_f) for s in SEEDS]
        if base is None:
            base = rt
        G.log(f"  {nm:10s} {solo:11.1f} {np.mean(rt):10.1f}   "
              f"[{', '.join(f'{x:.0f}' for x in rt)}]")

    G.log("\n  현행(w0.1) 대비 짝차이 — 퓨처스 행")
    for nm, *_ in plans[1:]:
        d = [sc(0.6 * ALL[s] + 0.4 * P[(nm, s)], is_f)
             - sc(0.6 * ALL[s] + 0.4 * P[("w0.1", s)], is_f) for s in SEEDS]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
        G.log(f"    {nm:10s} {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/{len(d)}  "
              f"[{', '.join(f'{x:+.0f}' for x in d)}]")

    G.log("\n  퓨처스 구간 +100 은 전체로 약 +12 다 (퓨처스 11.8%).")
    G.log("  퓨처스 ~630 을 1군 ~870 수준까지 올리면 전체 +28 이다. 그게 이 실험의 상한이다.")
