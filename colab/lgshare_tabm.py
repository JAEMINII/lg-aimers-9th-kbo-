# -*- coding: utf-8 -*-
"""lg_f_share 를 TabM 관문에서 잰다. pitcher_id 임베딩과 중복인지 판정한다.

배경
    lg_f_share = 그 투수가 지금까지 던진 공 중 퓨처스 비율 (그 행 이전까지).
    편상관(44열 통제) 전체 -0.0174, 퓨처스 행 -0.0325. 어제 탈락한 basic
    후보 12개가 전부 |r| < 0.0025 였던 것과 비교하면 6~13배다.

    CatBoost 3폴드 백테스트에서 전부 양수였다.
        2022 전체 +12.2  퓨처스 +106.0
        2023 전체 +77.8  퓨처스 +639.6   (체제 붕괴 해라 부풀려진 값)
        2024 전체  +3.5  퓨처스  +16.0   <- 배치와 구조가 같은 폴드

    그런데 함께 통과한 나머지 4개를 묶어 넣으면 3폴드 전부 음수였다.
    그래서 이 하나만 잰다.

왜 TabM 에서 다시 재나
    추론에서 이 값은 **투수당 상수**다 — 학습 이력이 2024년 말로 얼어붙는다.
    TabM 에는 pitcher_id 임베딩(카디널리티 712~793)이 있어 같은 정보를 이미
    담을 수 있다. 트랙맨 파생 5종이 그래서 전부 0 이었다.

    CatBoost 에서 통한 게 반증이 못 된다. 우리 CatBoost 는 pitcher_id 를
    **숫자 열**로 받는다(cat_features 를 뺐다). 트리에게 임의 정수 순서는
    거의 무의미하니, CatBoost 는 TabM 임베딩이 이미 하는 일을 못 하고
    있었을 뿐일 수 있다.

    어느 쪽이 나와도 배운다. 통하면 임베딩이 리그 이력을 못 배우고 있다는
    뜻이고, 안 통하면 이미 배우고 있다는 뜻이다.

설정
    지인 전처리 44열 (+ lg_f_share) , VS=2024, all 브랜치, 시드 4개 짝비교.
    1군 행과 퓨처스 행을 따로 채점한다 — 이 축은 퓨처스에서 먼저 보일 것이다.
    구간 점수는 그 구간 분모로 잰다.
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

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from screen_league import build                                 # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def run(seed, idx, season):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, 2e-3, seed=seed, tag="S1")
    s2 = season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed, tag="S2")
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
    order = pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()
    Xf = Xs.to_numpy(dtype=np.float32)[order]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    lg = build(tr_raw[["row_id", "season", "game_type", "pitcher_id"]])
    lg = lg.set_index("row_id").reindex(tr_raw["row_id"].to_numpy())
    v = lg["lg_f_share"].to_numpy(np.float64)
    m_tr = G.m_tr
    v = np.where(np.isnan(v), np.nanmedian(v[m_tr]), v).astype(np.float32)
    G.log(f"\n  lg_f_share  결측채움후 평균 {v.mean():.4f}  "
          f"1군행 {v[gate][R].mean():.4f}  퓨처스행 {v[gate][is_f].mean():.4f}")

    season = G.season.astype(np.int64)
    tr_idx = np.where(m_tr)[0]
    P = {}
    for nm, X, c in (("base", Xf, ci),
                     ("+lg_f_share", np.concatenate([Xf, v[:, None]], 1), ci)):
        Xn, Xc, cards = G.prep(X, m_tr, c)
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        t0 = time.time()
        for s in SEEDS:
            P[(nm, s)] = run(s, tr_idx, season)
            np.save(os.path.join(OUT, f"ls_{nm.replace('+','p')}_s{s}.npy"),
                    P[(nm, s)])
        G.log(f"  {nm:12s} 수치 {Xn.shape[1]}열  {time.time()-t0:.0f}s")

    G.log("\n  all 브랜치 단독, 최적 시프트, 시드별 계산 후 평균")
    G.log(f"  {'구성':12s} {'1군 행':>9s} {'퓨처스 행':>10s} {'전체':>9s}")
    FULL = np.ones(len(yv), bool)
    for nm in ("base", "+lg_f_share"):
        G.log(f"  {nm:12s} "
              f"{np.mean([sc(P[(nm,s)], R) for s in SEEDS]):9.1f} "
              f"{np.mean([sc(P[(nm,s)], is_f) for s in SEEDS]):10.1f} "
              f"{np.mean([sc(P[(nm,s)], FULL) for s in SEEDS]):9.1f}")

    G.log("\n  base 대비 짝차이")
    for tag, m in (("1군", R), ("퓨처스", is_f), ("전체", FULL)):
        d = [sc(P[("+lg_f_share", s)], m) - sc(P[("base", s)], m) for s in SEEDS]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
        G.log(f"    {tag:6s} {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/{len(d)}  "
              f"[{', '.join(f'{x:+.1f}' for x in d)}]")

    G.log("\n  퓨처스 구간 +100 은 전체로 약 +12 다 (퓨처스 11.8%).")
    G.log("  양수면 pitcher_id 임베딩이 리그 이력을 못 배우고 있다는 뜻이다.")
