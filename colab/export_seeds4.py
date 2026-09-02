# -*- coding: utf-8 -*-
"""배치용 reg4 TabM 을 시드 3개로 뽑는다 (VS=2025, 지인 전처리).

reg4 = 체제 코드 4단계 교차
    0 = 옛 퓨처스(<=2022)   1 = 옛 1군(<=2022)
    2 = 새 퓨처스(2023~)    3 = 새 1군(2023~)

지금까지 쓰던 2단계는 0 = 옛 퓨처스, 1 = 나머지 전부였다. 그러면 새 퓨처스가
옛 1군과 한 칸에 들어간다. 그게 1군을 망가뜨리고 있었다.

관문(VS=2024, 최적 시프트, 전체 혼합, 시드별 계산 후 평균)
    submit_20 (1군 all_base+0.4regular / 퓨처스 reg2)   895.1
    1군 all_reg4 단독 / 퓨처스 reg4 0.4                 906.4   +11.3  4/4
    3시드 앙상블로는 897.7 -> 907.3

    브랜치 단독 (최적 시프트)      1군행    퓨처스행
      all_base                   871.8    528.1
      all_reg (2단계)             851.1    609.5
      all_reg4                   882.8    612.4   <- 양쪽 다 최고

무엇을 뽑나
    all_reg4       1군 행에 단독으로 쓰고, 퓨처스 행에 0.6 으로 쓴다
    futures_reg4   퓨처스 행에 0.4
    시드 42 / 1 / 777

    regular_base 는 안 만든다. 1군에 브랜치를 섞는 게 손해라 w=0 이다.
    all_base 도 안 만든다. all_reg4 가 1군에서 그걸 11점 이긴다.

    옛 체제 퓨처스 학습가중 0.1 은 그대로 건다. 코드 단계만 바꾸는 것이지
    가중을 바꾸는 실험이 아니다.

추론에서 주의
    reg4 는 추론 때 상수가 아니다. 2025 는 전부 새 체제이므로
    퓨처스 행 = 2, 1군 행 = 3 이다. 2단계일 때는 전부 1 이었다.
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
VS = 2025
SEEDS = (42, 1, 777)
OLD_F_MAX = 2022
OLD_W = 0.1

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from export_ours import export, prep_stats                      # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

CFG = dict(k=32, n_blocks=3, d_block=256, dropout=0.1, d_emb=16,
           num_embeddings="linear_relu", backbone_kind="batch_ensemble",
           ep1=2, ep2=1, lr1=2e-3, lr2=2e-4, vs=VS, decay=0.0,
           regime="reg4_w0.1", prep="friend")


if __name__ == "__main__":
    t0 = time.time()
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xdf = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xdf.columns)
    Xf = Xdf.to_numpy(dtype=np.float32)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(Xf)), index=tr["row_id"].to_numpy())
    Xf = Xf[pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    code = np.where(old & isf, 0.0,
                    np.where(old & ~isf, 1.0,
                             np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    X = np.concatenate([Xf, code], 1)
    feats = cols + ["abs_regime"]
    ci_r = ci + [Xf.shape[1]]
    m_tr = np.ones(len(X), bool)
    old_f = isf & old

    u, cnt = np.unique(code.ravel(), return_counts=True)
    print(f"  코드 분포 {dict(zip(u.astype(int).tolist(), cnt.tolist()))}")
    print(f"  (0 옛F / 1 옛1군 / 2 새F / 3 새1군)   옛 체제 퓨처스 가중 {OLD_W}")

    Xn, Xc, cards, st = prep_stats(X, m_tr, ci_r)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    print(f"  수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열  카디널리티 {cards.tolist()}")

    idx_all = np.arange(len(X))
    for seed in SEEDS:
        for name, idx in (("all", idx_all), ("futures", idx_all[isf])):
            t1 = time.time()
            w = np.where(old_f[idx], OLD_W, 1.0).astype(np.float64)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            mdl = G.make_model()
            G.train(mdl, idx, 2, 2e-3, w=w, seed=seed, tag=f"reg4 {name} s{seed} S1")
            s2 = season[idx] == VS - 1
            G.train(mdl, idx[s2], 1, 2e-4, params=G.stage2_params(mdl),
                    seed=seed, tag=f"reg4 {name} s{seed} S2")
            p = os.path.join(OUT, f"r4_{name}_s{seed}.npz")
            export(mdl, p, st, cards, Xn.shape[1], dict(CFG, seed=seed), feats)
            print(f"  reg4 {name:8s} s{seed:<4d} {len(idx):>10,}행  "
                  f"{os.path.getsize(p)/1e6:.1f}MB  {time.time()-t1:.0f}s", flush=True)
            del mdl
            torch.cuda.empty_cache()
    print(f"  완료 {time.time()-t0:.0f}s")
