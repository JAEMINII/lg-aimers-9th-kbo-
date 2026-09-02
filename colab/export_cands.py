# -*- coding: utf-8 -*-
"""branch_redo 후보 브랜치를 배치용(VS=2025)으로 미리 뽑아 둔다.

이미 가진 것
    all_base / regular_base / futures_base   submit_18 zip 안에 지인 형식으로 있음
    all_reg / futures_reg                    fbregime_*.npz (submit_20 이 쓰는 것)

여기서 만드는 것
    futures_new   퓨처스 새 체제만 (2023~). 퓨처스 전용 모델의 65.4%가 폐지된
                  규칙으로 학습돼 있다는 문제를 가중이 아니라 표본으로 푸는 판본.
                  breaktest 에서 옛 체제가 섞이면 얼마나 망가지는지 봤다 —
                  2021~2023 학습 -> 2024 채점이 379.0 뿐이었다.
    all_drop      전부에서 옛 체제 퓨처스만 제거. 가중 0.1 대신 아예 뺀 판본.

    둘 다 44열이다(플래그 없음). 추론에서 지인 preprocess 결과를 그대로 넣으면 된다.

branch_redo 가 어느 조합을 고르든 바로 조립할 수 있게 미리 만든다.
어차피 GPU 시간은 두 판본 합쳐 4분이다.
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
SEED = 42
OLD_F_MAX = 2022

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from export_ours import export, prep_stats                      # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

CFG = dict(k=32, n_blocks=3, d_block=256, dropout=0.1, d_emb=16,
           num_embeddings="linear_relu", backbone_kind="batch_ensemble",
           ep1=2, ep2=1, lr1=2e-3, lr2=2e-4, vs=VS, decay=0.0,
           regime="none", prep="friend")


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
    X = Xf[pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    m_tr = np.ones(len(X), bool)                    # VS=2025 -> 전 행 학습
    old_f = isf & (season <= OLD_F_MAX)

    Xn, Xc, cards, st = prep_stats(X, m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    print(f"  학습 {len(X):,}행  수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열")
    print(f"  옛 체제 퓨처스 {old_f.sum():,}행")

    idx_all = np.arange(len(X))
    for name, sel in (("futures_new", isf & ~old_f),
                      ("all_drop", ~old_f)):
        idx = idx_all[sel]
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        mdl = G.make_model()
        G.train(mdl, idx, 2, 2e-3, seed=SEED, tag=f"{name} S1")
        s2 = season[idx] == VS - 1
        G.train(mdl, idx[s2], 1, 2e-4, params=G.stage2_params(mdl), seed=SEED,
                tag=f"{name} S2")
        p = os.path.join(OUT, f"cand_{name}_seed42.npz")
        export(mdl, p, st, cards, Xn.shape[1], CFG, cols)
        print(f"  {name:12s} {len(idx):>10,}행  Stage2 {int(s2.sum()):,}행  "
              f"-> {os.path.basename(p)}  {os.path.getsize(p)/1e6:.1f}MB")
        del mdl
        torch.cuda.empty_cache()
    print(f"  완료 {time.time()-t0:.0f}s")
