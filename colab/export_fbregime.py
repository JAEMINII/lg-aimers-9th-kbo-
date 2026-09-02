# -*- coding: utf-8 -*-
"""배치용 regime TabM 을 지인 전처리 위에서 학습해 내보낸다 (VS=2025).

왜 이것만 따로 만드나
    fb_arms 결과 (지인 전처리, 4시드 짝비교)
        regime   전체 +4.7   1군 -18.8 (0/4)   퓨처스 +157.6 (4/4, t=12.4)
    체제 처리는 퓨처스를 크게 올리고 1군을 내린다. 퓨처스가 11.8% 뿐이라
    전체로는 희석돼 무의미해진다. 그러니 **퓨처스 행에만** 쓰면 된다.
        1군  행 -> base   (submit_18 에 이미 있는 모델 그대로)
        퓨처스 행 -> regime (여기서 만드는 것)
    그래서 all + futures 두 브랜치만 만든다. regular_regime 은 쓸 데가 없다.

형식
    지인 피처 44열 + abs_regime 1열 = 45열.
    가중치는 우리 npz 형식(meta + prep 통계)으로 내보낸다. 지인 export 경로를
    쓰려면 그쪽 학습 API 를 통째로 태워야 하는데 sample_weight 를 못 넣는다.
    추론에서는 submit_18 의 preprocess.py 로 44열을 만들고 abs_regime=1 을
    붙여 넣으면 된다.
"""
import json
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
OLD_W = 0.1

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from export_ours import export, prep_stats                      # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

CFG = dict(k=32, n_blocks=3, d_block=256, dropout=0.1, d_emb=16,
           num_embeddings="linear_relu", backbone_kind="batch_ensemble",
           ep1=2, ep2=1, lr1=2e-3, lr2=2e-4, vs=VS, decay=0.0,
           regime="flag_w0.1", prep="friend")


if __name__ == "__main__":
    t0 = time.time()
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])     # 2019~2024 전부
    Xdf = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xdf.columns)
    Xf = Xdf.to_numpy(dtype=np.float32)
    order_f = tr["row_id"].to_numpy()

    # G 의 행 순서(features44)에 맞춘다. season / is_f 를 그쪽에서 가져오기 때문이다.
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(order_f)), index=order_f)
    Xf = Xf[pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season = G.season.astype(np.float64)
    isf = G.is_f
    flag = ((isf & (season > OLD_F_MAX)) | (~isf)).astype(np.float32)[:, None]
    X = np.concatenate([Xf, flag], 1)
    feats = cols + ["abs_regime"]
    ci_r = ci + [Xf.shape[1]]
    # tabm_gate_gpu 는 import 시점에 VS(기본 2024)로 데이터를 만든다. VS=2025 로
    # 올리면 cb_gate2025.npy 를 찾다가 죽는다. 그 모듈에서 쓰는 건 season / is_f /
    # 모델·학습 함수뿐이고 전부 VS 와 무관하므로, 기본값으로 올린 뒤 학습 구간만
    # 여기서 직접 준다. export_ours.py 도 같은 방식이다.
    m_tr = np.ones(len(X), bool)                 # VS=2025 -> 전 행이 학습
    print(f"  지인 피처 {Xf.shape} -> {X.shape}   학습 {m_tr.sum():,}행  "
          f"시즌 {int(season.min())}~{int(season.max())}")

    Xn, Xc, cards, st = prep_stats(X, m_tr, ci_r)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    print(f"  수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열  카디널리티 {cards.tolist()}")

    tr_idx = np.where(m_tr)[0]
    old_f = isf[tr_idx] & (season[tr_idx] <= OLD_F_MAX)
    print(f"  옛 체제 퓨처스 {old_f.sum():,}행 -> 학습가중 {OLD_W}")

    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", isf[tr_idx])):
        idx = tr_idx[sel]
        w = np.where(old_f[sel], OLD_W, 1.0).astype(np.float64)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        mdl = G.make_model()
        G.train(mdl, idx, 2, 2e-3, w=w, seed=SEED, tag=f"{br} S1")
        s2 = season[idx] == VS - 1                      # 2024 로 마무리
        G.train(mdl, idx[s2], 1, 2e-4, params=G.stage2_params(mdl),
                seed=SEED, tag=f"{br} S2")
        p = os.path.join(OUT, f"fbregime_{br}_seed42.npz")
        export(mdl, p, st, cards, Xn.shape[1], CFG, feats)
        print(f"  {br:8s} {len(idx):>10,}행  Stage2 {int(s2.sum()):,}행  "
              f"-> {os.path.basename(p)}  {os.path.getsize(p)/1e6:.1f}MB")
        del mdl
        torch.cuda.empty_cache()
    print(f"  완료 {time.time()-t0:.0f}s")
