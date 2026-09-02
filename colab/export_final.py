# -*- coding: utf-8 -*-
"""배치용(VS=2025) TabM 을 고친 스케줄러로 다시 뽑는다. 브랜치를 인자로 고른다.

왜 전부 다시 뽑나
    codegap 에서 학습 코드의 결함을 고쳤다 — 코사인을 배치마다 돌려 lr 을 0 까지
    감쇠시키느라 2에폭 중 후반이 버려지고 있었다. 에폭 단위로 바꾸니 1군 행
    869.5 -> 888.8 (+19.2, 3/3) 이고 지인 코드(874.6)를 넘는다.

    지금 submit_20 안의 TabM 은 출처가 셋이다.
        all_tabm_seed_42 / regular_tabm_seed_42   지인 학습 코드
        fbregime_*_seed42                          우리 코드, 망가진 스케줄러
    수정 후 우리 코드가 더 나으므로 전부 우리 코드로 통일한다. 그래야 reg4·
    시드·에폭 같은 개선을 88% 구간에도 실을 수 있다.

사용법
    python colab/export_final.py <브랜치목록> [시드목록]
예)
    python colab/export_final.py all_reg4,futures_reg4 42,1,777
    python colab/export_final.py all_base,regular_base,all_reg2,futures_reg2 42

브랜치 이름
    all_base      44열, 전체
    regular_base  44열, 1군만
    all_reg2      45열 2단계 코드, 전체, 옛 퓨처스 가중 0.1
    futures_reg2  45열 2단계 코드, 퓨처스만, 같은 가중
    all_reg4      45열 4단계 교차, 전체, 같은 가중
    futures_reg4  45열 4단계 교차, 퓨처스만, 같은 가중

추론에서 abs_regime 값
    2단계  전부 1        (2025 는 전부 새 체제)
    4단계  퓨처스 2, 1군 3
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
OLD_F_MAX = 2022
OLD_W = 0.1
# Stage1 학습률. lr_sweep 에서 2e-3 -> 3e-3 이 1군 +6.1 (t=4.61, 4/4),
# 퓨처스 +16.6. 4e-3/5e-3 은 1군에서 3e-3 보다 낮아 고원의 안쪽을 고른다.
LR1 = 3e-3
# Stage2 에폭. 퓨처스 브랜치는 2024 표본이 30,010행(15 스텝)뿐이라 1에폭으로는
# 부족했다. stage_ext 에서 e4~e8 이 고원이고 그 안의 차이는 SE 안쪽이라
# 덜 극단적인 e4 를 쓴다 (퓨처스 구간 +10.2, t=4.81, 4/4).
# 1군/all 브랜치는 223,497행(110 스텝)이라 1에폭으로 충분하다(e2 +0.6, 2/4).
EP2 = int(os.environ.get("EP2", "1"))
# 아키텍처. arch_sweep 결과로 정한다. tabm / tabm-mini / tabm-packed.
ARCH = os.environ.get("ARCH", "tabm")

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from export_ours import export, prep_stats                      # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

BASE_CFG = dict(k=32, n_blocks=3, d_block=256, dropout=0.1, d_emb=16,
                num_embeddings="linear_relu", backbone_kind="batch_ensemble", arch_type=ARCH,
                ep1=2, ep2=EP2, lr1=LR1, lr2=2e-4, vs=VS, decay=0.0,
                prep="friend", sched="per_epoch")


if __name__ == "__main__":
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else ["all_reg4",
                                                             "futures_reg4"]
    seeds = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
             else [42, 1, 777])
    t0 = time.time()

    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])       # 2019~2024 전부
    Xdf = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xdf.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xdf.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    old_f = isf & old
    m_tr = np.ones(len(Xf), bool)                          # VS=2025 -> 전 행
    idx_all = np.arange(len(Xf))
    w10 = np.where(old_f, OLD_W, 1.0).astype(np.float64)

    c2 = ((isf & ~old) | (~isf)).astype(np.float32)[:, None]
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]

    # (행렬, 범주열, 피처이름, 코드태그) 를 판본별로 준비한다
    variants = {
        "base": (Xf, ci, cols, "none"),
        "reg2": (np.concatenate([Xf, c2], 1), ci + [Xf.shape[1]],
                 cols + ["abs_regime"], "reg2_w0.1"),
        "reg4": (np.concatenate([Xf, c4], 1), ci + [Xf.shape[1]],
                 cols + ["abs_regime"], "reg4_w0.1"),
    }
    BR = {"all_base": ("base", idx_all, None),
          "regular_base": ("base", idx_all[~isf], None),
          "all_reg2": ("reg2", idx_all, w10),
          "futures_reg2": ("reg2", idx_all[isf], w10[isf]),
          "all_reg4": ("reg4", idx_all, w10),
          "futures_reg4": ("reg4", idx_all[isf], w10[isf])}
    bad = [w for w in want if w not in BR]
    if bad:
        raise SystemExit(f"모르는 브랜치: {bad}   가능: {list(BR)}")

    packs = {}
    for v in {BR[w][0] for w in want}:
        X, c, feats, tag = variants[v]
        Xn, Xc, cards, st = prep_stats(X, m_tr, c)
        packs[v] = (Xn, Xc, cards, st, feats, tag)
        print(f"  [{v}] 수치 {Xn.shape[1]}열 범주 {Xc.shape[1]}열 "
              f"카디널리티 {cards.tolist()}", flush=True)
    print(f"  학습 {len(Xf):,}행   옛 체제 퓨처스 {int(old_f.sum()):,} "
          f"-> 가중 {OLD_W}\n")

    for seed in seeds:
        for name in want:
            v, idx, w = BR[name]
            Xn, Xc, cards, st, feats, tag = packs[v]
            G.Xn, G.cards = Xn, cards
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            t1 = time.time()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            mdl = G.make_model(arch=ARCH)
            G.train(mdl, idx, 2, LR1, w=w, seed=seed, tag=f"{name} s{seed} S1")
            s2 = season[idx] == VS - 1
            for _e in range(EP2):
                G.train(mdl, idx[s2], 1, 2e-4, params=G.stage2_params(mdl),
                        seed=seed + _e, tag=f"{name} s{seed} S2e{_e+1}")
            p = os.path.join(OUT, f"fin_{name}_s{seed}.npz")
            export(mdl, p, st, cards, Xn.shape[1],
                   dict(BASE_CFG, seed=seed, regime=tag, branch=name), feats)
            print(f"  {name:14s} s{seed:<4d} {len(idx):>10,}행  "
                  f"Stage2 {int(s2.sum()):,}  {os.path.getsize(p)/1e6:.1f}MB  "
                  f"{time.time()-t1:.0f}s", flush=True)
            del mdl
            torch.cuda.empty_cache()
    print(f"  완료 {time.time()-t0:.0f}s   -> out/fin_*.npz")
