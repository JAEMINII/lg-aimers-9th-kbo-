# -*- coding: utf-8 -*-
"""1군 학습 격차의 범인을 가른다. 우리가 **추가한** 두 가지를 하나씩 뺀다.

무엇이 남았나
    스케줄러를 고친 뒤 지인 코드와 우리 코드를 끝까지 대조했더니 설정은 전부 같다.
        손실 0.5 BCE + 0.5 Brier / 클리핑 5.0 / wd 3e-4 / 배치 2048 /
        결측 지시자 / 범주 0=미지 / 중앙값 대치 후 표준화
    남은 차이는 우리가 **얹은** 두 가지뿐이다.
        (a) abs_regime 4단계 열   -> 지인은 44열, 우리는 45열
        (b) 옛퓨처스(<=2022) 가중 0.1 -> 105,308행을 사실상 뺀다

    둘 다 관문이 좋다고 해서 넣은 것이고, 둘 다 1군 행을 담당하는 all 브랜치에
    걸려 있다. 리더보드는 반대로 말한다.
        지인 1군   submit_20 1057   submit_26 1056
        우리 1군   submit_23 1048   submit_24 1049   submit_25 1052

왜 묶어서 내지 않나
    어제 퓨처스에 다섯 가지를 한꺼번에 실어 1057 -> 1056 을 받았다. 제출 한 장을
    쓰고 아무것도 못 갈랐다. 1군은 88.2% 라 해상도가 8배지만, 그래도 두 축을
    같이 움직이면 상쇄를 못 본다. 먼저 관문에서 크기를 재고 순서를 정한다.

관문을 얼마나 믿나 — 축마다 다르다
    (b) 가중은 **학습 분포** 질문이다. 시즌가중에서 관문이 맞혔던 부류다.
    (a) 열 추가는 **피처** 질문이다. 관문이 여섯 번 틀린 부류다.
    그래서 (b)는 관문 결과를 따르고, (a)는 크기만 읽고 판정은 리더보드로 넘긴다.

네 팔 (배치 구성, 1군 경로 = 0.6 all + 0.4 regular)
    cur       reg4 열 + 가중 0.1      현행
    no_col    44열   + 가중 0.1       열만 뺌
    no_w      reg4 열 + 무가중        가중만 뺌
    friend    44열   + 무가중         지인 학습 분포와 동일
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
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
ARMS = (("cur", True, True), ("no_col", False, True),
        ("no_w", True, False), ("friend", False, False))

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def one(seed, tr_idx, t_isf, w, season):
    """1군 경로만 만든다. all 과 regular 두 브랜치."""
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)), ("regular", ~t_isf)):
        idx = tr_idx[sel]
        ww = None if w is None else w[sel]
        torch.manual_seed(seed)                # 모델 생성 전에 건다
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"{br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        G.train(m, s2, 1, 2e-4, params=G.stage2_params(m), seed=seed,
                tag=f"{br} s{seed} S2")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return 0.6 * P["all"] + 0.4 * P["regular"]


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
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)

    # 열 유무로 전처리가 갈리므로 두 벌을 미리 만들어 둔다
    PACK = {}
    for use_col in (True, False):
        Xin = np.concatenate([Xf, c4], 1) if use_col else Xf
        cidx = ci + [Xf.shape[1]] if use_col else ci
        PACK[use_col] = G.prep(Xin, G.m_tr, cidx)
    G.log(f"  45열 벌 수치 {PACK[True][0].shape[1]}  "
          f"44열 벌 수치 {PACK[False][0].shape[1]}  "
          f"학습 {len(tr_idx):,}  가중 대상 {int((w10 < 1).sum()):,}행")

    RES = {}
    for nm, use_col, use_w in ARMS:
        Xn, Xc, cards = PACK[use_col]
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        t0 = time.time()
        w = w10 if use_w else None
        RES[nm] = [one(sd, tr_idx, t_isf, w, season) for sd in SEEDS]
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"a1_{nm}_s{sd}.npy"), RES[nm][i])
        G.log(f"  {nm:8s} 학습 끝 {time.time()-t0:.0f}s")

    G.log("\n" + "=" * 76)
    G.log(f"  {'팔':9s} {'열':>5s} {'가중':>5s} {'1군 행':>9s}"
          f"   현행 대비 (짝차이, 부호일치)")
    G.log("=" * 76)
    base = RES["cur"]
    for nm, use_col, use_w in ARMS:
        v = float(np.mean([sc(p, R) for p in RES[nm]]))
        col = "reg4" if use_col else "44"
        wt = "0.1" if use_w else "없음"
        if nm == "cur":
            G.log(f"  {nm:9s} {col:>5s} {wt:>5s} {v:9.1f}   <- 현행")
            continue
        d = [sc(a, R) - sc(b, R) for a, b in zip(RES[nm], base)]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
        G.log(f"  {nm:9s} {col:>5s} {wt:>5s} {v:9.1f}   "
              f"{mu:+7.1f}+-{se:5.1f}  t={mu/max(se,1e-9):5.2f}  "
              f"{sum(1 for x in d if x > 0)}/{len(d)}   "
              f"[{', '.join(f'{x:+.0f}' for x in d)}]")

    G.log("\n  읽는 법")
    G.log("    no_w 가 크게 이기면 -> 옛퓨처스 가중이 1군을 망치고 있었다.")
    G.log("      가중은 학습분포 축이라 관문을 따라도 된다. 바로 제출로 옮긴다.")
    G.log("    no_col 이 이기면 -> 크기만 읽는다. 열 추가는 관문이 여섯 번 틀린")
    G.log("      부류라 부호를 못 믿는다. 리더보드 한 장으로 확정해야 한다.")
    G.log("    friend 가 셋 다보다 높으면 두 축이 상호작용한다는 뜻이다.")
    G.log("    넷이 다 비슷하면 -> 범인은 이 둘이 아니다. 코드 대조를 더 판다.")
