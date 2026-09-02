# -*- coding: utf-8 -*-
"""TabM 아키텍처 변형과 용량을 훑는다. 한 번도 안 재본 축이다.

왜 지금인가
    오늘 codegap 에서 학습 코드의 결함을 고쳤다(코사인을 에폭 단위로).
    리더보드가 그걸 확인해 줬다 — submit_21(구스케줄러) 1024 -> submit_23(고침)
    1048, 시프트 몫 2.4 를 빼면 약 +21.6 이고 관문의 +19.2 와 맞는다.

    덜 학습된 모델은 용량을 못 쓴다. 그래서 용량·구조 판정은 학습이 제대로
    될 때 다시 해야 한다. arch_type 은 아예 재본 적이 없고, k=64 는
    구스케줄러에서 -7.4 로 기각됐다.

무엇을 재나 (지인 전처리, VS=2024, all 브랜치, 1군 행 채점, 시드 4개)
    tabm        현행 (BatchEnsemble, rank-1 어댑터 k개)
    tabm-mini   어댑터를 하나로 줄인 변형
    tabm-packed 완전 독립 앙상블
    k64         현행 구조에서 k 만 32 -> 64
    d512        d_block 256 -> 512
    blocks4     n_blocks 3 -> 4

    Stage1 2에폭 lr 3e-3(오늘 확정), Stage2 1에폭 2e-4 output+마지막블록.
    바뀌는 건 구조뿐이다.

    tabm-mini / tabm-packed 는 TabM 구현이 지원하는 값이다
    (official_tabm.py 의 arch_type 검증 목록과 같다).

판정
    시드 짝비교, 최적 시프트, 부호 일관성(4/4). 오늘 쓴 기준 그대로다.
    한 폴드이므로 이겨도 곧바로 배치하지 않고 다년으로 한 번 더 본다.
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
OUT = os.environ.get("AIMERS_OUT", "/workspace/aimers/out")
DATA = os.environ.get("AIMERS_DATA", "/workspace/aimers/data")
SEEDS = (42, 1, 777, 2)
LR1 = 3e-3

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings            # noqa: E402
from tabm import TabM                                           # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(yv), bool)


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def make(arch="tabm", k=32, n_blocks=3, d_block=256, dropout=0.1, d_emb=16):
    return TabM.make(
        n_num_features=G.Xn.shape[1],
        cat_cardinalities=[int(c) for c in G.cards],
        d_out=1,
        num_embeddings=LinearReLUEmbeddings(G.Xn.shape[1], d_embedding=d_emb),
        arch_type=arch, k=k, n_blocks=n_blocks,
        d_block=d_block, dropout=dropout,
    ).to(G.DEV)


def fit(seed, idx, season, **kw):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = make(**kw)
    G.train(m, idx, 2, LR1, seed=seed, tag=f"S1 {kw.get('arch','tabm')}")
    s2 = season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed, tag="S2")
    p = G.predict(m, gate)
    n_par = sum(x.numel() for x in m.parameters())
    del m
    torch.cuda.empty_cache()
    return p, n_par


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

    Xn, Xc, cards = G.prep(Xf, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    season = G.season.astype(np.float64)
    tr_idx = np.where(G.m_tr)[0]

    plans = [("tabm", {}),
             ("tabm-mini", dict(arch="tabm-mini")),
             ("tabm-packed", dict(arch="tabm-packed")),
             ("k64", dict(k=64)),
             ("d512", dict(d_block=512)),
             ("blocks4", dict(n_blocks=4))]
    only = os.environ.get("ARCH_ONLY")
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        plans = [p for p in plans if p[0] in wanted]
    P, NP = {}, {}
    for nm, kw in plans:
        t0 = time.time()
        try:
            for s in SEEDS:
                P[(nm, s)], NP[nm] = fit(s, tr_idx, season, **kw)
                np.save(os.path.join(OUT, f"ar_{nm}_s{s}.npy"), P[(nm, s)])
            G.log(f"  {nm:12s} 파라미터 {NP[nm]:>10,}  {time.time()-t0:.0f}s")
        except Exception as e:
            G.log(f"  {nm:12s} 실패: {type(e).__name__} {e}")

    ok = [nm for nm, _ in plans if (nm, SEEDS[0]) in P]
    G.log("\n  all 브랜치 단독, 최적 시프트, 시드별 계산 후 평균")
    G.log(f"  {'구성':12s} {'1군 행':>9s} {'전체':>9s} {'퓨처스':>9s}   시드별(1군)")
    for nm in ok:
        v = [sc(P[(nm, s)], R) for s in SEEDS]
        G.log(f"  {nm:12s} {np.mean(v):9.1f} "
              f"{np.mean([sc(P[(nm,s)], FULL) for s in SEEDS]):9.1f} "
              f"{np.mean([sc(P[(nm,s)], is_f) for s in SEEDS]):9.1f}   "
              f"[{', '.join(f'{x:.0f}' for x in v)}]")

    G.log("\n  tabm 대비 짝차이 (1군 행)")
    for nm in ok[1:]:
        d = [sc(P[(nm, s)], R) - sc(P[("tabm", s)], R) for s in SEEDS]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
        G.log(f"    {nm:12s} {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/4  "
              f"[{', '.join(f'{x:+.1f}' for x in d)}]")

    G.log("\n  k=64 는 구스케줄러에서 -7.4 로 기각됐다. 학습이 제대로 되면")
    G.log("  용량 판정이 달라질 수 있어 다시 넣었다.")
