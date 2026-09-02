# -*- coding: utf-8 -*-
"""구체제 퓨처스(2021-22 F) 학습가중 0.1 -> 0 절제. 외부 검토자 제안의 유일한
학습 실험 — 표적 기전이 다른 행이 공유 표현을 오염시키는지.

state6(1군 슬롯) 3폴드 + aux4 ms_s2(퓨처스 슬롯) 2폴드, 시드 42/1.
저장: msw0_{VS}_s6_s{sd}.npy / msw0_{VS}_ms_s{sd}.npy
판정은 로컬 route_diag 확장에서 믹스 델타로.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("AUDIT_MODE", "all")
os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
SEEDS = (42, 1)
import features44 as F                                          # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings            # noqa: E402
from tabm import TabM                                           # noqa: E402


def recover_state6(df):
    s = M.recover_state(df).copy()
    pid = df.pitcher_id.to_numpy()
    n = df.asof_pitcher_n.fillna(0.).to_numpy(float)
    nxt = (pid[1:] == pid[:-1]) & np.isclose(np.diff(n), 1., atol=1e-8)
    src = np.flatnonzero(nxt) + 1
    dst = src - 1
    cum = df["asof_pitcher_ball_rate"].fillna(0.).to_numpy(float) * n
    inc = cum[src] - cum[dst]
    lab = np.rint(inc)
    good = (np.abs(inc - lab) < .25) & ((lab == 0) | (lab == 1))
    ball = np.full(len(df), np.nan)
    ball[dst[good]] = lab[good]
    out = s.copy()
    m3 = s == 3
    out[m3 & (ball == 1)] = 3
    out[m3 & (ball == 0)] = 4
    out[m3 & ~np.isfinite(ball)] = -1
    return out


def make6(n_num, cards, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return TabM.make(n_num_features=n_num,
                     cat_cardinalities=[int(c) for c in cards], d_out=6,
                     num_embeddings=LinearReLUEmbeddings(n_num, d_embedding=16),
                     arch_type="tabm", k=32, n_blocks=3, d_block=512,
                     dropout=0.1).to(M.DEVICE)


raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
aux6 = recover_state6(raw)
aux4 = M.recover_state(raw)

for VS in (2024, 2023, 2022):
    built = F.build(DATA, VS=VS)
    X44 = built["X44"].astype(np.float32)
    names = list(built["F44"])
    xnum, _, xcat, _ = A.audit_features(raw, X44, names)
    c4 = np.where(old & isf, 0., np.where(old & ~isf, 1.,
                  np.where(isf, 2., 3.))).astype(np.float32)[:, None]
    Xin = np.concatenate([X44, xnum, c4, xcat], 1)
    ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    c4i = X44.shape[1] + xnum.shape[1]
    cat_idx = ci + [c4i] + list(range(c4i + 1, Xin.shape[1]))
    m_tr = season < VS
    Xn, Xc, cards = G.prep(Xin, m_tr, cat_idx)
    G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
    tr = np.flatnonzero(m_tr)
    va = np.flatnonzero(season == VS)
    yv = y[va].astype(float)
    isf_va = isf[va]
    w = np.ones(len(y), np.float32)
    w[tr] = np.where(isf[tr] & old[tr], 0.0, 1.0)          # <- 유일한 변경
    w[tr] = w[tr] * (1.5 ** (season[tr].astype(np.float32) - 2019.0))
    s2i = tr[season[tr] == VS - 1]

    def sc(p, m=None):
        q = np.ones(len(yv), bool) if m is None else m
        return M.best_shift(p[q], yv[q])[0]

    arms = [("s6", aux6, True)]
    if VS != 2023:
        arms.append(("ms", aux4, False))
    for nm, aux, use6 in arms:
        ps = []
        for sd in SEEDS:
            t0 = time.time()
            if use6:
                m = make6(Xn.shape[1], cards, sd)
            else:
                m = M.make_model(Xn.shape[1], cards, sd)
            m = M.train_model(m, tr, y, aux, w, sd, f"w0-{nm} VS{VS} s{sd}")
            lr0, ep0 = M.LR, M.EPOCHS
            M.LR, M.EPOCHS = 2e-4, 1
            m = M.train_model(m, s2i, y, aux, w, sd + 1,
                              f"w0-{nm} VS{VS} s{sd} S2")
            M.LR, M.EPOCHS = lr0, ep0
            d, _ = M.predict(m, va)
            ps.append(d)
            np.save(DL + f"/msw0_{VS}_{nm}_s{sd}.npy", d)
            print(f"  w0-{nm} VS{VS} s{sd}  {time.time()-t0:.0f}s  "
                  f"단독 {sc(d):.1f}  1군 {sc(d, ~isf_va):.1f}  "
                  f"퓨처스 {sc(d, isf_va):.1f}", flush=True)
            del m
            torch.cuda.empty_cache()
        p = np.mean(ps, 0)
        print(f"ARM w0-{nm} VS{VS} 시드평균 단독 {sc(p):8.1f}  "
              f"1군 {sc(p, ~isf_va):8.1f}  퓨처스 {sc(p, isf_va):8.1f}",
              flush=True)
