# -*- coding: utf-8 -*-
"""state-head 기부자 재료 — ms_s2/state6 폴드 재실행, direct+state 양 헤드 저장.

M.predict 의 둘째 반환(P(state=0))을 지금까지 버려 왔다. 잔차 기부자 검사
(r = p_state - p_direct, 수준·기울기 직교화)용으로 두 폴드에서 저장한다.
저장: mssh_{VS}_{fam}_{d|st}_s{seed}.npy
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
aux4 = M.recover_state(raw)
aux6 = recover_state6(raw)

for VS in (2024, 2022):
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
    w[tr] = np.where(isf[tr] & old[tr], .1, 1.)
    w[tr] = w[tr] * (1.5 ** (season[tr].astype(np.float32) - 2019.0))

    def sc(p, m=None):
        q = np.ones(len(yv), bool) if m is None else m
        return M.best_shift(p[q], yv[q])[0]

    s2i = tr[season[tr] == VS - 1]
    for nm in ("ms_s2", "state6"):
        for sd in (42, 1):
            t0 = time.time()
            if nm == "state6":
                m = make6(Xn.shape[1], cards, sd)
                aux_use = aux6
            else:
                m = M.make_model(Xn.shape[1], cards, sd)
                aux_use = aux4
            m = M.train_model(m, tr, y, aux_use, w, sd, f"{nm}-{VS} s{sd}")
            lr0, ep0 = M.LR, M.EPOCHS
            M.LR, M.EPOCHS = 2e-4, 1
            m = M.train_model(m, s2i, y, aux_use, w, sd + 1,
                              f"{nm}-{VS} s{sd} S2")
            M.LR, M.EPOCHS = lr0, ep0
            d, st = M.predict(m, va)
            np.save(DL + f"/mssh_{VS}_{nm}_d_s{sd}.npy", d)
            np.save(DL + f"/mssh_{VS}_{nm}_st_s{sd}.npy", st)
            print(f"  {nm} VS{VS} s{sd}  {time.time()-t0:.0f}s  "
                  f"direct {sc(d):.1f}  state {sc(st):.1f}", flush=True)
            del m
            torch.cuda.empty_cache()
print("statehead done", flush=True)
