# -*- coding: utf-8 -*-
"""MS 구조 2라운드 — 학습역학(1라운드 전멸)이 아니라 구조를 바꾼다. VS=2024.

팔  base       현행 (전구간 all 단일모델 + S2)          <- 대조군
    branch     f2b 설계 이식: all 학습 후 사본 둘을 regular/futures 행으로
               1ep lr2e-4 미세조정 -> 0.6 all + 0.4 (행 종류별 분기)
               (S2 와 같은 "부분 적응" 기제 — MS 에서 S2 는 +3.1 이었다)
    state6     보조 softmax 4 -> 5범주: other-실패를 ball/strike 로 분할
               (복원 라벨 활용 확대 — 이기는 기제의 심화)
저장: msr_{arm}_s{seed}.npy
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
VS = 2024
import features44 as F                                          # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings            # noqa: E402
from tabm import TabM                                           # noqa: E402
import copy                                                     # noqa: E402

raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
aux4 = M.recover_state(raw)


def recover_state6(df):
    """4범주의 other-실패를 ball/strike 로 가른다. 0=성공,1=rev,2=mid,3=other-ball,4=other-strike."""
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
    out[m3 & ~np.isfinite(ball)] = -1          # 분할 불가면 보조 마스크 제외
    return out


aux6 = recover_state6(raw)
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


def make6(n_num, cards, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return TabM.make(n_num_features=n_num,
                     cat_cardinalities=[int(c) for c in cards], d_out=6,
                     num_embeddings=LinearReLUEmbeddings(n_num, d_embedding=16),
                     arch_type="tabm", k=32, n_blocks=3, d_block=512,
                     dropout=0.1).to(M.DEVICE)


def finetune(m, idx, seed, aux):
    lr0, ep0 = M.LR, M.EPOCHS
    M.LR, M.EPOCHS = 2e-4, 1
    m = M.train_model(m, idx, y, aux, w, seed, "ft")
    M.LR, M.EPOCHS = lr0, ep0
    return m


s2i = tr[season[tr] == VS - 1]
reg_i = tr[~isf[tr]]
fut_i = tr[isf[tr]]

for nm in ("base", "branch", "state6"):
    ps = []
    for sd in M.SEEDS:
        t0 = time.time()
        aux_use = aux6 if nm == "state6" else aux4
        if nm == "state6":
            m = make6(Xn.shape[1], cards, sd)
        else:
            m = M.make_model(Xn.shape[1], cards, sd)
        m = M.train_model(m, tr, y, aux_use, w, sd, f"{nm} s{sd}")
        m = finetune(m, s2i, sd + 1, aux_use)          # S2 공통
        if nm == "branch":
            m_reg = finetune(copy.deepcopy(m), reg_i, sd + 2, aux_use)
            m_fut = finetune(copy.deepcopy(m), fut_i, sd + 3, aux_use)
            d_all, _ = M.predict(m, va)
            d_reg, _ = M.predict(m_reg, va)
            d_fut, _ = M.predict(m_fut, va)
            d = np.where(isf_va, 0.6 * d_all + 0.4 * d_fut,
                         0.6 * d_all + 0.4 * d_reg)
            del m_reg, m_fut
        else:
            d, _ = M.predict(m, va)
        ps.append(d)
        np.save(DL + f"/msr_{nm}_s{sd}.npy", d)
        print(f"  {nm} s{sd}  {time.time()-t0:.0f}s  단독 {sc(d):.1f}  "
              f"1군 {sc(d, ~isf_va):.1f}  퓨처스 {sc(d, isf_va):.1f}", flush=True)
        del m
        torch.cuda.empty_cache()
    p = np.mean(ps, 0)
    print(f"ARM {nm:8s} 시드평균 단독 {sc(p):8.1f}  1군 {sc(p, ~isf_va):8.1f}  "
          f"퓨처스 {sc(p, isf_va):8.1f}", flush=True)
