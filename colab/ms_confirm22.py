# -*- coding: utf-8 -*-
"""state6 의 VS=2022 확인 — 2024 단독 +14.8 / 혼합 +6.5(2시드 w.40)의 두 번째 폴드.

base 대조군은 msp_2022_ms_s2_s* 저장본으로 대신한다 (동일 구성).
저장: msr22_state6_s{seed}.npy
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
VS = 2022
import features44 as F                                          # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings            # noqa: E402
from tabm import TabM                                           # noqa: E402


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
    out[m3 & ~np.isfinite(ball)] = -1
    return out


raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
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


s2i = tr[season[tr] == VS - 1]
ps = []
for sd in M.SEEDS:
    t0 = time.time()
    m = make6(Xn.shape[1], cards, sd)
    m = M.train_model(m, tr, y, aux6, w, sd, f"s6-22 s{sd}")
    lr0, ep0 = M.LR, M.EPOCHS
    M.LR, M.EPOCHS = 2e-4, 1
    m = M.train_model(m, s2i, y, aux6, w, sd + 1, f"s6-22 s{sd} S2")
    M.LR, M.EPOCHS = lr0, ep0
    d, _ = M.predict(m, va)
    ps.append(d)
    np.save(DL + f"/msr22_state6_s{sd}.npy", d)
    print(f"  state6 VS22 s{sd}  {time.time()-t0:.0f}s  단독 {sc(d):.1f}  "
          f"1군 {sc(d, ~isf_va):.1f}  퓨처스 {sc(d, isf_va):.1f}", flush=True)
    del m
    torch.cuda.empty_cache()
p = np.mean(ps, 0)
print(f"ARM state6 VS22 시드평균 단독 {sc(p):8.1f}  1군 {sc(p, ~isf_va):8.1f}  "
      f"퓨처스 {sc(p, isf_va):8.1f}", flush=True)
