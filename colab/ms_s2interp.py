# -*- coding: utf-8 -*-
"""S2 체크포인트 보간 — θ_α=(1-α)θ0+αθS2, α∈{0.5,0.75}. 조언 7.

msr*_state6 과 같은 학습(결정론 재현 확인됨)을 다시 돌려, 이번엔
    msk_{VS}_s{sd}_z.npy   (n_va, 32) float16   멤버별 direct 로짓
    msk_{VS}_s{sd}_q.npy   (n_va, 5)  float32   보조 softmax (멤버 평균)
을 저장한다. 집계규칙/불확실도/보조센서 실험은 전부 이 위에서 CPU 로.
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
VS_LIST = (2024, 2023, 2022)
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


def predict_dump(model, idx):
    model.eval()
    ii = torch.from_numpy(idx.astype(np.int64))
    Z, Q = [], []
    for start in range(0, len(idx), 8192):
        b = ii[start:start + 8192]
        xn = G.XN[b].to(M.DEVICE, non_blocking=True)
        xc = G.XC[b].to(M.DEVICE, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda",
                                             dtype=torch.float16, enabled=M.AMP):
            dl, sl = M.forward_parts(model, xn, xc)
            Z.append(dl.float().cpu().numpy().astype(np.float16))
            Q.append(torch.softmax(sl, dim=2).mean(dim=1).float().cpu().numpy())
    return np.concatenate(Z), np.concatenate(Q)


raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
aux6 = recover_state6(raw)
for VS in VS_LIST:
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
    w = np.ones(len(y), np.float32)
    w[tr] = np.where(isf[tr] & old[tr], .1, 1.)
    w[tr] = w[tr] * (1.5 ** (season[tr].astype(np.float32) - 2019.0))
    s2i = tr[season[tr] == VS - 1]
    for sd in (42, 1):
        t0 = time.time()
        m = make6(Xn.shape[1], cards, sd)
        m = M.train_model(m, tr, y, aux6, w, sd, f"si-{VS} s{sd}")
        th0 = {k: v.detach().clone() for k, v in m.state_dict().items()}
        lr0, ep0 = M.LR, M.EPOCHS
        M.LR, M.EPOCHS = 2e-4, 1
        m = M.train_model(m, s2i, y, aux6, w, sd + 1, f"si-{VS} s{sd} S2")
        M.LR, M.EPOCHS = lr0, ep0
        th1 = {k: v.detach().clone() for k, v in m.state_dict().items()}
        for al in (0.5, 0.75):
            m.load_state_dict({k: (1 - al) * th0[k].float() + al * th1[k].float()
                               for k in th1})
            d, _ = M.predict(m, va)
            np.save(DL + f"/msi_{VS}_s{sd}_a{al}.npy", d)
        m.load_state_dict(th1)
        print(f"  si VS{VS} s{sd}  {time.time()-t0:.0f}s  저장", flush=True)
        del m
        torch.cuda.empty_cache()
print("덤프 끝", flush=True)
