# -*- coding: utf-8 -*-
"""DIN 에 state6 보조헤드 이식 — MS 에서 +15 를 낸 기제의 마지막 이식처. VS=DA_VS.

ctr_zoo.DINBST 의 출력헤드를 1->6 으로 바꾸고 (direct + 5범주 CE),
학습 손실에 SW·CE 를 더한다. 팔: SW 0 (대조군) / 0.45 / 0.9.
저장: da_{VS}_sw{SW}_s{seed}.npy
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
os.environ.setdefault("AIMERS_ROOT", "/root")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
import ctr_zoo as Z                                             # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("DA_VS", "2024"))
Z.VS = VS
D = Z.build_inputs()
season, isf, y = D["season"], D["isf"], D["y"].astype(np.float64)
gate = np.where(season == VS)[0]
yv, isf_g = y[gate], isf[gate]
tr_idx = np.where(D["m_tr"])[0]
old = season <= 2022


def recover_state6(df):
    import multistate_softmax as M
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


os.environ.setdefault("AUDIT_MODE", "all")
os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
aux6 = recover_state6(raw)
assert len(aux6) == len(y)

w = np.where(isf & old, 0.1, 1.0).astype(np.float64)
w[~D["m_tr"]] = 0.0
import features44 as F                                          # noqa: E402


def sc(p, m=None):
    q = np.ones(len(yv), bool) if m is None else m
    return F.best_shift(p[q], yv[q])[0]


def fit_aux(sw, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_num, cards = D["Xn"].shape[1], D["cards"]
    m = Z.DINBST(n_num, cards, D["n_state"], D["n_cell"], mode="din")
    d_hid = m.mlp[-1].in_features
    m.mlp[-1] = nn.Linear(d_hid, 6)
    m = m.to(Z.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=Z.LR, weight_decay=Z.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Z.EPOCHS)
    XN = torch.from_numpy(D["Xn"]).to(Z.DEV)
    XC = torch.from_numpy(D["Xc"]).to(Z.DEV)
    Y = torch.from_numpy(D["y"]).to(Z.DEV)
    W = torch.from_numpy(w.astype(np.float32)).to(Z.DEV)
    S6 = torch.from_numpy(aux6.astype(np.int64)).to(Z.DEV)
    SS = torch.from_numpy(D["SEQ_S"]).to(Z.DEV)
    SCq = torch.from_numpy(D["SEQ_C"]).to(Z.DEV)
    SM = torch.from_numpy(D["SEQ_M"]).to(Z.DEV)
    CU = torch.from_numpy(D["CUR"]).to(Z.DEV)
    for ep in range(Z.EPOCHS):
        m.train()
        perm = np.random.permutation(tr_idx)
        for a in range(0, len(perm), Z.BS):
            b = torch.from_numpy(perm[a:a + Z.BS]).to(Z.DEV)
            z = m(XN[b], XC[b], SS[b], SCq[b], SM[b], CU[b])
            zd = z[:, 0]
            p = torch.sigmoid(zd)
            per = 0.5 * nn.functional.binary_cross_entropy_with_logits(
                zd, Y[b], reduction="none") + 0.5 * (p - Y[b]) ** 2
            loss = (per * W[b]).sum() / W[b].sum()
            if sw > 0:
                sb = S6[b]
                mk = sb >= 0
                if bool(mk.any()):
                    ce = nn.functional.cross_entropy(z[:, 1:][mk], sb[mk],
                                                     reduction="none")
                    loss = loss + sw * (ce * W[b][mk]).sum() / W[b][mk].sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(gate), 8192):
            b = torch.from_numpy(gate[a:a + 8192]).to(Z.DEV)
            z = m(XN[b], XC[b], SS[b], SCq[b], SM[b], CU[b])
            out.append(torch.sigmoid(z[:, 0]).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


for sw in (0.0, 0.45, 0.9):
    ps = []
    for sd in (42, 1, 777):
        t0 = time.time()
        p = fit_aux(sw, sd)
        ps.append(p)
        np.save(DL + f"/da_{VS}_sw{sw}_s{sd}.npy", p)
        print(f"  da sw{sw} s{sd}  {time.time()-t0:.0f}s  단독 {sc(p):.1f}  "
              f"1군 {sc(p, ~isf_g):.1f}", flush=True)
    pm = np.mean(ps, 0)
    print(f"ARM da VS{VS} sw{sw:.2f} 시드평균 단독 {sc(pm):8.1f}  "
          f"1군 {sc(pm, ~isf_g):8.1f}  퓨처스 {sc(pm, isf_g):8.1f}", flush=True)
