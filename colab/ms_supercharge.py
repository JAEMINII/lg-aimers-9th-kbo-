# -*- coding: utf-8 -*-
"""MS 구조 개선 1라운드 — 스케줄러/에폭/용량. VS=2024, 시드3, 감쇠15+S2 기반.

발견: multistate_softmax.train_model 에 lr 스케줄러가 없다 (상수 3e-3).
TabM 은 '에폭 단위 코사인' 수정 하나로 관문 +19.2 였다 — 그 수정이 지금 판의
엔진(MS, 가중 0.30)에 미적용 상태다.

팔  base      현행 (상수 lr, 2ep) + S2       <- 52 배치판 재현
    cos2      코사인(T_max=2), 2ep + S2
    cos3      코사인(T_max=3), 3ep + S2
    cos4      코사인(T_max=4), 4ep + S2
    d768cos   d_block 768, 코사인 2ep + S2   (d256->d512 가 +105 였던 축의 연장)
저장: mss_{arm}_s{seed}.npy — 로컬에서 혼합/가중 스윕에 쓴다.
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
VS = int(os.environ.get("MSS_VS", "2024"))
import features44 as F                                          # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings            # noqa: E402
from tabm import TabM                                           # noqa: E402

raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
aux = M.recover_state(raw)
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
w = np.ones(len(y), np.float32)
w[tr] = np.where(isf[tr] & old[tr], .1, 1.)
w[tr] = w[tr] * (1.5 ** (season[tr].astype(np.float32) - 2019.0))


def make_model_d(n_num, cards, seed, d_block):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return TabM.make(n_num_features=n_num,
                     cat_cardinalities=[int(c) for c in cards], d_out=5,
                     num_embeddings=LinearReLUEmbeddings(n_num, d_embedding=16),
                     arch_type="tabm", k=32, n_blocks=3, d_block=d_block,
                     dropout=0.1).to(M.DEVICE)


def train_cos(model, idx, seed, epochs, lr0=3e-3, tag=""):
    """M.train_model 과 동일하되 에폭 단위 코사인을 얹는다 (+19.2 의 그 수정)."""
    torch.manual_seed(seed)
    ii = torch.from_numpy(idx.astype(np.int64))
    yy = torch.from_numpy(y.astype(np.float32))
    ss = torch.from_numpy(aux.astype(np.int64))
    ww = torch.from_numpy(w.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=M.AMP)
    import torch.nn.functional as Fnn
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        for start in range(0, len(idx), M.BS):
            b = ii[perm[start:start + M.BS]]
            xn = G.XN[b].to(M.DEVICE, non_blocking=True)
            xc = G.XC[b].to(M.DEVICE, non_blocking=True)
            yb = yy[b].to(M.DEVICE); sb = ss[b].to(M.DEVICE)
            wb = ww[b].to(M.DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=M.AMP):
                dl, sl = M.forward_parts(model, xn, xc)
                dm = dl.mean(dim=1)
                dp = torch.sigmoid(dm)
                bce = Fnn.binary_cross_entropy_with_logits(dm, yb,
                                                           reduction="none")
                brier = (dp - yb).square()
                mmask = sb >= 0
                if bool(mmask.any()):
                    ce_each = Fnn.cross_entropy(
                        sl[mmask].reshape(-1, sl.shape[-1]),
                        sb[mmask, None].expand(-1, sl.shape[1]).reshape(-1),
                        reduction="none").reshape(-1, sl.shape[1]).mean(dim=1)
                    ce = (ce_each * wb[mmask]).sum() / wb[mmask].sum()
                else:
                    ce = torch.zeros((), device=M.DEVICE)
                loss = ((M.DIRECT_W * bce + M.Brier_W * brier) * wb).sum() \
                    / wb.sum() + M.STATE_W * ce
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
        sch.step()
        M.log(f"  {tag} ep{ep+1}/{epochs} lr={opt.param_groups[0]['lr']:.5f}")
    return model


def s2_tune(model, seed):
    s2i = tr[season[tr] == VS - 1]
    return train_cos(model, s2i, seed + 1, 1, lr0=2e-4, tag="S2")


def sc(p):
    return M.best_shift(p, yv)[0]


ARMS = (("base", "const", 2, 512), ("cos2", "cos", 2, 512),
        ("cos3", "cos", 3, 512), ("cos4", "cos", 4, 512),
        ("d768cos", "cos", 2, 768))
for nm, kind, ep, db in ARMS:
    ps = []
    for sd in M.SEEDS:
        t0 = time.time()
        m = make_model_d(Xn.shape[1], cards, sd, db)
        if kind == "const":
            lr0, ep0 = M.LR, M.EPOCHS
            M.EPOCHS = ep
            m = M.train_model(m, tr, y, aux, w, sd, f"{nm} s{sd}")
            M.EPOCHS = ep0
        else:
            m = train_cos(m, tr, sd, ep, tag=f"{nm} s{sd}")
        # S2 (마지막 시즌 1에폭 lr2e-4) — 52 배치와 동일
        lr0, ep0 = M.LR, M.EPOCHS
        M.LR, M.EPOCHS = 2e-4, 1
        s2i = tr[season[tr] == VS - 1]
        m = M.train_model(m, s2i, y, aux, w, sd + 1, f"{nm} s{sd} S2")
        M.LR, M.EPOCHS = lr0, ep0
        d, _ = M.predict(m, va)
        ps.append(d)
        np.save(DL + f"/mss_{VS}_{nm}_s{sd}.npy", d)
        print(f"  {nm} s{sd}  {time.time()-t0:.0f}s  단독 {sc(d):.1f}",
              flush=True)
        del m
        torch.cuda.empty_cache()
    print(f"ARM {nm:8s} 시드평균 단독 {sc(np.mean(ps, 0)):8.1f}", flush=True)
