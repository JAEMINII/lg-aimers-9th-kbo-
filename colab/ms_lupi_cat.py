# -*- coding: utf-8 -*-
"""LUPI-범주 보조헤드 — state6 위에 Trackman 특권정보를 softmax 로 얹는다. VS=2024.

증류(예측 전달) 4형태·물리 회귀 보조는 전멸했지만, 이긴 기제는 '범주형 보조
라벨 세분화'(state6 1군 +19~23)다. 그 형태로 특권정보를 주입한다.
Trackman 은 1군 70~79% 커버 — state6 가 배치되는 라우팅 구간과 일치.

팔  ptype4    pitch_type_group 4범주, w2=0.05  (49의 유일 생존 보조를 MS 로)
    phys8     (rel_speed, ivb, hb, spin) 표준화 KMeans-8, w2=0.05
    phys8w2   같은 라벨, w2=0.15
대조군: ms_round2 의 state6 (같은 시드/데이터, 1군 940.6).
저장: msl_{arm}_s{seed}.npy
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

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
from sklearn.cluster import KMeans                              # noqa: E402


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


raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
aux6 = recover_state6(raw)

# ---- 특권 범주 라벨 (row_id 매핑) --------------------------------------
mp = pd.read_csv(DL + "/lupi_match2.csv.gz")
rid = raw.row_id.to_numpy()
mpi = mp.set_index("row_id")
mpi = mpi[~mpi.index.duplicated(keep="first")]
al = mpi.reindex(rid)
pt_map = {"fastball": 0, "breaking": 1, "offspeed": 2, "other": 3}
aux_pt = np.array([pt_map.get(v, -1) for v in al.pitch_type_group.tolist()],
                  np.int64)
PH = ["rel_speed", "induced_vert_break", "horz_break", "spin_rate"]
Pv = al[PH].to_numpy(float)
okp = np.isfinite(Pv).all(1)
tr_fit = okp & (season < VS)
mu, sd = Pv[tr_fit].mean(0), Pv[tr_fit].std(0) + 1e-9
Z = (Pv - mu) / sd
km = KMeans(n_clusters=8, n_init=4, random_state=0)
sub = np.random.default_rng(0).choice(np.flatnonzero(tr_fit), 200_000,
                                      replace=False)
km.fit(Z[sub])
aux_ph = np.full(len(raw), -1, np.int64)
aux_ph[okp] = km.predict(Z[okp])
print(f"라벨 커버  ptype {np.mean(aux_pt >= 0):.3f}  "
      f"phys8 {np.mean(aux_ph >= 0):.3f}", flush=True)

# ---- 입력 -------------------------------------------------------------
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


def make_d(n_num, cards, seed, d_out):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return TabM.make(n_num_features=n_num,
                     cat_cardinalities=[int(c) for c in cards], d_out=d_out,
                     num_embeddings=LinearReLUEmbeddings(n_num, d_embedding=16),
                     arch_type="tabm", k=32, n_blocks=3, d_block=512,
                     dropout=0.1).to(M.DEVICE)


def train2(model, idx, aux2, w2, seed, lr, epochs, tag):
    """M.train_model + 둘째 CE 헤드 (out[...,6:], 마스크 -1 제외)."""
    torch.manual_seed(seed)
    ii = torch.from_numpy(idx.astype(np.int64))
    yy = torch.from_numpy(y.astype(np.float32))
    ss = torch.from_numpy(aux6.astype(np.int64))
    s2 = torch.from_numpy(aux2.astype(np.int64))
    ww = torch.from_numpy(w.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=M.AMP)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        total = 0.0
        for start in range(0, len(idx), M.BS):
            b = ii[perm[start:start + M.BS]]
            xn = G.XN[b].to(M.DEVICE, non_blocking=True)
            xc = G.XC[b].to(M.DEVICE, non_blocking=True)
            yb = yy[b].to(M.DEVICE, non_blocking=True)
            sb = ss[b].to(M.DEVICE, non_blocking=True)
            s2b = s2[b].to(M.DEVICE, non_blocking=True)
            wb = ww[b].to(M.DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=M.AMP):
                out = model(xn, xc).float()
                dm = out[..., 0].mean(dim=1)
                dp = torch.sigmoid(dm)
                bce = Fnn.binary_cross_entropy_with_logits(dm, yb,
                                                           reduction="none")
                brier = (dp - yb).square()
                st = out[..., 1:6]
                mk = sb >= 0
                if bool(mk.any()):
                    ce_each = Fnn.cross_entropy(
                        st[mk].reshape(-1, 5),
                        sb[mk, None].expand(-1, st.shape[1]).reshape(-1),
                        reduction="none").reshape(-1, st.shape[1]).mean(dim=1)
                    ce = (ce_each * wb[mk]).sum() / wb[mk].sum()
                else:
                    ce = torch.zeros((), device=M.DEVICE)
                c2 = out[..., 6:]
                mk2 = s2b >= 0
                if bool(mk2.any()):
                    ce2_each = Fnn.cross_entropy(
                        c2[mk2].reshape(-1, c2.shape[-1]),
                        s2b[mk2, None].expand(-1, c2.shape[1]).reshape(-1),
                        reduction="none").reshape(-1, c2.shape[1]).mean(dim=1)
                    ce2 = (ce2_each * wb[mk2]).sum() / wb[mk2].sum()
                else:
                    ce2 = torch.zeros((), device=M.DEVICE)
                loss = ((M.DIRECT_W * bce + M.Brier_W * brier) * wb).sum() \
                    / wb.sum() + M.STATE_W * ce + w2 * ce2
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(b)
        M.log(f"  {tag} ep{ep+1}/{epochs} loss={total/len(idx):.6f}")
    return model


s2i = tr[season[tr] == VS - 1]
ARMS = (("ptype4", aux_pt, 4, 0.05), ("phys8", aux_ph, 8, 0.05),
        ("phys8w2", aux_ph, 8, 0.15))
for nm, aux2, C2, w2 in ARMS:
    ps = []
    for sd in M.SEEDS:
        t0 = time.time()
        m = make_d(Xn.shape[1], cards, sd, 6 + C2)
        m = train2(m, tr, aux2, w2, sd, M.LR, M.EPOCHS, f"{nm} s{sd}")
        m = train2(m, s2i, aux2, w2, sd + 1, 2e-4, 1, f"{nm} s{sd} S2")
        d, _ = M.predict(m, va)
        ps.append(d)
        np.save(DL + f"/msl_{nm}_s{sd}.npy", d)
        print(f"  {nm} s{sd}  {time.time()-t0:.0f}s  단독 {sc(d):.1f}  "
              f"1군 {sc(d, ~isf_va):.1f}", flush=True)
        del m
        torch.cuda.empty_cache()
    p = np.mean(ps, 0)
    print(f"ARM {nm:8s} 시드평균 단독 {sc(p):8.1f}  1군 {sc(p, ~isf_va):8.1f}  "
          f"퓨처스 {sc(p, isf_va):8.1f}", flush=True)
