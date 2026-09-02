# -*- coding: utf-8 -*-
"""phys8w2 배치 학습 — 퓨처스 슬롯용. 전 데이터(≤2024), decay15+S2.

state6 구조(d_out 6) + phys KMeans-8 보조헤드(w2=0.15, d_out 14).
추론은 direct 헤드만 쓰므로 npz 변환·추론 경로는 기존과 동일하다.
저장: /root/msa_pw2/msa_seed{seed}.pt (ms_deploy_audit 와 같은 payload 형식)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

os.environ.setdefault("AUDIT_MODE", "all")
os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
ROOT = "/root/aimers"
sys.path.insert(0, "/root")
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
OUT = Path(os.environ.get("MSA_OUT", "/root/msa_pw2"))
OUT.mkdir(parents=True, exist_ok=True)
import features44 as FF                                         # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings            # noqa: E402
from tabm import TabM                                           # noqa: E402
from sklearn.cluster import KMeans                              # noqa: E402
from ms_deploy_audit import recover_state6                      # noqa: E402

W2 = 0.15
C2 = 8

raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
aux6 = recover_state6(raw)

mp = pd.read_csv(DL + "/lupi_match2.csv.gz")
mpi = mp.set_index("row_id")
mpi = mpi[~mpi.index.duplicated(keep="first")]
al = mpi.reindex(raw.row_id.to_numpy())
PH = ["rel_speed", "induced_vert_break", "horz_break", "spin_rate"]
Pv = al[PH].to_numpy(float)
okp = np.isfinite(Pv).all(1)
mu, sd = Pv[okp].mean(0), Pv[okp].std(0) + 1e-9
Z = (Pv - mu) / sd
km = KMeans(n_clusters=C2, n_init=4, random_state=0)
sub = np.random.default_rng(0).choice(np.flatnonzero(okp), 200_000,
                                      replace=False)
km.fit(Z[sub])
aux2 = np.full(len(raw), -1, np.int64)
aux2[okp] = km.predict(Z[okp])
M.log(f"phys8 라벨 커버 {np.mean(aux2 >= 0):.3f}")

built = FF.build(DATA, VS=2025)
X44 = built["X44"].astype(np.float32)
names = list(built["F44"])
xnum, xnum_names, xcat, xcat_names = A.audit_features(raw, X44, names)
c4 = np.where(old & isf, 0., np.where(old & ~isf, 1.,
              np.where(isf, 2., 3.))).astype(np.float32)[:, None]
Xin = np.concatenate([X44, xnum, c4, xcat], 1)
ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
c4i = X44.shape[1] + xnum.shape[1]
cat_idx = ci + [c4i] + list(range(c4i + 1, Xin.shape[1]))
mask = np.ones(len(y), bool)
Xn, Xc, cards = G.prep(Xin, mask, cat_idx)
G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
idx = np.arange(len(y), dtype=np.int64)
w = np.where(isf & old, 0.1, 1.0).astype(np.float32)
w = (w * 1.5 ** (season.astype(np.float32) - 2019.0)).astype(np.float32)

all_names = names + xnum_names + ["abs_regime"] + xcat_names
ni = [j for j in range(Xin.shape[1]) if j not in set(cat_idx)]
numcols = [all_names[j] for j in ni]
catcols = [all_names[j] for j in cat_idx]
catvals = {c: sorted(set(Xin[mask, j][np.isfinite(Xin[mask, j])].tolist()))
           for c, j in zip(catcols, cat_idx)}
med = np.nanmedian(Xin[:, ni], 0)
med = np.where(np.isfinite(med), med, 0.0)
filled = np.where(np.isnan(Xin[:, ni]), med, Xin[:, ni])
mu2, sd2 = filled.mean(0), filled.std(0) + 1e-6
missing = [numcols[k] for k, j in enumerate(ni) if np.isnan(Xin[:, j]).any()]
prep = {"feature_names": all_names, "cat_cols": catcols, "num_cols": numcols,
        "cat_values": catvals,
        "medians": dict(zip(numcols, med.tolist())),
        "means": dict(zip(numcols, mu2.tolist())),
        "stds": dict(zip(numcols, sd2.tolist())),
        "missing_cols": missing,
        "audit_mode": os.environ["AUDIT_MODE"],
        "extras_num": xnum_names, "extras_cat": xcat_names}


def make_d(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return TabM.make(n_num_features=Xn.shape[1],
                     cat_cardinalities=[int(c) for c in cards], d_out=6 + C2,
                     num_embeddings=LinearReLUEmbeddings(Xn.shape[1],
                                                         d_embedding=16),
                     arch_type="tabm", k=32, n_blocks=3, d_block=512,
                     dropout=0.1).to(M.DEVICE)


def train2(model, ii_np, seed, lr, epochs, tag):
    torch.manual_seed(seed)
    ii = torch.from_numpy(ii_np.astype(np.int64))
    yy = torch.from_numpy(y.astype(np.float32))
    ss = torch.from_numpy(aux6.astype(np.int64))
    s2 = torch.from_numpy(aux2.astype(np.int64))
    ww = torch.from_numpy(w.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=M.AMP)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(ii_np))
        total = 0.0
        for start in range(0, len(ii_np), M.BS):
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
                c2h = out[..., 6:]
                mk2 = s2b >= 0
                if bool(mk2.any()):
                    ce2_each = Fnn.cross_entropy(
                        c2h[mk2].reshape(-1, C2),
                        s2b[mk2, None].expand(-1, c2h.shape[1]).reshape(-1),
                        reduction="none").reshape(-1, c2h.shape[1]).mean(dim=1)
                    ce2 = (ce2_each * wb[mk2]).sum() / wb[mk2].sum()
                else:
                    ce2 = torch.zeros((), device=M.DEVICE)
                loss = ((M.DIRECT_W * bce + M.Brier_W * brier) * wb).sum() \
                    / wb.sum() + M.STATE_W * ce + W2 * ce2
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(b)
        M.log(f"  {tag} ep{ep+1}/{epochs} loss={total/len(ii_np):.6f}")
    return model


s2i = idx[season == 2024]
for seed in M.SEEDS:
    t0 = time.time()
    m = make_d(seed)
    m = train2(m, idx, seed, M.LR, M.EPOCHS, f"PW2-EXPORT s{seed}")
    m = train2(m, s2i, seed + 1, 2e-4, 1, f"PW2-EXPORT s{seed} S2")
    payload = {"config": {"k": M.K, "arch_type": "tabm", "n_blocks": 3,
                          "d_block": M.DBLOCK, "num_embeddings": "linear_relu",
                          "loss": "state6_plus_phys8_aux_w015"},
               "preprocessor": prep,
               "model_state": {k: v.detach().cpu()
                               for k, v in m.state_dict().items()}}
    torch.save(payload, OUT / f"msa_seed{seed}.pt")
    M.log(f"saved msa_seed{seed}.pt ({time.time()-t0:.0f}s)")
    del m
    torch.cuda.empty_cache()
print("PW2 export done")
