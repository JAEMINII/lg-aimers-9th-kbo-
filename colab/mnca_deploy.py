# -*- coding: utf-8 -*-
"""MNCA 를 배치용(VS=2025)으로 학습하고 추론에 필요한 것만 내보낸다.

왜 MNCA 인가
    후보를 열 몇 개 재서 살아남은 유일한 계열이다. 필요조건이 둘인데
    동시에 만족한 게 이것뿐이었다.
        상관 < 0.97   (0.98 위는 뭘 넣어도 0 이거나 음수)
        단독 > 880    (상관이 낮아도 품질이 낮으면 안 됨)

        TabPFN  상관 0.689  단독 342   전 비중 음수
        XGBoost 상관 0.892  단독 680   전 비중 음수
        FT-T    상관 0.941  단독 770   전 비중 음수
        TabR    상관 0.951  단독 814   전 비중 음수
        ResNet  상관 0.935  단독 813   전 비중 음수
        MNCA    상관 0.954  단독 901   +6.6 ± 2.7   <- 유일

후보 수를 16384 로 잡은 이유
    관문        C=16384  +4.3 ± 1.4  t=2.98
                C=32768  +4.0 ± 4.0  t=1.02
                C=65536  +6.6 ± 2.7  t=2.48
    증분은 65536 이 크지만 t 는 16384 가 높다. 그리고 추론 비용이 후보 수에
    비례한다 — 245,789 x C x 256 거리 계산이다.
        C=16384  약 1~2분,   C=65536  약 4~8분  (L4 추정)
    지금 예산 10분 중 1.5분을 쓰고 있어 65536 은 총 6~10분으로 위험하다.
    증분 2.3 을 포기하고 안전을 산다.

내보내는 것 — 후보를 **미리 인코딩해서** 저장한다
    추론에서 후보를 다시 인코딩할 필요가 없어진다.
        cand_z  (C, 256) float32   후보 키 벡터
        cand_y  (C,)     float32   후보 라벨
        인코더 가중치               시험 행을 인코딩할 때만 쓴다
        전처리 통계                 script.py 의 _fm_prep 형식 그대로
    추론은 z = encode(test) -> cdist(z, cand_z) -> softmax(-d/tau) -> @ cand_y.

규칙 4
    후보는 **학습 행에서만** 뽑는다. 시험 행끼리 참조하면 즉시 위반이다.
    후보가 고정 배열로 박혀 있으므로 시험 행 순서나 개수와 무관하다 —
    구조적으로 안전하고, 감사에서 확인한다.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
DATA = "/workspace/aimers/data"
OUT = "/workspace/aimers/mnca_npz"
SEEDS = (42, 1, 777)
VS = 2025                 # 배치: 2019~2024 전부 학습
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
D = 256
C_CAND = 16384
BS = 2048

import rtdl_num_embeddings as rne                               # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


class MNCA(nn.Module):
    def __init__(self, n_num, cards, d=D, d_emb=16, dropout=0.1):
        super().__init__()
        self.num_emb = rne.LinearReLUEmbeddings(n_num, d_embedding=d_emb)
        self.cat_emb = nn.ModuleList([nn.Embedding(int(c), d_emb) for c in cards])
        d_in = n_num * d_emb + len(cards) * d_emb
        self.E = nn.Sequential(nn.Linear(d_in, d), nn.BatchNorm1d(d), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(d, d))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def encode(self, xn, xc):
        e = [self.num_emb(xn).flatten(1)]
        for j, emb in enumerate(self.cat_emb):
            e.append(emb(xc[:, j]))
        return self.E(torch.cat(e, 1))

    def probs(self, z, cz, cy, drop_self=None):
        dist = torch.cdist(z, cz)
        s = -dist / self.log_tau.exp().clamp(1e-3, 1e3)
        if drop_self is not None:
            s = s.masked_fill(drop_self, float("-inf"))
        w = torch.softmax(s, 1)
        return (w * cy[None, :]).sum(1).clamp(1e-6, 1 - 1e-6)


def prep_stats(X, m_tr, cat_idx):
    """export_ours.prep_stats 와 같다. script.py 의 _fm_prep 이 읽는 형식."""
    n, p = X.shape
    ci = np.asarray(cat_idx)
    ni = np.asarray([j for j in range(p) if j not in set(cat_idx)])
    st = {"cat_idx": ci, "num_idx": ni}
    Xc = np.zeros((n, len(ci)), dtype=np.int64)
    cards = []
    for a, j in enumerate(ci):
        vals = np.unique(X[m_tr, j])
        vals = vals[~np.isnan(vals)]
        st[f"catkey_{a}"] = vals.astype(np.float64)
        st[f"catval_{a}"] = np.arange(1, len(vals) + 1, dtype=np.int64)
        pos = np.clip(np.searchsorted(vals, X[:, j]), 0, max(len(vals) - 1, 0))
        hit = (vals[pos] == X[:, j]) if len(vals) else np.zeros(n, bool)
        Xc[:, a] = np.where(hit, pos + 1, 0)
        cards.append(len(vals) + 1)
    Xn = X[:, ni].astype(np.float64)
    miss = np.isnan(Xn)
    med = np.nanmedian(Xn[m_tr], 0)
    Xn = np.where(miss, med, Xn)
    mu, sd = Xn[m_tr].mean(0), Xn[m_tr].std(0) + 1e-6
    Xn = ((Xn - mu) / sd).astype(np.float32)
    hn = miss[m_tr].any(0)
    st.update(med=med, mu=mu, sd=sd, has_nan=hn)
    if hn.any():
        Xn = np.concatenate([Xn, miss[:, hn].astype(np.float32)], 1)
    return Xn, Xc, np.asarray(cards), st


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")

    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr)          # VS=2025 이므로 전부가 학습 구간
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    y = tr[PP.TARGET].to_numpy(np.float32)
    season = tr["season"].to_numpy(np.float64)
    isf = tr["game_type"].astype(str).to_numpy() == "F"
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    Xf = Xs.to_numpy(dtype=np.float32)

    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xall = np.concatenate([Xf, c4], 1)
    m_tr = np.ones(len(Xall), bool)
    Xn, Xc, cards, st = prep_stats(Xall.astype(np.float64), m_tr,
                                   ci + [Xf.shape[1]])
    XN = torch.from_numpy(Xn)
    XC = torch.from_numpy(Xc)
    YY = torch.from_numpy(y)
    w10 = np.where(isf & old, OLD_W, 1.0).astype(np.float32)
    WW = torch.from_numpy(w10)
    print(f"  학습 {len(Xn):,}행  수치 {Xn.shape[1]}열  범주 {len(cards)}열  "
          f"후보 {C_CAND:,}\n")

    feats = list(Xs.columns) + ["abs_regime"]
    for sd in SEEDS:
        t0 = time.time()
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        g = torch.Generator().manual_seed(sd)
        model = MNCA(Xn.shape[1], cards).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR1, weight_decay=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
        idx = torch.arange(len(Xn))
        model.train()
        for ep in range(2):
            perm = torch.randperm(len(Xn), generator=g)
            tot = n = 0.0
            for s in range(0, len(Xn), BS):
                sel = perm[s:s + BS]
                cpick = torch.randint(0, len(Xn), (C_CAND,), generator=g)
                xn, xc = XN[sel].to(dev), XC[sel].to(dev)
                cn, cc = XN[cpick].to(dev), XC[cpick].to(dev)
                yb, cy = YY[sel].to(dev), YY[cpick].to(dev)
                ds = (sel[:, None] == cpick[None, :]).to(dev)
                ds = ds if ds.any() else None
                opt.zero_grad(set_to_none=True)
                allz = model.encode(torch.cat([xn, cn]), torch.cat([xc, cc]))
                z, cz = allz[:len(sel)], allz[len(sel):]
                p = model.probs(z, cz, cy, ds)
                per = (0.5 * nn.functional.binary_cross_entropy(
                            p, yb, reduction="none")
                       + 0.5 * (p - yb).square())
                wb = WW[sel].to(dev)
                loss = (per * wb).sum() / wb.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach()) * len(sel)
                n += len(sel)
            sch.step()
            print(f"    seed {sd} ep{ep+1}/2  loss {tot/n:.6f}  "
                  f"tau {float(model.log_tau.exp()):.3f}")

        # ---- 후보를 뽑아 미리 인코딩한다 (학습 행에서만)
        model.eval()
        rng = np.random.default_rng(sd)
        cnp = np.sort(rng.choice(len(Xn), size=C_CAND, replace=False))
        cb = torch.from_numpy(cnp)
        with torch.no_grad():
            cz = torch.cat([model.encode(XN[cb[i:i + 8192]].to(dev),
                                         XC[cb[i:i + 8192]].to(dev))
                            for i in range(0, len(cb), 8192)])
        s = model.state_dict()
        a = {"num_embedding_weight": s["num_emb.linear.weight"].cpu().numpy(),
             "num_embedding_bias": s["num_emb.linear.bias"].cpu().numpy(),
             "E0_weight": s["E.0.weight"].cpu().numpy(),
             "E0_bias": s["E.0.bias"].cpu().numpy(),
             "bn_weight": s["E.1.weight"].cpu().numpy(),
             "bn_bias": s["E.1.bias"].cpu().numpy(),
             "bn_mean": s["E.1.running_mean"].cpu().numpy(),
             "bn_var": s["E.1.running_var"].cpu().numpy(),
             "E4_weight": s["E.4.weight"].cpu().numpy(),
             "E4_bias": s["E.4.bias"].cpu().numpy(),
             "tau": np.float32(float(model.log_tau.exp())),
             "cand_z": cz.cpu().numpy().astype(np.float32),
             "cand_y": y[cnp].astype(np.float32)}
        for j in range(len(cards)):
            a[f"cat_emb_{j}"] = s[f"cat_emb.{j}.weight"].cpu().numpy()
        a.update({k: np.asarray(v) for k, v in st.items()})
        a["meta"] = np.asarray(json.dumps(
            {"features": feats, "cards": [int(c) for c in cards],
             "n_num": int(Xn.shape[1]), "d": D, "d_emb": 16,
             "C": int(C_CAND), "vs": VS, "seed": int(sd)}, ensure_ascii=False))
        path = os.path.join(OUT, f"mnca_s{sd}.npz")
        np.savez_compressed(path, **a)
        print(f"    seed {sd} 저장 {os.path.getsize(path)/1e6:.1f}MB  "
              f"{time.time()-t0:.0f}s\n")
    print(f"  완료. {OUT}")
