# -*- coding: utf-8 -*-
"""ModernNCA 를 직접 구현해 잰다. 이건 내 기각 논거를 반박할 수 있는 후보다.

왜 TabR 기각이 이걸 안 덮나
    TabR 을 기각한 근거는 이웃 수였다.
        이웃 96개의 라벨 평균 잡음  sqrt(0.25/96) = 0.0510
        우리 신호 폭                             0.0502     1.02배 -> 순증 0
        잡음을 신호의 1/3 로 줄이려면 m ~ 900,  1/5 이면 m ~ 2,500

    MNCA 는 상위 m 개를 고르지 않고 **후보 전체에 소프트맥스 가중**을 준다.
    즉 내 계산이 '필요하다' 고 한 쪽에 서 있다. 내 논거를 내 손으로 반박할 수
    있는 유일한 후보라서 이웃 수를 직접 훑는다.

        C 를 키울수록 좋아지다 꺾이면  -> 논거가 맞다 (국소성과 잡음의 맞교환)
        끝까지 안 오르면              -> 내가 틀렸고 다른 이유다

구조 (Ye et al., ICLR 2025 의 ModernNCA)
    인코더 E 하나뿐이다. TabR 의 value 모듈 같은 게 없다.
        z   = E(x)
        d_i = ||z - E(c_i)||          <- 제곱이 아니라 거리
        w   = softmax(-d / tau)
        p   = sum_i w_i * y_i         <- 이웃 라벨의 가중평균, 그게 곧 예측

    예측기 머리가 없으므로 출력이 이웃 라벨에 갇힌다. 내 SNR 분석이 겨냥한
    바로 그 양이다. 이 모델은 그 분석의 직접 시험대다.

TabR 때 두 가지를 고친다
    ① 후보를 기울기까지 태워 인코딩한다. TabR 에서는 비용 때문에 문맥을
       동결(50스텝마다 갱신)했는데, MNCA 는 배치+후보만 통과시키면 되므로
       2배 비용이면 된다. 거리 척도가 제대로 학습된다.
    ② 거리를 torch.cdist 로 잰다. TabR 에서 ||a||^2 - 2ab + ||b||^2 로 전개했다가
       float32 파국적 상쇄로 C=32768 팔이 -1028 로 무너졌다.

규칙 4
    후보셀은 **학습 행에서만** 뽑는다. 시험 행끼리 참조하면 즉시 위반이다.
    학습 중에는 배치의 자기 자신을 후보에서 뺀다 (라벨 복사 방지).

판정
    TabR / TabPFN 과 동일하게 1057 구성 위 증분으로 본다. +5 미만이면 안 싣는다.
    단독 점수가 낮아도 상관이 낮으면 값어치가 있을 수 있으나, TabPFN 이
    상관 0.689 로 충분히 직교했는데도 품질 격차(342 vs 917) 때문에 전 비중에서
    음수였다. 직교성만으로는 구제되지 않는다.
"""
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
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
D = 256
BS = 2048
CANDS = (16384, 32768, 65536)      # 8192 에서 아직 오르는 중이라 더 민다
INF_CHUNK = 4096

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
import rtdl_num_embeddings as rne                               # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


class MNCA(nn.Module):
    def __init__(self, n_num, cards, d=D, d_emb=16, dropout=0.1):
        super().__init__()
        self.num_emb = rne.LinearReLUEmbeddings(n_num, d_embedding=d_emb)
        self.cat_emb = nn.ModuleList([nn.Embedding(int(c), d_emb) for c in cards])
        d_in = n_num * d_emb + len(cards) * d_emb
        # 인코더 하나뿐. 예측기 머리가 없다 — 그게 NCA 다.
        self.E = nn.Sequential(nn.Linear(d_in, d), nn.BatchNorm1d(d), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(d, d))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def encode(self, xn, xc):
        e = [self.num_emb(xn).flatten(1)]
        for j, emb in enumerate(self.cat_emb):
            e.append(emb(xc[:, j]))
        return self.E(torch.cat(e, 1))

    def probs(self, z, cz, cy, drop_self=None):
        dist = torch.cdist(z, cz)                       # (B,C) 수치적으로 안전
        s = -dist / self.log_tau.exp().clamp(1e-3, 1e3)
        if drop_self is not None:
            s = s.masked_fill(drop_self, float("-inf"))
        w = torch.softmax(s, 1)
        return (w * cy[None, :]).sum(1).clamp(1e-6, 1 - 1e-6)


def run(C, seed, tr_idx, w, season):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MNCA(G.Xn.shape[1], G.cards).to(G.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR1, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    ii = torch.from_numpy(tr_idx)
    ww = torch.from_numpy(w.astype(np.float32))
    yt = G.YY
    g = torch.Generator().manual_seed(seed)

    model.train()
    for ep in range(2):
        perm = torch.randperm(len(tr_idx), generator=g)
        tot = n = 0.0
        for s in range(0, len(tr_idx), BS):
            sel = perm[s:s + BS]
            b = ii[sel]
            # 후보는 학습 행에서만 뽑는다 (규칙 4). 매 스텝 새로 뽑는다.
            cpick = torch.randint(0, len(tr_idx), (C,), generator=g)
            cb = ii[cpick]
            xn = G.XN[b].to(G.DEV)
            xc = G.XC[b].to(G.DEV)
            cn = G.XN[cb].to(G.DEV)
            cc = G.XC[cb].to(G.DEV)
            yb = yt[b].to(G.DEV)
            cy = yt[cb].to(G.DEV)
            # 자기 자신이 후보에 섞이면 라벨 복사가 된다
            ds = (b[:, None] == cb[None, :]).to(G.DEV)
            ds = ds if ds.any() else None

            opt.zero_grad(set_to_none=True)
            allz = model.encode(torch.cat([xn, cn]), torch.cat([xc, cc]))
            z, cz = allz[:len(b)], allz[len(b):]
            p = model.probs(z, cz, cy, ds)
            per = (0.5 * nn.functional.binary_cross_entropy(p, yb,
                                                            reduction="none")
                   + 0.5 * (p - yb).square())
            wb = ww[sel].to(G.DEV)
            loss = (per * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        sch.step()
        G.log(f"      MNCA C={C} s{seed} ep{ep+1}/2  loss {tot/n:.6f}  "
              f"tau {float(model.log_tau.exp()):.3f}")

    # 추론 — 후보는 학습 행에서 한 번 뽑아 고정한다
    model.eval()
    r = np.random.default_rng(seed)
    cnp = np.sort(r.choice(tr_idx, size=min(C, len(tr_idx)), replace=False))
    cb = torch.from_numpy(cnp)
    with torch.no_grad():
        cz = torch.cat([model.encode(G.XN[cb[i:i + 8192]].to(G.DEV),
                                     G.XC[cb[i:i + 8192]].to(G.DEV))
                        for i in range(0, len(cb), 8192)])
        cy = yt[cb].to(G.DEV)
        out = []
        for s in range(0, len(gate), INF_CHUNK):
            b = torch.from_numpy(gate[s:s + INF_CHUNK])
            z = model.encode(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV))
            out.append(model.probs(z, cz, cy).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


if __name__ == "__main__":
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xn, Xc, cards = G.prep(np.concatenate([Xf, c4], 1), G.m_tr,
                           ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
    G.log(f"  학습 {len(tr_idx):,}   후보는 학습 행에서만 (규칙 4)")

    TM = np.mean([np.where(is_f,
                           np.load(f"{OUT}/emb2_linear_relu_f_s{s}.npy"),
                           np.load(f"{OUT}/emb2_linear_relu_r_s{s}.npy"))
                  for s in SEEDS], 0)
    HG = np.load(f"{OUT}/hg_route_d2.0.npy")
    REF = 0.10 * G.CB + 0.10 * HG + 0.80 * TM
    r0 = sc(REF)
    G.log(f"  기준 혼합 {r0:.1f}   TabM 단독 {sc(TM):.1f}\n")

    AL = (0.10, 0.20, 0.30, 0.40, 0.50)   # 0.30 에서 아직 올라 확장
    RES = {}
    for C in CANDS:
        t0 = time.time()
        try:
            ps = [run(C, sd, tr_idx, w10, season) for sd in SEEDS]
        except Exception as e:
            G.log(f"  C={C} 실패: {type(e).__name__} {str(e)[:120]}")
            continue
        RES[C] = ps                       # 시드별로 보관해야 오차막대가 나온다
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"mnca2_c{C}_s{sd}.npy"), ps[i])
        G.log(f"  C={C} 끝 {time.time()-t0:.0f}s")

    G.log("\n" + "=" * 92)
    G.log(f"  {'후보수':>7s} {'단독':>8s} {'1군':>8s} {'퓨처스':>8s} "
          f"{'기준상관':>8s}   증분  " + "  ".join(f"a={a:.2f}" for a in AL))
    G.log("=" * 92)
    G.log(f"  {'기준':>7s} {r0:8.1f}")
    for C in CANDS:
        if C not in RES:
            continue
        ps = RES[C]
        p = np.mean(ps, 0)
        # 시드별 짝차이 -> 평균, 표준오차, 부호일치. 오차막대 없는 숫자는 못 쓴다.
        cells = []
        for a in AL:
            d = [sc((1 - a) * REF + a * q) - r0 for q in ps]
            mu = float(np.mean(d)); se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            cells.append((a, mu, se, sum(1 for x in d if x > 0)))
        best = max(cells, key=lambda c: c[1])
        G.log(f"  {C:7d} {sc(p):8.1f} {sc(p, R):8.1f} {sc(p, is_f):8.1f} "
              f"{np.corrcoef(p, REF)[0,1]:8.4f}   "
              + "  ".join(f"{mu:+5.1f}+-{se:4.1f}" for _, mu, se, _ in cells))
        G.log(f"  {'':7s} {'':8s} 최적 a={best[0]:.2f}  {best[1]:+.1f}+-{best[2]:.1f}"
              f"  t={best[1]/max(best[2],1e-9):.2f}  {best[3]}/{len(ps)}"
              f"   (3시드평균 예측으로 재면 {max(sc((1-a)*REF + a*p) - r0 for a in AL):+.1f})")

    G.log("\n  읽는 법")
    G.log("    C 를 키울수록 단독이 오르면 이웃 잡음 논거가 맞다.")
    G.log("    끝까지 평평하면 내 SNR 논거가 원인이 아니었다는 뜻이다.")
    G.log("    판정은 증분 하나로 한다 — +5 미만이면 안 싣는다.")
