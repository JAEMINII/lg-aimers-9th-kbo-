# -*- coding: utf-8 -*-
"""TabR 을 제대로 구현해 다시 잰다. 내가 부당하게 기각했는지 확인한다.

1차에서 내가 한 절충 두 가지
    ① 후보 인코딩을 no_grad 로 동결하고 50스텝마다 갱신했다.
       비용을 아끼려는 절충이었는데, 검색으로 사는 모델에서 **거리 척도가
       제대로 학습되지 않는다**는 뜻이다. 치명적일 수 있다.
    ② 거리를 ||a||^2 - 2ab + ||b||^2 로 전개했다. float32 파국적 상쇄로
       C=32768 팔이 -1028 로 무너졌고, 살아남은 팔도 거리가 오염됐을 수 있다.

    둘 다 모델 성질이 아니라 내 구현 선택이다. 그 위에서 나온 '단독 754,
    증분 전 비중 음수' 를 기각 근거로 적었으니 다시 봐야 한다.

이번에 고치는 것
    ① 후보를 매 스텝 **기울기까지 태워** 인코딩한다. 후보 풀을 4096 으로
       줄여 감당한다.
    ② torch.cdist 를 쓴다.

고치지 않는 것 — 이웃 수
    TabR 은 이웃마다 4층 MLP 를 통과시킨다 (v_i = W_Y(y_i) + T(k_x - k_i)).
    비용이 O(B x m x d) 라 m 을 키우면 메모리가 터진다. m=8192, B=2048 이면
    4.3e9 원소가 MLP 를 지나야 한다. 논문이 m=96 을 기본으로 쓰는 이유다.

    MNCA 는 p = sum w_i y_i 로 MLP 가 없어 O(B x C) 다. 그래서 C=65536 이 된다.
    **이웃 수 우위는 MNCA 의 구조적 장점이지 내 구현 탓이 아니다.**

    그래서 여기서는 TabR 이 감당하는 범위(m 96~512)만 훑는다. 답하는 질문은
    "TabR 이 MNCA 를 이길 수 있나" 가 아니라 "내가 부당하게 기각했나" 다.

메모리를 맞추려고 배치를 1024 로 낮춘다
    B x m 이 메모리를 지배한다. m=512, B=1024 면 (1024, 512, 256) = 1.3e8 원소
    = 537MB/텐서다. 세 팔이 같은 B 를 써야 비교가 깨끗하므로 전부 1024 로 둔다.

판정
    TabR 1차: 단독 754.7 / 753.7, 기준상관 0.9138 / 0.9294, 증분 전 비중 음수.
    MNCA:     단독 897 (C=8192), 최적 증분 +5.6 ~ +9.0.
    고친 TabR 이 MNCA 근처면 기각이 부당했던 것이고, 그래도 배치는 MNCA 다
    (이웃을 크게 쓸 수 있고 추론도 싸다). 크게 넘으면 갈아탄다.
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
OUT = os.environ.get("AIMERS_OUT", "/workspace/aimers/out")
DATA = os.environ.get("AIMERS_DATA", "/workspace/aimers/data")
SEEDS = (42, 1, 777)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
D = 256
BS = 1024
C_POOL = 4096
MS = (96, 256, 512)

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


class TabR(nn.Module):
    def __init__(self, n_num, cards, d=D, d_emb=16, dropout=0.1):
        super().__init__()
        self.num_emb = rne.LinearReLUEmbeddings(n_num, d_embedding=d_emb)
        self.cat_emb = nn.ModuleList([nn.Embedding(int(c), d_emb) for c in cards])
        d_in = n_num * d_emb + len(cards) * d_emb
        self.E = nn.Linear(d_in, d)
        self.WK = nn.Linear(d, d)
        self.WY = nn.Linear(1, d)
        self.T = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(d, d))
        self.P = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(d, 1))
        self.d = d

    def encode(self, xn, xc):
        e = [self.num_emb(xn).flatten(1)]
        for j, emb in enumerate(self.cat_emb):
            e.append(emb(xc[:, j]))
        return self.E(torch.cat(e, 1))

    def head(self, z, ck, cy, m, drop_self=None):
        k = self.WK(z)
        dist = torch.cdist(k, ck)                       # 수치적으로 안전
        s = -dist / (self.d ** 0.5)
        if drop_self is not None:
            s = s.masked_fill(drop_self, float("-inf"))
        sv, si = torch.topk(s, min(m, ck.shape[0]), dim=1)
        w = torch.softmax(sv, 1)
        kn = ck[si]
        # cy 는 (C,) 1차원이다. WY 가 Linear(1, d) 라 (B, m, 1) 로 만들어 넣는다.
        v = self.WY(cy[si].unsqueeze(-1)) + self.T(k[:, None, :] - kn)
        return self.P(z + (w[..., None] * v).sum(1)).squeeze(-1)


def run(m, seed, tr_idx, w):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = TabR(G.Xn.shape[1], G.cards).to(G.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR1, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    ii = torch.from_numpy(tr_idx)
    ww = torch.from_numpy(w.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    model.train()
    for ep in range(2):
        perm = torch.randperm(len(tr_idx), generator=g)
        tot = n = 0.0
        for s in range(0, len(tr_idx), BS):
            sel = perm[s:s + BS]
            b = ii[sel]
            cb = ii[torch.randint(0, len(tr_idx), (C_POOL,), generator=g)]
            xn, xc = G.XN[b].to(G.DEV), G.XC[b].to(G.DEV)
            cn, cc = G.XN[cb].to(G.DEV), G.XC[cb].to(G.DEV)
            yb, cy = G.YY[b].to(G.DEV), G.YY[cb].to(G.DEV)
            ds = (b[:, None] == cb[None, :]).to(G.DEV)
            ds = ds if ds.any() else None
            opt.zero_grad(set_to_none=True)
            # 후보도 같은 순전파에 태운다 — 기울기가 흐른다
            allz = model.encode(torch.cat([xn, cn]), torch.cat([xc, cc]))
            z, cz = allz[:len(b)], allz[len(b):]
            lg = model.head(z, model.WK(cz), cy, m, ds)
            per = (0.5 * nn.functional.binary_cross_entropy_with_logits(
                        lg, yb, reduction="none")
                   + 0.5 * (lg.sigmoid() - yb).square())
            wb = ww[sel].to(G.DEV)
            loss = (per * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        sch.step()
        G.log(f"      TabR m={m} s{seed} ep{ep+1}/2  loss {tot/n:.6f}")

    model.eval()
    r = np.random.default_rng(seed)
    cnp = np.sort(r.choice(tr_idx, size=C_POOL, replace=False))
    cb = torch.from_numpy(cnp)
    with torch.no_grad():
        cz = model.encode(G.XN[cb].to(G.DEV), G.XC[cb].to(G.DEV))
        ck = model.WK(cz)
        cy = G.YY[cb].to(G.DEV)
        out = []
        for s in range(0, len(gate), 2048):
            b = torch.from_numpy(gate[s:s + 2048])
            z = model.encode(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV))
            out.append(model.head(z, ck, cy, m).sigmoid().float().cpu().numpy())
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

    # 예전 실행 환경의 emb2_*/hg_route 파일이 있으면 그 기준을 재현하고,
    # 현재 인스턴스처럼 파일이 없으면 tabm_gate_gpu가 준비한 동일 관문 기준선
    # (VS=2024: 0.8 CB + 0.2 TabM, 그 외: CB)을 사용한다.
    tm_files = [(os.path.join(OUT, f"emb2_linear_relu_f_s{s}.npy"),
                 os.path.join(OUT, f"emb2_linear_relu_r_s{s}.npy"))
                for s in SEEDS]
    hg_file = os.path.join(OUT, "hg_route_d2.0.npy")
    if all(os.path.exists(a) and os.path.exists(b) for a, b in tm_files) \
            and os.path.exists(hg_file):
        TM = np.mean([np.where(is_f, np.load(a), np.load(b))
                      for a, b in tm_files], 0)
        HG = np.load(hg_file)
        REF = 0.10 * G.CB + 0.10 * HG + 0.80 * TM
        ref_name = "legacy CB/HG/TabM"
    else:
        REF = np.asarray(G.CUR, dtype=np.float64)
        ref_name = "current gate baseline"
    r0 = sc(REF)
    G.log(f"  기준 혼합({ref_name}) {r0:.1f}   후보풀 {C_POOL}  배치 {BS}  "
          f"(1차: 단독 754.7, 증분 전 비중 음수)\n")

    AL = (0.10, 0.20, 0.30, 0.40)
    for m in MS:
        t0 = time.time()
        try:
            ps = [run(m, sd, tr_idx, w10) for sd in SEEDS]
        except Exception as e:
            G.log(f"  m={m} 실패: {type(e).__name__} {str(e)[:120]}")
            continue
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"tabrf_m{m}_s{sd}.npy"), ps[i])
        p = np.mean(ps, 0)
        cells = []
        for a in AL:
            d = [sc((1 - a) * REF + a * q) - r0 for q in ps]
            mu = float(np.mean(d)); se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            cells.append((a, mu, se, sum(1 for x in d if x > 0)))
        best = max(cells, key=lambda c: c[1])
        G.log(f"  m={m:4d}  단독 {sc(p):7.1f}  상관 {np.corrcoef(p, REF)[0,1]:.4f}"
              f"  {time.time()-t0:5.0f}s   "
              + "  ".join(f"a{a:.1f} {mu:+5.1f}+-{se:4.1f}"
                          for a, mu, se, _ in cells))
        G.log(f"  {'':10s} 최적 a={best[0]:.2f}  {best[1]:+.1f}+-{best[2]:.1f}"
              f"  t={best[1]/max(best[2],1e-9):.2f}  {best[3]}/{len(ps)}")

    G.log("\n  1차 TabR: 단독 754.7 / 753.7, 증분 전 비중 음수")
    G.log("  MNCA:     단독 897 (C=8192), 최적 증분 +5.6 ~ +9.0")
    G.log("  고친 TabR 이 MNCA 근처면 내 기각이 부당했던 것이다.")
    G.log("  그래도 배치는 MNCA 다 — 이웃을 크게 쓸 수 있고 추론이 싸다.")
