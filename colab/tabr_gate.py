# -*- coding: utf-8 -*-
"""TabR 을 직접 구현해, 1057 구성 위에 **얹을 수 있는지**만 본다.

쓰임새를 먼저 고정한다
    TabR 을 단독 주력으로 쓰자는 게 아니다. '비슷한 상황을 찾아 확률을 보완하는'
    부품으로 기존 혼합(= submit_20, LB 1057)에 얹는다. 그래서 판정 기준은
    단독 점수가 아니라 **기존 혼합 대비 증분**이다.

        기준 혼합 = 0.10 CatBoost + 0.10 HistGB + 0.80 TabM   (submit_20 구성)
        후보     = (1-a) x 기준 + a x TabR,  a 를 훑는다

    단독이 낮아도 상관이 낮으면 값어치가 있다. 반대로 단독이 높아도 TabM 을
    따라하면 0 이다 — CatBoost cat_features 에서 정확히 그렇게 됐다
    (단독 -15.9 인데 상관 0.9412 -> 0.9312, 혼합 기여 0).

왜 PLR 이 선택이 아니라 전제인가
    TabM 에서 periodic 은 -43.5 로 졌다. 그 결과는 여기로 전이되지 않는다.
    TabM 에서 임베딩은 입력 표현일 뿐이지만, TabR 에서 임베딩은 **검색 거리의
    기하 그 자체**다. 표준화된 스칼라를 선형+ReLU 로 펴면 L2 거리가 원래 크기
    차이에 지배된다. PLR 은 스칼라를 여러 주파수로 흩어 국소 차이를 거리에
    살린다. 이웃을 잘못 고르면 그 뒤가 전부 무의미하다.

    그래서 같은 후보수에서 plr 과 linear_relu 를 나란히 두고 이 가설을 직접 잰다.

규칙 4 — 이 모델에서 제일 위험한 지점
    각 평가 행은 독립 예측이어야 한다. TabR 은 구조적으로 '다른 행을 참조'하는
    모델이라 여기서 갈린다.
        후보셀 = 학습 행만     -> 안전. 시험 행은 (자기 자신 + 고정된 학습
                                데이터)만 본다. 순서를 섞어도 20%만 넣어도 같다.
        후보셀에 시험 행 포함  -> 즉시 위반. 절대 하지 않는다.
    CAND 는 항상 G.m_tr 에서만 뽑고, 학습 중에는 자기 자신을 가린다
    (자기 라벨을 끌어오면 정답 복사가 된다).

배치 비용 — 생각보다 나쁘지 않다
    시험 253,507행 x 후보 C x d=256 의 거리 계산.
        C=32768   4.3e12 FLOP   numpy BLAS 로 대략 2~5분   거리행렬은 청크로
        C= 8192   1.1e12 FLOP   대략 1분
    submit_25 가 9분 26초였으니 C=8192 는 확실히 들어가고 32768 도 여지가 있다.
    메모리는 청크 4096행씩 끊으면 (4096 x 32768) 537MB 로 잡힌다.

구현 (논문 구조, 효율 변형 하나)
    E  인코더   PLR 또는 linear_relu 수치임베딩 + 범주임베딩 -> Linear -> d
    R  검색     s_i = -||k_x - k_i||^2 / sqrt(d),  상위 m 개 softmax
                v_i = W_Y(y_i) + T(k_x - k_i)
                out = z_x + sum w_i v_i
    P  예측기   LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> 1

    효율 변형: 후보 인코딩을 no_grad 로 두고 REFRESH 스텝마다 갱신한다
    (논문의 context freeze). 후보마다 매 스텝 역전파하면 8배 이상 느려진다.
    탐색 단계의 절충이고, 이기면 정밀 재측정한다.
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
M = 96
REFRESH = 50
BS = 2048
#    (수치임베딩, 후보수, 메모)
ARMS = (("plr", 32768, "본선"),
        ("plr", 8192, "배치 확실"),
        ("linear_relu", 32768, "PLR 가설 대조"))

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
    def __init__(self, kind, n_num, cards, d=D, d_emb=16, dropout=0.1):
        super().__init__()
        if kind == "plr":
            # PeriodicEmbeddings(lite=False) = 주기 -> 선형 -> ReLU (PLR)
            self.num_emb = rne.PeriodicEmbeddings(n_num, d_embedding=d_emb,
                                                  lite=False)
        else:
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

    def retrieve(self, z, ck, cy, m, drop_self=None):
        k = self.WK(z)                                   # (B,d)
        d2 = (k.square().sum(1, keepdim=True)
              - 2.0 * k @ ck.T
              + ck.square().sum(1)[None, :])             # (B,C)
        s = -d2 / (self.d ** 0.5)
        if drop_self is not None:
            s = s.masked_fill(drop_self, float("-inf"))
        sv, si = torch.topk(s, m, dim=1)
        w = torch.softmax(sv, 1)
        kn = ck[si]                                      # (B,m,d)
        v = self.WY(cy[si]) + self.T(k[:, None, :] - kn)
        return z + (w[..., None] * v).sum(1)

    def forward(self, xn, xc, ck, cy, m, drop_self=None):
        z = self.encode(xn, xc)
        return self.P(self.retrieve(z, ck, cy, m, drop_self)).squeeze(-1)


def cand_keys(model, cidx, bs=16384):
    """후보 키를 no_grad 로 계산한다. 학습 행만 들어온다."""
    was = model.training
    model.eval()
    ks = []
    with torch.no_grad():
        for s in range(0, len(cidx), bs):
            b = cidx[s:s + bs]
            ks.append(model.WK(model.encode(G.XN[b].to(G.DEV),
                                            G.XC[b].to(G.DEV))))
    model.train(was)
    return torch.cat(ks)


def run(kind, C, seed, tr_idx, w, m=M, epochs=2, lr=LR1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    # 후보셀: 학습 행에서만. 시험 행은 절대 안 들어간다.
    cnp = np.sort(rng.choice(tr_idx, size=min(C, len(tr_idx)), replace=False))
    cidx = torch.from_numpy(cnp)
    cy = torch.from_numpy(G.y[cnp].astype(np.float32))[:, None].to(G.DEV)
    # 전역 행번호 -> 후보 위치. 배치의 자기 자신을 가릴 때 쓴다.
    lut = np.full(len(G.y), -1, dtype=np.int64)
    lut[cnp] = np.arange(len(cnp))

    model = TabR(kind, G.Xn.shape[1], G.cards).to(G.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ii = torch.from_numpy(tr_idx)
    ww = None if w is None else torch.from_numpy(w.astype(np.float32))
    model.train()
    ck = cand_keys(model, cidx)
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(tr_idx))
        tot = n = 0.0
        for s in range(0, len(tr_idx), BS):
            sel = perm[s:s + BS]
            b = ii[sel]
            if step and step % REFRESH == 0:
                ck = cand_keys(model, cidx)
            xn, xc = G.XN[b].to(G.DEV), G.XC[b].to(G.DEV)
            yb = G.YY[b].to(G.DEV)
            hit = lut[b.numpy()]
            ds = None
            if (hit >= 0).any():
                rr = np.flatnonzero(hit >= 0)
                ds = torch.zeros(len(b), len(cnp), dtype=torch.bool, device=G.DEV)
                ds[torch.from_numpy(rr).to(G.DEV),
                   torch.from_numpy(hit[rr]).to(G.DEV)] = True
            opt.zero_grad(set_to_none=True)
            lg = model(xn, xc, ck, cy, m, ds)
            per = (0.5 * nn.functional.binary_cross_entropy_with_logits(
                        lg, yb, reduction="none")
                   + 0.5 * (lg.sigmoid() - yb).square())
            loss = per.mean() if ww is None else \
                (per * ww[sel].to(G.DEV)).sum() / ww[sel].to(G.DEV).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
            step += 1
        sch.step()
        G.log(f"      TabR {kind} C={C} s{seed} ep{ep+1}/{epochs} "
              f"loss {tot/n:.6f}")

    ck = cand_keys(model, cidx)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(gate), 8192):
            b = torch.from_numpy(gate[s:s + 8192])
            lg = model(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV), ck, cy, m)
            out.append(lg.sigmoid().float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


if __name__ == "__main__":
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
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
    G.log(f"  수치 {Xn.shape[1]}열 범주 {len(cards)}열  학습 {len(tr_idx):,}  "
          f"이웃 m={M}  갱신 {REFRESH}스텝")
    G.log("  후보셀은 학습 행에서만 뽑는다 (규칙 4). 시험 행 0개.")

    # ---- 기준 혼합 = submit_20 구성 (0.10 CB + 0.10 HistGB + 0.80 TabM)
    TMs = []
    for sd in SEEDS:
        pr = os.path.join(OUT, f"emb2_linear_relu_r_s{sd}.npy")
        pf = os.path.join(OUT, f"emb2_linear_relu_f_s{sd}.npy")
        if os.path.exists(pr):
            TMs.append(np.where(is_f, np.load(pf), np.load(pr)))
    if not TMs:
        raise SystemExit("TabM 기준선이 없다. emb_redo.py 산출물이 필요하다")
    tm = np.mean(TMs, 0)
    hgp = os.path.join(OUT, "hg_route_d2.0.npy")
    if os.path.exists(hgp):
        HG = np.load(hgp)
        REF = 0.10 * G.CB + 0.10 * HG + 0.80 * tm
        rname = "0.10 CB + 0.10 HistGB + 0.80 TabM"
    else:
        REF = 0.10 * G.CB + 0.90 * tm
        rname = "0.10 CB + 0.90 TabM  (HistGB 없음)"
    r0 = sc(REF)
    G.log(f"  기준 혼합 [{rname}]  전체 {r0:.1f}  "
          f"1군 {sc(REF, R):.1f}  퓨처스 {sc(REF, is_f):.1f}")
    G.log(f"    (TabM 단독 {sc(tm):.1f})")

    RES = {}
    for kind, C, note in ARMS:
        t0 = time.time()
        try:
            ps = [run(kind, C, sd, tr_idx, w10) for sd in SEEDS]
        except Exception as e:
            G.log(f"  {kind} C={C} 실패: {type(e).__name__} {str(e)[:140]}")
            continue
        RES[(kind, C)] = (np.mean(ps, 0), ps)
        np.save(os.path.join(OUT, f"tabr_{kind}_c{C}.npy"), RES[(kind, C)][0])
        G.log(f"  {kind} C={C} ({note}) 끝 {time.time()-t0:.0f}s")

    AL = (0.05, 0.10, 0.15, 0.20, 0.30)
    G.log("\n" + "=" * 92)
    G.log(f"  {'임베딩':11s} {'후보':>6s} {'단독':>7s} {'기준상관':>8s}"
          f"   기준 혼합 대비 증분  " + "  ".join(f"a={a:.2f}" for a in AL))
    G.log("=" * 92)
    G.log(f"  {'기준 혼합':11s} {'':>6s} {r0:7.1f}")
    for kind, C, note in ARMS:
        if (kind, C) not in RES:
            continue
        p, ps = RES[(kind, C)]
        inc = [sc((1 - a) * REF + a * p) - r0 for a in AL]
        # 시드 부호일치: 각 시드 단독으로 최적 a 부근에서 봤을 때
        ab = AL[int(np.argmax(inc))]
        sg = sum(1 for q in ps if sc((1 - ab) * REF + ab * q) - r0 > 0)
        G.log(f"  {kind:11s} {C:6d} {sc(p):7.1f} "
              f"{np.corrcoef(p, REF)[0,1]:8.4f}   {'':18s}"
              + "  ".join(f"{v:+6.1f}" for v in inc)
              + f"   최적 a={ab:.2f} {max(inc):+.1f}  {sg}/{len(ps)}")

    G.log("\n  읽는 법")
    G.log("    판정은 '기준 혼합 대비 증분' 하나로 한다. 단독 점수는 참고다.")
    G.log("    plr C=32768 과 linear_relu C=32768 의 차이가 PLR 가설의 답이다.")
    G.log("    plr C=8192 가 32768 과 비슷하면 배치가 편해진다.")
    G.log("    증분이 +5 미만이면 부품으로 안 싣는다 — 추론 시간을 2~5분 더")
    G.log("    쓰는 대가가 있고, 혼합 비중을 CatBoost/HistGB 에서 빼와야 한다.")
