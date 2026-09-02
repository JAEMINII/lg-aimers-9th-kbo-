# -*- coding: utf-8 -*-
"""MLP 를 제대로 다시 만든다 — 우리가 안 해본 것들.

현재 우리 MLP 의 문제
    2024 한 시즌만 학습 (24.6만 행), 시드 1개, pytabkit 기본 설정.
    단독 관문 776.4 로 CatBoost(906.7)보다 130 낮은데도 상관이 0.907 로
    가장 낮아서 앙상블에 +6 을 준다. 다양성이 이미 확인된 재료다.

    그런데 학습 구간이 **극단**이다. CatBoost 에서 배운 것이 여기 안 들어갔다.
        CatBoost  "최근 시즌만" -27  ->  "시즌 가중치 2.0" +22   (리더보드 996->1021 로 검증)
        MLP       "2024 한 시즌만"   ->  가중치는 한 번도 안 해봄

왜 지금 이걸 하나
    TabM 에서 관문이 크게 틀렸다(관문 883 -> LB 980, 지인 ep2 는 881 -> 1047).
    원인은 'epoch 을 늘려 학습 구간 마지막 시즌에 과적합' 이었다.
    시즌 가중치는 그것과 다른 축이고 CatBoost 에서 리더보드로 이미 검증됐다.

재는 것
    · 학습 구간: 한 시즌만 vs 전체+시즌가중 vs 전체+Stage2 fine-tune
    · 상관: 낮게 유지되는가 (다양성이 MLP 의 존재 이유다)
    · 앙상블 이득

관문 2019~2023 -> 2024, 1군.  현 제출본(CatBoost+MLP) 912.6 = 리더보드 1021.
"""
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.optimize import minimize_scalar

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "/workspace/aimers/data"
OUT = "/workspace/aimers/out"
PROG = os.path.join(OUT, "mlp_progress.txt")


def log(s):
    print(s, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


import features44 as F                                        # noqa: E402
from rtdl_num_embeddings import PeriodicEmbeddings            # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = F.build(DATA, VS=2024)
X, y = d["X44"], d["y"]
m_tr, m_va, is_f, season = d["m_tr"], d["m_va"], d["is_f"], d["season"]
gate = np.where(m_va)[0]
mm = ~is_f[gate]
yv = y[gate].astype(np.float64)


def prep(X, m_tr, cat_idx):
    n, p = X.shape
    ci = np.asarray(cat_idx)
    ni = np.asarray([j for j in range(p) if j not in set(cat_idx)])
    Xc = np.zeros((n, len(ci)), dtype=np.int64)
    cards = []
    for a, j in enumerate(ci):
        vals = np.unique(X[m_tr, j])
        vals = vals[~np.isnan(vals)]
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
    has_nan = miss[m_tr].any(0)
    if has_nan.any():
        Xn = np.concatenate([Xn, miss[:, has_nan].astype(np.float32)], 1)
    return Xn, Xc, np.asarray(cards)


Xn, Xc, cards = prep(X, m_tr, d["cat_idx"])
XN, XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
YY = torch.from_numpy(y.astype(np.float32))

CB = np.load(os.path.join(SC, "cb_gate.npy"))
MLP0 = np.load(os.path.join(SC, "mlp_gate.npy")).astype(np.float64)
CUR = 0.8 * CB + 0.2 * MLP0
log(f"수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열   관문 1군 {mm.sum():,}")


def bss(p):
    r = yv[mm].mean()
    return 100000 * (1 - ((p[mm] - yv[mm]) ** 2).mean() / (r * (1 - r)))


def best(p):
    def sh(q, c):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))
    r = minimize_scalar(lambda c: -bss(sh(p, c)), bounds=(-0.3, 0.3), method="bounded")
    return -r.fun


BASE = best(CUR)
log(f"기준  CatBoost {best(CB):.1f} / 기존MLP {best(MLP0):.1f} / "
    f"현 제출본 {BASE:.1f} (= LB 1021)\n")


class MLP_PLR(nn.Module):
    """기존 제출본과 같은 계열 (PLR 수치임베딩 + 3층 MLP + 범주임베딩)."""

    def __init__(self, n_num, cards, d_emb=24, d_hidden=(128, 256, 128), d_cat=8):
        super().__init__()
        self.num_emb = PeriodicEmbeddings(n_num, d_embedding=d_emb, n_frequencies=48,
                                          frequency_init_scale=0.01, activation=True,
                                          lite=False)
        self.cat_embs = nn.ModuleList([nn.Embedding(int(c), d_cat) for c in cards])
        d_in = n_num * d_emb + len(cards) * d_cat
        layers, prev = [], d_in
        for h in d_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, xn, xc):
        parts = [self.num_emb(xn).flatten(1)]
        for j, e in enumerate(self.cat_embs):
            parts.append(e(xc[:, j]))
        return self.head(self.body(torch.cat(parts, 1)))

    def last_params(self):
        return list(self.body[-2:].parameters()) + list(self.head.parameters())


def train(model, idx, epochs, lr, params=None, w=None, bs=1024, wd=1e-4, tag=""):
    torch.manual_seed(42)
    ps = params if params is not None else [p for p in model.parameters()
                                            if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    nstep = max(epochs * int(np.ceil(len(idx) / bs)), 1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nstep)
    ii = torch.from_numpy(idx)
    ww = None if w is None else torch.from_numpy(w.astype(np.float32))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        tot = 0.0
        for s in range(0, len(idx), bs):
            sel = perm[s:s + bs]
            b = ii[sel]
            xn, xc = XN[b].to(DEV), XC[b].to(DEV)
            yb = YY[b].to(DEV)
            opt.zero_grad(set_to_none=True)
            lg = model(xn, xc).squeeze(-1)
            # 채점이 Brier 이므로 BCE 와 반반 섞는다 (TabM 에서 쓰는 방식)
            per = (0.5 * Fn.binary_cross_entropy_with_logits(lg, yb, reduction="none")
                   + 0.5 * (torch.sigmoid(lg) - yb) ** 2)
            loss = per.mean() if ww is None else \
                (per * ww[sel].to(DEV)).sum() / ww[sel].to(DEV).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 5.0)
            opt.step()
            sch.step()
            tot += float(loss.detach()) * len(b)
        log(f"      {tag} ep{ep+1}/{epochs}  loss {tot/len(idx):.6f}")
    return model


@torch.no_grad()
def predict(model, idx, bs=8192):
    model.eval()
    out = np.empty(len(idx), np.float64)
    ii = torch.from_numpy(idx)
    for s in range(0, len(idx), bs):
        b = ii[s:min(s + bs, len(idx))]
        lg = model(XN[b].to(DEV), XC[b].to(DEV)).squeeze(-1)
        out[s:s + len(b)] = torch.sigmoid(lg).double().cpu().numpy()
    return out


def run(name, years=None, decay=None, stage2=False, ep1=6, ep2=2,
        lr1=1e-3, lr2=1e-4, seed=42):
    tr_idx = np.where(m_tr)[0]
    if years is not None:
        tr_idx = tr_idx[np.isin(season[tr_idx], years)]
    w = None if decay is None else decay ** (season[tr_idx].astype(np.float64) - 2019)
    t0 = time.time()
    torch.manual_seed(seed)
    m = MLP_PLR(Xn.shape[1], cards).to(DEV)
    train(m, tr_idx, ep1, lr1, w=w, tag=f"{name} S1")
    if stage2:
        s2 = tr_idx[season[tr_idx] == 2023]
        for p in m.parameters():
            p.requires_grad_(False)
        ps = m.last_params()
        for p in ps:
            p.requires_grad_(True)
        train(m, s2, ep2, lr2, params=ps, tag=f"{name} S2")
    p = predict(m, gate)
    np.save(os.path.join(OUT, f"mlpnew_{name}.npy"), p)
    del m
    gc.collect()
    torch.cuda.empty_cache()

    solo = best(p)
    cc = np.corrcoef(p[mm], CB[mm])[0, 1]
    co = np.corrcoef(p[mm], MLP0[mm])[0, 1]
    # 기존 MLP 를 이걸로 교체했을 때
    bw, bs_ = 0.0, best(CB)
    for wv in np.arange(0.05, 0.60, 0.05):
        s = best((1 - wv) * CB + wv * p)
        if s > bs_:
            bs_, bw = s, wv
    log(f"\n  [{name}]  {time.time()-t0:.0f}s   단독 {solo:.1f}  "
        f"CB상관 {cc:.4f}  기존MLP상관 {co:.4f}")
    log(f"    CatBoost 와 둘이서: 비중 {bw:.2f} -> {bs_:.1f}   "
        f"(기존 MLP 로는 {BASE:.1f})   {bs_-BASE:+.1f}\n")
    return dict(name=name, solo=solo, cb=cc, old=co, w=bw, sc=bs_)


CASES = {
    # flat 주변 탐색. 학습이 26초라 거의 공짜다.
    # epoch 을 늘리면 TabM 처럼 최근 시즌 과적합이 올 수 있어 양쪽으로 본다.
    "flat3":      dict(ep1=3),
    "flat10":     dict(ep1=10),
    "flat16":     dict(ep1=16),
    "flat_s1":    dict(ep1=6, seed=1),
    "flat_s777":  dict(ep1=6, seed=777),

    # 현재 제출본 재현 (한 시즌만) — 기준점
    "one_season": dict(years=[2023], ep1=6),
    # CatBoost 에서 +22 였던 시즌가중을 MLP 에 적용
    "sw2":        dict(decay=2.0, ep1=6),
    "sw3":        dict(decay=3.0, ep1=6),
    "sw4":        dict(decay=4.5, ep1=6),
    # 균등 전체 (기존 실측 455 를 재현하는지 확인)
    "flat":       dict(ep1=6),
    # TabM 이 쓰는 구조: 전체 학습 후 최근 시즌 fine-tune
    "stage2":     dict(ep1=6, stage2=True),
    "sw2_stage2": dict(decay=2.0, ep1=6, stage2=True),
}

if __name__ == "__main__":
    want = sys.argv[1:] or ["one_season", "sw2", "flat"]
    res = []
    for nm in want:
        res.append(run(nm, **CASES[nm]))
    log("=" * 70)
    log(f"  {'구성':<12s}{'단독':>9s}{'CB상관':>9s}{'구MLP상관':>10s}"
        f"{'비중':>7s}{'CB와둘이':>9s}{'현제출대비':>10s}")
    for r in res:
        log(f"  {r['name']:<12s}{r['solo']:9.1f}{r['cb']:9.4f}{r['old']:10.4f}"
            f"{r['w']:7.2f}{r['sc']:9.1f}{r['sc']-BASE:+10.1f}")
