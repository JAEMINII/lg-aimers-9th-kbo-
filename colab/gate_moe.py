# -*- coding: utf-8 -*-
"""행별 게이팅 g(x) 의 결정 실험 — MoE 제안의 알맹이를 최상급 조건에서 가른다.

제안: 공유몸통+어댑터들 위에 행별 소프트 게이트. 어댑터 가지는 상관 0.99+ 라
     다양성이 없으므로, 게이트의 값어치는 **진짜 다양한 전문가 위에서** 상한이다.
     그래서 기존 4원 구성원(상관 0.93~0.96, cb/tab/din/ms) 위에 g(x) 를 학습한다.
     여기서 못 이기면 어댑터판 MoE 는 지을 이유가 없다.

설계
    g: 표준화한 44열 -> MLP(32) -> softmax 4 가중 (행 내부 함수, 규칙 4 적합)
    보수형: softmax(logit + log(고정가중)) — 고정 가중 주변의 편차만 배운다
    p = sum_i g_i(x) p_i,  Brier 직접 최소화
    폴드 교차: 2022에서 g 학습 -> 2024 채점, 반대도. 기준 = 고정 (0.31/0.13/0.31/0.25)
"""
import os
import sys

import numpy as np
import torch
from torch import nn

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, SC)
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(SC, "_dl")
FIX = np.array([0.31, 0.13, 0.31, 0.25])

import features44 as F                                          # noqa: E402


def fold(vs):
    d = F.build(DATA, VS=vs)
    g = np.where(d["season"] == vs)[0]
    y = d["y"].astype(np.float64)[g]
    X = d["X44"].astype(np.float64)[g]
    med = np.nanmedian(X, 0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = ((X - mu) / sd).astype(np.float32)
    if vs == 2024:
        cb = np.load(f"{DL}/ta2024_base.npy").mean(0)
        tab = 0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy") + 0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")
    else:
        cb = np.load(f"{DL}/cb50fixed_2022.npy").astype(np.float64)
        tab = np.load(f"{DL}/h2h_2022_friend_s42.npy").astype(np.float64)
    din = np.load(f"{DL}/dg_{vs}_DIN.npy").mean(0)
    ms = np.load(f"{DL}/msaudit/audit_all_vs{vs}_direct.npy").astype(np.float64)
    P = np.column_stack([cb, tab, din, ms]).astype(np.float32)
    return X, P, y


class Gate(nn.Module):
    def __init__(self, n_in, conservative):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(32, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.cons = conservative
        self.register_buffer("logfix", torch.log(torch.tensor(FIX, dtype=torch.float32)))

    def forward(self, x):
        z = self.net(x)
        if self.cons:
            z = z + self.logfix
        return torch.softmax(z, 1)


def train_gate(X, P, y, conservative, seed=42, epochs=5, bs=8192, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Gate(X.shape[1], conservative)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    XT, PT = torch.from_numpy(X), torch.from_numpy(P)
    YT = torch.from_numpy(y.astype(np.float32))
    n = len(y)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for a in range(0, n, bs):
            b = torch.from_numpy(perm[a:a + bs])
            g = m(XT[b])
            p = (g * PT[b]).sum(1)
            loss = ((p - YT[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def apply_gate(m, X, P):
    out = []
    for a in range(0, len(X), 16384):
        g = m(torch.from_numpy(X[a:a + 16384]))
        out.append((g * torch.from_numpy(P[a:a + 16384])).sum(1).numpy())
    return np.concatenate(out).astype(np.float64)


if __name__ == "__main__":
    D = {vs: fold(vs) for vs in (2022, 2024)}
    sc = lambda p, y: F.best_shift(p, y)[0]
    for src, tgt in ((2022, 2024), (2024, 2022)):
        Xs, Ps, ys = D[src]
        Xt, Pt, yt = D[tgt]
        ref = sc(Pt @ FIX, yt)
        print(f"  {src}->{tgt}   고정가중 {ref:.1f}")
        for nm, cons in (("자유 게이트", False), ("보수 게이트", True)):
            outs = []
            for sd in (42, 1, 777):
                m = train_gate(Xs, Ps, ys, cons, seed=sd)
                outs.append(sc(apply_gate(m, Xt, Pt), yt))
            mu = float(np.mean(outs))
            print(f"    {nm}   {mu:8.1f}  (시드별 "
                  + " ".join(f"{v:.1f}" for v in outs)
                  + f")   고정 대비 {mu-ref:+.1f}", flush=True)
        # 참고: 그 폴드 자신에서 적합하면 얼마나 나오나 (과적합 상한)
        m = train_gate(Xt, Pt, yt, False, seed=42)
        print(f"    (인폴드 상한 {sc(apply_gate(m, Xt, Pt), yt):.1f} — 참고용, 못 쓰는 값)")
