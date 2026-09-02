# -*- coding: utf-8 -*-
"""TabM (ICLR 2025) PyTorch 구현 — 지인 TRAINING_SPEC 재현용.

공식 `tabm` 패키지 API 를 추측으로 쓰면 Colab 에서 깨지므로 직접 구현했다.
구조가 단순해서 재현이 정확하고, Stage2 fine-tuning 도 자유롭게 된다.

TabM 의 핵심 — BatchEnsemble
    하나의 가중치 W 를 k개 멤버가 공유하고, 멤버마다 rank-1 어댑터 (r, s) 만 다르다.

        y_i = ((x_i * r_i) @ W + b_i) * s_i          i = 1..k

    r, s 는 ±1 랜덤 부호로 초기화한다. 이 부호 차이만으로 k개 멤버가 서로 다른
    해로 갈라지고, 학습 1회로 앙상블 효과를 얻는다.
    시드 k개를 따로 학습하는 것과 달리 **붕괴 멤버가 잘 안 생긴다** —
    가중치를 공유하므로 한 멤버가 망가지려면 전체가 망가져야 한다.
    (우리 MLP 시드 앙상블이 실패한 이유가 시드2의 -2759 붕괴였다)

지인 설정
    k=32, n_blocks=3, d_block=256, dropout=0.1
    num_embeddings = LinearReLU, d_embedding=16
    loss = 0.5 x BCE + 0.5 x Brier
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearReLUEmbeddings(nn.Module):
    """수치 피처마다 독립적인 Linear -> ReLU 임베딩.

    스칼라 하나를 d차원으로 펼친다. 트리가 만드는 계단 경계를 MLP 가
    흉내내기 어려운 문제를 완화한다 (PLR 임베딩과 같은 동기, 더 단순한 형태).
    """

    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        bound = 1 / math.sqrt(2)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):                      # x: (B, n_features)
        return F.relu(x[..., None] * self.weight + self.bias)   # (B, n, d)


class EnsembleLinear(nn.Module):
    """BatchEnsemble 선형층. W 는 공유, (r, s, b) 만 멤버별."""

    def __init__(self, k: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(k, d_out))
        self.r = nn.Parameter(torch.empty(k, d_in))
        self.s = nn.Parameter(torch.empty(k, d_out))
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        # ±1 랜덤 부호. 이 초기화가 멤버를 갈라놓는다
        with torch.no_grad():
            self.r.copy_(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
            self.s.copy_(torch.randint(0, 2, (k, d_out)).float() * 2 - 1)

    def forward(self, x):                      # x: (B, k, d_in)
        return (x * self.r) @ self.weight * self.s + self.bias


class TabM(nn.Module):
    def __init__(self, n_num, cat_cards, k=32, n_blocks=3, d_block=256,
                 dropout=0.1, d_embedding=16, n_classes=2):
        super().__init__()
        self.k = k
        self.num_emb = LinearReLUEmbeddings(n_num, d_embedding) if n_num else None
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(int(c), d_embedding) for c in cat_cards])
        d_in = n_num * d_embedding + len(cat_cards) * d_embedding
        blocks, d = [], d_in
        for _ in range(n_blocks):
            blocks.append(nn.ModuleDict({
                "lin": EnsembleLinear(k, d, d_block),
            }))
            d = d_block
        self.blocks = nn.ModuleList(blocks)
        self.dropout = dropout
        self.head = EnsembleLinear(k, d, n_classes)

    def forward(self, x_num, x_cat):
        parts = []
        if self.num_emb is not None and x_num.shape[1]:
            parts.append(self.num_emb(x_num).flatten(1))
        for j, emb in enumerate(self.cat_embs):
            parts.append(emb(x_cat[:, j]))
        x = torch.cat(parts, 1)                        # (B, d_in)
        x = x[:, None].expand(-1, self.k, -1)          # (B, k, d_in)
        for blk in self.blocks:
            x = F.relu(blk["lin"](x))
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        return self.head(x)                            # (B, k, n_classes)

    def last_block_params(self):
        """Stage2 fine-tuning 대상: 마지막 block + head."""
        return list(self.blocks[-1].parameters()) + list(self.head.parameters())


def bce_brier(logits, target):
    """0.5 x BCE + 0.5 x Brier.  최종 채점이 Brier 기반이라 절반을 그쪽에 준다."""
    B, k, _ = logits.shape
    t = target[:, None].expand(B, k).reshape(-1)
    lg = logits.reshape(-1, 2)
    bce = F.cross_entropy(lg, t)
    p1 = torch.softmax(lg, -1)[:, 1]
    brier = ((p1 - t.float()) ** 2).mean()
    return 0.5 * bce + 0.5 * brier


def train_stage(model, Xn, Xc, y, idx, epochs, lr, params=None,
                bs=2048, wd=3e-4, clip=5.0, seed=42, log=print):
    """한 단계 학습. params 를 주면 그것만 갱신한다(fine-tuning)."""
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    ps = params if params is not None else list(model.parameters())
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    steps = max(epochs * int(np.ceil(len(idx) / bs)), 1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    xn = torch.as_tensor(Xn[idx]).to(dev)
    xc = torch.as_tensor(Xc[idx]).to(dev)
    yy = torch.as_tensor(y[idx].astype(np.int64)).to(dev)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx), device=dev)
        tot = 0.0
        for s in range(0, len(idx), bs):
            b = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            loss = bce_brier(model(xn[b], xc[b]), yy[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, clip)
            opt.step()
            sch.step()
            tot += loss.item() * len(b)
        log(f"      epoch {ep+1}/{epochs}  loss {tot/len(idx):.5f}")
    del xn, xc, yy
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def predict(model, Xn, Xc, idx, bs=8192):
    """k개 멤버 확률의 평균."""
    dev = next(model.parameters()).device
    model.eval()
    out = np.empty(len(idx), np.float64)
    for s in range(0, len(idx), bs):
        e = min(s + bs, len(idx))
        xn = torch.as_tensor(Xn[idx[s:e]]).to(dev)
        xc = torch.as_tensor(Xc[idx[s:e]]).to(dev)
        p = torch.softmax(model(xn, xc), -1)[..., 1].mean(1)
        out[s:e] = p.double().cpu().numpy()
    return out


def prep(X, m_tr, cat_idx):
    """TabM 입력 전처리. 통계는 전부 학습 구간에서만 뽑는다.

    범주형 9개  학습 구간 값 목록으로 1..K 인코딩, 미지값 0
    수치형      학습 중앙값 대체 -> 학습 평균/표준편차로 표준화
                결측이 있던 열에는 결측 표시자를 덧붙인다
    """
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
