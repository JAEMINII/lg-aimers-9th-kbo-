# -*- coding: utf-8 -*-
"""
MLP-PLR + TabM — TabReD(ICLR 2025) 가 시간분할 환경에서 최고라고 보고한 조합.

왜 이 조합인가
  TabReD 는 '학술 벤치마크는 무작위 분할, 실제 문제는 시간 분할' 이라는 결함을 지적한다.
  우리 문제가 정확히 그 케이스다 (2019~2024 학습 -> 2025 예측).
  그 벤치마크에서 GBDT 의 우위가 줄고, MLP + 수치임베딩이 GBDT 와 함께 최고로 나온다.

  PLR (On Embeddings for Numerical Features, NeurIPS 2022)
    지금 FC 는 연속형 44개를 표준화해 그냥 벡터로 넣는다. 각 피처가 1차원이다.
    트리는 x > 0.53 같은 계단 경계를 자유롭게 만드는데 MLP 는 매끄러워서 못 만든다.
    주기 임베딩이 그 표현력을 준다:
        z = 2π c x           c 는 학습되는 주파수 (피처마다 k개)
        [sin z, cos z]  ->  Linear  ->  ReLU  ->  d차원
    범주형에 임베딩을 주듯 연속형에도 준다.

  TabM (ICLR 2025)
    우리 FC 실측: 단일시드 506.8/545.8/712.1 -> 3시드 앙상블 752.2 (+164).
    앙상블 이득이 거대하다. TabM 은 이걸 구조로 흡수한다.
    가중치를 공유하고 k개 rank-1 어댑터로 k개 예측을 동시에 낸다(BatchEnsemble).
    학습 1회로 앙상블 효과를 얻고, 학습 중 앙상블 성능을 직접 볼 수 있다.

비교 기준 (같은 관문, 1군 채점, shift 0 / −0.05)
  FC 평범 3시드      752.2 / 785.5
  sklearn 전체단독   814.2 / 856.2
  CatBoost 60+40     850.6 / 886.3
목표: 850 이상. 거기 닿으면 앙상블 비중이 0.05 -> 0.3 대로 올라가 실제 이득이 난다.

주의: 에포크는 내부검증(학습구간 마지막 5%)으로 고른다. 관문 점수도 매 에포크
      찍지만 그건 '내부검증이 좋은 안내자인가' 를 보는 진단용이고 선택에는 안 쓴다.
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
SC = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SC, "nn_cache.npz")
def H(t):
    print("\n" + "=" * 88); print(t); print("=" * 88); sys.stdout.flush()

z = np.load(CACHE, allow_pickle=False)
Xc, Xn, y = z["Xc"], z["Xn"], z["y"]
season, is_f = z["season"], z["is_f"]
cat_card = z["cat_card"].tolist()
n_con = Xn.shape[1]
m_tr = season < 2024
i_tr = np.where(m_tr)[0]
cut = int(len(i_tr) * 0.95)
i_fit, i_val = i_tr[:cut], i_tr[cut:]
i_gate = np.where(season == 2024)[0]
fv = is_f[i_gate]
yv = y[i_gate]
print(f"학습 {len(i_fit):,} + 내부검증 {len(i_val):,} -> 관문 {len(i_gate):,}")
print(f"범주 {len(cat_card)}개, 연속 {n_con}차원")


class PLR(nn.Module):
    """연속형 피처 하나하나에 주기 임베딩을 준다.

    z = 2π c x  ->  [sin z, cos z]  ->  피처별 Linear  ->  ReLU
    c 는 학습되는 주파수. 초기값 N(0, sigma) 로, sigma 가 작으면 매끄럽고
    크면 급한 경계를 만들 수 있다. 논문 권장대로 0.1 근처에서 시작한다.
    """
    def __init__(self, n_feat, k=24, d=8, sigma=0.1):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_feat, k) * sigma)
        self.w = nn.Parameter(torch.randn(n_feat, 2 * k, d) / math.sqrt(2 * k))
        self.b = nn.Parameter(torch.zeros(n_feat, d))
        self.n_feat, self.d = n_feat, d

    def forward(self, x):                       # x: (B, n_feat)
        zz = 2 * math.pi * x.unsqueeze(-1) * self.c        # (B, F, k)
        p = torch.cat([torch.sin(zz), torch.cos(zz)], -1)  # (B, F, 2k)
        out = torch.einsum("bfk,fkd->bfd", p, self.w) + self.b
        return torch.relu(out).flatten(1)                  # (B, F*d)


class TabMLinear(nn.Module):
    """BatchEnsemble 선형층. 가중치는 공유하고 헤드별 rank-1 어댑터만 따로 둔다.

        y_i = ((x * r_i) @ W) * s_i + b_i

    파라미터가 k배가 아니라 (in+out+out)*k 만 늘어난다. 그래서 '효율적 앙상블'이다.
    """
    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        # r 은 ±1 로 초기화하는 것이 BatchEnsemble 관례. 헤드 간 다양성의 원천이다.
        self.r = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
        self.s = nn.Parameter(torch.ones(k, d_out))
        self.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):                        # x: (B, k, d_in)
        return self.W(x * self.r) * self.s + self.b


class TabM(nn.Module):
    def __init__(self, cards, n_con_feat, k=8, d_emb=8, hid=(512, 256), p=0.1):
        super().__init__()
        dims = [max(2, min(32, int(round(c ** 0.4)))) for c in cards]
        self.emb = nn.ModuleList([nn.Embedding(c, dd) for c, dd in zip(cards, dims)])
        self.plr = PLR(n_con_feat, d=d_emb)
        d_in = sum(dims) + n_con_feat * d_emb
        self.k = k
        L, d = [], d_in
        for h in hid:
            L.append(nn.ModuleDict({
                "lin": TabMLinear(d, h, k),
                "bn": nn.BatchNorm1d(k * h),
                "do": nn.Dropout(p)}))
            d = h
        self.blocks = nn.ModuleList(L)
        self.head = TabMLinear(d, 1, k)
        self.d_in = d_in

    def forward(self, xc, xn):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.emb)]
        h = torch.cat(e + [self.plr(xn)], 1)          # (B, d_in)
        h = h.unsqueeze(1).expand(-1, self.k, -1)     # (B, k, d_in)
        for blk in self.blocks:
            h = blk["lin"](h)
            B, k, d = h.shape
            h = blk["bn"](h.reshape(B, k * d)).reshape(B, k, d)
            h = blk["do"](torch.nn.functional.gelu(h))
        return self.head(h).squeeze(-1)               # (B, k) 로짓 k개


def bss(p, yy):
    r = yy.mean()
    return 100000 * (1 - ((p - yy) ** 2).mean() / (r * (1 - r)))
def sh(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))

def to_t(idx):
    return (torch.from_numpy(Xc[idx].astype(np.int64)),
            torch.from_numpy(Xn[idx]), torch.from_numpy(y[idx]))
Tc_f, Tn_f, Ty_f = to_t(i_fit)
Tc_v, Tn_v, Ty_v = to_t(i_val)
Tc_g, Tn_g, _ = to_t(i_gate)

def predict(net, tc, tn, bs=32768):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tc), bs):
            out.append(torch.sigmoid(net(tc[i:i+bs], tn[i:i+bs])).mean(1).numpy())
    return np.concatenate(out)

K, EPOCHS, BS, SEEDS = 8, 10, 4096, (1, 42)
H(f"TabM(k={K}) + PLR   에포크 {EPOCHS}  배치 {BS}  시드 {SEEDS}")
preds, t00 = [], time.time()
for sd_ in SEEDS:
    torch.manual_seed(sd_); np.random.seed(sd_)
    net = TabM(cat_card, n_con, k=K)
    npar = sum(p.numel() for p in net.parameters())
    print(f"  seed{sd_}  입력 {net.d_in}차원  파라미터 {npar/1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    nstep = (len(i_fit) + BS - 1) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EPOCHS, steps_per_epoch=nstep)
    lossf = nn.BCEWithLogitsLoss()
    best, best_p, hist = 1e9, None, []
    t1 = time.time()
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(i_fit))
        for b in range(nstep):
            j = perm[b*BS:(b+1)*BS]
            opt.zero_grad()
            # 헤드마다 독립적으로 손실을 준다 (TabM 방식)
            lg = net(Tc_f[j], Tn_f[j])                       # (B, k)
            loss = lossf(lg, Ty_f[j].unsqueeze(1).expand_as(lg))
            loss.backward(); opt.step(); sched.step()
        pv = predict(net, Tc_v, Tn_v)
        vl = float(((pv - Ty_v.numpy()) ** 2).mean())
        pg = predict(net, Tc_g, Tn_g)
        g0, g5 = bss(sh(pg, 0.0)[~fv], yv[~fv]), bss(sh(pg, -0.05)[~fv], yv[~fv])
        hist.append(vl)
        mark = " <최저" if vl == min(hist) else ""
        if vl < best:
            best, best_p = vl, pg
        print(f"    ep{ep+1:02d} 내부Brier {vl:.6f}{mark:5s}"
              f"   [진단] 관문1군 {g0:7.1f} / {g5:7.1f}", flush=True)
    preds.append(best_p)
    print(f"  seed{sd_} 완료 {time.time()-t1:.0f}s", flush=True)

H("결과 — 관문 1군 채점")
p = np.mean(preds, 0)
for c in (0.0, -0.05):
    q = sh(p, c)
    print(f"  TabM+PLR {len(SEEDS)}시드  shift {c:+.2f}   1군 {bss(q[~fv], yv[~fv]):8.1f}")
print("\n  비교 (1군, shift 0 / −0.05)")
print("    FC 평범 3시드      752.2 / 785.5")
print("    sklearn 전체단독   814.2 / 856.2")
print("    CatBoost 60+40     850.6 / 886.3")
np.save(os.path.join(SC, "tabm_gate_pred.npy"), p)
print(f"\n  예측 저장: tabm_gate_pred.npy")
print(f"총 {time.time()-t00:.0f}s")
