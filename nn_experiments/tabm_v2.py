# -*- coding: utf-8 -*-
"""
TabM + PLR v2 — 두 가지를 고친다.

v1 에서 나온 문제
  내부검증(학습구간 마지막 5% = 2023 후반)과 관문(2024)이 정반대로 움직였다.
      ep    내부Brier    관문 shift0 / −0.05
      01    0.248784      742.6 / 770.4      <- 관문 최고
      02    0.248490      678.7 / 762.5
      03    0.248412      724.3 / 745.2
      04    0.248662      694.3 / 621.7
      05    0.248320      627.0 / 736.3      <- 내부 최저
  내부검증으로 에포크를 고르면 관문 최악을 고르게 된다.
  그리고 shift 0 과 −0.05 의 순서가 에포크마다 뒤집힌다(ep04 vs ep05).
  즉 점수 변동의 지배 성분이 '예측 수준' 이지 판별력이 아니다.

고침 1 — 예측 재중심화
  로짓 평균을 학습구간에서 정한 상수로 강제 정렬한다.
      logit(p) - mean(logit(p)) + C
  C 는 학습 마지막 시즌(2023)의 성공률 로짓. 학습 데이터만 쓰므로 규칙에 안전하다.
  에포크마다 흔들리는 수준을 없애면 판별력만 남아 비교가 깨끗해진다.

고침 2 — 시간이동 내부검증
  학습 2019~2022 -> 검증 2023 (1년 간격) 으로 에포크를 고른다.
  검증이 관문과 같은 성질을 가져야 올바른 에포크가 나온다.
  고른 에포크 수로 2019~2023 전체를 다시 학습해 2024 를 예측한다.
  관문(2024)은 선택에 일절 쓰지 않는다.
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
SC = os.path.dirname(os.path.abspath(__file__))
def H(t):
    print("\n" + "=" * 88); print(t); print("=" * 88); sys.stdout.flush()

z = np.load(os.path.join(SC, "nn_cache.npz"), allow_pickle=False)
Xc, Xn, y, season, is_f = z["Xc"], z["Xn"], z["y"], z["season"], z["is_f"]
cat_card = z["cat_card"].tolist()
n_con = Xn.shape[1]

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
def bss(p, yy):
    r = yy.mean()
    return 100000 * (1 - ((p - yy) ** 2).mean() / (r * (1 - r)))

# 재중심화 목표: 학습 마지막 시즌의 성공률. 학습 데이터만 사용.
def recenter(p, C):
    lp = logit(p)
    return 1 / (1 + np.exp(-(lp - lp.mean() + C)))


class PLR(nn.Module):
    def __init__(self, n_feat, k=24, d=8, sigma=0.1):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_feat, k) * sigma)
        self.w = nn.Parameter(torch.randn(n_feat, 2 * k, d) / math.sqrt(2 * k))
        self.b = nn.Parameter(torch.zeros(n_feat, d))
    def forward(self, x):
        zz = 2 * math.pi * x.unsqueeze(-1) * self.c
        p = torch.cat([torch.sin(zz), torch.cos(zz)], -1)
        return torch.relu(torch.einsum("bfk,fkd->bfd", p, self.w) + self.b).flatten(1)


class TabMLinear(nn.Module):
    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.r = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
        self.s = nn.Parameter(torch.ones(k, d_out))
        self.b = nn.Parameter(torch.zeros(k, d_out))
    def forward(self, x):
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
            L.append(nn.ModuleDict({"lin": TabMLinear(d, h, k),
                                    "bn": nn.BatchNorm1d(k * h),
                                    "do": nn.Dropout(p)}))
            d = h
        self.blocks = nn.ModuleList(L)
        self.head = TabMLinear(d, 1, k)
        self.d_in = d_in
    def forward(self, xc, xn):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.emb)]
        h = torch.cat(e + [self.plr(xn)], 1).unsqueeze(1).expand(-1, self.k, -1)
        for blk in self.blocks:
            h = blk["lin"](h)
            B, k, d = h.shape
            h = blk["bn"](h.reshape(B, k * d)).reshape(B, k, d)
            h = blk["do"](torch.nn.functional.gelu(h))
        return self.head(h).squeeze(-1)


def to_t(idx):
    return (torch.from_numpy(Xc[idx].astype(np.int64)),
            torch.from_numpy(Xn[idx]), torch.from_numpy(y[idx]))

def predict(net, tc, tn, bs=32768):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tc), bs):
            out.append(torch.sigmoid(net(tc[i:i+bs], tn[i:i+bs])).mean(1).numpy())
    return np.concatenate(out)

K, BS, LR = 8, 4096, 2e-3

def run(i_fit, i_eval, epochs, seed, C, tag, y_eval=None, mask=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = TabM(cat_card, n_con, k=K)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    nstep = (len(i_fit) + BS - 1) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=epochs, steps_per_epoch=nstep)
    lossf = nn.BCEWithLogitsLoss()
    Tc_f, Tn_f, Ty_f = to_t(i_fit)
    Tc_e, Tn_e, _ = to_t(i_eval)
    curve, snaps = [], []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(i_fit))
        for b in range(nstep):
            j = perm[b*BS:(b+1)*BS]
            opt.zero_grad()
            lg = net(Tc_f[j], Tn_f[j])
            lossf(lg, Ty_f[j].unsqueeze(1).expand_as(lg)).backward()
            opt.step(); sched.step()
        pe = recenter(predict(net, Tc_e, Tn_e), C)
        snaps.append(pe)
        if y_eval is not None:
            m = mask if mask is not None else slice(None)
            s = bss(pe[m], y_eval[m])
            curve.append(s)
            print(f"    {tag} ep{ep+1:02d}  {s:8.1f}", flush=True)
    return curve, snaps

# ── 1단계: 시간이동 검증으로 에포크 고르기 ────────────────────────────────
EPOCHS = 8
i_a = np.where(season < 2023)[0]
i_b = np.where(season == 2023)[0]
r23 = float(y[season == 2022].mean())          # 학습 마지막 시즌 성공률(2022) 로 중심 맞춤
C_a = math.log(r23 / (1 - r23))
H(f"1단계  학습 2019~2022 ({len(i_a):,}) -> 검증 2023 ({len(i_b):,})   에포크 선택")
print(f"  재중심화 목표 C = logit({r23:.4f}) = {C_a:+.4f}   (2022 성공률)")
mb = ~is_f[i_b]
cur, _ = run(i_a, i_b, EPOCHS, 1, C_a, "검증2023", y[i_b], mb)
best_ep = int(np.argmax(cur)) + 1
print(f"\n  선택: ep{best_ep}  (검증2023 1군 {max(cur):.1f})")

# ── 2단계: 그 에포크 수로 2019~2023 학습 -> 2024 예측 ────────────────────
i_f = np.where(season < 2024)[0]
i_g = np.where(season == 2024)[0]
r24 = float(y[season == 2023].mean())
C_b = math.log(r24 / (1 - r24))
fv, yv = is_f[i_g], y[i_g]
H(f"2단계  학습 2019~2023 ({len(i_f):,}) -> 관문 2024 ({len(i_g):,})   ep{best_ep} 고정")
print(f"  재중심화 목표 C = logit({r24:.4f}) = {C_b:+.4f}   (2023 성공률)")
preds, t0 = [], time.time()
for sd in (1, 42, 777):
    _, snaps = run(i_f, i_g, best_ep, sd, C_b, f"seed{sd}")
    p = snaps[-1]
    preds.append(p)
    print(f"  seed{sd} 단독 관문 1군 {bss(p[~fv], yv[~fv]):.1f}  ({time.time()-t0:.0f}s)",
          flush=True)

H("결과 — 관문 1군 채점")
p = np.mean(preds, 0)
print(f"  TabM+PLR v2  3시드 재중심화   1군 {bss(p[~fv], yv[~fv]):8.1f}   "
      f"전체 {bss(p, yv):8.1f}")
print("\n  비교 (1군)")
print("    TabM+PLR v1 (내부검증 선택)   ~745 (ep03) / 관문최고 770 (ep01)")
print("    FC 평범 3시드                752.2 / 785.5")
print("    sklearn 전체단독              814.2 / 856.2")
print("    CatBoost 60+40                850.6 / 886.3")
np.save(os.path.join(SC, "tabm_v2_gate_pred.npy"), p)
print("\n  예측 저장: tabm_v2_gate_pred.npy")
