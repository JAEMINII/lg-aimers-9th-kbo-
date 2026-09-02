# -*- coding: utf-8 -*-
"""LUPI 재시도 — 같은 계열 증류 + 물리량 보조 헤드. CPU 전용 스크린.

어제의 기각과 이번의 차이
    기각된 판   교사 = CatBoost(x44+물리량) -> 학생 = TabM
                교사의 x44-투영이 CatBoost 라서 TabM 을 CatBoost 쪽으로 끌었다.
                λ 내릴수록 단조 악화 (-0.3/-6.3/-14.7). 계열 교차가 사인이었다.
    이번 판     교사 = SENet(x44+물리량) -> 학생 = SENet   **같은 계열**
                내 설명이 맞다면 최소한 안 죽어야 하고, 표적 잡음 제거분이 남으면
                양수가 나온다. 재민님 지시: 텍스트로 버리지 말고 실측으로 가른다.
    보조 헤드    SENet 몸통 + 물리량 8종 회귀 헤드 (매칭행만 마스킹).
                확률을 베끼는 게 아니라 표현을 빚는다 — MS(복원 라벨 보조헤드)가
                이 방식으로 우리 최강 단독 구성원이 된 전례가 있다.

원료   lupi_match2 (976,137쌍, 74.3%, 타자손 일치 0.9999)
팔     base / kdNN07 / kdNN05 / aux05 / aux20   (VS=2024, 시드 2, all 브랜치)
       스크린이다 — 통과하면 풀 프로토콜(두 폴드·3시드·라우팅)로 확정한다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, SC)
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(SC, "_dl")
VS = 2024
SEEDS = (42, 1)
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]

import features44 as F                                          # noqa: E402
import ctr_zoo as Z                                             # noqa: E402


class SENetAux(nn.Module):
    """ctr_zoo.SENet 과 같은 몸통 + (main 1 + 물리량 8) 헤드."""

    def __init__(self, n_num, cards, d=128, red=3, n_aux=8):
        super().__init__()
        EMB = Z.EMB
        self.emb = nn.ModuleList([nn.Embedding(c, EMB) for c in cards])
        for e in self.emb:
            nn.init.normal_(e.weight, 0.0, 0.02)
        self.nf = len(cards)
        self.num_proj = nn.Linear(n_num, EMB)
        f = self.nf + 1
        self.se = nn.Sequential(nn.Linear(f, max(f // red, 2)), nn.ReLU(),
                                nn.Linear(max(f // red, 2), f), nn.Sigmoid())
        self.body = nn.Sequential(nn.Linear(2 * f * EMB + n_num, d), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(d, d), nn.ReLU(),
                                  nn.Dropout(0.1))
        self.head = nn.Linear(d, 1)
        self.aux = nn.Linear(d, n_aux)

    def forward(self, xn, xc):
        e = torch.stack([m(xc[:, i]) for i, m in enumerate(self.emb)], 1)
        e = torch.cat([e, self.num_proj(xn).unsqueeze(1)], 1)
        w = self.se(e.mean(2))
        v = e * w.unsqueeze(2)
        h = self.body(torch.cat([e.flatten(1), v.flatten(1), xn], 1))
        return self.head(h).squeeze(1), self.aux(h)


def train_generic(model, Xn, Xc, tgt, w, idx, seed, aux_t=None, aux_m=None,
                  aux_w=0.0, epochs=2, lr=3e-3):
    """ctr_zoo.fit 의 학습 절차 + 선택적 보조 회귀."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=Z.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    XN, XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    T = torch.from_numpy(tgt.astype(np.float32))
    W = torch.from_numpy(w.astype(np.float32))
    if aux_t is not None:
        AT = torch.from_numpy(aux_t.astype(np.float32))
        AM = torch.from_numpy(aux_m.astype(np.float32))
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(idx)
        for a in range(0, len(perm), Z.BS):
            b = torch.from_numpy(perm[a:a + Z.BS])
            z, az = model(XN[b], XC[b])
            p = torch.sigmoid(z)
            per = 0.5 * nn.functional.binary_cross_entropy_with_logits(
                z, T[b], reduction="none") + 0.5 * (p - T[b]) ** 2
            loss = (per * W[b]).sum() / W[b].sum()
            if aux_t is not None and aux_w > 0:
                m = AM[b]
                if float(m.sum()) > 0:
                    ae = ((az - AT[b]) ** 2).mean(1)
                    loss = loss + aux_w * (ae * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
    model.eval()
    return model


@torch.no_grad()
def predict_main(model, Xn, Xc, rows):
    XN, XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    out = []
    for a in range(0, len(rows), 8192):
        b = torch.from_numpy(rows[a:a + 8192])
        z, _ = model(XN[b], XC[b])
        out.append(torch.sigmoid(z).numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    Z.VS = VS
    D = Z.build_inputs()
    season, isf = D["season"], D["isf"]
    y = D["y"].astype(np.float64)
    gate = np.where(season == VS)[0]
    yv, isf_g = y[gate], isf[gate]
    tr_idx = np.where(D["m_tr"])[0]
    old = season <= 2022
    w = np.where(isf & old, 0.1, 1.0)

    # ---- 물리량 (v2 매칭) 을 features44 행 순서로
    fr_rid = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])["row_id"].to_numpy()
    M = pd.read_csv(os.path.join(DL, "lupi_match2.csv.gz"), encoding="utf-8-sig")
    M["ptg"] = M["pitch_type_group"].map(
        {"fastball": 0, "breaking": 1, "offspeed": 2}).fillna(3)
    pos = pd.Series(np.arange(len(fr_rid)), index=fr_rid)
    mrow = pos.reindex(M["row_id"].to_numpy()).to_numpy()
    ok = np.isfinite(mrow)
    M, mrow = M[ok], mrow[ok].astype(int)
    P = np.full((len(fr_rid), len(PHYS) + 1), np.nan, np.float32)
    P[mrow, :len(PHYS)] = M[PHYS].to_numpy(np.float32)
    P[mrow, len(PHYS)] = M["ptg"].to_numpy(np.float32)
    has = np.zeros(len(fr_rid), bool)
    has[mrow] = True
    mm = has & (season < VS)
    mu = np.nanmean(P[mm], 0)
    sd = np.nanstd(P[mm], 0) + 1e-6
    Pz = np.where(np.isfinite(P), (P - mu) / sd, 0.0).astype(np.float32)
    print(f"  물리량 v2  매칭 {int(has.sum()):,}  학습구간 {int(mm.sum()):,}",
          flush=True)

    Xn, Xc = D["Xn"], D["Xc"]
    cards = D["cards"]
    Xn_t = np.concatenate([Xn, Pz], 1)          # 교사 입력 (물리량 포함)

    def sc(p, m=None):
        q = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[q], yv[q])[0]

    # ---- 1) 교사 (같은 계열, +물리량) — 5폴드 OOF 소프트 타깃
    t0 = time.time()
    tr_m_idx = np.where(mm)[0]
    rng = np.random.default_rng(7)
    fold = rng.integers(0, 5, len(tr_m_idx))
    cache_f = os.path.join(DL, f"nl_soft_{VS}.npy")
    if os.path.exists(cache_f):
        p_soft = np.load(cache_f)
        print("  NN교사 OOF 캐시 재사용", flush=True)
    else:
        p_soft = np.full(len(fr_rid), np.nan)
    for k in (range(5) if not os.path.exists(cache_f) else range(0)):
        tr_k = tr_m_idx[fold != k]
        te_k = tr_m_idx[fold == k]
        mdl = SENetAux(Xn_t.shape[1], cards)
        train_generic(mdl, Xn_t, Xc, y, w, tr_k, seed=100 + k)
        p_soft[te_k] = predict_main(mdl, Xn_t, Xc, te_k)
    np.save(os.path.join(DL, f"nl_soft_{VS}.npy"), p_soft)
    # 교사 품질 확인 (관문 밖 채점: 관문행 물리량으로 직접)
    t_gate = SENetAux(Xn_t.shape[1], cards)
    train_generic(t_gate, Xn_t, Xc, y, w, tr_m_idx, seed=42)
    g_has = gate[has[gate]]
    pg = predict_main(t_gate, Xn_t, Xc, g_has)
    print(f"  NN교사  OOF 완료 {time.time()-t0:.0f}s   관문(매칭행) "
          f"{F.best_shift(pg, y[g_has])[0]:.1f}", flush=True)

    # ---- 2) 팔 실행
    aux_m = (has & (season < VS)).astype(np.float32)
    ARMS = [("base", dict(lam=1.0, aux_w=0.0)),
            ("kdNN07", dict(lam=0.7, aux_w=0.0)),
            ("kdNN05", dict(lam=0.5, aux_w=0.0)),
            ("aux05", dict(lam=1.0, aux_w=0.5)),
            ("aux20", dict(lam=1.0, aux_w=2.0))]
    R = {}
    for nm, cfg in ARMS:
        lam = cfg["lam"]
        tgt = np.where(np.isfinite(p_soft) & (lam < 1.0),
                       lam * y + (1 - lam) * np.nan_to_num(p_soft), y)
        ps = []
        t0 = time.time()
        for sd_ in SEEDS:
            mdl = SENetAux(Xn.shape[1], cards)
            train_generic(mdl, Xn, Xc, tgt, w, tr_idx, seed=sd_,
                          aux_t=Pz[:, :len(PHYS)], aux_m=aux_m,
                          aux_w=cfg["aux_w"])
            ps.append(predict_main(mdl, Xn, Xc, gate))
        R[nm] = ps
        np.save(os.path.join(DL, f"nl_{VS}_{nm}.npy"), np.asarray(ps))
        print(f"    {nm:7s} 끝 {time.time()-t0:5.0f}s", flush=True)

    # ---- 3) 채점 — 단독 + 48식 4원 혼합에 5원째로 소량 추가
    cb = (0.70 * np.load(DL + "/pcgpu2024_c12_cmh_10.npy")
          + 0.30 * np.load(DL + "/pcgpu2024_base44_10.npy")).astype(np.float64) \
        if os.path.exists(DL + "/pcgpu2024_c12_cmh_10.npy") else \
        np.load(DL + "/ta2024_base.npy").mean(0)
    tab = 0.6 * np.load(DL + "/c4g_2024_c4_all_s42.npy") \
        + 0.4 * np.load(DL + "/c4g_2024_c4_regular_s42.npy")
    din = np.load(DL + f"/dg_{VS}_DIN.npy").mean(0)
    ms = np.load(DL + f"/msaudit/audit_all_vs{VS}_direct.npy").astype(np.float64)
    b48 = 0.31 * cb + 0.13 * tab + 0.31 * din + 0.25 * ms
    s48 = sc(b48)
    print("\n" + "=" * 88)
    print(f"  같은 계열 증류 + 보조헤드 (SENet, VS={VS})   48혼합 {s48:.1f}")
    print("=" * 88)
    ref = R["base"]
    for nm, cfg in ARMS:
        ps = R[nm]
        p = np.mean(ps, 0)
        line = f"  {nm:7s} 단독 {sc(p):8.1f}"
        if nm != "base":
            dd = [sc(a) - sc(b) for a, b in zip(ps, ref)]
            line += ("   base 대비 " + " ".join(f"{v:+6.1f}" for v in dd)
                     + f"  {sum(1 for v in dd if v > 0)}/{len(dd)}")
        for wt in (0.05, 0.10):
            line += f"   +5원 w={wt:.2f}: {sc((1-wt)*b48 + wt*p) - s48:+.1f}"
        print(line, flush=True)
    print("\n  같은 계열인데도 kdNN 이 음수면 증류 축은 완전히 닫는다.")
    print("  aux 가 양수면 TabM/MS 몸통으로 확대할 근거가 된다.")
