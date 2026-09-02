# -*- coding: utf-8 -*-
"""주최측이 뺀 사후 정보를 되찾아 **보조 표적**으로 쓴다.

발견
    데이터 설명에 "Target 산출에 사용되는 현재 투구의 사후 정보는 입력 피처로
    제공되지 않습니다" 라고 적혀 있다. 그런데 as-of 비율 열들이 **누적 비율**이라
        누적개수 = rate x n
        직전 투구의 지표 = 누적개수(i) - 누적개수(i-1)
    로 되돌릴 수 있다. 같은 투수이고 n 이 정확히 1 증가한 쌍에서만 쓴다.

검증 (137.6만 쌍)
    범주      0/1 근접   평균     직전 control_success 와 일치
    success    99.94%   0.5204        **1.0000**   <- 자기 자신이라 1 이어야 맞다
    reverse    99.94%   0.2298         0.2499
    middle     99.94%   0.1465         0.3331
    success=1 이면서 reverse=1  0.00000   (상호배타)
    success=1 이면서 middle=1   0.00000   (상호배타)
    success + reverse + middle = 1 인 비율 0.830, 셋 다 0 이 0.137

    즉 제구 실패는 최소 세 갈래다 — 의도반대 23.0% / 가운데·위험 14.7% /
    이름 없는 네 번째 13.7%. 지금은 이 셋을 **하나로 뭉개서** 학습한다.

왜 이게 피처 추가와 다른가
    피처는 모델이 **무엇을 보는가**를 바꾼다. 44열의 변환이면 정보량이 안 는다.
    보조 표적은 모델이 **무엇을 설명해야 하는가**를 바꾼다. 입력은 그대로인데
    행마다 라벨이 1개에서 2~3개가 된다. 같은 표현이 셋을 동시에 설명해야 하므로
    표현이 제약을 받는다 — **용량을 늘리는 게 아니라 감독을 늘리는 것**이라
    오늘 확인한 "신호 0.93%" 벽에 안 걸린다.

규칙
    학습 데이터 안의 산수다. 추론 때는 주 표적 머리(head 0)만 쓴다.
    다른 평가 행을 안 본다. 규칙 4 위반이 아니다.

설정
    base       d_out=1, 현행
    rev0.3     d_out=2, reverse 보조, 가중 0.3
    rev1.0     d_out=2, reverse 보조, 가중 1.0
    revmid0.5  d_out=3, reverse+middle 보조, 각 가중 0.5

    라벨이 없는 행(다음 투구가 없는 투수의 마지막 행 등, 약 6.7%)은
    보조 손실에서 뺀다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)
FOLDS = (2022, 2024)
OLD_F_MAX, OLD_W, LR1 = 2022, 0.1, 3e-3
EP2 = {"all": 1, "regular": 1, "futures": 4}
# (이름, 보조 라벨 목록, 보조 가중)
ARMS = [("base", [], 0.0), ("rev0.3", ["reverse"], 0.3),
        ("rev1.0", ["reverse"], 1.0), ("revmid0.5", ["reverse", "middle"], 0.5)]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
import rtdl_num_embeddings as rne                               # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def recover_labels(data_dir):
    """as-of 누적 비율에서 투구 단위 라벨을 되돌린다. 없으면 nan."""
    cols = {"success": "asof_pitcher_success_rate",
            "reverse": "asof_pitcher_reverse_rate",
            "middle": "asof_pitcher_middle_rate"}
    t = pd.read_csv(os.path.join(data_dir, "train.csv"), encoding="utf-8-sig",
                    usecols=["pitcher_id", "asof_pitcher_n", "control_success"]
                    + list(cols.values()))
    n = t["asof_pitcher_n"].to_numpy(np.float64)
    pid = t["pitcher_id"].to_numpy()
    # 행 i 의 라벨은 행 i+1 의 누적 증가분에서 나온다
    nxt_ok = np.r_[(pid[1:] == pid[:-1]) & (np.diff(n) == 1), False]
    src = np.where(nxt_ok)[0] + 1          # 증가분을 읽을 행
    dst = src - 1                          # 라벨이 붙을 행
    out = {}
    for nm, c in cols.items():
        cum = t[c].to_numpy(np.float64) * n
        inc = cum[src] - cum[dst]
        lab = np.round(inc)
        good = (np.abs(inc - lab) < 0.25) & ((lab == 0) | (lab == 1))
        v = np.full(len(t), np.nan)
        v[dst[good]] = lab[good]
        out[nm] = v
    y = t["control_success"].to_numpy(np.float64)
    m = np.isfinite(out["success"])
    agree = float((out["success"][m] == y[m]).mean())
    print(f"  라벨 복원  {int(m.sum()):,}/{len(t):,} 행 ({m.mean()*100:.1f}%)")
    print(f"  검증: 복원한 success 가 control_success 와 일치 {agree:.6f} "
          f"(1.000000 이어야 맞다)")
    for nm in ("reverse", "middle"):
        v = out[nm]
        print(f"    {nm:8s} 있음 {np.isfinite(v).mean()*100:5.1f}%  "
              f"평균 {np.nanmean(v):.4f}")
    if agree < 0.9999:
        raise ValueError("복원 검증 실패 — success 가 자기 자신과 안 맞는다")
    return out


def make_model(d_out):
    return G.TabM.make(
        n_num_features=G.Xn.shape[1],
        cat_cardinalities=[int(c) for c in G.cards], d_out=d_out,
        num_embeddings=rne.LinearReLUEmbeddings(G.Xn.shape[1], d_embedding=16),
        arch_type="tabm", k=32, n_blocks=3, d_block=256, dropout=0.1,
    ).to(G.DEV)


def train_mt(model, idx, epochs, lr, YA, wa, params=None, w=None, bs=2048,
             wd=3e-4, clip=5.0, seed=42, tag=""):
    """주 표적 + 보조 표적. 보조는 라벨 있는 행에만 건다."""
    torch.manual_seed(seed)
    ps = params if params is not None else [p for p in model.parameters()
                                            if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ii = torch.from_numpy(idx)
    ww = None if w is None else torch.from_numpy(w.astype(np.float32))
    YAT = None if YA is None else torch.from_numpy(YA)     # (n, n_aux) float32
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        tot = 0.0
        for s in range(0, len(idx), bs):
            b = ii[perm[s:s + bs]]
            xn = G.XN[b].to(G.DEV, non_blocking=True)
            xc = G.XC[b].to(G.DEV, non_blocking=True)
            yb = G.YY[b].to(G.DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lg = model(xn, xc)                       # (B, k, d_out)
            m0 = lg[:, :, 0:1]
            t = yb[:, None, None].expand_as(m0)
            per = (0.5 * bce(m0, t, reduction="none")
                   + 0.5 * (m0.sigmoid() - t).square()).mean(dim=(1, 2))
            if ww is None:
                loss = per.mean()
            else:
                wb = ww[perm[s:s + bs]].to(G.DEV)
                loss = (per * wb).sum() / wb.sum()
            if YAT is not None and wa > 0:
                ya = YAT[b].to(G.DEV, non_blocking=True)     # (B, n_aux)
                for j in range(ya.shape[1]):
                    mk = torch.isfinite(ya[:, j])
                    if mk.any():
                        hj = lg[mk][:, :, j + 1:j + 2]
                        tj = ya[mk, j][:, None, None].expand_as(hj)
                        loss = loss + wa * bce(hj, tj)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, clip)
            opt.step()
            tot += float(loss.detach()) * len(b)
        sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} loss {tot/len(idx):.6f}")
    return model


@torch.no_grad()
def predict0(model, idx, bs=8192):
    """주 표적 머리만 쓴다."""
    model.eval()
    out = np.empty(len(idx), np.float64)
    ii = torch.from_numpy(idx)
    for s in range(0, len(idx), bs):
        b = ii[s:min(s + bs, len(idx))]
        lg = model(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV))
        out[s:s + len(b)] = lg[:, :, 0].sigmoid().mean(dim=1).double().cpu().numpy()
    return out


if __name__ == "__main__":
    LAB = recover_labels(DATA)
    d0 = F.build(DATA, VS=2024)
    season = d0["season"].astype(np.float64)
    isf, y = d0["is_f"], d0["y"].astype(np.float64)
    F44 = list(d0["F44"])
    rid = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id"])["row_id"].to_numpy()
    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(rid).to_numpy()

    ALL = {}
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        G.gate, G.yv = gate, yv
        m_tr = season < VS
        hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
        Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
        cols = list(Xs.columns)
        Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
        ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        dv = F.build(DATA, VS=VS)
        Xfr[:, cols.index("plat_dev")] = \
            dv["X44"][:, F44.index("plat_dev")].astype(np.float32)
        old = season <= OLD_F_MAX
        c4 = np.where(old & isf, 0.0,
                      np.where(old & ~isf, 1.0,
                               np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
        Xin = np.concatenate([Xfr, c4], 1)
        Xn, Xc, cards = G.prep(Xin, m_tr, ci + [Xfr.shape[1]])
        G.Xn, G.cards, G.m_tr = Xn, cards, m_tr
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}")

        for nm, auxs, wa in ARMS:
            YA = (None if not auxs else
                  np.stack([LAB[a] for a in auxs], 1).astype(np.float32))
            D = 1 + len(auxs)
            t0 = time.time()
            ps = []
            for sd in SEEDS:
                P = {}
                for br, sel in (("all", np.ones(len(tr_idx), bool)),
                                ("regular", ~t_isf), ("futures", t_isf)):
                    idx_b, wb = tr_idx[sel], w10[sel]
                    torch.manual_seed(sd)
                    torch.cuda.manual_seed_all(sd)
                    m = make_model(D)
                    train_mt(m, idx_b, 2, LR1, YA, wa, w=wb, seed=sd,
                             tag=f"VS{VS} {nm} {br} s{sd} S1")
                    s2 = idx_b[season[idx_b] == VS - 1]
                    pr = G.stage2_params(m)
                    for e in range(EP2[br]):
                        train_mt(m, s2, 1, 2e-4, YA, wa, params=pr, seed=sd + e,
                                 tag=f"VS{VS} {nm} {br} s{sd} S2e{e+1}")
                    P[br] = predict0(m, gate)
                    del m
                    torch.cuda.empty_cache()
                ps.append(np.where(isf_g, 0.6 * P["all"] + 0.4 * P["futures"],
                                   0.6 * P["all"] + 0.4 * P["regular"]))
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"at_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:10s} d_out={D} 보조가중={wa} 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m):
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 92)
    print("  보조 표적 (복원한 reverse / middle) — 입력은 그대로, 감독만 늘린다")
    print("=" * 92)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        allm = np.ones(len(yv), bool)
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}   (기준 = base, d_out=1)")
        for nm, auxs, wa in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:10s} 전체 {sc(p, yv, allm):8.1f}  "
                    f"1군 {sc(p, yv, ~isf_g):8.1f}  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv, allm) - sc(b, yv, allm) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   전체차 {mu:+6.1f}+-{se:4.1f} "
                         f"t={mu/max(se,1e-9):5.2f} "
                         f"{sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)
    print("\n  판정선 — 두 폴드 같은 부호 & 3/3.")
    print("  되면 middle 을 더하고 네 번째 범주(13.7%)도 라벨로 만들어 본다.")
