# -*- coding: utf-8 -*-
"""FT-Transformer 와 ResNet 을 잰다. 벤치마크에서 남은 주요 계열이다.

왜 이 둘인가
    이미 잰 것: TabM(906), CatBoost(903), HistGB, MLP(776), TabR(803),
    TabPFN(342), MNCA(897~901), XGBoost(680).

    남은 것 중 우리 TabM 근처(900)까지 올라올 가능성이 있는 건 둘이다.
        FT-Transformer  피처 간 어텐션. 편향이 진짜 다르다.
        ResNet (RTDL)   MLP 계열이지만 잔차 연결. 같은 패키지에 있다.

    나머지(DCNv2, AutoInt, SAINT, NODE, TabNet)는 RTDL 벤치마크에서 일관되게
    MLP 아래였다. 여기서 열지 않는다.

오늘 배운 판정 기준
    후보가 값어치를 가지려면 **품질과 직교성을 동시에** 만족해야 한다.
        TabPFN   상관 0.689 로 충분히 직교했지만 단독 342 -> 전 비중 음수
        XGBoost  상관 0.892 / 0.752 인데 단독 680 -> 전 비중 음수
        MNCA     상관 0.954 로 별로 안 직교한데 단독 901 -> +6.6
    직교성만으로는 구제되지 않는다. 단독이 880 아래면 기대하지 않는다.

설정
    배치 구성에 맞춘다 — 지인 전처리 + abs_regime, 옛퓨처스 가중 0.1,
    2에폭 lr 3e-3, 브랜치 라우팅. TabM 과 같은 조건에서 비교해야 한다.
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
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def make(kind, n_num, cards):
    import rtdl_revisiting_models as rtdl
    if kind == "ftt":
        return rtdl.FTTransformer(
            n_cont_features=n_num, cat_cardinalities=[int(c) for c in cards],
            d_out=1, **rtdl.FTTransformer.get_default_kwargs(n_blocks=3),
        ).to(G.DEV)
    if kind == "resnet":
        return rtdl.ResNet(
            d_in=n_num + len(cards), d_out=1, n_blocks=3, d_block=256,
            d_hidden=None, d_hidden_multiplier=2.0,
            dropout1=0.15, dropout2=0.0,
        ).to(G.DEV)
    raise ValueError(kind)


def one(kind, seed, tr_idx, t_isf, w, season):
    """TabM 과 같은 라우팅으로 브랜치 3개를 만든다."""
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = make(kind, G.Xn.shape[1], G.cards)
        _train(m, kind, idx, 2, LR1, ww, seed, f"{kind} {br} s{seed} S1")
        # Stage2 — 마지막 시즌만. 가중도 같은 부분집합으로 맞춘다.
        m2 = season[idx] == VS - 1
        s2, w2 = idx[m2], ww[m2]
        for e in range(EP2[br]):
            _train(m, kind, s2, 1, 2e-4, w2, seed + e,
                   f"{kind} {br} s{seed} S2e{e+1}")
        P[br] = _predict(m, kind, gate)
        del m
        torch.cuda.empty_cache()
    return (0.6 * P["all"] + 0.4 * P["regular"],
            0.6 * P["all"] + 0.4 * P["futures"])


def _feed(m, kind, b):
    xn = G.XN[b].to(G.DEV)
    xc = G.XC[b].to(G.DEV)
    if kind == "ftt":
        return m(xn, xc).squeeze(-1)
    return m(torch.cat([xn, xc.float()], 1)).squeeze(-1)


def _train(m, kind, idx, epochs, lr, w, seed, tag, head_only=False, bs=2048):
    torch.manual_seed(seed)
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ii = torch.from_numpy(idx)
    ww = None if w is None else torch.from_numpy(w.astype(np.float32))
    m.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        tot = n = 0.0
        for s in range(0, len(idx), bs):
            sel = perm[s:s + bs]
            b = ii[sel]
            yb = G.YY[b].to(G.DEV)
            opt.zero_grad(set_to_none=True)
            lg = _feed(m, kind, b)
            per = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                        lg, yb, reduction="none")
                   + 0.5 * (lg.sigmoid() - yb).square())
            loss = per.mean() if ww is None else \
                (per * ww[sel].to(G.DEV)).sum() / ww[sel].to(G.DEV).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 5.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} loss {tot/n:.6f}")


def _predict(m, kind, idx, bs=8192):
    m.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = torch.from_numpy(idx[s:s + bs])
            out.append(_feed(m, kind, b).sigmoid().float().cpu().numpy())
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

    TM = np.mean([np.load(f"{OUT}/sc_s{s}.npy") for s in
                  (42, 1, 777, 2, 7, 13, 99, 2024)
                  if os.path.exists(f"{OUT}/sc_s{s}.npy")], 0)
    REF = 0.30 * G.CB + 0.70 * TM
    r0 = sc(REF)
    G.log(f"  기준 혼합 (CB 0.30 / TabM8 0.70) {r0:.1f}\n")

    AL = (0.05, 0.10, 0.20, 0.30)
    for kind in ("resnet", "ftt"):
        t0 = time.time()
        try:
            res = [one(kind, sd, tr_idx, t_isf, w10, season) for sd in SEEDS]
        except Exception as e:
            G.log(f"  {kind} 실패: {type(e).__name__} {str(e)[:140]}")
            continue
        ps = [np.where(is_f, r[1], r[0]) for r in res]
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"ftt_{kind}_s{sd}.npy"), ps[i])
        p = np.mean(ps, 0)
        cells = []
        for a in AL:
            d = [sc((1 - a) * REF + a * q) - r0 for q in ps]
            mu = float(np.mean(d)); se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            cells.append((a, mu, se, sum(1 for x in d if x > 0)))
        best = max(cells, key=lambda c: c[1])
        G.log(f"  {kind:8s} 단독 {sc(p):7.1f}  1군 {sc(p, R):7.1f}  "
              f"퓨처스 {sc(p, is_f):7.1f}  상관 {np.corrcoef(p, REF)[0,1]:.4f}"
              f"  {time.time()-t0:5.0f}s")
        G.log(f"  {'':8s} " + "  ".join(f"a{a:.2f} {mu:+5.1f}+-{se:4.1f}"
                                        for a, mu, se, _ in cells)
              + f"   최적 {best[1]:+.1f} t={best[1]/max(best[2],1e-9):.2f} "
                f"{best[3]}/{len(ps)}")

    G.log("\n  단독이 880 아래면 기대하지 않는다 — TabPFN(342)/XGBoost(680) 이")
    G.log("  상관은 낮았지만 전 비중에서 음수였다. 직교성만으로는 안 된다.")
