# -*- coding: utf-8 -*-
"""정규화를 **푸는** 쪽을 훑는다. 같은 진단의 안 해본 처방이다.

진단
    모델이 덜 학습돼 있다. 증거가 둘이다.
        스케줄러 결함 수정        관문 +19.2  ->  리더보드 +21.6
        배치 절반 + lr x0.7       관문  +5.7,  3/3   (스텝 두 배)
    둘 다 '학습을 더 시키는' 처방이고 둘 다 이겼다.

    같은 진단에서 나오는 다른 처방은 **규제를 푸는 것**인데 한 번도 안 봤다.
        weight_decay 3e-4,  dropout 0.1  둘 다 지인 config 에서 물려받은 값

    그리고 그 config 는 시드 하나로 골랐다. 우리가 잰 시드 편차가 6.45 이므로
    차이의 표준편차가 9.1, 2σ 로 약 18점이다. 그 안에 숨은 건 못 봤을 값이다.

주의 — 배치와 상호작용한다
    배치를 줄이면 경사 잡음이 커지고 그 자체가 정규화로 작동한다. 그래서
    bs1024 위에서 규제를 더 풀면 과할 수 있다. 두 배치 모두에서 본다.

팔 (지인 피처, plat_dev 는 **누출판 그대로** — 관문 기준선을 기존과 맞춘다)
    base        bs2048 lr3e-3  wd 3e-4  do 0.1     현행
    do05        dropout 0.05
    do00        dropout 0.0
    wd1e4       weight_decay 1e-4
    wd0         weight_decay 0
    b1024       bs1024 lr2.1e-3            (opt_sweep 승자, 재현 확인용)
    b1024_do05  bs1024 + dropout 0.05      두 처방을 겹침
    b1024_wd1e4 bs1024 + weight_decay 1e-4

판정
    최적화 축이라 관문 부호를 믿는다. 판정선 +5 & 3/3.
    다만 전이율이 0.16 이라 관문 +5 는 리더보드 +1 이다. 제출 슬롯을 쓸지는
    이득이 bs1024_lr 과 **겹치는지 별개인지** 를 보고 정한다.
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
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
#      (이름, 배치, lr, weight_decay, dropout)
ARMS = (("base",        2048, 3e-3, 3e-4, 0.10),
        ("do05",        2048, 3e-3, 3e-4, 0.05),
        ("do00",        2048, 3e-3, 3e-4, 0.00),
        ("wd1e4",       2048, 3e-3, 1e-4, 0.10),
        ("wd0",         2048, 3e-3, 0.0,  0.10),
        ("b1024",       1024, 2.1e-3, 3e-4, 0.10),
        ("b1024_do05",  1024, 2.1e-3, 3e-4, 0.05),
        ("b1024_wd1e4", 1024, 2.1e-3, 1e-4, 0.10))

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


def train(model, idx, epochs, lr, bs, wd, params=None, w=None, seed=42, tag=""):
    torch.manual_seed(seed)
    ps = params if params is not None else [p for p in model.parameters()
                                            if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ii = torch.from_numpy(idx)
    ww = None if w is None else torch.from_numpy(w.astype(np.float32))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        tot = n = 0.0
        for s in range(0, len(idx), bs):
            sel = perm[s:s + bs]
            b = ii[sel]
            xn = G.XN[b].to(G.DEV, non_blocking=True)
            xc = G.XC[b].to(G.DEV, non_blocking=True)
            yb = G.YY[b].to(G.DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lg = model(xn, xc)
            t = yb[:, None, None].expand_as(lg)
            per = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                        lg, t, reduction="none")
                   + 0.5 * (lg.sigmoid() - t).square()).mean(dim=(1, 2))
            loss = (per.mean() if ww is None
                    else (per * ww[sel].to(G.DEV)).sum() / ww[sel].to(G.DEV).sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 5.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} loss {tot/n:.6f}")


def one(seed, tr_idx, t_isf, w, season, bs, lr, wd, do):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model(dropout=do)
        train(m, idx, 2, lr, bs, wd, w=ww, seed=seed, tag=f"{br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            train(m, s2, 1, 2e-4, bs, wd, params=ps, seed=seed + e,
                  tag=f"{br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return (0.6 * P["all"] + 0.4 * P["regular"],
            0.6 * P["all"] + 0.4 * P["futures"])


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
    G.log(f"  학습 {len(tr_idx):,}   팔 {len(ARMS)}개 x 시드 {len(SEEDS)}개\n")

    RES = {}
    for nm, bs, lr, wd, do in ARMS:
        t0 = time.time()
        RES[nm] = [one(sd, tr_idx, t_isf, w10, season, bs, lr, wd, do)
                   for sd in SEEDS]
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"rg_{nm}_r_s{sd}.npy"), RES[nm][i][0])
            np.save(os.path.join(OUT, f"rg_{nm}_f_s{sd}.npy"), RES[nm][i][1])
        G.log(f"  {nm:12s} 끝 {time.time()-t0:.0f}s")

    def full(x):
        return np.where(is_f, x[1], x[0])

    G.log("\n" + "=" * 96)
    G.log(f"  {'팔':12s} {'bs':>5s} {'wd':>7s} {'do':>5s} {'전체':>8s} "
          f"{'1군':>8s} {'퓨처스':>8s}   현행 대비")
    G.log("=" * 96)
    base = RES["base"]
    for nm, bs, lr, wd, do in ARMS:
        r = RES[nm]
        v = [float(np.mean([sc(full(x)) for x in r])),
             float(np.mean([sc(x[0], R) for x in r])),
             float(np.mean([sc(x[1], is_f) for x in r]))]
        head = f"  {nm:12s} {bs:5d} {wd:7.0e} {do:5.2f} {v[0]:8.1f} {v[1]:8.1f} {v[2]:8.1f}"
        if nm == "base":
            G.log(head + "   <- 현행")
            continue
        tail = []
        for lab, fn, msk in (("전체", full, FULL),
                             ("1군", lambda x: x[0], R),
                             ("퓨처스", lambda x: x[1], is_f)):
            d = [sc(fn(a), msk) - sc(fn(b), msk) for a, b in zip(r, base)]
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            tail.append(f"{lab} {mu:+6.1f}+-{se:4.1f} "
                        f"{sum(1 for x in d if x > 0)}/{len(d)}")
        G.log(head + "   " + "  ".join(tail))

    G.log("\n  읽는 법")
    G.log("    do05/do00/wd1e4/wd0 가 이기면 '덜 학습됨' 진단의 두 번째 처방이 맞다.")
    G.log("    b1024_* 가 b1024 보다 더 이기면 두 처방이 **별개**로 쌓인다 —")
    G.log("      그러면 합쳐서 제출할 값어치가 생긴다.")
    G.log("    b1024 수준에서 멈추면 겹치는 것이고, 배치 하나만 쓰면 된다.")
    G.log("    전이율 0.16 이라 관문 +5 는 리더보드 +1 이다. 그걸 감안해 판단한다.")
