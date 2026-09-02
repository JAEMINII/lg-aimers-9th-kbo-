# -*- coding: utf-8 -*-
"""최적화 축을 훑는다. 여기만 안 건드렸고, 여기서 최대 이득이 나왔다.

지금까지의 분포
    구조 축 (관문이 못 맞히는 곳)   전부 패배
        임베딩 0/3,  활성함수 0/3,  용량 10구성 전패,  구조 전패
    최적화 축 (관문이 맞히는 곳)     두 개만 건드림
        스케줄러 에폭단위  -> 리더보드 +21.6   <- 지금까지 최대 이득
        lr 2e-3 -> 3e-3   -> 4/4,  리더보드 +1

    나머지는 전부 지인 selected_config.json 에서 물려받은 값이다.
        batch_size 2048 / weight_decay 3e-4 / dropout 0.1 / d_embedding 16
        bce_weight 0.5 / lr 3e-3 (한 걸음만)

왜 batch_size 가 제일 유망한가
    스케줄러 결함의 정체가 '덜 학습됨' 이었다. 에폭을 늘리는 건 이미 졌다
    (ep3 -6.5, ep4 -37.6) — 과적합 때문이다. 배치를 줄이는 건 다르다.

        2에폭 x 배치 2048  ->  1,194 스텝
        2에폭 x 배치  512  ->  4,776 스텝

    같은 데이터를 더 잘게 나눠 도는 것이라 데이터를 더 보는 게 아니다.
    게다가 작은 배치의 경사 잡음은 정규화로도 작용한다.

bce_weight 는 별개 축이 아니다 (수식으로 확인)
    BCE 기울기   (p - y)
    Brier 기울기 2(p - y) p(1-p)
    우리 예측은 0.40~0.58 에 갇혀 있고 그 구간에서 2p(1-p) = 0.48~0.50 이다.
    즉 두 손실은 기울기가 상수배 차이일 뿐이다.
        0.5 BCE + 0.5 Brier ~ 0.745 BCE,   순수 Brier ~ 0.49 BCE
    bce_weight 를 0.5 -> 0 으로 바꾸는 건 lr 을 0.66배 하는 것과 거의 같다.
    그래도 한 팔 넣는다 — 예측대로면 lr 스윕과 겹쳐 나올 것이고, 그 확인
    자체가 위 분석이 맞는지를 검증한다.

배치와 lr 은 함께 움직여야 한다
    배치를 1/4 로 줄이면 스텝당 경사 잡음이 2배가 된다. 관례대로 lr 을
    sqrt 로 줄인 팔과 그대로 둔 팔을 같이 본다.

판정
    이 축은 관문 부호를 믿는다 — 스케줄러/시즌가중이 맞았던 부류다.
    조합 축의 +15 가 아니라 원래 기준 **+5 & 3/3** 을 쓴다.
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
#      (이름, 배치, lr, bce_weight)
ARMS = (("base",      2048, 3e-3, 0.5),
        ("bs1024",    1024, 3e-3, 0.5),
        ("bs1024_lr", 1024, 2.1e-3, 0.5),
        ("bs512",      512, 3e-3, 0.5),
        ("bs512_lr",   512, 1.5e-3, 0.5),
        ("lr4e3",     2048, 4e-3, 0.5),
        ("brier",     2048, 3e-3, 0.0))

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


def train(model, idx, epochs, lr, bs, bw, params=None, w=None, seed=42, tag=""):
    """G.train 과 같되 배치와 bce_weight 를 받는다. 코사인은 에폭 단위."""
    torch.manual_seed(seed)
    ps = params if params is not None else [p for p in model.parameters()
                                            if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=3e-4)
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
            per = (bw * torch.nn.functional.binary_cross_entropy_with_logits(
                        lg, t, reduction="none")
                   + (1.0 - bw) * (lg.sigmoid() - t).square()).mean(dim=(1, 2))
            if ww is None:
                loss = per.mean()
            else:
                wb = ww[sel].to(G.DEV)
                loss = (per * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 5.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} loss {tot/n:.6f} "
              f"lr {opt.param_groups[0]['lr']:.6f}")


def one(seed, tr_idx, t_isf, w, season, bs, lr, bw):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        train(m, idx, 2, lr, bs, bw, w=ww, seed=seed, tag=f"{br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            train(m, s2, 1, 2e-4, bs, bw, params=ps, seed=seed + e,
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
    G.log(f"  학습 {len(tr_idx):,}\n")
    G.log(f"  {'팔':11s} {'배치':>6s} {'lr':>8s} {'bce_w':>6s} {'2에폭 스텝':>10s}")
    for nm, bs, lr, bw in ARMS:
        G.log(f"  {nm:11s} {bs:6d} {lr:8.2e} {bw:6.2f} "
              f"{int(np.ceil(len(tr_idx)/bs))*2:10,d}")

    RES = {}
    for nm, bs, lr, bw in ARMS:
        t0 = time.time()
        RES[nm] = [one(sd, tr_idx, t_isf, w10, season, bs, lr, bw)
                   for sd in SEEDS]
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"opt_{nm}_r_s{sd}.npy"), RES[nm][i][0])
            np.save(os.path.join(OUT, f"opt_{nm}_f_s{sd}.npy"), RES[nm][i][1])
        G.log(f"  {nm:11s} 끝 {time.time()-t0:.0f}s")

    def full(x):
        return np.where(is_f, x[1], x[0])

    G.log("\n" + "=" * 92)
    G.log(f"  {'팔':11s} {'전체':>8s} {'1군행':>8s} {'퓨처스행':>9s}"
          f"   현행 대비 (짝차이, 부호일치)")
    G.log("=" * 92)
    base = RES["base"]
    for nm, *_ in ARMS:
        r = RES[nm]
        v = [float(np.mean([sc(full(x)) for x in r])),
             float(np.mean([sc(x[0], R) for x in r])),
             float(np.mean([sc(x[1], is_f) for x in r]))]
        if nm == "base":
            G.log(f"  {nm:11s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   <- 현행")
            continue
        tail = []
        for lab, fn, msk in (("전체", full, FULL),
                             ("1군", lambda x: x[0], R),
                             ("퓨처스", lambda x: x[1], is_f)):
            d = [sc(fn(a), msk) - sc(fn(b), msk) for a, b in zip(r, base)]
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            tail.append(f"{lab} {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f} "
                        f"{sum(1 for x in d if x > 0)}/{len(d)}")
        G.log(f"  {nm:11s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   "
              + "  ".join(tail))

    G.log("\n  읽는 법")
    G.log("    배치를 줄이면 같은 2에폭에 스텝이 늘어난다 — 에폭 늘리기(과적합으로")
    G.log("    졌음)와 다른 경로로 '덜 학습됨' 을 푸는 시도다.")
    G.log("    brier 팔이 lr4e3 나 bs*_lr 과 비슷하게 나오면 '손실은 lr 의 재탕'")
    G.log("    이라는 분석이 맞은 것이다 (기울기 상수배 0.49).")
    G.log("    판정선 +5 & 3/3. 최적화 축이라 관문 부호를 믿는다.")
