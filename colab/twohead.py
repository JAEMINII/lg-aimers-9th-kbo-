# -*- coding: utf-8 -*-
"""TabM 하나에 리그별 헤드 둘. 브랜치 블렌드를 구조로 대체한다.

지금 구조
    all      전 행으로 학습
    regular  1군 행만
    futures  퓨처스 행만
    1군  행 = 0.6 x all + 0.4 x regular
    퓨처스행 = 0.6 x all + 0.4 x futures
    -> 1군 행에 2패스, 퓨처스 행에 2패스. 시드 3개면 6패스.

제안 구조
    backbone 공유 + d_out=2 (헤드 0 = 1군, 헤드 1 = 퓨처스)
    각 행의 손실은 그 행 리그의 헤드에만 흐른다.
    -> 리그 무관하게 1패스. 시드 3개면 3패스.

왜 이게 우리 실패들과 다른가
    '리그 간에 표본을 빌린다' 를 세 번 시도해 다 실패했다
        futures_new  새 체제만 학습            퓨처스 286.7 (vs 587.5)
        preft        사전학습 -> 새 체제 미세조정   -46.3  0/4
        xleague      전 리그 -> 퓨처스 미세조정     -42.7  0/4
    셋 다 **순차** 방식이다. 작은 표본으로 미세조정하다 backbone 이 망가진다.
    공동 학습은 그 실패 모드가 없다 — 두 헤드가 동시에 학습하므로 backbone 이
    퓨처스 쪽으로 무너질 수 없고, 퓨처스 헤드는 1.47M 행으로 배운 표현 위에 얹힌다.

    그리고 0.6/0.4 라는 손으로 정한 혼합비가 사라지고 모델이 배운다.

구현이 싼 이유
    추론 코드가 이미 d_out 차원을 쓰고 0번만 뽑고 있다.
        logits = einsum("bki,kio->bko", h, output_weight) + output_bias
        pred   = sigmoid(logits[:, :, 0]).mean(axis=1)
    d_out=2 로 만들면 행마다 열만 골라 뽑으면 된다. 내보내기 형식도 그대로다.

무엇을 재나 (지인 전처리, VS=2024, 시드 4개)
    blend2   현행. all + 리그별 브랜치, 0.6/0.4          (2패스)
    head2    공유 backbone + 헤드 2개                    (1패스)
    head2_nc 헤드 2개인데 abs_regime 을 2단계로 줄인 판본  (1패스)
             리그가 구조에 들어갔으니 코드에서 리그 축을 빼도 되는지 본다

    1군 행과 퓨처스 행을 구간 분모로 따로 채점하고, 추론 시간도 같이 찍는다.

판정
    혼합비가 사라지므로 **조합 변경**이다. 관문이 부호를 못 맞히는 부류다.
    그래서 '이기면 채택' 이 아니라 **'안 지면서 2배 빠르면 채택'** 으로 본다.
    속도 이득은 관문과 무관하게 확실하다.
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
SEEDS = (42, 1, 777, 2)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings            # noqa: E402
from tabm import TabM                                           # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
HEAD = torch.from_numpy(G.is_f.astype(np.int64))     # 0 = 1군, 1 = 퓨처스


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def make2(d_out=2, k=32, n_blocks=3, d_block=256, dropout=0.1, d_emb=16):
    return TabM.make(
        n_num_features=G.Xn.shape[1],
        cat_cardinalities=[int(c) for c in G.cards],
        d_out=d_out,
        num_embeddings=LinearReLUEmbeddings(G.Xn.shape[1], d_embedding=d_emb),
        arch_type="tabm", k=k, n_blocks=n_blocks,
        d_block=d_block, dropout=dropout,
    ).to(G.DEV)


def loss2(logits, target, head):
    """행마다 그 리그의 헤드에만 손실을 흘린다.

    logits (b, k, 2) 에서 head (b,) 로 열을 골라 (b, k, 1) 로 만든 뒤
    기존 손실(0.5 BCE + 0.5 Brier)을 그대로 쓴다.
    """
    idx = head[:, None, None].expand(-1, logits.shape[1], 1)
    sel = logits.gather(2, idx)
    t = target[:, None, None].expand_as(sel)
    brier = (sel.sigmoid() - t).square().mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(sel, t)
    return 0.5 * bce + 0.5 * brier


def train2(model, idx, epochs, lr, params=None, w=None, bs=2048, wd=3e-4,
           clip=5.0, seed=42, tag=""):
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
        tot = 0.0
        for s in range(0, len(idx), bs):
            b = ii[perm[s:s + bs]]
            xn = G.XN[b].to(G.DEV, non_blocking=True)
            xc = G.XC[b].to(G.DEV, non_blocking=True)
            yb = G.YY[b].to(G.DEV, non_blocking=True)
            hd = HEAD[b].to(G.DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lg = model(xn, xc)
            if ww is None:
                loss = loss2(lg, yb, hd)
            else:
                i2 = hd[:, None, None].expand(-1, lg.shape[1], 1)
                sel = lg.gather(2, i2)
                t = yb[:, None, None].expand_as(sel)
                per = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                            sel, t, reduction="none")
                       + 0.5 * (sel.sigmoid() - t).square()).mean(dim=(1, 2))
                wb = ww[perm[s:s + bs]].to(G.DEV)
                loss = (per * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, clip)
            opt.step()
            tot += float(loss.detach()) * len(b)
        sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} loss {tot/len(idx):.6f}")
    return model


@torch.no_grad()
def predict2(model, idx, bs=8192):
    """행마다 그 리그의 헤드에서 뽑는다."""
    model.eval()
    out = np.empty(len(idx), np.float64)
    ii = torch.from_numpy(idx)
    for s in range(0, len(idx), bs):
        b = ii[s:min(s + bs, len(idx))]
        lg = model(G.XN[b].to(G.DEV), G.XC[b].to(G.DEV))
        hd = HEAD[b].to(G.DEV)
        sel = lg.gather(2, hd[:, None, None].expand(-1, lg.shape[1], 1))
        out[s:s + len(b)] = sel.sigmoid().mean(dim=1).squeeze(-1).double().cpu().numpy()
    return out


def fit_head2(seed, idx, w, season, ncode):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = make2(d_out=2)
    t0 = time.time()
    train2(m, idx, 2, LR1, w=w, seed=seed, tag=f"head2({ncode}) S1")
    s2 = season[idx] == VS - 1
    train2(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed, tag="S2")
    t_tr = time.time() - t0
    t1 = time.time()
    p = predict2(m, gate)
    t_pr = time.time() - t1
    del m
    torch.cuda.empty_cache()
    return p, t_tr, t_pr


def fit_branch(seed, idx, w, season, tag):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, LR1, w=w, seed=seed, tag=f"{tag} S1")
    s2 = season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed, tag="S2")
    t1 = time.time()
    p = G.predict(m, gate)
    t_pr = time.time() - t1
    del m
    torch.cuda.empty_cache()
    return p, t_pr


if __name__ == "__main__":
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    c2 = (~old).astype(np.float32)[:, None]     # 리그는 헤드가 맡으니 시대만
    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)

    res, tim = {}, {}

    # --- 현행: all + 리그별 브랜치, reg4 코드
    Xn, Xc, cards = G.prep(np.concatenate([Xf, c4], 1), G.m_tr, ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    A, Rg, Fu, tp = {}, {}, {}, []
    t0 = time.time()
    for s in SEEDS:
        A[s], a = fit_branch(s, tr_idx, w10, season, "all")
        Rg[s], b = fit_branch(s, tr_idx[~t_isf], w10[~t_isf], season, "regular")
        Fu[s], c = fit_branch(s, tr_idx[t_isf], w10[t_isf], season, "futures")
        tp.append(a + b)           # 한 행이 실제로 타는 패스 2개분
    tim["blend2"] = float(np.mean(tp))
    res["blend2"] = {s: np.where(is_f, 0.6 * A[s] + 0.4 * Fu[s],
                                 0.6 * A[s] + 0.4 * Rg[s]) for s in SEEDS}
    G.log(f"  blend2   {time.time()-t0:.0f}s   추론 {tim['blend2']:.1f}s/시드(2패스)")

    # --- 헤드 2개, reg4 코드
    t0 = time.time()
    tp = []
    res["head2"] = {}
    for s in SEEDS:
        p, _, pr = fit_head2(s, tr_idx, w10, season, "reg4")
        res["head2"][s] = p
        tp.append(pr)
    tim["head2"] = float(np.mean(tp))
    G.log(f"  head2    {time.time()-t0:.0f}s   추론 {tim['head2']:.1f}s/시드(1패스)")

    # --- 헤드 2개, 코드를 시대 2단계로 축소
    Xn, Xc, cards = G.prep(np.concatenate([Xf, c2], 1), G.m_tr, ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    t0 = time.time()
    tp = []
    res["head2_nc"] = {}
    for s in SEEDS:
        p, _, pr = fit_head2(s, tr_idx, w10, season, "era2")
        res["head2_nc"][s] = p
        tp.append(pr)
    tim["head2_nc"] = float(np.mean(tp))
    G.log(f"  head2_nc {time.time()-t0:.0f}s   추론 {tim['head2_nc']:.1f}s/시드(1패스)")

    for nm, d in res.items():
        for s in SEEDS:
            np.save(os.path.join(OUT, f"th_{nm}_s{s}.npy"), d[s])

    G.log("\n  최적 시프트, 시드별 계산 후 평균")
    G.log(f"  {'구성':10s} {'1군 행':>9s} {'퓨처스 행':>10s} {'추론':>7s} {'속도':>7s}")
    for nm in ("blend2", "head2", "head2_nc"):
        r = np.mean([sc(res[nm][s], R) for s in SEEDS])
        f_ = np.mean([sc(res[nm][s], is_f) for s in SEEDS])
        G.log(f"  {nm:10s} {r:9.1f} {f_:10.1f} {tim[nm]:7.1f} "
              f"{tim['blend2']/max(tim[nm],1e-9):6.2f}x")

    G.log("\n  blend2 대비 짝차이")
    for nm in ("head2", "head2_nc"):
        for tag, m in (("1군", R), ("퓨처스", is_f)):
            d = [sc(res[nm][s], m) - sc(res["blend2"][s], m) for s in SEEDS]
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            G.log(f"    {nm:9s} {tag:6s} {mu:+7.1f}+-{se:5.1f} "
                  f"t={mu/max(se,1e-9):5.2f} {sum(1 for x in d if x > 0)}/4  "
                  f"[{', '.join(f'{x:+.1f}' for x in d)}]")

    G.log("\n  혼합비가 사라지므로 조합 변경이다. 관문 부호를 못 믿는다.")
    G.log("  '이기면 채택' 이 아니라 '안 지면서 2배 빠르면 채택' 으로 읽는다.")
