# -*- coding: utf-8 -*-
"""퓨처스를 **앵커의 보정 모델**로 학습한다. anchor penalty 로 수축을 조절한다.

제안 (지인)
    z_futures = z_all + delta
    Loss = Brier(y, sigmoid(z_futures)) + lambda x (z_futures - z_all)^2

왜 이게 오늘 발견의 '처방' 인가
    fut_stage2 에서 e4_lb 가 퓨처스 구간 +14.4 인데 전역 시프트에서 -1.73 이었다.
    퓨처스 예측 **수준이 밀려서** 전역 시프트와 어긋난 것이다.
    anchor penalty 는 정확히 그 이동을 막는다 — z_f 를 z_a 근처에 묶는다.

왜 0.6/0.4 의 재탕이 아닌가
    현행   퓨처스 모델이 17.4만 행(11.8%)으로 **전부 새로 배운다** -> 그 뒤 혼합
    제안   앵커는 147만 행으로 배운 걸 그대로 쓰고 퓨처스는 **차이만** 배운다
    배워야 할 양이 훨씬 적다. 신호가 0.93% 인 문제에서 이건 큰 차이다.

과거 실패와의 차이
    preft   사전학습 -> 새 체제 미세조정   -46.3  0/4
    xleague 전 리그 -> 퓨처스 미세조정      -42.7  0/4
    둘 다 순차 미세조정이라 작은 표본에서 backbone 이 망가졌다.
    penalty 가 그 실패 모드를 막는다. lambda -> inf 면 앵커 그대로다.

구현
    앵커 로짓 z_a 는 행마다 스칼라다 (앵커의 확률 평균을 로짓으로).
    퓨처스 모델의 k개 서브모델 각각을 z_a 로 당긴다.
        penalty = mean_{batch,k} (z_f - z_a)^2

팔 (1군 행은 항상 0.6 x all + 0.4 x regular. 퓨처스 행만 바뀐다)
    blend   현행 0.6 x all + 0.4 x futures(독립 학습)      <- 기준
    res0.1  z_f 직접 사용, lambda=0.1
    res0.3  lambda=0.3
    res1.0  lambda=1.0
    res3.0  lambda=3.0

채점
    **전체**를 주 잣대로 한다. 팔마다 전역 시프트를 다시 잡는다.
    구간 점수는 참고로만 찍는다 — 구간 시프트가 구간 이득을 부풀리기 때문이다.
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
LAMS = (0.1, 0.3, 1.0, 3.0)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def train_res(model, idx, epochs, lr, z_a, lam, w=None, bs=2048, wd=3e-4,
              clip=5.0, seed=42, tag=""):
    """G.train 과 같되 앵커 수축 항을 더한다. z_a 는 행마다 스칼라 로짓."""
    torch.manual_seed(seed)
    ps = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ii = torch.from_numpy(idx)
    ww = None if w is None else torch.from_numpy(w.astype(np.float32))
    ZA = torch.from_numpy(z_a.astype(np.float32))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        tot = 0.0
        for s in range(0, len(idx), bs):
            b = ii[perm[s:s + bs]]
            xn = G.XN[b].to(G.DEV, non_blocking=True)
            xc = G.XC[b].to(G.DEV, non_blocking=True)
            yb = G.YY[b].to(G.DEV, non_blocking=True)
            za = ZA[b].to(G.DEV, non_blocking=True)[:, None, None]
            opt.zero_grad(set_to_none=True)
            lg = model(xn, xc)
            t = yb[:, None, None].expand_as(lg)
            per = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                        lg, t, reduction="none")
                   + 0.5 * (lg.sigmoid() - t).square()).mean(dim=(1, 2))
            pen = (lg - za).square().mean(dim=(1, 2))
            per = per + lam * pen
            if ww is None:
                loss = per.mean()
            else:
                wb = ww[perm[s:s + bs]].to(G.DEV)
                loss = (per * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, clip)
            opt.step()
            tot += float(loss.detach()) * len(b)
        sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} loss {tot/len(idx):.6f}")
    return model


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


if __name__ == "__main__":
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
        fut_idx, reg_idx = tr_idx[t_isf], tr_idx[~t_isf]
        w_all = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        w_f = np.where(old[fut_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  퓨처스 {len(fut_idx):,}  "
              f"검증 {len(gate):,}")

        P_all, P_reg, ZA = {}, {}, {}
        for sd in SEEDS:
            for nmb, idxb, wb, store in (("all", tr_idx, w_all, P_all),
                                         ("regular", reg_idx, None, P_reg)):
                torch.manual_seed(sd)
                torch.cuda.manual_seed_all(sd)
                m = G.make_model()
                G.train(m, idxb, 2, LR1, w=wb, seed=sd, tag=f"VS{VS} {nmb} s{sd}")
                s2 = idxb[season[idxb] == VS - 1]
                G.train(m, s2, 1, 2e-4, params=G.stage2_params(m), seed=sd,
                        tag=f"VS{VS} {nmb} s{sd} S2")
                store[sd] = G.predict(m, gate)
                if nmb == "all":       # 앵커 로짓 — 학습 퓨처스 행에도 필요하다
                    za = np.zeros(len(season), np.float64)
                    za[fut_idx] = _logit(G.predict(m, fut_idx))
                    za[gate] = _logit(store[sd])
                    ZA[sd] = za
                del m
                torch.cuda.empty_cache()
        G.log("    all / regular 완료 (팔 공통)")

        def route(p_fut, sd):
            return np.where(isf_g, p_fut, 0.6 * P_all[sd] + 0.4 * P_reg[sd])

        # ---- 기준: 현행 0.6 all + 0.4 futures(독립)
        t0 = time.time()
        ps = []
        for sd in SEEDS:
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            m = G.make_model()
            G.train(m, fut_idx, 2, LR1, w=w_f, seed=sd, tag=f"VS{VS} futB s{sd}")
            s2 = fut_idx[season[fut_idx] == VS - 1]
            for e in range(4):
                G.train(m, s2, 1, 2e-4, params=G.stage2_params(m), seed=sd + e,
                        tag=f"VS{VS} futB s{sd} S2e{e+1}")
            ps.append(route(0.6 * P_all[sd] + 0.4 * G.predict(m, gate), sd))
            del m
            torch.cuda.empty_cache()
        ALL[(VS, "blend")] = ps
        G.log(f"    blend 끝 {time.time()-t0:.0f}s")

        for lam in LAMS:
            nm = f"res{lam}"
            t0 = time.time()
            ps = []
            for sd in SEEDS:
                torch.manual_seed(sd)
                torch.cuda.manual_seed_all(sd)
                m = G.make_model()
                train_res(m, fut_idx, 2, LR1, ZA[sd], lam, w=w_f, seed=sd,
                          tag=f"VS{VS} {nm} s{sd}")
                ps.append(route(G.predict(m, gate), sd))
                del m
                torch.cuda.empty_cache()
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"fr_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:8s} 끝 {time.time()-t0:.0f}s")
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"fr_{VS}_blend_s{sd}.npy"),
                    ALL[(VS, "blend")][i])

    def sc(p, yv, m):
        return F.best_shift(p[m], yv[m])[0]

    NAMES = ["blend"] + [f"res{l}" for l in LAMS]
    print("\n" + "=" * 96)
    print("  퓨처스 residual expert (anchor penalty) — **전체**가 주 잣대")
    print("=" * 96)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        allm = np.ones(len(yv), bool)
        base = ALL[(VS, "blend")]
        print(f"\n  VS={VS}   (기준 = blend, 현행 0.6all+0.4fut)")
        for nm in NAMES:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:8s} 전체 {sc(p, yv, allm):8.1f}  "
                    f"1군 {sc(p, yv, ~isf_g):8.1f}  "
                    f"퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "blend":
                dd = [sc(a, yv, allm) - sc(b, yv, allm) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   전체차 {mu:+6.1f}+-{se:4.1f} "
                         f"t={mu/max(se,1e-9):5.2f} "
                         f"{sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)
    print("\n  lambda 가 클수록 앵커에 가깝다. lambda -> inf 면 all 모델 그대로다.")
    print("  퓨처스 구간이 올라도 전체가 안 오르면 수준 이동 문제다 —")
    print("  그때는 penalty 를 더 키워야 한다.")
