# -*- coding: utf-8 -*-
"""미탐색 축 스윕. 인자로 설정 이름을 받는다.

    python sweep_runner.py oldw   옛 퓨처스 가중 (한 번도 안 훑었다)
    python sweep_runner.py loss   BCE/Brier 비율 (하드코딩돼 있었다)
    python sweep_runner.py bslr   배치 절반 + lr x0.7 (관문 통과했는데 미배치)

왜 이 셋인가
    이 프로젝트에서 통한 부류는 둘뿐이다 — **결함 수정**과 **학습 분포 변경**.
    피처 추가는 일곱 번 졌다 (오늘 트랙맨 편차형이 일곱 번째다).

    oldw / loss 는 학습 분포·목적함수 축인데 상수로만 박혀 있고 스윕 기록이 없다.
        OLD_W = 0.1   2022 이하 퓨처스 행의 가중. 체제전환(70.9%->47.3%) 대응으로
                      넣은 값인데 0.1 이 어디서 왔는지 근거가 없다.
        손실   0.5*BCE + 0.5*Brier. 지인 코드에서 물려받았다. 채점이 Brier 인데
                      BCE 를 절반 섞는 근거를 잰 적이 없다.
    bslr 은 관문 +5.7 (3/3) 로 통과했는데 안 실었다. 배치 구성에서 재확인한다.

주의
    시드는 모델 생성 **전에** 걸어야 한다. one() 안에서 manual_seed 를 먼저 부른다.
    비교는 항상 최적 시프트에서 한다 (수준 보정과 판별력이 섞이면 안 된다).
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
OLD_F_MAX = 2022
EP2 = {"all": 1, "regular": 1, "futures": 4}

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

CFG = sys.argv[1] if len(sys.argv) > 1 else "oldw"
# 팔 = (이름, OLD_W, BCE비중, 배치, lr배수)
SPECS = {
    "oldw": [("w000", 0.00, 0.5, 2048, 1.0), ("w005", 0.05, 0.5, 2048, 1.0),
             ("base", 0.10, 0.5, 2048, 1.0), ("w020", 0.20, 0.5, 2048, 1.0),
             ("w040", 0.40, 0.5, 2048, 1.0), ("w100", 1.00, 0.5, 2048, 1.0)],
    "loss": [("bce100", 0.10, 1.00, 2048, 1.0), ("bce075", 0.10, 0.75, 2048, 1.0),
             ("base", 0.10, 0.50, 2048, 1.0), ("bce025", 0.10, 0.25, 2048, 1.0),
             ("bce000", 0.10, 0.00, 2048, 1.0)],
    "bslr": [("base", 0.10, 0.5, 2048, 1.0), ("bs1024", 0.10, 0.5, 1024, 0.7),
             ("bs1024f", 0.10, 0.5, 1024, 1.0), ("bs4096", 0.10, 0.5, 4096, 1.4)],
}
ARMS = SPECS[CFG]
LR1 = 3e-3
_orig_loss = G.loss_fn


def make_loss(wb):
    def f(logits, target):
        t = target[:, None, None].expand_as(logits)
        brier = (logits.sigmoid() - t).square().mean()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, t)
        return wb * bce + (1.0 - wb) * brier
    return f


def one(seed, tr_idx, t_isf, w, season, VS, gate, isf_g, bs, lrm):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1 * lrm, w=ww, bs=bs, seed=seed,
                tag=f"VS{VS} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        pr = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4 * lrm, params=pr, bs=bs, seed=seed + e,
                    tag=f"VS{VS} {br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(isf_g, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


if __name__ == "__main__":
    G.log(f"\n=== 스윕 {CFG}  팔 {[a[0] for a in ARMS]} ===")
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
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}")

        for nm, ow, wb, bs, lrm in ARMS:
            G.loss_fn = make_loss(wb) if abs(wb - 0.5) > 1e-9 else _orig_loss
            w10 = np.where(t_isf & old[tr_idx], ow, 1.0).astype(np.float64)
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g, bs, lrm)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"{CFG}_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:8s} OLD_W={ow} BCE={wb} bs={bs} lr x{lrm} "
                  f"끝 {time.time()-t0:.0f}s")
        G.loss_fn = _orig_loss

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 96)
    print(f"  스윕 {CFG} — plat 고친 기준선 위, 두 폴드, 시드 3개 짝비교")
    print("=" * 96)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm, ow, wb, bs, lrm in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:8s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)
    print("\n  판정 — 두 폴드에서 같은 부호이고 3/3 이어야 후보다.")
    print("  관문은 최적화 축에서는 맞히지만 그래도 전이율이 0.16 수준이다.")
