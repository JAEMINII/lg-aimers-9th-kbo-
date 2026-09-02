# -*- coding: utf-8 -*-
"""스크린 통과 최강 1개와 '신뢰도 가중'을 배치 구성에서 잰다.

두 축을 한 번에, 단 각 팔은 기준에서 **한 가지만** 다르다
    base      현행 배치 구성 (지인 전처리 44 + abs_regime, 옛퓨처스 가중 0.1)
    bhand     + pdev_bhand 한 열                      <- 피처 축
    relw      + 신뢰도 가중 w = n_p/(n_p+300)          <- 학습분포 축

왜 pdev_bhand 하나만 넣나
    편상관 스크린에서 11개가 통과했지만 5개를 묶으면 3폴드 전부 음수, 최강
    1개만 넣으면 전부 양수였던 전례가 있다. 가장 강한 것부터 하나씩 넣는다.

        pdev_bhand     +0.0198  (퓨처스 +0.0197 / 저표본 +0.0224 / 고표본 +0.0229)
        pdev_twostrike +0.0082
        pdev_ahead     +0.0080
        pdev_strikes   +0.0078
        (기존 유일 통과였던 lg_f_share 가 -0.0174 였다)

    네 부분집합에서 크기가 같은 게 중요하다. 특정 구간에서만 뜨는 값이 아니다.

pdev_bhand 가 plat_dev 와 뭐가 다른가 — 44열에 이미 플래툰 편차가 있다
    plat_dev   고정 이력표에서 뽑은 matchup - overall.  **무수축**
    pdev_bhand as-of 확장(그 행 이전만) + 경험적 베이즈 수축 K=150

    그리고 44열에 asof_pitcher_n 이 없다. 모델은 plat_dev 를 얼마나 믿어야
    할지 알 방법이 없다. 수축은 그 판단을 값 안에 미리 넣어 주는 것이다.
    통제 후에도 편상관이 남은 이유가 여기라고 본다.

왜 신뢰도 가중인가 — 시즌가중과 기전이 다르다
    시즌가중이 통한 건 분포가 실제로 이동해서다 (2023 ABS 로 퓨처스 70.9->47.3%).
    상황은 안 움직인다. 2019년 2-1 카운트와 2025년 2-1 카운트는 같은 물건이다.

    하지만 **피처 신뢰도**는 행마다 다르다. 통산 100구 미만 투수(관문의 2.7%)는
    asof_pitcher_* 아홉 열이 사실상 잡음인데, 모델은 asof_pitcher_n 을 못 봐서
    그 잡음을 신호로 착각해 학습한다. 그 행들을 누르는 건 기전이 명확하다.

        w_rel = n_p / (n_p + 300)      n_p=100 -> 0.25,  n_p=3000 -> 0.91
    기존 옛퓨처스 가중 0.1 과 곱해서 쓴다.
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
REL_K = 300.0
EP2 = {"all": 1, "regular": 1, "futures": 4}
NEW = "pdev_bhand"

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


def one(seed, tr_idx, t_isf, w, season):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"{br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e,
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
                         usecols=["row_id", "asof_pitcher_n"])
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    cond = pd.read_csv(os.path.join(SC, "_dl", "cond_pass.csv.gz"))
    assert len(cond) == len(tr_raw)
    assert (cond["row_id"].to_numpy() == tr_raw["row_id"].to_numpy()).all(), \
        "cond_pass 의 행 순서가 train.csv 와 다르다"
    newcol = cond[NEW].to_numpy(np.float32)[:, None]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]

    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
    npz = tr_raw["asof_pitcher_n"].to_numpy(np.float64)[tr_idx]
    w_rel = w10 * (npz / (npz + REL_K))

    # 두 벌의 전처리 — 열 유무로 갈린다
    PACK = {}
    for use_new in (False, True):
        Xin = np.concatenate([Xf, c4, newcol], 1) if use_new \
            else np.concatenate([Xf, c4], 1)
        PACK[use_new] = G.prep(Xin, G.m_tr, ci + [Xf.shape[1]])
    G.log(f"  기준 수치 {PACK[False][0].shape[1]}열 / "
          f"+{NEW} {PACK[True][0].shape[1]}열   범주 {len(PACK[True][2])}열")
    G.log(f"  신뢰도 가중  중앙값 {np.median(npz/(npz+REL_K)):.3f}   "
          f"0.5 미만 {float(np.mean(npz/(npz+REL_K) < 0.5))*100:.1f}%")

    ARMS = (("base", False, w10), ("bhand", True, w10), ("relw", False, w_rel))
    RES = {}
    for nm, use_new, w in ARMS:
        Xn, Xc, cards = PACK[use_new]
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        t0 = time.time()
        RES[nm] = [one(sd, tr_idx, t_isf, w, season) for sd in SEEDS]
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"cg_{nm}_r_s{sd}.npy"), RES[nm][i][0])
            np.save(os.path.join(OUT, f"cg_{nm}_f_s{sd}.npy"), RES[nm][i][1])
        G.log(f"  {nm:6s} 끝 {time.time()-t0:.0f}s")

    def full(x):
        return np.where(is_f, x[1], x[0])

    G.log("\n" + "=" * 88)
    G.log(f"  {'팔':8s} {'전체':>8s} {'1군행':>8s} {'퓨처스행':>9s}"
          f"   기준 대비 (짝차이, 부호일치)")
    G.log("=" * 88)
    base = RES["base"]
    for nm, *_ in ARMS:
        r = RES[nm]
        v = [float(np.mean([sc(full(x)) for x in r])),
             float(np.mean([sc(x[0], R) for x in r])),
             float(np.mean([sc(x[1], is_f) for x in r]))]
        if nm == "base":
            G.log(f"  {nm:8s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   <- 기준")
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
        G.log(f"  {nm:8s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   "
              + "  ".join(tail))

    G.log("\n  읽는 법")
    G.log("    bhand 는 피처 축이다. 관문이 여섯 번 틀린 부류라 부호를 그대로")
    G.log("      못 믿는다. 크게 이기면 리더보드 한 장으로 확정한다.")
    G.log("    relw 는 학습분포 축이다. 시즌가중이 맞았던 부류라 관문을 따라도")
    G.log("      된다. 이기면 바로 배치로 옮긴다.")
