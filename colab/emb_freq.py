# -*- coding: utf-8 -*-
"""주기 임베딩의 frequency_init_scale 을 훑는다. 이 축은 한 점에서만 시험했었다.

왜 다시 여는가 (기억에는 "임베딩 축 닫음" 이라 적혀 있다)
    TabReD 논문(같은 연구팀, 시간분할 + 분포이동 벤치마크 — 우리 설정과 동일)에서
    **MLP-PLR 이 GBDT 와 함께 최상위**다. PLR = Periodic -> Linear -> ReLU.

    우리는 emb_redo.py 에서 PeriodicEmbeddings(lite=False) 로 **PLR 을 제대로**
    시험했다. 그런데 서명을 보면
        frequency_init_scale: float = 0.01     <- 넘기지 않았다. 기본값을 썼다.
    표준화된 피처에서 sigma=0.01 이면 주기 성분이 거의 선형이라 임베딩이 낭비된다.
    rtdl 논문은 이 값을 0.01~100 로그 범위로 훑는다. 가장 민감한 하이퍼다.

    스케줄러 고친 뒤 재측정에서 periodic 이 -43.5 였는데, 그 크기는 sigma 가
    잘못 잡혔을 때 나오는 크기와 모순되지 않는다.

    "기각 근거가 설정에 묶여 있으면 그게 바뀔 때 다시 재라" — CatBoost -539 가
    설정 실수였고 제대로 재니 +20.3 이었던 전례가 있다.

팔
    base    linear_relu (현행)
    s001    periodic sigma=0.01   <- 이미 진 구성. 재현 대조군으로 넣는다
    s005    periodic sigma=0.05
    s020    periodic sigma=0.2
    s100    periodic sigma=1.0
    s500    periodic sigma=5.0

    s001 이 다시 -40 근처로 나와야 이 실험이 앞선 것과 이어진다.
    다른 sigma 가 base 를 넘으면 축이 다시 열린다.
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
# (이름, 종류, 값)   lr=linear_relu / per=periodic sigma / pw=piecewise n_bins
ARMS = [("base", "lr", None),
        ("s001", "per", 0.01), ("s005", "per", 0.05), ("s020", "per", 0.2),
        ("s100", "per", 1.0), ("s500", "per", 5.0),
        ("pw008", "pw", 8), ("pw096", "pw", 96)]

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
import rtdl_num_embeddings as rne                               # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

KIND, PARAM = "lr", None


def make_model():
    """G.make_model 과 같되 임베딩만 바꾼다. d_emb 는 16 으로 통일한다.

    piecewise 는 경계를 학습 구간에서 뽑아야 한다 (G.m_tr). 앞선 실험은
    n_bins=48 한 점만 봤는데 rtdl 튜닝 범위는 2~128 이다.
    """
    n_num = G.Xn.shape[1]
    if KIND == "lr":
        emb = rne.LinearReLUEmbeddings(n_num, d_embedding=16)
    elif KIND == "per":
        emb = rne.PeriodicEmbeddings(n_num, d_embedding=16, lite=False,
                                     frequency_init_scale=PARAM)
    else:
        bins = rne.compute_bins(torch.from_numpy(G.Xn[G.m_tr]), n_bins=PARAM)
        emb = rne.PiecewiseLinearEmbeddings(bins, d_embedding=16,
                                            activation=False,
                                            version="B")
    # G.make_model 과 동일한 생성자·설정. 임베딩만 갈아끼운다.
    return G.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=[int(c) for c in G.cards],
        d_out=1, num_embeddings=emb, arch_type="tabm",
        k=32, n_blocks=3, d_block=256, dropout=0.1,
    ).to(G.DEV)


def one(seed, tr_idx, t_isf, w, season, VS, gate, isf_g):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"VS{VS} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        pr = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=pr, seed=seed + e,
                    tag=f"VS{VS} {br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(isf_g, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


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
        G.Xn, G.cards = Xn, cards
        # piecewise 경계를 이 폴드의 학습 구간에서 뽑아야 한다. G.m_tr 은 모듈
        # 임포트 시점 값이라 갱신하지 않으면 2022 폴드에서 미래를 본다.
        G.m_tr = m_tr
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}"
              f"  수치열 {Xn.shape[1]}")

        for nm, kd, pv in ARMS:
            KIND, PARAM = kd, pv
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"ef_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:6s} {kd}={pv} 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 96)
    print("  주기 임베딩 frequency_init_scale 스윕 — 기본값 0.01 만 시험했던 축")
    print("=" * 96)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm, kd, pv in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:6s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)
    print("\n  s001 이 -40 근처면 앞선 실험(periodic -43.5)과 이어진 것이다.")
    print("  다른 sigma 가 base 를 넘으면 임베딩 축을 다시 연다.")
    print("  TabReD 에서 MLP-PLR 이 GBDT 와 함께 최상위인 게 이 실험의 근거다.")
