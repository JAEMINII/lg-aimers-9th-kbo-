# -*- coding: utf-8 -*-
"""수치 임베딩 세 벌을 **현행 배치 구성에서** 다시 잰다.

왜 다시 재나 — 앞선 기각이 무효다
    plr_gate.py 는 스케줄러를 고치기 전에 돌렸다. 그때 코사인이 배치마다 돌아
    2에폭 중 후반이 통째로 lr 약 0 이었다. 그 결함은 임베딩을 **차별적으로**
    불리하게 만든다. linear_relu 는 열당 선형 하나뿐이지만
        periodic    열당 주파수 48개 + 선형        (파라미터가 훨씬 많다)
        piecewise   열당 구간 48개의 기울기/절편
    라서 새로 배울 게 많은 쪽일수록 학습이 끊긴 손해가 크다.
    periodic -91.1 은 '나쁘다' 가 아니라 '안 배워졌다' 에 가까운 숫자였다.

    그리고 plr_gate 는 배치 구성도 아니었다.
        lr 2e-3 (현행 3e-3) / reg4 열 없음 / 옛퓨처스 가중 없음 /
        혼합을 0.10 CB + 0.30 flatMLP + 0.60 TabM 으로 쟀는데 flatMLP 는 뺐다.
    구성이 다르면 부품 값이 바뀌는 걸 오늘 HistGB(+14.7 -> +0.1)에서 봤다.

배치 구성 그대로 (export_final.py 와 일치)
    지인 전처리 + abs_regime 4단계 열 + 옛퓨처스(<=2022) 가중 0.1
    Stage1 2에폭 lr 3e-3,  Stage2 lr 2e-4  (all/regular 1에폭, futures 4에폭)
    1군 행    0.6 x all + 0.4 x regular
    퓨처스 행 0.6 x all + 0.4 x futures

무엇을 믿을 것인가
    임베딩 교체는 같은 TabM 계열 안의 구조 변경이라 계열간 순위 함정은 없다.
    다만 조합 축은 아니어도 최적화 축도 아니다. 용량 스윕과 같은 부류다.
    그래서 판정선을 높게 잡는다 — 시드 짝차이가 3/3 이고 +10 이상일 때만
    제출로 옮긴다. 그 미만은 '무효였던 것을 되살렸다' 로만 기록한다.

배치 비용 (이겼을 때만 문제가 된다)
    script.py 는 numpy 만으로 추론한다.
        periodic   cos/sin(x * freq) 후 선형 — 10줄이면 된다
        piecewise  구간 탐색 + 구간내 비율 — 25줄쯤, 경계값을 같이 저장해야 한다
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
KINDS = ("linear_relu", "periodic", "piecewise")

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def make(kind, n_num, cards, d_emb=16):
    import rtdl_num_embeddings as rne
    from tabm import TabM
    if kind == "linear_relu":
        emb = rne.LinearReLUEmbeddings(n_num, d_embedding=d_emb)
    elif kind == "periodic":
        emb = rne.PeriodicEmbeddings(n_num, d_embedding=d_emb, lite=False)
    elif kind == "piecewise":
        # 분위수 경계는 **학습 구간에서만** 만든다. 검증/시험 행이 새면 규칙 4 다.
        bins = rne.compute_bins(torch.from_numpy(G.Xn[G.m_tr]), n_bins=48)
        emb = rne.PiecewiseLinearEmbeddings(bins, d_embedding=d_emb,
                                            activation=False, version="B")
    else:
        raise ValueError(kind)
    return TabM.make(n_num_features=n_num,
                     cat_cardinalities=[int(c) for c in cards], d_out=1,
                     num_embeddings=emb, arch_type="tabm", k=32,
                     n_blocks=3, d_block=256, dropout=0.1).to(G.DEV)


def one(kind, seed, cards, tr_idx, t_isf, w10, season):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, w = tr_idx[sel], w10[sel]
        torch.manual_seed(seed)                 # 모델 생성 전에 건다
        torch.cuda.manual_seed_all(seed)
        m = make(kind, G.Xn.shape[1], cards)
        G.train(m, idx, 2, LR1, w=w, seed=seed, tag=f"{kind} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e,
                    tag=f"{kind} {br} s{seed} S2e{e+1}")
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
    Xn, Xc, cards = G.prep(np.concatenate([Xf, c4], 1), G.m_tr,
                           ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
    G.log(f"  수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열  "
          f"학습 {len(tr_idx):,}  가중 0.1 적용 {int((w10 < 1).sum()):,}행")

    RES = {}
    for kind in KINDS:
        t0 = time.time()
        try:
            RES[kind] = [one(kind, sd, cards, tr_idx, t_isf, w10, season)
                         for sd in SEEDS]
        except Exception as e:
            G.log(f"  {kind:12s} 실패: {type(e).__name__} {str(e)[:100]}")
            continue
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"emb2_{kind}_r_s{sd}.npy"), RES[kind][i][0])
            np.save(os.path.join(OUT, f"emb2_{kind}_f_s{sd}.npy"), RES[kind][i][1])
        G.log(f"  {kind:12s} 학습 끝 {time.time()-t0:.0f}s")

    if "linear_relu" not in RES:
        raise SystemExit("기준선이 실패했다. 비교 불가")

    def full(pair):
        p = np.where(is_f, pair[1], pair[0])
        return p

    G.log("\n" + "=" * 78)
    G.log(f"  {'임베딩':12s} {'전체':>8s} {'1군행':>8s} {'퓨처스행':>9s}"
          f"   기준 대비 (짝차이, 부호일치)")
    G.log("=" * 78)
    base = RES["linear_relu"]
    for kind in KINDS:
        if kind not in RES:
            continue
        r = RES[kind]
        v_all = float(np.mean([sc(full(x), np.ones(len(gate), bool)) for x in r]))
        v_r = float(np.mean([sc(x[0], R) for x in r]))
        v_f = float(np.mean([sc(x[1], is_f) for x in r]))
        if kind == "linear_relu":
            G.log(f"  {kind:12s} {v_all:8.1f} {v_r:8.1f} {v_f:9.1f}   <- 기준")
            continue
        tail = []
        for nm, fn, msk in (("전체", full, np.ones(len(gate), bool)),
                            ("1군", lambda x: x[0], R),
                            ("퓨처스", lambda x: x[1], is_f)):
            d = [sc(fn(a), msk) - sc(fn(b), msk) for a, b in zip(r, base)]
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            tail.append(f"{nm} {mu:+6.1f}+-{se:4.1f} "
                        f"{sum(1 for x in d if x > 0)}/{len(d)}")
        G.log(f"  {kind:12s} {v_all:8.1f} {v_r:8.1f} {v_f:9.1f}   "
              + "  ".join(tail))

    G.log("\n  판정선: 전체 짝차이 +10 이상 & 3/3 일 때만 제출로 옮긴다.")
    G.log("  그 미만이면 '스케줄러 고친 뒤에도 안 이겼다' 로 축을 닫는다.")
    G.log("  같은 계열 안의 구조 변경이라 계열간 함정은 없지만, 용량 스윕과")
    G.log("  같은 부류라 관문을 그대로 리더보드로 환산하지는 않는다.")
