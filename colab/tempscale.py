# -*- coding: utf-8 -*-
"""보정을 절편 하나에서 (온도, 절편) 둘로 늘린다. 그리고 폴드 간 전이를 잰다.

지금 우리 보정
    p_out = sigmoid(logit(p) + c),  c = -0.005
    기울기가 없다. 예측이 너무 평평하거나 너무 뾰족해도 못 고친다.

이게 놀고 있다는 증거 (2026-08-24 로컬 실측)
    같은 60,000행에서 submit_18 예측SD 0.03202, submit_19 0.02837.
    submit_19 의 로짓을 1.129배 늘리자 점수가 +68.9 올랐다.
    (그 채점은 누출된 값이라 크기는 못 믿지만 방향은 분명하다.)

무엇을 재나
    A. 폴드 안에서 (T, c) 를 맞췄을 때 절편만 맞춘 것보다 얼마나 오르나
       -> 이건 상한이다. 같은 데이터에서 맞추고 같은 데이터에서 채점한다.
    B. 다른 폴드에서 맞춘 (T, c) 를 가져다 쓰면 얼마나 남나
       -> 이게 배치에서 실제로 받는 값이다. 2025 의 (T, c) 는 못 보니까.
    C. 1군/퓨처스에 따로 맞추면 더 나은가 (지인 제안 6번 '브랜치별 보정')

    폴드 셋 VS=2022 / 2023 / 2024. 폴드마다 시드 2개.
    시드를 2개만 쓰는 이유는 재는 대상이 점수 차이가 아니라 보정 계수라서다.
    계수는 25만 행에서 추정하므로 시드 잡음에 훨씬 덜 흔들린다.

    학습 파이프라인은 지인 전처리다. submit_18(1048)이 그걸 쓰고,
    리더보드에서 우리 전처리가 이긴 적이 없다.

사용법
    VS=2022 python3 colab/tempscale.py     (폴드마다 따로. import 시점에 VS 고정)
    VS=2023 python3 colab/tempscale.py
    VS=2024 python3 colab/tempscale.py
    python3 colab/tempscale.py --report    (세 폴드 결과를 모아 전이를 잰다)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1)
EPS = 1e-6

import features44 as F                                          # noqa: E402


def lg(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def bss(p, t):
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


def apply_tc(z, T, c):
    return 1.0 / (1.0 + np.exp(-np.clip(T * z + c, -60.0, 60.0)))


def fit_tc(z, t):
    """(T, c) 를 브라이어 최소로 맞춘다. T=1 에서 출발한다."""
    def obj(v):
        return ((apply_tc(z, v[0], v[1]) - t) ** 2).mean()
    r = minimize(obj, np.array([1.0, 0.0]), method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-12, maxiter=2000))
    return float(r.x[0]), float(r.x[1])


def fit_c(z, t):
    """절편만. 지금 방식."""
    def obj(v):
        return ((apply_tc(z, 1.0, v[0]) - t) ** 2).mean()
    r = minimize(obj, np.array([0.0]), method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-12, maxiter=2000))
    return float(r.x[0])


# ------------------------------------------------------------------ 학습
def train_fold(VS):
    import tabm_gate_gpu as G
    from train_chan_3 import preprocess as PP

    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xdf = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xdf.columns)
    Xf = Xdf.to_numpy(dtype=np.float32)
    order_f = tr["row_id"].to_numpy()
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(order_f)), index=order_f)
    Xf = Xf[pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    gate, is_f = G.gate, G.is_f[G.gate]
    Xn, Xc, cards = G.prep(Xf, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    tr_idx = np.where(G.m_tr)[0]

    ps = []
    for seed in SEEDS:
        t0 = time.time()
        P = {}
        for br, m in (("all", np.ones(len(tr_idx), bool)),
                      ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
            idx = tr_idx[m]
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            mdl = G.make_model()
            G.train(mdl, idx, 2, 2e-3, seed=seed, tag=f"{br} S1")
            s2 = G.season[idx] == VS - 1
            G.train(mdl, idx[s2], 1, 2e-4, params=G.stage2_params(mdl),
                    seed=seed, tag=f"{br} S2")
            P[br] = G.predict(mdl, gate)
            del mdl
            torch.cuda.empty_cache()
        ps.append(np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                           0.4 * P["regular"] + 0.6 * P["all"]))
        G.log(f"    seed {seed}  {time.time()-t0:.0f}s")
    return np.mean(ps, 0), is_f, G.yv


# ------------------------------------------------------------------ 보고
def report():
    folds = {}
    for VS in (2022, 2023, 2024):
        f = os.path.join(OUT, f"ts{VS}.npz")
        if not os.path.exists(f):
            print(f"  VS={VS} 없음 — 먼저 VS={VS} python3 colab/tempscale.py")
            continue
        z = np.load(f)
        folds[VS] = (z["p"], z["is_f"], z["y"])
    if len(folds) < 2:
        return

    print(f"\n  A. 폴드 안에서 맞췄을 때 (상한. 같은 데이터에서 맞추고 채점한다)")
    print(f"  {'폴드':6s} {'절편만 c':>10s} {'점수':>9s}   "
          f"{'T':>7s} {'c':>9s} {'점수':>9s} {'차이':>7s}")
    pars = {}
    for VS, (p, isf, y) in folds.items():
        z = lg(p)
        c0 = fit_c(z, y)
        s0 = bss(apply_tc(z, 1.0, c0), y)
        T, c = fit_tc(z, y)
        s1 = bss(apply_tc(z, T, c), y)
        pars[VS] = (T, c)
        print(f"  {VS:6d} {c0:10.4f} {s0:9.1f}   {T:7.4f} {c:9.4f} "
              f"{s1:9.1f} {s1-s0:+7.1f}")

    print(f"\n  B. 다른 폴드에서 맞춘 값을 가져다 쓰면 (배치에서 실제로 받는 값)")
    print(f"  {'적용폴드':8s} {'출처':16s} {'T':>7s} {'c':>9s} "
          f"{'절편만':>9s} {'온도포함':>9s} {'차이':>7s}")
    for VS, (p, isf, y) in folds.items():
        z = lg(p)
        src = [v for v in folds if v != VS]
        if not src:
            continue
        # 다른 폴드들의 (로짓, 라벨) 을 모아 한 벌의 (T, c) 를 만든다.
        zz = np.concatenate([lg(folds[v][0]) for v in src])
        yy = np.concatenate([folds[v][2] for v in src])
        T, c = fit_tc(zz, yy)
        c0 = fit_c(zz, yy)
        s0 = bss(apply_tc(z, 1.0, c0), y)
        s1 = bss(apply_tc(z, T, c), y)
        print(f"  {VS:8d} {'+'.join(map(str, src)):16s} {T:7.4f} {c:9.4f} "
              f"{s0:9.1f} {s1:9.1f} {s1-s0:+7.1f}")

    print(f"\n  C. 1군/퓨처스에 따로 맞추면 (지인 제안 6번)")
    print(f"  {'폴드':6s} {'1군 T':>8s} {'1군 c':>9s} {'F T':>8s} {'F c':>9s} "
          f"{'공통':>9s} {'분리':>9s} {'차이':>7s}")
    for VS, (p, isf, y) in folds.items():
        z = lg(p)
        T, c = fit_tc(z, y)
        s1 = bss(apply_tc(z, T, c), y)
        TR, cR = fit_tc(z[~isf], y[~isf])
        TF, cF = fit_tc(z[isf], y[isf])
        q = np.where(isf, apply_tc(z, TF, cF), apply_tc(z, TR, cR))
        s2 = bss(q, y)
        print(f"  {VS:6d} {TR:8.4f} {cR:9.4f} {TF:8.4f} {cF:9.4f} "
              f"{s1:9.1f} {s2:9.1f} {s2-s1:+7.1f}")

    print("\n  판정 기준")
    print("    A 가 커도 B 가 0 근처면 온도는 못 쓴다 — 그 해에만 맞는 값이다.")
    print("    B 가 세 폴드에서 모두 양수여야 배치에 넣는다.")
    print("    T 자체가 폴드마다 크게 흔들리면 (예: 0.9 / 1.2 / 1.05) 역시 못 쓴다.")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        VS = int(os.environ.get("VS", 2024))
        t0 = time.time()
        p, isf, y = train_fold(VS)
        np.savez(os.path.join(OUT, f"ts{VS}.npz"), p=p, is_f=isf, y=y)
        z = lg(p)
        c0 = fit_c(z, y)
        T, c = fit_tc(z, y)
        print(f"\n  VS={VS}  {time.time()-t0:.0f}s")
        print(f"    절편만  c={c0:+.4f}  {bss(apply_tc(z,1.0,c0), y):8.1f}")
        print(f"    온도    T={T:.4f} c={c:+.4f}  {bss(apply_tc(z,T,c), y):8.1f}")
        print(f"    예측SD {p.std():.5f}   실제SD {y.std():.5f}")
