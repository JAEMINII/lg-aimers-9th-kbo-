# -*- coding: utf-8 -*-
"""fb_hb_norm(속구 수평 무브먼트)를 퓨처스 브랜치에 얹어 관문에서 잰다.

왜 이것 하나인가
    v5 후보 7개를 편상관으로 걸렀다. 귀무 기대값으로 보정한 배율(퓨처스 행)
        fb_hb_norm      4.1x   <- 최강
        fb_shape        2.8x   (fb_hb_norm 의 이산화. 중복)
        cluster_v5      1.9x   (원재료의 압축. 라벨이 원재료보다 약하다)
        max_speed       1.4x   잡음
        n_pitch_types   1.4x   잡음
        fb_ivb / n_breaking_types  통과 못 함
    전체 행에서는 전부 1.3x 근처로 사실상 0 이고 **퓨처스에서만** 뜬다.
    퓨처스 투수는 커리어 표본이 적어 성적 집계가 못 담는 걸 신체 프로파일이
    메우는 구조로 읽힌다. 그리고 퓨처스가 우리 약점 구간(630 vs 870)이다.

    스크린 통과분을 묶어 넣으면 안 된다는 걸 어제 배웠다 — 리그 이동 피처
    5개를 묶었더니 3폴드 전부 음수, 최강 1개만 넣으니 전부 양수였다.

as-of 규약이 핵심이다
    배치(2025)에는 트랙맨이 없다. 평가행은 그 투수의 **직전 시즌 이하** 최신
    프로파일을 받는다. 학습에서도 같은 규칙을 써야 train/serve 가 안 어긋난다.
    같은 시즌 프로파일을 쓰면 관문이 부풀려지고 배치에서 무너진다.

    대가: 트랙맨 클러스터가 2022~2024 뿐이라 학습 구간(2019~2023)에서
    값을 받는 건 2023 행뿐이다. 커버리지가 얇다. 그래도 이게 정직한 조건이고,
    안 통하면 '커버리지가 얇아서' 라는 설명이 남는다.

설정
    지인 전처리 44열 + abs_regime(4단계) + fb_hb_norm = 46열.
    퓨처스 브랜치에만 얹는다. all 브랜치는 시드마다 한 번만 학습해 공유한다.
    채점은 퓨처스 행, 라우팅 0.6 x all + 0.4 x 퓨처스브랜치, 최적 시프트.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
OUT = os.environ.get("AIMERS_OUT", "/workspace/aimers/out")
DATA = os.environ.get("AIMERS_DATA", "/workspace/aimers/data")
V5 = os.environ.get("AIMERS_V5", os.path.join(SC, "_dl", "pitcher_cluster_v5.csv"))
SEEDS = tuple(int(s) for s in os.environ.get("FS_SEEDS", "42,1,777,2").split(","))
OLD_F_MAX = 2022
OLD_W = 0.1

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def asof_shape(raw):
    """각 행에 그 투수의 **직전 시즌 이하** 최신 fb_hb_norm 을 붙인다."""
    v5 = pd.read_csv(V5)
    v5 = v5[v5["pitcher_id"].notna()].copy()
    v5["pitcher_id"] = v5["pitcher_id"].astype(int)
    prof = (v5.groupby(["pitcher_id", "season"])["fb_hb_norm"]
            .mean().reset_index())
    # (투수, 시즌) -> 그 시즌까지의 최신값을 다음 시즌부터 쓸 수 있게 민다
    prof = prof.sort_values(["pitcher_id", "season"])
    prof["avail_from"] = prof["season"] + 1
    out = np.full(len(raw), np.nan)
    key = raw["pitcher_id"].to_numpy()
    ssn = raw["season"].to_numpy()
    look = {}
    for pid, g in prof.groupby("pitcher_id"):
        look[pid] = (g["avail_from"].to_numpy(), g["fb_hb_norm"].to_numpy())
    for i in range(len(raw)):
        e = look.get(key[i])
        if e is None:
            continue
        av, val = e
        j = np.searchsorted(av, ssn[i], side="right") - 1   # 가장 최근 사용가능분
        if j >= 0:
            out[i] = val[j]
    return out


def fit(seed, idx, w, season, tag):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, 2e-3, w=w, seed=seed, tag=f"{tag} S1")
    s2 = season[idx] == VS - 1
    G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
            tag=f"{tag} S2")
    p = G.predict(m, gate)
    del m
    torch.cuda.empty_cache()
    return p


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

    hb = asof_shape(tr_raw[["pitcher_id", "season"]])
    season, isf = G.season.astype(np.float64), G.is_f
    m_tr = G.m_tr
    cov_tr = np.isfinite(hb[m_tr]).mean()
    cov_g = np.isfinite(hb[gate]).mean()
    cov_gf = np.isfinite(hb[gate][is_f]).mean()
    G.log(f"\n  fb_hb_norm as-of  학습 커버리지 {cov_tr*100:.1f}%   "
          f"관문 {cov_g*100:.1f}%   관문 퓨처스 {cov_gf*100:.1f}%")
    fill = np.nanmedian(hb[m_tr]) if np.isfinite(hb[m_tr]).any() else 0.0
    hbf = np.where(np.isfinite(hb), hb, fill).astype(np.float32)
    miss = (~np.isfinite(hb)).astype(np.float32)

    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    X45 = np.concatenate([Xf, c4], 1)
    ci45 = ci + [Xf.shape[1]]
    # 결측 지시자를 같이 넣는다. 안 넣으면 중앙값이 실제값인 척 들어간다.
    X47 = np.concatenate([X45, hbf[:, None], miss[:, None]], 1)

    tr_idx = np.where(m_tr)[0]
    t_isf = isf[tr_idx]
    t_old = t_isf & old[tr_idx]
    w10 = np.where(t_old, OLD_W, 1.0).astype(np.float64)

    # all 브랜치는 45열로 한 번만 (시드마다). 비교 대상은 퓨처스 브랜치뿐이다.
    Xn, Xc, cards = G.prep(X45, m_tr, ci45)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    ALL, BASE = {}, {}
    for s in SEEDS:
        t0 = time.time()
        ALL[s] = fit(s, tr_idx, w10, season, "all")
        BASE[s] = fit(s, tr_idx[t_isf], w10[t_isf], season, "fut base")
        G.log(f"  seed {s} base {time.time()-t0:.0f}s")

    Xn, Xc, cards = G.prep(X47, m_tr, ci45)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    G.log(f"  +fb_hb_norm  수치 {Xn.shape[1]}열")
    HB = {}
    for s in SEEDS:
        t0 = time.time()
        HB[s] = fit(s, tr_idx[t_isf], w10[t_isf], season, "fut +hb")
        G.log(f"  seed {s} +hb {time.time()-t0:.0f}s")

    for nm, D in (("all", ALL), ("futbase", BASE), ("futhb", HB)):
        for s in SEEDS:
            np.save(os.path.join(OUT, f"fs_{nm}_s{s}.npy"), D[s])

    G.log("\n  퓨처스 행, 라우팅 0.6 x all + 0.4 x 브랜치, 최적 시프트")
    b = [sc(0.6 * ALL[s] + 0.4 * BASE[s], is_f) for s in SEEDS]
    h = [sc(0.6 * ALL[s] + 0.4 * HB[s], is_f) for s in SEEDS]
    G.log(f"    base        {np.mean(b):8.1f}   [{', '.join(f'{x:.0f}' for x in b)}]")
    G.log(f"    +fb_hb_norm {np.mean(h):8.1f}   [{', '.join(f'{x:.0f}' for x in h)}]")
    d = [x - y for x, y in zip(h, b)]
    mu = float(np.mean(d))
    se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
    G.log(f"    짝차이 {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
          f"{sum(1 for x in d if x > 0)}/{len(d)}  "
          f"[{', '.join(f'{x:+.1f}' for x in d)}]")
    G.log("\n  퓨처스 구간 +100 은 전체로 약 +12 다 (퓨처스 11.8%).")
    G.log("  커버리지가 얇으니 음수가 나와도 '이 축이 무용' 이 아니라")
    G.log("  '이 커버리지로는 못 쓴다' 로 읽어야 한다.")
