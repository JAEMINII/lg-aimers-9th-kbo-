# -*- coding: utf-8 -*-
"""MS 폴드 비대칭(2022 약함 / 2024 강함)의 원인이 학습량인지 가른다.

가설
    MS 는 210만 파라미터 (TabM 의 7배). 2022 폴드는 학습 73만 행, 2024 폴드는
    122만 행이다. 약함이 체제가 아니라 데이터 부족이면, 배치(148만 행)는
    2024 폴드보다도 강하고 w_ms 를 0.15 -> 0.20~0.25 로 올릴 근거가 된다.

방법 — VS=2024 폴드 하나에서 학습량만 조작한다 (체제 구성은 유지)
    full    2019~2023 전부 (122만)
    sub60   같은 구간 무작위 60% (73만 = 2022 폴드와 같은 크기)
    sub35   같은 구간 무작위 35% (43만)
    학습량-성능 곡선이 가파르면 가설 지지. sub60 이 2022 폴드 수준(-20~-60)으로
    떨어지면 강한 지지다.

같은 시드 3개, 감사판 구성(58열, K32 d512) 그대로.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("AUDIT_MODE", "all")
os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
ROOT = Path("/root")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "aimers"))
DATA = Path(os.environ.get("AIMERS_DATA", "/root/open (1)/data"))
DL = Path("/root/aimers/_dl")
VS = 2024

import features44 as FF                                         # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def main():
    raw = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    isf = raw.game_type.astype(str).to_numpy() == "F"
    old = season <= 2022
    aux = M.recover_state(raw)
    built = FF.build(str(DATA), VS=VS)
    X44 = built["X44"].astype(np.float32)
    names = list(built["F44"])
    xnum, xnn, xcat, xcn = A.audit_features(raw, X44, names)
    c4 = np.where(old & isf, 0.0, np.where(old & ~isf, 1.0,
                  np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xin = np.concatenate([X44, xnum, c4, xcat], axis=1)
    ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    c4i = X44.shape[1] + xnum.shape[1]
    cat_idx = ci + [c4i] + list(range(c4i + 1, Xin.shape[1]))
    m_tr = season < VS
    Xn, Xc, cards = G.prep(Xin, m_tr, cat_idx)
    G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
    tr_all = np.flatnonzero(m_tr)
    va = np.flatnonzero(season == VS)
    w = np.ones(len(y), np.float32)
    w[tr_all] = np.where(isf[tr_all] & old[tr_all], 0.1, 1.0)
    yv = y[va].astype(float)

    # 채점 준비 — 4원 혼합 기준
    cb = (0.70 * np.load(DL / "pcgpu2024_c12_cmh_10.npy")
          + 0.30 * np.load(DL / "pcgpu2024_base44_10.npy")).astype(np.float64)
    tab = np.load(DL / "h2h_2024_friend_s42.npy").astype(np.float64)
    din = np.load(DL / "dg_2024_DIN.npy").mean(0)

    def bs(p):
        return M.best_shift(p, yv)[0]

    b3 = 0.35 * cb + 0.15 * tab + 0.35 * din
    ARMS = (("full", 1.00), ("sub60", 0.60), ("sub35", 0.35))
    M.log(f"DATASIZE VS={VS} full={len(tr_all):,} X={Xin.shape}")
    for nm, frac in ARMS:
        preds = []
        for seed in M.SEEDS:
            rng = np.random.default_rng(1000 + seed)
            idx = (tr_all if frac >= 1.0 else
                   np.sort(rng.choice(tr_all, int(len(tr_all) * frac),
                                      replace=False)))
            t0 = time.time()
            m = M.train_model(M.make_model(Xn.shape[1], cards, seed), idx, y,
                              aux, w, seed, f"DS-{nm} s{seed}")
            d, _ = M.predict(m, va)
            preds.append(d)
            M.log(f"  DS-{nm} s{seed} n={len(idx):,} {time.time()-t0:.0f}s "
                  f"solo={bs(d):.1f}")
            del m
            torch.cuda.empty_cache()
        p = np.mean(preds, 0)
        line = f"ARM {nm:6s} n={int(len(tr_all)*frac):>9,}  solo={bs(p):7.1f}  "
        for wm in (0.15, 0.25):
            mix = (1 - wm) / 0.85 * b3 + wm * p
            line += f"w{wm:.2f}: {bs(mix):7.1f}  "
        M.log(line)
        np.save(DL / f"ds24_{nm}.npy", p)
    M.log("끝. 곡선이 가파르면 학습량 가설 지지 -> 배치는 2024 폴드보다 강하다")


if __name__ == "__main__":
    main()
