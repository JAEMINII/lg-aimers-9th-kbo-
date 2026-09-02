# -*- coding: utf-8 -*-
"""LUPI 학생 관문 — 교사(현재 투구 물리량)의 OOF 소프트 타깃으로 TabM 을 증류한다.

근거
    교사 상한 (VS=2024, 매칭행 CatBoost): base 869.7 -> phys 1167.8 (+298)
    FAQ 공식 허용 (2026.08.20 DACON.GM: 교사-학생 증류, 보조헤드, 부분 매칭 모두 가능)

증류 = 표적 치환
    학습 손실(0.5BCE+0.5Brier)이 표적에 선형이라
        λ·L(y) + (1-λ)·L(p_t)  =  L(λ·y + (1-λ)·p_t) + 상수
    G.train 을 그대로 두고 G.YY 만 혼합 표적으로 바꾼다. 배치와 같은 절차 보장.
    소프트 타깃이 없는 행(미매칭·퓨처스)은 y 그대로 (부분 매칭 허용 확인됨).

팔    base   λ=1.0 (현행)     kd07 λ=0.7     kd05 λ=0.5     kd03 λ=0.3
관문  VS=2022 / 2024, 3시드, 브랜치 라우팅 + Stage2 — 배치 TabM 과 동일 구성
채점  TabM 라우팅 단독 + 47식 4원 혼합에서 tab 자리 교체
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
for p in (SC, "/root/aimers", os.path.dirname(SC)):
    if p not in sys.path:
        sys.path.insert(0, p)
DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
SEEDS = tuple(int(x) for x in os.environ.get("LS_SEEDS", "42,1,777").split(","))
FOLDS = tuple(int(x) for x in os.environ.get("LS_FOLDS", "2024,2022").split(","))
SOFT_TAG = os.environ.get("LUPI_SOFT_TAG", "")
OLD_F_MAX, OLD_W, LR1 = 2022, 0.1, 3e-3
EP2 = {"all": 1, "regular": 1, "futures": 4}
_ARM_LAM = {"base": 1.0, "kd085": .85, "kd07": .7, "kd05": .5, "kd03": .3}
ARM_NAMES = tuple(x for x in os.environ.get("LS_ARMS", "base,kd07,kd05,kd03").split(",") if x)
if any(x not in _ARM_LAM for x in ARM_NAMES):
    raise ValueError(f"unknown LS_ARMS={ARM_NAMES}; choices={sorted(_ARM_LAM)}")
ARMS = tuple((x, _ARM_LAM[x]) for x in ARM_NAMES)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def fit(idx, ep2, w, season, VS, gate, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    G.train(m, idx, 2, LR1, w=w, seed=seed, tag="VS%d s%d" % (VS, seed))
    s2 = idx[season[idx] == VS - 1]
    pr = G.stage2_params(m)
    for e in range(ep2):
        G.train(m, s2, 1, 2e-4, params=pr, seed=seed + e,
                tag="VS%d s%d S2e%d" % (VS, seed, e + 1))
    p = G.predict(m, gate)
    del m
    torch.cuda.empty_cache()
    return p


def blend_members(VS):
    if VS == 2024:
        cb = (0.70 * np.load(DL + "/pcgpu2024_c12_cmh_10.npy")
              + 0.30 * np.load(DL + "/pcgpu2024_base44_10.npy")).astype(np.float64)
        ms = np.load(DL + "/ms24_audit_direct.npy").astype(np.float64)
    else:
        cb = np.load(DL + "/cb50fixed_%d.npy" % VS).astype(np.float64)
        ms = np.load(DL + "/ms22_audit_direct.npy").astype(np.float64)
    din = np.load(DL + "/dg_%d_DIN.npy" % VS).mean(0)
    return cb, din, ms


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

    for VS in FOLDS:
        stem = ("lupi_soft_%s_%d" % (SOFT_TAG, VS)) if SOFT_TAG else ("lupi_soft_%d" % VS)
        soft = pd.read_csv(DL + "/" + stem + ".csv.gz",
                           encoding="utf-8-sig")
        pt = pd.Series(soft["p_t"].to_numpy(),
                       index=soft["row_id"].to_numpy()).reindex(rid).to_numpy()
        have = np.isfinite(pt)
        G.log("\n  VS=%d  소프트 타깃 %d행 (학습구간의 %.1f%%)"
              % (VS, int(have.sum()),
                 have[season < VS].mean() * 100))

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

        P = {}
        for nm, lam in ARMS:
            tmix = np.where(have, lam * y + (1.0 - lam) * np.nan_to_num(pt),
                            y).astype(np.float32)
            G.YY = torch.from_numpy(tmix)
            t0 = time.time()
            for br, idxb, wb in (("all", tr_idx, w_all),
                                 ("regular", reg_idx, None),
                                 ("futures", fut_idx, w_f)):
                P[(nm, br)] = [fit(idxb, EP2[br], wb, season, VS, gate, sd)
                               for sd in SEEDS]
            G.log("    %s (λ=%.1f) 끝 %.0fs" % (nm, lam, time.time() - t0))
        G.YY = torch.from_numpy(y.astype(np.float32))

        def sc(p, m=None):
            mm = np.ones(len(yv), bool) if m is None else m
            return F.best_shift(p[mm], yv[mm])[0]

        cb, din, ms = blend_members(VS)
        print("\n" + "=" * 96)
        print("  LUPI 학생 (TabM 증류) — VS=%d   판정선 두 폴드 & 3/3" % VS)
        print("=" * 96)
        BL = {}
        for nm, lam in ARMS:
            BL[nm] = [np.where(isf_g,
                               0.6 * P[(nm, "all")][i] + 0.4 * P[(nm, "futures")][i],
                               0.6 * P[(nm, "all")][i] + 0.4 * P[(nm, "regular")][i])
                      for i in range(len(SEEDS))]
            np.save(DL + "/ls_%d_%s.npy" % (VS, nm), np.asarray(BL[nm]))
        ref = BL["base"]
        ref_mix = [F.best_shift(0.31 * cb + 0.13 * q + 0.31 * din + 0.25 * ms,
                                yv)[0] for q in ref]
        for nm, lam in ARMS:
            ps = BL[nm]
            p = np.mean(ps, 0)
            line = ("  %-5s λ=%.1f  단독 %8.1f  1군 %8.1f  퓨처스 %8.1f"
                    % (nm, lam, sc(p), sc(p, ~isf_g), sc(p, isf_g)))
            if nm != "base":
                dd = [sc(a) - sc(b) for a, b in zip(ps, ref)]
                mix = [F.best_shift(0.31 * cb + 0.13 * q + 0.31 * din + 0.25 * ms,
                                    yv)[0] for q in ps]
                dm = [a - b for a, b in zip(mix, ref_mix)]
                line += ("   단독차 %+6.1f %d/3   4원혼합차 %+6.1f %d/3"
                         % (float(np.mean(dd)), sum(1 for v in dd if v > 0),
                            float(np.mean(dm)), sum(1 for v in dm if v > 0)))
            print(line, flush=True)
        print("  (4원혼합 = 0.31cb + 0.13tab + 0.31din + 0.25ms, tab 자리를 학생으로 교체)")
