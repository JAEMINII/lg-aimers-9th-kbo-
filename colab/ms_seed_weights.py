# -*- coding: utf-8 -*-
"""남은 두 최적화를 한 번에 잰다 — MS 시드 확장 + 3폴드 가중 재최적화.

① MS 시드 확장
    MS 시드 편차가 874~948 로 거대하다 (full 팔, datasize 실험). 시드평균(958)이
    최고 단독시드보다 +10 높았다. 편차 큰 모델일수록 시드평균 이득이 크다 —
    TabM 시드평균 실패(편차 작음)와 다른 상황. VS=2024 감사판 MS 를 시드 3개
    더 학습해 1~6시드 평균 곡선과 4원 혼합 효과를 잰다.

② 3폴드 가중 재최적화 (2023 구성원이 생겨 처음 가능)
    2023 에서 MS 단독 +433.9 (나머지 전부 음수권) — 체제 격변에 MS 가 가장
    강건하다. 세 폴드에서 leave-one-year-out 으로 **전역 가중**을 다시 푼다.
    가중은 게이팅과 달리 연도를 넘는 게 실증돼 있다 (교차 +6.7/+13.1 적중).
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, SC)
sys.path.insert(0, ROOT)
os.environ.setdefault("AUDIT_MODE", "all")
os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
DATA = os.environ.get("AIMERS_DATA", os.path.join(ROOT, "data"))
DL = os.path.join(SC, "_dl")
FIX = np.array([0.31, 0.13, 0.31, 0.25])
FOLDS = (2022, 2023, 2024)
NEW_SEEDS = (7, 123, 2024)

import features44 as F                                          # noqa: E402


def members(vs):
    if vs == 2024:
        cb = np.load(f"{DL}/ta2024_base.npy").mean(0)
        tab = 0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy") \
            + 0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")
    elif vs == 2022:
        cb = np.load(f"{DL}/cb50fixed_2022.npy").astype(np.float64)
        tab = np.load(f"{DL}/h2h_2022_friend_s42.npy").astype(np.float64)
    else:
        cb = np.load(f"{DL}/cb50_2023.npy").astype(np.float64)
        tab = np.load(f"{DL}/h2h_2023_friend_s42.npy").astype(np.float64)
    din = np.load(f"{DL}/dg_{vs}_DIN.npy").mean(0)
    ms = np.load(f"{DL}/msaudit/audit_all_vs{vs}_direct.npy").astype(np.float64)
    return cb, tab, din, ms


def ensure_extra_ms_seeds():
    """VS=2024 감사판 MS 를 시드 3개 더 학습한다 (multistate 스택 재사용)."""
    out = os.path.join(DL, "ms24_extra_seeds.npy")
    if os.path.exists(out):
        return np.load(out)
    import pandas as pd
    import multistate_softmax as M
    import multistate_auditfeat as A
    import tabm_gate_gpu as G
    from train_chan_3 import preprocess as PP
    raw = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                        encoding="utf-8-sig"))
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    isf = raw.game_type.astype(str).to_numpy() == "F"
    old = season <= 2022
    aux = M.recover_state(raw)
    built = F.build(DATA, VS=2024)
    X44 = built["X44"].astype(np.float32)
    names = list(built["F44"])
    xnum, _, xcat, _ = A.audit_features(raw, X44, names)
    c4 = np.where(old & isf, 0., np.where(old & ~isf, 1.,
                  np.where(isf, 2., 3.))).astype(np.float32)[:, None]
    Xin = np.concatenate([X44, xnum, c4, xcat], 1)
    ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    c4i = X44.shape[1] + xnum.shape[1]
    cat_idx = ci + [c4i] + list(range(c4i + 1, Xin.shape[1]))
    m_tr = season < 2024
    Xn, Xc, cards = G.prep(Xin, m_tr, cat_idx)
    G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
    tr = np.flatnonzero(m_tr)
    va = np.flatnonzero(season == 2024)
    w = np.ones(len(y), np.float32)
    w[tr] = np.where(isf[tr] & old[tr], 0.1, 1.0)
    ps = []
    for sd in NEW_SEEDS:
        t0 = time.time()
        m = M.train_model(M.make_model(Xn.shape[1], cards, sd), tr, y, aux, w,
                          sd, f"MSX s{sd}")
        d, _ = M.predict(m, va)
        ps.append(d)
        print(f"  MS 추가시드 {sd}  {time.time()-t0:.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()
    arr = np.asarray(ps)
    np.save(out, arr)
    return arr


if __name__ == "__main__":
    D = {vs: members(vs) for vs in FOLDS}
    Y = {}
    for vs in FOLDS:
        d = F.build(DATA, VS=vs)
        Y[vs] = d["y"].astype(np.float64)[np.where(d["season"] == vs)[0]]
    sc = lambda p, y: F.best_shift(p, y)[0]

    # ---------- ① MS 시드 곡선 (VS=2024)
    extra = ensure_extra_ms_seeds()
    cb, tab, din, ms3 = D[2024]
    y24 = Y[2024]
    # 기존 msaudit 은 3시드 평균 저장본. 추가 3시드와 합쳐 곡선을 만든다.
    print("\n  ① MS 시드 확장 (VS=2024)")
    print(f"    기존 3시드 평균  단독 {sc(ms3, y24):8.1f}")
    pool = [ms3]                      # 3시드 평균을 한 덩어리로
    for i in range(len(extra)):
        k = 3 + i + 1
        msk = (ms3 * 3 + extra[:i+1].sum(0)) / k
        b = 0.31*cb + 0.13*tab + 0.31*din + 0.25*msk
        print(f"    {k}시드 평균     단독 {sc(msk, y24):8.1f}   "
              f"4원혼합 {sc(b, y24):8.1f}  "
              f"(3시드 혼합 대비 {sc(b, y24)-sc(0.31*cb+0.13*tab+0.31*din+0.25*ms3, y24):+.1f})",
              flush=True)

    # ---------- ② 3폴드 LOYO 전역 가중
    print("\n  ② 전역 가중 LOYO (두 해에서 격자 -> 남은 해 채점)")

    def grid_two(folds):
        best, bw = -1e18, None
        for wc in np.arange(0.05, 0.60, 0.05):
            for wd in np.arange(0.05, 0.60, 0.05):
                for wm in np.arange(0.05, 0.60, 0.05):
                    wt = 1 - wc - wd - wm
                    if wt < 0.02:
                        continue
                    s = 0.0
                    for vs in folds:
                        cbf, tabf, dinf, msf = D[vs]
                        p = wc*cbf + wt*tabf + wd*dinf + wm*msf
                        s += F.best_shift(p, Y[vs])[0]
                    if s > best:
                        best, bw = s, (wc, wt, wd, wm)
        return bw

    for hold in FOLDS:
        tr_vs = [v for v in FOLDS if v != hold]
        w = grid_two(tr_vs)
        cbf, tabf, dinf, msf = D[hold]
        yt = Y[hold]
        ref = sc(cbf*FIX[0] + tabf*FIX[1] + dinf*FIX[2] + msf*FIX[3], yt)
        g = sc(w[0]*cbf + w[1]*tabf + w[2]*dinf + w[3]*msf, yt)
        print(f"    {tr_vs} -> {hold}   가중 (cb {w[0]:.2f}/tab {w[1]:.2f}/"
              f"din {w[2]:.2f}/ms {w[3]:.2f})   {g:8.1f}   고정 대비 {g-ref:+.1f}",
              flush=True)
