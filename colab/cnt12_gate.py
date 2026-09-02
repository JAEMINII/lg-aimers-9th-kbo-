# -*- coding: utf-8 -*-
"""count12 를 TabM 범주형으로 추가해 관문에 올린다. c4_1gun.py 의 복제 변형이다.

근거
    1군 잔여신호 1위 축이 카운트다 (보정 상한 14.2, 여섯 시즌 부호 6/6).
    TabM 은 balls/strikes 를 숫자 두 개로만 받는다. 12칸이 하나의 범주로
    들어간 적이 없다. 3-2 (-0.027) 와 0-1 (+0.012) 은 두 축의 합이 아니다.
주의
    피처 추가는 관문이 여섯 번 틀린 부류다. 통과해도 전이 기대를 낮게 잡는다.
    다만 c4(체제 범주형 추가)는 통했다 — 범주형 한 칸 추가는 성격이 다를 수 있다.

팔   base  45열 (44 + c4)          <- 배치 현행
     cnt   46열 (44 + c4 + cnt12)  <- cnt12 를 범주형으로
채점  TabM 라우팅 단독 + 전체 혼합(재조정 가중 0.40/0.20/0.40, cb/din 은 저장 배열).
판정  두 폴드 같은 부호 & 3/3.
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
SEEDS = (42, 1, 777)
FOLDS = tuple(int(x) for x in os.environ.get("CG_FOLDS", "2022,2024").split(","))
OLD_F_MAX, OLD_W, LR1 = 2022, 0.1, 3e-3
EP2 = {"all": 1, "regular": 1, "futures": 4}

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


def blend_cb_din(VS):
    if VS == 2024:
        cb = (0.70 * np.load(DL + "/pcgpu2024_c12_cmh_10.npy")
              + 0.30 * np.load(DL + "/pcgpu2024_base44_10.npy")).astype(np.float64)
    else:
        cb = np.load(DL + "/cb50fixed_%d.npy" % VS).astype(np.float64)
    din = np.load(DL + "/dg_%d_DIN.npy" % VS).mean(0)
    return cb, din


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
    cnt_sorted = (tr_sorted["balls_before"].astype(int) * 3
                  + tr_sorted["strikes_before"].astype(int)).to_numpy()
    cnt12 = np.clip(cnt_sorted, 0, 11).astype(np.float32)[pos_map]

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
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        fut_idx, reg_idx = tr_idx[t_isf], tr_idx[~t_isf]
        w_all = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        w_f = np.where(old[fut_idx], OLD_W, 1.0).astype(np.float64)
        G.log("\n  VS=%d  all %d  regular %d  futures %d"
              % (VS, len(tr_idx), len(reg_idx), len(fut_idx)))

        P = {}
        for tag, Xin, cidx in (
                ("base", np.concatenate([Xfr, c4], 1),
                 ci + [Xfr.shape[1]]),
                ("cnt", np.concatenate([Xfr, c4, cnt12[:, None]], 1),
                 ci + [Xfr.shape[1], Xfr.shape[1] + 1])):
            Xn, Xc, cards = G.prep(Xin, m_tr, cidx)
            G.Xn, G.cards, G.m_tr = Xn, cards, m_tr
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            G.log("    [%s] 수치 %d열  범주 %d개  카디널리티 %s"
                  % (tag, Xn.shape[1], len(cards), cards))
            t0 = time.time()
            for br, idxb, wb in (("all", tr_idx, w_all),
                                 ("regular", reg_idx, None),
                                 ("futures", fut_idx, w_f)):
                P[(tag, br)] = [fit(idxb, EP2[br], wb, season, VS, gate, sd)
                                for sd in SEEDS]
            G.log("    [%s] 끝 %.0fs" % (tag, time.time() - t0))

        def sc(p, m=None):
            mm = np.ones(len(yv), bool) if m is None else m
            return F.best_shift(p[mm], yv[mm])[0]

        cb, din = blend_cb_din(VS)
        print("\n" + "=" * 92)
        print("  count12 범주형 — VS=%d   판정선 두 폴드 & 3/3" % VS)
        print("=" * 92)
        blends = {}
        for tag in ("base", "cnt"):
            blends[tag] = [
                np.where(isf_g,
                         0.6 * P[(tag, "all")][i] + 0.4 * P[(tag, "futures")][i],
                         0.6 * P[(tag, "all")][i] + 0.4 * P[(tag, "regular")][i])
                for i in range(len(SEEDS))]
            np.save(DL + "/cg_%d_%s.npy" % (VS, tag),
                    np.asarray(blends[tag]))
        for tag in ("base", "cnt"):
            ps = blends[tag]
            p = np.mean(ps, 0)
            line = ("  %-5s TabM단독 %8.1f  1군 %8.1f  퓨처스 %8.1f"
                    % (tag, sc(p), sc(p, ~isf_g), sc(p, isf_g)))
            if tag == "cnt":
                dd = [sc(a) - sc(b) for a, b in zip(ps, blends["base"])]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += ("   차이 %+6.1f+-%4.1f t=%5.2f %d/3"
                         % (mu, se, mu / max(se, 1e-9),
                            sum(1 for v in dd if v > 0)))
            print(line)
        b_ref = [F.best_shift(0.40 * cb + 0.20 * q + 0.40 * din, yv)[0]
                 for q in blends["base"]]
        b_cnt = [F.best_shift(0.40 * cb + 0.20 * q + 0.40 * din, yv)[0]
                 for q in blends["cnt"]]
        dd = [a - b for a, b in zip(b_cnt, b_ref)]
        print("  전체혼합(재조정 .40/.20/.40)  차이 %+6.1f   시드별 %s  %d/3"
              % (float(np.mean(dd)), " ".join("%+.1f" % v for v in dd),
                 sum(1 for v in dd if v > 0)), flush=True)
