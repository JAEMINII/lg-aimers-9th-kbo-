# -*- coding: utf-8 -*-
"""1110 -> +20 탐색 1부: MS 팔들의 폴드 예측을 저장하며 새 팔 둘을 잰다.

51 = 1110 (+21) 로 전이율 0.85 확정. 2024 관문 +24 를 모으면 +20 이다.
w_ms 곡선(0.30 에서도 +19.5, 고원 미확인)이 최대 광맥 — 그 스윕을 하려면
감쇠 MS 의 폴드 예측 저장본이 필요하다. 겸사겸사 새 팔 둘:
    ms_s2     감쇠15 + 마지막시즌 1에폭 미세조정 (TabM 성공/DIN 실패, MS 미검)
    (base/decay15 는 저장 목적 재실행)

저장: msp_{vs}_{arm}_s{seed}.npy  (관문행 직접예측)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("AUDIT_MODE", "all")
os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
import features44 as F                                          # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
aux = M.recover_state(raw)

for VS in (2024, 2022):
    built = F.build(DATA, VS=VS)
    X44 = built["X44"].astype(np.float32)
    names = list(built["F44"])
    xnum, _, xcat, _ = A.audit_features(raw, X44, names)
    c4 = np.where(old & isf, 0., np.where(old & ~isf, 1.,
                  np.where(isf, 2., 3.))).astype(np.float32)[:, None]
    Xin = np.concatenate([X44, xnum, c4, xcat], 1)
    ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    c4i = X44.shape[1] + xnum.shape[1]
    cat_idx = ci + [c4i] + list(range(c4i + 1, Xin.shape[1]))
    m_tr = season < VS
    Xn, Xc, cards = G.prep(Xin, m_tr, cat_idx)
    G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
    tr = np.flatnonzero(m_tr)
    va = np.flatnonzero(season == VS)
    yv = y[va].astype(float)
    base_w = np.ones(len(y), np.float32)
    base_w[tr] = np.where(isf[tr] & old[tr], .1, 1.)

    def sc(p):
        return M.best_shift(p, yv)[0]

    for nm, dec, s2 in (("decay15", 1.5, 0), ("ms_s2", 1.5, 1)):
        ps = []
        for sd in M.SEEDS:
            w = base_w.copy()
            w[tr] = w[tr] * (dec ** (season[tr].astype(np.float32) - 2019.0))
            t0 = time.time()
            m = M.train_model(M.make_model(Xn.shape[1], cards, sd), tr, y, aux,
                              w, sd, f"P-{nm} VS{VS} s{sd}")
            if s2:
                s2i = tr[season[tr] == VS - 1]
                lr0, ep0 = M.LR, M.EPOCHS
                M.LR, M.EPOCHS = 2e-4, 1
                m = M.train_model(m, s2i, y, aux, w, sd + 1,
                                  f"P-{nm} VS{VS} s{sd} S2")
                M.LR, M.EPOCHS = lr0, ep0
            d, _ = M.predict(m, va)
            ps.append(d)
            np.save(DL + f"/msp_{VS}_{nm}_s{sd}.npy", d)
            print(f"  {nm} VS{VS} s{sd}  {time.time()-t0:.0f}s  "
                  f"단독 {sc(d):.1f}", flush=True)
            del m
            torch.cuda.empty_cache()
        print(f"ARM VS={VS} {nm:8s} 시드평균 단독 {sc(np.mean(ps,0)):8.1f}",
              flush=True)
