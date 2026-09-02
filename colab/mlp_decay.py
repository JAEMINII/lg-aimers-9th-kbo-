# -*- coding: utf-8 -*-
"""flatMLP 에 시즌가중을 다시 건다. TabM 에서 배운 값으로.

왜 다시 하나
    flatMLP 은 지금 시즌가중이 없다. 예전에 재보고 "주면 나빠진다" 고 결론냈는데
    (관문 929.6 -> 921.1) 그 판단에 두 가지 문제가 있었다.
      1. 당시 근거가 "CatBoost 가 이미 시즌가중 2.0 을 쓰니 두 번 보정하면
         상관이 올라가고 다양성이 죽는다" 였다. 그런데 지금 혼합의 주력은
         TabM(0.60)이고 CatBoost 는 0.10 이다. 전제가 바뀌었다.
      2. 당시 시험한 decay 가 2~4.5 였는데 단일 실행 비교였고, 그때는 초기화
         시드가 안 잡혀 있어 편차가 컸다.

    TabM 은 시즌가중 3.5 에서 관문 단독 875 -> 911 (+36) 이었다. 곡선이 3~4 에서
    포화했고 8시드 짝비교로 t=6.60 이었다. 모델마다 최적점이 다르므로 flatMLP 에도
    직접 재본다.

판정
    혼합 기준으로 본다. flatMLP 은 단독 802.9 로 셋 중 가장 약한데 비중이 0.30 이다.
    단독이 올라도 혼합에 안 오면 의미가 없고, 그 반대도 마찬가지다.
    시드 3개로 스크리닝하고 유망하면 늘린다.
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)
EPOCHS = 6

import features44 as F                                          # noqa: E402
import mlp_gpu as M                                             # noqa: E402


def bss_full(p, y):
    r = y.mean()
    return 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)
    X, y, m_tr = d["X44"], d["y"], d["m_tr"]
    season = d["season"]
    gate = np.where(d["m_va"])[0]
    yv = y[gate].astype(np.float64)
    Xn, Xc, cards = M.prep(X, m_tr, d["cat_idx"])
    M.XN, M.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    M.YY = torch.from_numpy(y.astype(np.float32))
    tr_idx = np.where(m_tr)[0]

    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    T = np.load(os.path.join(OUT, "sw2conf_sw3.5.npy"))       # TabM 8시드 sw3.5

    # cb_gate.npy 와 sw2conf 는 이미 관문 행만 담고 있다(253,507).
    # gate 는 전체 프레임(147만)에 대한 인덱스라 다시 인덱싱하면 안 된다.
    def blend(m):
        return F.best_shift(0.10 * CB + 0.30 * m + 0.60 * T, yv)[0]

    print("\n  decay   단독      혼합(TabM sw3.5 와)   개별 시드")
    for decay in (0.0, 2.0, 3.5, 5.0):
        w = None if not decay else (decay ** (season[tr_idx].astype(np.float64) - 2019)
                                    ).astype(np.float32)
        ps, solos = [], []
        t0 = time.time()
        for sd in SEEDS:
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            mdl = M.MLP_PLR(Xn.shape[1], cards).to(M.DEV)
            M.train(mdl, tr_idx, EPOCHS, 1e-3, w=w, tag=f"d{decay} s{sd}")
            p = M.predict(mdl, gate)
            ps.append(p)
            solos.append(F.best_shift(p, yv)[0])
            del mdl
            torch.cuda.empty_cache()
        r = np.mean(ps, 0)
        b = blend(r)
        np.save(os.path.join(OUT, f"mlpdecay_{decay}.npy"), r)
        print(f"   {decay:4.1f}  {F.best_shift(r, yv)[0]:7.1f}   {b:9.1f}          "
              f"[{', '.join(f'{v:.0f}' for v in solos)}]   {time.time()-t0:.0f}s")

    print("\n  현재 제출본은 decay 0 (가중 없음) 이다.")
    print("  TabM 은 3.5 에서 +36 이었다. flatMLP 최적점은 다를 수 있다.")
