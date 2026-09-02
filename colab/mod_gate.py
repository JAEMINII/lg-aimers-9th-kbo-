# -*- coding: utf-8 -*-
"""상황 변조 2열이 현재 최선 구성 위에서 점수를 올리는지 짝비교로 잰다.

무엇을 더하나
    tm_mod_speed   그 투수가 2스트라이크에서 구속을 얼마나 바꾸는가
    tm_mod_ivb     상하무브를 얼마나 바꾸는가
    시즌 Y 행에는 시즌 < Y 누적만 쓴다. 관문 2024 커버리지 72.2%.

왜 이 둘인가
    투구 단위로 붙인 121만 행에서 근거를 확인했다.
      기존 성공률을 통제해도 물리량이 남는다 (구속 편상관 +0.046, 상하무브 +0.054)
      변조 성향이 시즌을 넘어 이어진다 (지속성 +0.67 / +0.71)
      변조가 제구 성공과 관계있다 (+0.22 / +0.23, 전체수준 통제 후에도 유지)
    어제 넣은 물리량 평균 13개가 실패한 이유도 같이 설명된다 — 그건 성공률과
    중복이었다. 변조는 44열 어디에도 없는 축이다.

기준선은 시즌가중 3.5 다. 지금 최선 구성 위에서 더하는지가 문제이기 때문이다.

시드 6개로 짝지어 잰다. 짝지은 차이의 표준편차가 22.3 이라
    n=3  ±25.7    n=6  ±18.2    n=8  ±15.8
3개로는 결론이 안 난다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777, 2, 3, 5)
DECAY = 3.5

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    d = F.build(DATA, VS=2024)
    base = d["X44"]
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["pitcher_id", "season"])
    mod = pd.read_csv(os.path.join(SC, "mod_pitcher_season.csv"))
    m = raw.merge(mod, on=["pitcher_id", "season"], how="left")
    add = m[["tm_mod_speed", "tm_mod_ivb"]].to_numpy(dtype=np.float32)
    G.log(f"\n  추가 2열   전체 커버리지 {np.isfinite(add[:, 0]).mean()*100:.1f}%   "
          f"관문 커버리지 {np.isfinite(add[gate, 0]).mean()*100:.1f}%")
    X2 = np.concatenate([base, add], 1)

    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    def run(X, seed):
        Xn, Xc, cards = G.prep(X, G.m_tr, d["cat_idx"])
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        return one(seed, decay=DECAY)

    P0, P2, dif = {}, {}, []
    G.log("")
    G.log("  시드    기준(sw3.5)   +변조2열      차이")
    for sd in SEEDS:
        t0 = time.time()
        P0[sd] = run(base, sd)
        P2[sd] = run(X2, sd)
        a, b = sc(P0[sd]), sc(P2[sd])
        dif.append(b - a)
        G.log(f"  {sd:4d}   {a:9.1f}   {b:9.1f}   {b - a:+8.1f}   "
              f"{time.time() - t0:.0f}s")

    mm, s = float(np.mean(dif)), float(np.std(dif, ddof=1))
    se = s / np.sqrt(len(dif))
    G.log("")
    G.log(f"  짝지은 차이  평균 {mm:+.1f}  표준편차 {s:.1f}  표준오차 {se:.1f}  "
          f"양수 {sum(1 for x in dif if x > 0)}/{len(dif)}   t={mm/se:.2f}")
    G.log(f"  판정: {'확정' if abs(mm) > 2 * se else '미확정'}")
    for tag, P in (("기준", P0), ("+변조", P2)):
        r = np.mean([P[k] for k in SEEDS], 0)
        np.save(os.path.join(OUT, f"mod_{tag.strip('+')}.npy"), r)
        G.log(f"  {tag:6s} 6시드평균 단독 {sc(r):7.1f}   제출비중 "
              f"{sc(0.10 * CB + 0.30 * MLPF + 0.60 * r):7.1f}")
