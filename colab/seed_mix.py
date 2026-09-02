# -*- coding: utf-8 -*-
"""브랜치마다 '다른' 시드를 쓰면 공짜로 나아지는가.

지금 제출본은 all/futures/regular 세 브랜치가 모두 시드 42 다. 그런데 브랜치별로
다른 시드를 배정해도 추론 비용은 똑같다 — 각 브랜치를 한 번씩 도는 건 동일하니까.
그게 이미 작은 앙상블이라면 시간을 한 푼도 안 쓰고 이득을 본다.

    같은 시드 3가지 (42/42/42, 1/1/1, 777/777/777)
    다른 시드 24가지

두 집단의 분포를 비교한다. 채점은 전체(R+F).
"""
import itertools
import os
import sys

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
SEEDS = (42, 1, 777)
BR = ("all", "futures", "regular")

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    S = {s: {br: np.load(os.path.join(OUT, f"br{G.VS}_base_s{s}_{br}.npy"))
             for br in BR} for s in SEEDS}
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    same, diff = [], []
    G.log("\n  배정 (all/fut/reg)      단독     제출비중")
    for combo in itertools.product(SEEDS, repeat=3):
        P = {br: S[combo[i]][br] for i, br in enumerate(BR)}
        r = np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                     0.4 * P["regular"] + 0.6 * P["all"])
        solo, bl = sc(r), sc(0.10 * CB + 0.30 * MLPF + 0.60 * r)
        (same if len(set(combo)) == 1 else diff).append((combo, solo, bl))
    for combo, solo, bl in same:
        mark = "  <- 현 제출본" if combo == (42, 42, 42) else ""
        G.log(f"    {str(combo):18s} {solo:7.1f}   {bl:7.1f}{mark}")
    G.log("")
    for combo, solo, bl in sorted(diff, key=lambda t: -t[2])[:6]:
        G.log(f"    {str(combo):18s} {solo:7.1f}   {bl:7.1f}")
    G.log("      ... (다른시드 24가지 중 상위 6개)")

    for name, grp in (("같은 시드 3가지", same), ("다른 시드 24가지", diff)):
        b = [x[2] for x in grp]
        G.log(f"\n  {name}   제출비중 평균 {np.mean(b):7.1f}   "
              f"{min(b):.1f} ~ {max(b):.1f}")
    G.log("\n  같은 비용이다. 다른 시드 쪽이 높으면 시간을 안 쓰고 얻는 이득이다.")
    G.log("  단, 조합을 관문 점수로 고르면 안 된다 — 그건 시드를 고르는 것과 같다.")
