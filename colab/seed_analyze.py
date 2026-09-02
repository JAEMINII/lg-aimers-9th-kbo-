# -*- coding: utf-8 -*-
"""저장된 시드별 브랜치 예측으로 구성안을 비교한다. 재학습 없음.

추론 시간이 시드 수를 제한한다 (10분 제한, 시드당 2패스).
    전부 3시드   서버 10.8분   초과
    all만 3시드  서버  7.4분   가능
    전부 2시드   서버  7.4분   가능   <- 비용이 같다
어느 쪽이 나은지 여기서 정한다.

채점은 전체(R+F) 기준이다. 1군만 채점하면 퓨처스가 무너지는 걸 못 본다 —
ep4 를 골랐다가 980 을 받은 게 그 때문이다.
"""
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
ONE = ~is_f


def sc(p, msk=None):
    m = FULL if msk is None else msk
    return F.best_shift(p[m], yv[m])[0]


def route(P, wb=0.4):
    return np.where(is_f,
                    wb * P["futures"] + (1 - wb) * P["all"],
                    wb * P["regular"] + (1 - wb) * P["all"])


def avg(seeds, branches):
    """branches 에 든 브랜치만 여러 시드로 평균낸다. 나머지는 시드 42."""
    out = {}
    for br in BR:
        use = seeds if br in branches else (42,)
        out[br] = np.mean([S[s][br] for s in use], 0)
    return out


if __name__ == "__main__":
    S = {s: {br: np.load(os.path.join(OUT, f"br{G.VS}_base_s{s}_{br}.npy"))
             for br in BR} for s in SEEDS}
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    G.log("\n" + "=" * 76)
    G.log("  구성                    단독(전체/1군)      제출비중     최적TabM비중")
    G.log("=" * 76)

    def show(label, P, cost):
        r = route(P)
        blend = 0.10 * CB + 0.30 * MLPF + 0.60 * r
        best = max(((sc((1 - w) / 4 * CB + (1 - w) * 3 / 4 * MLPF + w * r), w)
                    for w in np.arange(0.30, 0.91, 0.05)), key=lambda t: t[0])
        G.log(f"  {label:22s} {sc(r):7.1f} / {sc(r, ONE):7.1f}   "
              f"{sc(blend):7.1f}    {best[1]:.2f} -> {best[0]:7.1f}   [{cost}]")
        return sc(r), sc(blend), best[0]

    base = show("시드1 (현 제출본)", avg((42,), BR), "서버 4.6분")
    show("시드2 전부", avg(SEEDS[:2], BR), "서버 7.4분")
    show("시드3 전부", avg(SEEDS, BR), "서버 10.8분 초과")
    show("시드2 all만", avg(SEEDS[:2], ("all",)), "서버 6.2분")
    show("시드3 all만", avg(SEEDS, ("all",)), "서버 7.4분")

    G.log("\n  시드별 단독 (전체채점) — 편차가 크면 평균의 값어치가 크다")
    for s in SEEDS:
        r = route(S[s])
        G.log(f"    seed {s:3d}   {sc(r):7.1f}   상관(seed42) "
              f"{np.corrcoef(r, route(S[42]))[0,1]:.4f}")

    G.log("\n  브랜치별로 시드가 얼마나 흔들리나 (전체채점, 그 브랜치만 3시드)")
    for br in BR:
        one = show(f"    {br} 만 3시드", avg(SEEDS, (br,)), "-")[0]
        G.log(f"      -> 시드1 대비 {one - base[0]:+.1f}")

    G.log("\n  주의: 관문을 리더보드로 환산하지 않는다. 계열을 섞은 구성에서")
    G.log("  그 환산이 30점 빗나간 적이 있다(1076 추정 -> 실제 1046).")
