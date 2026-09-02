# -*- coding: utf-8 -*-
"""고정 시프트 -0.0145 가 지금 혼합에 맞는지 본다.

의심하는 것
    관문은 후보마다 '최적 시프트'에서 점수를 매긴다. 그런데 제출은 -0.0145 고정이다.
    그 값은 submit_12(CatBoost 0.8 + 옛MLP 0.2)에 맞춰 구한 것인데, 지금 혼합은
    구성이 완전히 다르다. 혼합이 바뀌면 예측 평균이 움직이고, 맞는 시프트도 움직인다.

    관문은 TabM 단독 875.7 -> 혼합 928.6 (+53) 이라고 하는데
    리더보드는 TabM 단독 1041 -> 혼합 1046 (+5) 다. 10배 차이다.
    최적 시프트로 잰 이득을 고정 시프트로 실현하지 못하고 있다면 여기가 원인이다.

각 구성에 대해
    최적 시프트에서의 점수 / 최적 시프트 값 / -0.0145 에서의 점수 / 그 손해
를 찍는다. 채점은 전체(R+F).
"""
import os
import sys

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
SEEDS = (42, 1, 777)
BR = ("all", "futures", "regular")
SHIPPED = -0.0145

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def at(p, c):
    """시프트 c 를 걸었을 때의 점수."""
    return F.bss(F.shift(p[FULL], c), yv[FULL])


def best(p):
    """최적 시프트와 그 점수."""
    s, c = F.best_shift(p[FULL], yv[FULL])
    return s, c


if __name__ == "__main__":
    S = {s: {br: np.load(os.path.join(OUT, f"br{G.VS}_base_s{s}_{br}.npy"))
             for br in BR} for s in SEEDS}
    A = {br: np.mean([S[s][br] for s in SEEDS], 0) for br in BR}
    tabm = np.where(is_f, 0.4 * A["futures"] + 0.6 * A["all"],
                    0.4 * A["regular"] + 0.6 * A["all"])
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    cases = {
        "CatBoost 단독": CB,
        "flatMLP 단독": MLPF,
        "TabM 단독": tabm,
        "혼합 .10/.30/.60": 0.10 * CB + 0.30 * MLPF + 0.60 * tabm,
        "혼합 .00/.00/1.0": tabm,
        "혼합 .20/.30/.50": 0.20 * CB + 0.30 * MLPF + 0.50 * tabm,
    }
    G.log("\n  구성                최적시프트 점수 / 시프트값 | -0.0145 점수 | 손해 | 예측평균")
    G.log("  " + "-" * 82)
    for name, p in cases.items():
        s, c = best(p)
        f = at(p, SHIPPED)
        G.log(f"  {name:18s} {s:7.1f} / {c:+.4f}   |  {f:7.1f}  | "
              f"{f-s:6.1f} | {p[FULL].mean():.4f}")
    G.log(f"\n  실제 성공률 {yv[FULL].mean():.4f}")
    G.log("\n  손해가 크면 고정 시프트가 원인이다. 관문에서 잰 이득을")
    G.log("  제출에서 실현하지 못하고 있다는 뜻이 된다.")
