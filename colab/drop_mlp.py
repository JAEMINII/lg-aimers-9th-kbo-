# -*- coding: utf-8 -*-
"""flatMLP 를 뺀 구성을 관문에서 잰다.

왜 이걸 보나
    관문은 CatBoost+flatMLP 가 TabM 에 +53 을 얹는다고 하는데 리더보드는 +5 였다.
    남은 유력 후보 중 하나가 flatMLP 다.
      - 설정을 관문 최고로 골랐다(10개 후보 중). 선택 편향이 있다.
      - 단독 관문 802.9 로 셋 중 가장 약한데 비중이 0.30 이다.
      - 예측 평균이 실제보다 1.7%p 높다. 관문의 최적시프트가 그걸 공짜로 고쳐준다.
        제출은 고정 시프트라 못 고친다 — 단독 802.9 가 -0.0145 에서 728.6 이 된다.

    flatMLP 가 관문 착시였다면 빼도 리더보드가 안 떨어진다. 한 가지만 바꾸는
    구성이라 다음 제출 결과가 읽힌다.

시프트를 두 가지로 찍는다
    최적    관문이 후보를 비교할 때 쓰는 값. 배치에서는 알 수 없다.
    -0.0145 실제로 제출에 건 값.
    둘의 차이가 크면 그 구성은 '관문에서만 좋은' 것이다.
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

if __name__ == "__main__":
    S = {s: {br: np.load(os.path.join(OUT, f"br{G.VS}_base_s{s}_{br}.npy"))
             for br in BR} for s in SEEDS}
    A = {br: np.mean([S[s][br] for s in SEEDS], 0) for br in BR}
    T = np.where(is_f, 0.4 * A["futures"] + 0.6 * A["all"],
                 0.4 * A["regular"] + 0.6 * A["all"])
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    M = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    def row(label, p):
        s, c = F.best_shift(p[FULL], yv[FULL])
        f = F.bss(F.shift(p[FULL], SHIPPED), yv[FULL])
        G.log(f"  {label:24s} {s:7.1f} / {c:+.4f}   {f:7.1f}   {f-s:6.1f}")

    G.log("\n  구성                      최적시프트 점수/값   -0.0145   차이")
    G.log("  " + "-" * 66)
    row("TabM 단독", T)
    row("CB .10 / MLP .30 / T .60", 0.10 * CB + 0.30 * M + 0.60 * T)
    G.log("")
    for w in (0.10, 0.20, 0.30, 0.40, 0.50):
        row(f"CB {w:.2f} / T {1-w:.2f}  (MLP 없음)", w * CB + (1 - w) * T)
    G.log("")
    for w in (0.10, 0.20, 0.30):
        row(f"MLP {w:.2f} / T {1-w:.2f}  (CB 없음)", w * M + (1 - w) * T)
    G.log("\n  '차이' 가 크면 관문의 최적시프트에 기대는 구성이다.")
    G.log("  배치에서는 그 시프트를 모르므로 그만큼 못 받는다.")
