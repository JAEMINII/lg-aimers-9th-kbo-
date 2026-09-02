# -*- coding: utf-8 -*-
"""혼합 점수가 TabM 시드 뽑기에 얼마나 흔들리는지 잰다.

왜 필요한가
    submit_16(시즌가중 3.5, 시드 2개)이 1044 로, submit_15 의 1046 보다 2점 낮았다.
    관문은 +3.9 를 예상했다. 이 -2 가 노이즈인지 신호인지 가려야 다음 수가 정해진다.

    TabM 단독의 시드 편차(짝지은 차이 표준편차 22.3)는 알고 있다. 그런데 혼합에서
    얼마나 남는지는 모른다. 혼합은 CatBoost·flatMLP 가 고정이라 TabM 의 흔들림을
    상당히 흡수한다. 그 흡수율을 직접 잰다.

방법
    8개 시드의 TabM 예측이 있다. 거기서 2개씩 뽑는 모든 조합(28가지)의 혼합
    점수를 계산해 산포를 본다. 제출본이 시드 2개였으므로 그 조건과 같다.
"""
import itertools
import os
import sys

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
SEEDS = (42, 1, 777, 2, 3, 5, 11, 23)
BR = ("all", "futures", "regular")

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402
import torch                                                    # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    DATA = "/workspace/aimers/data"
    d = F.build(DATA, VS=2024)
    Xn, Xc, cards = G.prep(d["X44"], G.m_tr, d["cat_idx"])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    # 8시드 확정 실험에서 쓴 것과 같은 설정으로 시드별 예측을 만든다.
    P0, P2 = {}, {}
    for sd in SEEDS:
        P0[sd] = one(sd)                 # 가중 없음
        P2[sd] = one(sd, decay=3.5)      # 시즌가중 3.5
        G.log(f"  seed {sd} 완료")

    def blend(P, subset):
        r = np.mean([P[s] for s in subset], 0)
        return sc(0.10 * CB + 0.30 * MLPF + 0.60 * r)

    for tag, P in (("가중없음", P0), ("가중3.5", P2)):
        for k in (1, 2, 8):
            combos = list(itertools.combinations(SEEDS, k))
            if k == 8:
                combos = [tuple(SEEDS)]
            vals = [blend(P, c) for c in combos]
            sd_ = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            G.log(f"  {tag:8s} 시드 {k}개  혼합 평균 {np.mean(vals):7.1f}  "
                  f"표준편차 {sd_:5.2f}  범위 {min(vals):.1f}~{max(vals):.1f}  "
                  f"(조합 {len(vals)})")

    # 시드 2개 조건에서, 가중 유무의 차이를 짝지어 본다
    G.log("")
    pairs = list(itertools.combinations(SEEDS, 2))
    dif = [blend(P2, c) - blend(P0, c) for c in pairs]
    m, s = float(np.mean(dif)), float(np.std(dif, ddof=1))
    G.log(f"  시드 2개 조건 혼합 이득  평균 {m:+.2f}  표준편차 {s:.2f}  "
          f"양수 {sum(1 for x in dif if x > 0)}/{len(dif)}")
    G.log(f"  범위 {min(dif):+.2f} ~ {max(dif):+.2f}")
    G.log("")
    G.log("  실측은 1046 -> 1044 로 -2 였다. 위 산포와 견줘 판단한다.")
    G.log("  단, 제출본은 전처리와 시드 수도 같이 바뀌어 완전한 대조가 아니다.")
