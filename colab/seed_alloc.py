# -*- coding: utf-8 -*-
"""브랜치별 시드 배분을 공정하게 비교한다. 재학습 없음.

앞선 seed_analyze.py 에는 함정이 있었다
    시드 순서를 (42, 1, 777) 로 고정해 놓고 "시드 2개" 를 42+1 로 쟀다.
    그런데 1 이 가장 좋은 시드(876.8), 777 이 가장 나쁜 시드(866.3)다.
    즉 "시드 2개" 가 사실은 "좋은 시드 두 개" 였다. 관문 점수로 시드를 고른 셈이고,
    flatMLP 설정을 관문 보고 골랐다가 30점 틀린 것과 같은 실수다.

여기서는 시드 k개를 쓸 때 **가능한 모든 조합의 평균**으로 잰다. 실제로 시드를
고를 방법이 없으므로 그 평균이 기대값이다. 최선·최악도 같이 찍어 폭을 본다.

시간 모델 (로컬 253,507행 실측 -> 서버 배율 0.51)
    all 패스 3.21분 / regular 2.89 / futures 0.39   (분기 패스 3.28 을 행수로 나눔)
    배율은 submit_jaemin_14 로 보정했다: 로컬 6.49분 -> 서버 3분20초
    그 밖(CatBoost + flatMLP + 피처) 서버 0.90분
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

# 로컬 분 단위 패스 비용. 서버 환산은 x0.51.
COST = {"all": 3.21, "regular": 2.89, "futures": 0.39}
SCALE, FIXED = 0.51, 0.90
LIMIT = 10.0

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


def route(P, wb=0.4):
    return np.where(is_f,
                    wb * P["futures"] + (1 - wb) * P["all"],
                    wb * P["regular"] + (1 - wb) * P["all"])


def evaluate(nseed, S, CB, MLPF):
    """브랜치별 시드 개수 nseed={"all":k,...}. 모든 시드조합의 평균/최소/최대."""
    choices = {br: list(itertools.combinations(SEEDS, k)) for br, k in nseed.items()}
    vals, blends = [], []
    for combo in itertools.product(*(choices[br] for br in BR)):
        P = {br: np.mean([S[s][br] for s in combo[i]], 0)
             for i, br in enumerate(BR)}
        r = route(P)
        vals.append(sc(r))
        blends.append(sc(0.10 * CB + 0.30 * MLPF + 0.60 * r))
    t = SCALE * sum(COST[br] * k for br, k in nseed.items()) + FIXED
    return (np.mean(vals), min(vals), max(vals),
            np.mean(blends), min(blends), max(blends), t, len(vals))


if __name__ == "__main__":
    S = {s: {br: np.load(os.path.join(OUT, f"br{G.VS}_base_s{s}_{br}.npy"))
             for br in BR} for s in SEEDS}
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    plans = []
    for a in (1, 2, 3):
        for r in (1, 2, 3):
            for f in (1, 2, 3):
                plans.append({"all": a, "regular": r, "futures": f})

    G.log("\n" + "=" * 82)
    G.log("  all/reg/fut   단독 평균 (최소~최대)      제출비중    서버분   조합수")
    G.log("=" * 82)
    rows = []
    for nseed in plans:
        m, lo, hi, bm, blo, bhi, t, n = evaluate(nseed, S, CB, MLPF)
        rows.append((nseed, m, lo, hi, bm, t, blo, bhi))
        flag = "" if t <= LIMIT else "  초과"
        G.log(f"  {nseed['all']}/{nseed['regular']}/{nseed['futures']}   "
              f"{m:7.1f} ({lo:6.1f}~{hi:6.1f})   "
              f"{bm:7.1f} ({blo:6.1f}~{bhi:6.1f})  폭 {bhi-blo:5.1f}   "
              f"{t:5.2f}{flag}")

    ok = [r for r in rows if r[5] <= 8.0]        # 여유 2분 두고 8분 이내
    ok.sort(key=lambda r: -r[4])
    G.log("\n  8분 이내에서 제출비중 기준 상위 5개")
    for nseed, m, lo, hi, bm, t in ok[:5]:
        G.log(f"    {nseed['all']}/{nseed['regular']}/{nseed['futures']}   "
              f"단독 {m:7.1f}   제출비중 {bm:7.1f}   {t:.2f}분")
    b = rows[0]
    G.log(f"\n  현 제출본 1/1/1   단독 {b[1]:.1f}   제출비중 {b[4]:.1f}   {b[5]:.2f}분")
    G.log("  (시드 하나뿐이라 조합이 1개, 평균=최소=최대)")
