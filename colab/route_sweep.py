# -*- coding: utf-8 -*-
"""분기 비중을 3시드 평균 위에서 훑는다. 재학습 없음.

왜 다시 보나
    지인은 "regular/futures 구분 학습" 으로 +35 를 얻었다고 했다. 그게 각 행을
    해당 브랜치 모델로 그냥 보내는 것(비중 1.0)이면, 우리는 다른 걸 하고 있다.
    우리는 all 모델을 0.6 섞는다. 그 0.4 는 우리 CatBoost 에서 가져온 값이지
    TabM 에서 재서 정한 값이 아니다.

    예전에 훑어보긴 했는데 전부 단일 시드였다. 단일 시드는 24점씩 흔들리므로
    그 스윕들은 노이즈를 읽은 것이다. 3시드 평균으로 다시 본다.

비중 1.0 이면 all 패스를 아예 안 돌려도 된다. 253,507행에서 로컬 3.21분,
서버 1.6분이 그대로 남는다. 그 시간을 시드에 쓸 수 있다.
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


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


if __name__ == "__main__":
    S = {s: {br: np.load(os.path.join(OUT, f"br{G.VS}_base_s{s}_{br}.npy"))
             for br in BR} for s in SEEDS}
    A = {br: np.mean([S[s][br] for s in SEEDS], 0) for br in BR}
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    G.log("\n  분기비중   TabM단독   +CB/MLP혼합   all패스   서버분(3시드)")
    G.log("  " + "-" * 62)
    best = (-1e9, None)
    for wb in np.arange(0.0, 1.01, 0.1):
        r = np.where(is_f, wb * A["futures"] + (1 - wb) * A["all"],
                     wb * A["regular"] + (1 - wb) * A["all"])
        solo = sc(r)
        bl = sc(0.10 * CB + 0.30 * MLPF + 0.60 * r)
        need_all = wb < 1.0
        # 로컬 패스비용 all 3.21 / regular 2.89 / futures 0.39, 서버 배율 0.51
        cost = 0.51 * (3 * (3.21 if need_all else 0.0) + 3 * 2.89 + 3 * 0.39) + 0.90
        mark = "  <- 현재" if abs(wb - 0.4) < 1e-9 else ""
        G.log(f"    {wb:.1f}     {solo:7.1f}     {bl:7.1f}      "
              f"{'필요' if need_all else '불필요':6s}   {cost:5.2f}{mark}")
        if solo > best[0]:
            best = (solo, wb)
    G.log(f"\n  단독 기준 최적 분기비중 {best[1]:.1f}  ({best[0]:.1f})")

    G.log("\n  시드별로도 최적점이 같은 자리인지 (한 시드 결과에 끌려가면 안 된다)")
    for s in SEEDS:
        vals = []
        for wb in np.arange(0.0, 1.01, 0.1):
            r = np.where(is_f, wb * S[s]["futures"] + (1 - wb) * S[s]["all"],
                         wb * S[s]["regular"] + (1 - wb) * S[s]["all"])
            vals.append((sc(r), wb))
        b = max(vals)
        G.log(f"    seed {s:3d}   최적 {b[1]:.1f} ({b[0]:.1f})   "
              f"비중0.4 {vals[4][0]:.1f}   비중1.0 {vals[10][0]:.1f}")
