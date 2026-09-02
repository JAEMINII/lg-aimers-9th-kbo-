# -*- coding: utf-8 -*-
"""flatMLP 이득 중 선택 편향이 얼마인지 잰다. 제출 없이.

논리
    flatMLP 설정은 10여 개 후보 중 관문 최고로 골랐다. 고른 것의 관문 점수는
    참값보다 높다. 그런데 리더보드는 그 편향을 안 물려받는다.

    편향 크기는 '안 고른 후보들' 로 잰다. 그들도 비슷한 이득을 준다면 편향은
    작고, 고른 것만 튀면 편향이 크다. 후보들의 평균이 편향 없는 추정치다.

각 후보를 TabM(3시드 평균)에 0.30 으로 얹어서 잰다. 나머지는 CatBoost 0.10.
"""
import glob
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

    base_opt = F.best_shift((0.10 * CB + 0.90 * T)[FULL], yv[FULL])[0]
    G.log(f"\n  MLP 없이 (CB .10 / T .90)   최적시프트 {base_opt:.1f}")
    G.log("\n  MLP 후보                단독    +0.30 넣으면   이득    예측평균")
    G.log("  " + "-" * 64)
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "mlpnew_*.npy"))):
        name = os.path.basename(p)[7:-4]
        m = np.load(p)
        solo = F.best_shift(m[FULL], yv[FULL])[0]
        q = 0.10 * CB + 0.30 * m + 0.60 * T
        s = F.best_shift(q[FULL], yv[FULL])[0]
        chosen = "  <- 채택한 것" if name in ("flat_s3",) else ""
        G.log(f"  {name:18s} {solo:8.1f}   {s:9.1f}   {s-base_opt:+6.1f}   "
              f"{m[FULL].mean():.4f}{chosen}")
        rows.append((name, s - base_opt))
    v = [r[1] for r in rows if r[0] != "flat_s3"]
    ch = [r[1] for r in rows if r[0] == "flat_s3"]
    G.log(f"\n  안 고른 후보 {len(v)}개 평균 이득 {np.mean(v):+.1f}  "
          f"({min(v):+.1f} ~ {max(v):+.1f})")
    if ch:
        G.log(f"  채택한 flat_s3 이득 {ch[0]:+.1f}")
        G.log(f"  선택 편향 추정치 {ch[0]-np.mean(v):+.1f}")
    G.log("\n  실제 성공률 " + f"{yv[FULL].mean():.4f}")
