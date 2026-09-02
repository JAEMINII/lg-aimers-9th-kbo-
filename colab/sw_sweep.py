# -*- coding: utf-8 -*-
"""시즌가중을 TabM 에서 훑고, 릴리스 흔들림과 겹치는지 본다.

발견
    시즌가중 2.0 을 TabM 에 걸었더니 단독 867.5 -> 895.5 (+28.0).
    오늘 잰 모든 것을 합친 것보다 크다. 개별 시드도 짝지어 +7/+41/+40 전부 양수.

    이 장치는 우리 CatBoost 를 996 -> 1021 로 올린 것과 같다. 그런데 TabM 에는
    한 번도 안 걸어봤다 — 지인 설정에 없어서 그대로 물려받았다.

    관문을 믿을 근거도 있다. 계열도 학습량도 안 바뀌고 손실 가중만 바뀐다.
    그리고 다른 계열에서 관문 +22 -> 리더보드 +25 로 전이가 확인된 장치다.

여기서 보는 것
    decay 값이 2.0 이 최적인가 (CatBoost 에서 나온 값을 그대로 쓰고 있다)
    릴리스 흔들림(+11.6)과 더해지는가, 아니면 같은 일을 하는가
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
SEEDS = (42, 1, 777)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from hp_sweep import one                                        # noqa: E402
from trackman_gate2 import trackman_matrix                      # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)

if __name__ == "__main__":
    d = F.build(DATA, VS=int(os.environ.get("VS", 2024)))
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["pitcher_id", "season"])
    M, _ = trackman_matrix(raw.season.to_numpy(), raw.pitcher_id.to_numpy(),
                           cols={"tm_relh_sd", "tm_rels_sd"})
    X2 = np.concatenate([d["X44"], M], 1)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    def prep_into(X):
        Xn, Xc, cards = G.prep(X, G.m_tr, d["cat_idx"])
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    G.log("\n" + "=" * 74)
    G.log("  구성              시드평균 단독   제출비중   개별 시드")
    G.log("=" * 74)
    plan = [("기준", d["X44"], {}),
            ("sw1.5", d["X44"], dict(decay=1.5)),
            ("sw2.0", d["X44"], dict(decay=2.0)),
            ("sw3.0", d["X44"], dict(decay=3.0)),
            ("sw4.0", d["X44"], dict(decay=4.0)),
            ("sd2", X2, {}),
            ("sw2.0+sd2", X2, dict(decay=2.0))]
    res = {}
    for name, X, kw in plan:
        prep_into(X)
        t0 = time.time()
        ps = [one(sd, **kw) for sd in SEEDS]
        solos = [F.best_shift(p[FULL], yv[FULL])[0] for p in ps]
        r = np.mean(ps, 0)
        solo = F.best_shift(r[FULL], yv[FULL])[0]
        bl = F.best_shift((0.10 * CB + 0.30 * MLPF + 0.60 * r)[FULL], yv[FULL])[0]
        np.save(os.path.join(OUT, f"sw_{name}.npy"), r)
        res[name] = (solo, bl, r)
        b = res["기준"][0]
        G.log(f"  {name:16s} {solo:7.1f}      {bl:7.1f}   "
              f"[{', '.join(f'{v:.0f}' for v in solos)}]   {time.time()-t0:.0f}s"
              f"{'' if name == '기준' else f'   기준대비 {solo-b:+6.1f}'}")

    G.log("\n  더해지는가")
    b = res["기준"][0]
    if "sw2.0" in res and "sd2" in res and "sw2.0+sd2" in res:
        a1, a2 = res["sw2.0"][0] - b, res["sd2"][0] - b
        both = res["sw2.0+sd2"][0] - b
        G.log(f"    시즌가중 {a1:+.1f}  +  릴리스흔들림 {a2:+.1f}  =  {a1+a2:+.1f} (따로 더하면)")
        G.log(f"    같이 걸면 {both:+.1f}   -> {'겹친다' if both < a1 + a2 - 5 else '더해진다'}")

    G.log("\n  최고 구성에서 TabM 비중을 올리면 (방향만 본다)")
    best = max((v[0], k) for k, v in res.items())[1]
    r = res[best][2]
    G.log(f"    [{best}]")
    for w in (0.6, 0.7, 0.8, 0.9, 1.0):
        rest = 1 - w
        q = (rest / 4) * CB + (rest * 3 / 4) * MLPF + w * r
        G.log(f"      TabM {w:.1f}   {F.best_shift(q[FULL], yv[FULL])[0]:7.1f}")
