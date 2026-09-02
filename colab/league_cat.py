# -*- coding: utf-8 -*-
"""리그 이동 피처를 CatBoost 다년 백테스트로 1차 확인한다. 로컬 CPU, GPU 불필요.

편상관 통과분 (44열 통제 후)
                전체      퓨처스행
    lg_f_share  -0.0174   -0.0325
    lg_run      -0.0059   -0.0315
    lg_1gun_n   +0.0107   +0.0281
    lg_switches +0.0043   +0.0193
    lg_f_n      -0.0019   -0.0148

무엇을 재나
    base        44열
    +league     44열 + 통과분 5개
    +f_share    44열 + lg_f_share 하나만 (가장 강한 것)

    폴드 셋 2022 / 2023 / 2024. 시드 3개, 같은 시드끼리 짝비교.
    전체(R+F)와 퓨처스 구간을 따로 찍는다. 이 축은 퓨처스에서 먼저 보일 것이다.
    구간 점수는 그 구간 분모로 잰다.

한계
    CatBoost 한 계열의 판정이다. TabM 이 쓰는 걸 트리가 못 쓸 수 있고 그 반대도
    있다. 그래서 여기서 양수면 GPU 로 TabM 관문을 돌리고, 음수여도 바로 버리지는
    않는다 — 예전에 CatBoost 기각이 모델에 묶인 판정이었던 적이 있다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
SEEDS = (42, 1234, 2025)
DECAY = 2.0
HP = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, thread_count=6)
LG = ["lg_f_share", "lg_run", "lg_switches", "lg_1gun_n", "lg_f_n"]

import features44 as F                                          # noqa: E402
from screen_league import build                                 # noqa: E402


def run(X, y, m_tr, m_va, w, seed):
    m = CatBoostClassifier(random_seed=seed, **HP)
    m.fit(X[m_tr], y[m_tr].astype(int), sample_weight=w)
    return m.predict_proba(X[m_va])[:, 1].astype(np.float64)


if __name__ == "__main__":
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id", "season", "game_type", "pitcher_id"])
    cand = build(raw).set_index("row_id").reindex(raw["row_id"].to_numpy())
    LGX = cand[LG].to_numpy(np.float64)
    print(f"  리그 피처 {LGX.shape}   결측 "
          f"{np.isnan(LGX).any(1).mean()*100:.1f}%\n")

    print(f"  {'폴드':>6s} {'구성':10s} {'전체':>9s} {'퓨처스':>9s} "
          f"{'전체 짝차이':>22s} {'퓨처스 짝차이':>22s}")
    for VS in (2022, 2023, 2024):
        d = F.build(DATA, VS=VS)
        X44 = d["X44"].astype(np.float64)
        y = d["y"].astype(np.float64)
        m_tr, m_va = d["m_tr"], d["m_va"]
        isf = d["is_f"][m_va]
        yv = y[m_va]
        w = DECAY ** (d["season"][m_tr].astype(np.float64) - 2019)

        sets = {"base": X44,
                "+league": np.c_[X44, LGX],
                "+f_share": np.c_[X44, LGX[:, [0]]]}
        P = {}
        for nm, X in sets.items():
            t0 = time.time()
            P[nm] = [run(X, y, m_tr, m_va, w, s) for s in SEEDS]
            sc = np.mean([F.best_shift(p, yv)[0] for p in P[nm]])
            scf = np.mean([F.best_shift(p[isf], yv[isf])[0] for p in P[nm]])
            if nm == "base":
                print(f"  {VS:6d} {nm:10s} {sc:9.1f} {scf:9.1f} "
                      f"{'':>22s} {'':>22s}   {time.time()-t0:.0f}s")
            else:
                da = [F.best_shift(a, yv)[0] - F.best_shift(b, yv)[0]
                      for a, b in zip(P[nm], P["base"])]
                df = [F.best_shift(a[isf], yv[isf])[0]
                      - F.best_shift(b[isf], yv[isf])[0]
                      for a, b in zip(P[nm], P["base"])]
                fm = lambda v: (f"{np.mean(v):+7.1f} "                # noqa: E731
                                f"[{', '.join(f'{x:+.0f}' for x in v)}] "
                                f"{sum(1 for x in v if x > 0)}/{len(v)}")
                print(f"  {VS:6d} {nm:10s} {sc:9.1f} {scf:9.1f} "
                      f"{fm(da):>22s} {fm(df):>22s}   {time.time()-t0:.0f}s")
        print()

    print("  세 폴드에서 부호가 일관돼야 채택한다. 한 폴드 판정은 리더보드에서")
    print("  여러 번 뒤집혔다. 퓨처스 구간 +100 은 전체로 약 +12 다.")
