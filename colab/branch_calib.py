# -*- coding: utf-8 -*-
"""브랜치별 보정이 **연도를 넘어 전이되는가**. 재학습 없이 잰다.

왜 이걸 먼저 재나
    지인 문서의 6번(conditional residual head)은
        z_final = z_tabm + delta_game_type + delta_regime
    인데, delta 가 스칼라면 이건 **브랜치별 시프트**와 같다. 문서 1번(브랜치별
    calibration)의 다른 표현이다.

    독립 헤드 판(twohead.py)은 이미 4시드로 졌다 — 퓨처스 -42.4, t=-8.67, 0/4.
    공유 backbone 위 헤드는 퓨처스 11.8% 에서만 기울기를 받아서다.
    남은 건 **저용량 판**이고, 신호가 0.93% 인 이 문제에선 그쪽이 유망하다.

무엇이 관건인가
    브랜치마다 (T, b) 를 그 해 자료로 맞추면 당연히 그 해 점수가 오른다.
    문제는 **다른 해에도 통하는가** 다. 기억에 "온도는 그 해에만 맞는 값" 이라
    적혀 있고, 다른 폴드에서 맞춘 T 를 가져다 쓰면 2022 에서 -744 였다.

    그래서 정직하게 잰다 — **VS=2022 에서 맞추고 VS=2024 에서 채점**한다.
    그 반대 방향도 본다.

비교
    none      보정 없음 (원 예측)
    global    전역 (T, b) 한 벌
    branch    1군/퓨처스 각각 (T, b)
    oracle    채점하는 해에서 직접 맞춘 값 — 상한이지 쓸 수 있는 값이 아니다
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(SC, "_dl")
SEEDS = (42, 1, 777)

import features44 as F                                          # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply(p, T, b):
    return 1.0 / (1.0 + np.exp(-(T * logit(p) + b)))


def brier_score(p, y, r):
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1 - r)))


def fit_Tb(p, y):
    """그 자료에서 (T, b) 를 Brier 최소로 맞춘다."""
    z = logit(p)

    def obj(v):
        q = 1.0 / (1.0 + np.exp(-(v[0] * z + v[1])))
        return ((q - y) ** 2).mean()

    r = minimize(obj, [1.0, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 2000})
    return float(r.x[0]), float(r.x[1])


FOLD = {}
for VS in (2022, 2024):
    d = F.build(DATA, VS=VS)
    season = d["season"].astype(int)
    g = np.where(season == VS)[0]
    ps = [np.load(f"{DL}/pcd_{VS}_base_s{s}.npy") for s in SEEDS]
    FOLD[VS] = dict(p=np.mean(ps, 0), y=d["y"].astype(np.float64)[g],
                    isf=d["is_f"][g])
    yv = FOLD[VS]["y"]
    FOLD[VS]["r"] = float(yv.mean())
    print(f"  VS={VS}  {len(g):,}행  실제 {yv.mean():.4f}  "
          f"예측 {FOLD[VS]['p'].mean():.4f}  퓨처스 {FOLD[VS]['isf'].mean()*100:.1f}%")

print()
for src, dst in ((2022, 2024), (2024, 2022)):
    S, D = FOLD[src], FOLD[dst]
    r = D["r"]
    # 맞추는 쪽
    gT, gb = fit_Tb(S["p"], S["y"])
    rT, rb = fit_Tb(S["p"][~S["isf"]], S["y"][~S["isf"]])
    fT, fb = fit_Tb(S["p"][S["isf"]], S["y"][S["isf"]])
    # 적용하는 쪽
    none = brier_score(D["p"], D["y"], r)
    glob = brier_score(apply(D["p"], gT, gb), D["y"], r)
    br = D["p"].copy()
    br[~D["isf"]] = apply(D["p"][~D["isf"]], rT, rb)
    br[D["isf"]] = apply(D["p"][D["isf"]], fT, fb)
    bran = brier_score(br, D["y"], r)
    # 상한 (채점하는 해에서 직접 맞춤 — 못 쓰는 값)
    oT, ob = fit_Tb(D["p"], D["y"])
    orT, orb = fit_Tb(D["p"][~D["isf"]], D["y"][~D["isf"]])
    ofT, ofb = fit_Tb(D["p"][D["isf"]], D["y"][D["isf"]])
    og = brier_score(apply(D["p"], oT, ob), D["y"], r)
    ob_ = D["p"].copy()
    ob_[~D["isf"]] = apply(D["p"][~D["isf"]], orT, orb)
    ob_[D["isf"]] = apply(D["p"][D["isf"]], ofT, ofb)
    obr = brier_score(ob_, D["y"], r)

    print(f"  {src} 에서 맞추고 {dst} 에서 채점")
    print(f"    맞춘 값   전역 T={gT:.4f} b={gb:+.4f}   "
          f"1군 T={rT:.4f} b={rb:+.4f}   퓨처스 T={fT:.4f} b={fb:+.4f}")
    print(f"    보정 없음 {none:9.1f}")
    print(f"    전역      {glob:9.1f}   {glob-none:+7.1f}")
    print(f"    브랜치별  {bran:9.1f}   {bran-none:+7.1f}   "
          f"전역 대비 {bran-glob:+6.1f}")
    print(f"    [상한] 그 해에서 직접  전역 {og:9.1f}  브랜치별 {obr:9.1f}   "
          f"차 {obr-og:+6.1f}")
    print()

print("  읽는 법")
print("    '브랜치별 - 전역' 이 양수여야 문서 6번(저용량 판)에 값어치가 있다.")
print("    상한의 '브랜치별 - 전역' 이 작으면 그 해에서 직접 맞춰도 얻을 게 없다는 뜻이다.")
print("    독립 헤드 판(twohead)은 이미 퓨처스 -42.4 (t=-8.67, 0/4) 로 졌다.")
