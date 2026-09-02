# -*- coding: utf-8 -*-
"""cm 계열 3열의 as-of 판을 만들고 누출 크기를 잰다.

창 의존 검사에서 걸린 4열 중 plat_dev 는 이미 고쳤고 셋이 남았다.
    lg_cm_eff  cm_rel  p_adj_cm

기전
    cm = (balls*3 + strikes)*4 + pitcher_hand*2 + batter_hand   -> 48칸
    cm_lg  = 칸별 y 평균 - 전체평균
    cm_rel = 칸별 (y ~ p_is_succ) 기울기 / 전체 기울기
    지인 코드는 이 표를 **학습창 전체**로 한 번 만들어 모든 행에 붙인다.

plat_dev 와 크기가 다르다
    plat_dev  792칸, 칸당 1,860행 -> 그 투수 자신의 미래가 들어간다
    cm         48칸, 칸당 30,700행 -> 리그 수준. 자기누출 1/30,700

    그래도 학습/추론 의미 어긋남은 같다. 2019 학습행은 2020~2023 을 본 표를
    받는데 2025 추론행은 못 받는다.

as-of 판
    각 행보다 **앞선 행만** 으로 칸 통계를 만든다 (칸별 cumsum 후 shift(1)).
    표본이 적은 초반은 중립값으로 수축한다 (원본의 fillna 와 같은 방향).

진단 (plat_dev 때 쓴 서명)
    누출판과 as-of 판의 **표적 상관**을 비교한다. 누출판이 더 높으면 그 차이가
    미래 정보다. plat_dev 는 +0.02368 (누출) -> +0.01478 (as-of) 이었다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"
ALPHA = 300.0

from train_chan_3 import preprocess as PP                       # noqa: E402

tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                   encoding="utf-8-sig"))
y = tr[PP.TARGET].to_numpy(np.float64)
season = tr["season"].to_numpy()
VS = 2024
m_tr = season < VS

# 배치 경로 그대로 누출판을 만든다
h = PP.fit_history_tables(tr[tr.season < VS])
Xl = PP.transform_features(tr, h, train_mode=True)
cm = ((tr["balls_before"].astype("int32") * 3
       + tr["strikes_before"].astype("int32")) * 4
      + tr["pitcher_hand"].astype("int32") * 2
      + tr["batter_hand"].astype("int32")).to_numpy()
x = Xl["p_is_succ"].to_numpy(np.float64)
print(f"  cm 칸 {len(np.unique(cm))}개   칸당 중앙 "
      f"{int(np.median(np.bincount(cm)[np.bincount(cm) > 0])):,}행")


def cum_prev(v, key):
    """그 행 **직전까지**의 칸별 누적합."""
    s = pd.Series(v).groupby(key).cumsum().to_numpy() - v
    return s


one = np.ones(len(tr))
n_c = cum_prev(one, cm)
sy_c = cum_prev(y, cm)
sx_c = cum_prev(x, cm)
sxy_c = cum_prev(x * y, cm)
sxx_c = cum_prev(x * x, cm)
# 전체(칸 무관) 누적 — as-of 전역 평균/기울기
gz = np.zeros(len(tr), np.int64)
n_g = cum_prev(one, gz)
sy_g = cum_prev(y, gz)
sx_g = cum_prev(x, gz)
sxy_g = cum_prev(x * y, gz)
sxx_g = cum_prev(x * x, gz)

with np.errstate(invalid="ignore", divide="ignore"):
    gmean_a = np.where(n_g > 0, sy_g / np.maximum(n_g, 1), y.mean())
    vg = sxx_g / np.maximum(n_g, 1) - (sx_g / np.maximum(n_g, 1)) ** 2
    cg = sxy_g / np.maximum(n_g, 1) - (sx_g / np.maximum(n_g, 1)) * gmean_a
    slope_g = np.where(vg > 1e-12, cg / vg, 1.0)

    mc = sy_c / np.maximum(n_c, 1)
    vc = sxx_c / np.maximum(n_c, 1) - (sx_c / np.maximum(n_c, 1)) ** 2
    cc = sxy_c / np.maximum(n_c, 1) - (sx_c / np.maximum(n_c, 1)) * mc
    slope_c = np.where(vc > 1e-12, cc / vc, np.nan)

w = n_c / (n_c + ALPHA)                       # 표본 적으면 중립으로
lg_a = w * (mc - gmean_a)                     # 없으면 0 (원본 fillna 0.0)
lg_a = np.where(n_c > 0, lg_a, 0.0)
rel_raw = np.where(np.isfinite(slope_c) & (np.abs(slope_g) > 1e-12),
                   slope_c / slope_g, 1.0)
rel_a = 1.0 + w * (rel_raw - 1.0)             # 없으면 1 (원본 fillna 1.0)
padj_a = gmean_a + (x - gmean_a) * rel_a

NEW = {"lg_cm_eff": lg_a, "cm_rel": rel_a, "p_adj_cm": padj_a}
print(f"\n  {'열':12s} {'누출판 표적상관':>16s} {'as-of 표적상관':>16s} "
      f"{'차이':>10s} {'두 판 상관':>11s}")
for c, v in NEW.items():
    o = Xl[c].to_numpy(np.float64)
    ok = np.isfinite(o) & np.isfinite(v) & m_tr
    ro = float(np.corrcoef(o[ok], y[ok])[0, 1])
    rn = float(np.corrcoef(v[ok], y[ok])[0, 1])
    rr = float(np.corrcoef(o[ok], v[ok])[0, 1])
    print(f"  {c:12s} {ro:+16.5f} {rn:+16.5f} {rn-ro:+10.5f} {rr:11.4f}")
print("\n  참고 — plat_dev 는 +0.02368(누출) -> +0.01478(as-of), 차이 -0.0089")
print("  누출판이 더 높으면 그 차이가 미래 정보다.")

out = pd.DataFrame({f"asof_{c}": v for c, v in NEW.items()})
out.insert(0, "row_id", tr["row_id"].to_numpy())
out.to_csv(os.path.join(SC, "_dl", "cm_asof.csv.gz"), index=False)
print(f"\n  저장  colab/_dl/cm_asof.csv.gz")
