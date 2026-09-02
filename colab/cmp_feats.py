# -*- coding: utf-8 -*-
"""두 전처리가 만드는 44열을 열 단위로 대조한다. 'ourfeat' 이 뭔지 확인용.

배경
    features44.py (우리) 와 train_chan_3/preprocess.py (지인) 는 둘 다 44열을
    내놓고 **열 이름까지 같다**. 다른 건 같은 이름의 열을 계산하는 방식이다.

    피처셋 정면비교(feat_head2head)에서 VS=2024 기준 +23.8 (3/3, t=6.73) 이
    나왔는데, 그 차이가 어느 열에서 오는지 알아야 판단할 수 있다.

무엇을 보나
    같은 창(season < VS)으로 둘을 만들어
      ① 열 이름 집합이 같은가
      ② 열마다 값이 얼마나 다른가 (상관, 최대차)
      ③ 표적과의 상관이 어느 쪽이 강한가
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"
VS = 2024

import features44 as F                                          # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

d = F.build(DATA, VS=VS)
F44 = list(d["F44"])
Xou = d["X44"].astype(np.float64)
y = d["y"].astype(np.float64)
m_tr = d["m_tr"]

tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                          encoding="utf-8-sig"))
hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id"])
pos = pd.Series(np.arange(len(tr_sorted)), index=tr_sorted["row_id"].to_numpy())
Xfr = Xs.to_numpy(dtype=np.float64)[
    pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
FFR = list(Xs.columns)

print(f"\n  우리 {len(F44)}열   지인 {len(FFR)}열")
so, sf = set(F44), set(FFR)
print(f"  우리에만: {sorted(so - sf)}")
print(f"  지인에만: {sorted(sf - so)}")
print(f"  공통 {len(so & sf)}열\n")


def corr(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 100 or np.nanstd(a[ok]) < 1e-12 or np.nanstd(b[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


rows = []
for c in F44:
    if c not in sf:
        continue
    a = Xou[:, F44.index(c)]
    b = Xfr[:, FFR.index(c)]
    r = corr(a, b)
    ok = ~(np.isnan(a) | np.isnan(b))
    md = float(np.abs(a[ok] - b[ok]).max()) if ok.any() else np.nan
    ta = corr(np.where(np.isnan(a), np.nanmedian(a), a)[m_tr], y[m_tr])
    tb = corr(np.where(np.isnan(b), np.nanmedian(b), b)[m_tr], y[m_tr])
    rows.append((c, r, md, ta, tb))

rows.sort(key=lambda x: (1.0 if np.isnan(x[1]) else x[1]))
print(f"  {'열':30s} {'두 판본 상관':>11s} {'최대차':>10s} "
      f"{'표적상관(우리)':>13s} {'표적상관(지인)':>13s}")
print("  " + "-" * 82)
for c, r, md, ta, tb in rows:
    flag = "  <- 다름" if (np.isnan(r) or r < 0.9999) else ""
    print(f"  {c:30s} {r:11.5f} {md:10.4f} {ta:13.5f} {tb:13.5f}{flag}")

diff = [c for c, r, *_ in rows if np.isnan(r) or r < 0.9999]
print(f"\n  값이 다른 열 {len(diff)} / {len(rows)}: {diff}")
print("\n  나머지 열은 주어진 원본을 그대로 쓰므로 당연히 같다.")
print("  차이는 파생 열의 계산 방식에서만 나온다 — 그게 'ourfeat' 의 정체다.")
