# -*- coding: utf-8 -*-
"""배치 패키지의 history 테이블이 어느 구간에서 만들어졌나.

왜 중요한가
    추론에서 2025 인시즌 피처를 이렇게 복원한다.
        base_p_n  = pitcher_id.map(history["pitcher_n"])   그 투수의 train 총 투구수
        pn_cur    = asof_pitcher_n - base_p_n              = 2025 인시즌 투구수
        p_is_succ = (asof 성공수 - base 성공수) / pn_cur    = 2025 인시즌 성공률

    history 가 2019~2024 **전체**로 만들어져야 맞다. 2019~2023 이면 pn_cur 이
    '2024+2025 합산' 이 되어 인시즌 피처가 통째로 틀린다.

    p_is_succ 는 표적상관 0.102 로 44열 중 가장 강한 축에 속한다. 여기가 틀리면
    plat_dev(0.015)보다 훨씬 크게 잃는다.

무엇을 재나
    투수별 총 투구수를 여러 창에서 직접 계산해서 저장된 표와 대조한다.
    가장 잘 맞는 창이 그 표가 만들어진 구간이다.

    파일 두 개를 다 본다.
        model/history.json   TabM 경로 (preprocess.build_inference_features)
        model/trees.npz      CatBoost 경로 (script.py 의 build_features)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"
PKG = "submit_30"

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                 usecols=["season", "pitcher_id", "batter_id",
                          "control_success"])
print(f"  train {len(tr):,}행   시즌 {sorted(tr.season.unique())}\n")


def truth(upto, key):
    d = tr[tr.season <= upto]
    n = d.groupby(key).size()
    s = d.groupby(key)["control_success"].sum()
    return n, s


def compare(name, table, key, kind):
    """저장된 표를 여러 창의 진짜 값과 대조해 가장 맞는 창을 찾는다."""
    # history.json 은 [[id, val], ...] 리스트, trees.npz 는 dict 로 들어온다
    items = table if isinstance(table, list) else list(table.items())
    ks = np.array([int(a) for a, _ in items])
    vs = np.array([float(b) for _, b in items])
    print(f"  [{name}]  항목 {len(ks)}개")
    best = None
    for upto in (2021, 2022, 2023, 2024):
        n, s = truth(upto, key)
        t = (n if kind == "n" else s).reindex(ks)
        ok = t.notna().to_numpy()
        if ok.sum() < 10:
            continue
        d = np.abs(vs[ok] - t.to_numpy()[ok])
        hit = float((d < 0.5).mean())
        r = float(np.corrcoef(vs[ok], t.to_numpy()[ok])[0, 1])
        print(f"    <= {upto}   일치율 {hit*100:6.2f}%   상관 {r:.6f}   "
              f"평균차 {d.mean():10.1f}")
        if best is None or hit > best[1]:
            best = (upto, hit)
    print(f"    -> 가장 맞는 창: <= {best[0]}  (일치율 {best[1]*100:.2f}%)\n")
    return best[0]


print("=" * 70)
print("  model/history.json  (TabM 추론 경로)")
print("=" * 70)
h = json.load(open(os.path.join(PKG, "model", "history.json"),
                   encoding="utf-8"))
print(f"  키: {sorted(h.keys())}\n")
w1 = compare("pitcher_n", h["pitcher_n"], "pitcher_id", "n")
w2 = compare("pitcher_s", h["pitcher_s"], "pitcher_id", "s")
w3 = compare("batter_n", h["batter_n"], "batter_id", "n")

print("=" * 70)
print("  model/trees.npz  (CatBoost 추론 경로)")
print("=" * 70)
z = np.load(os.path.join(PKG, "model", "trees.npz"), allow_pickle=False)
t_pn = dict(zip(z["pitcher_id"].tolist(), z["pitcher_n"].tolist()))
t_ps = dict(zip(z["pitcher_id"].tolist(), z["pitcher_s"].tolist()))
t_bn = dict(zip(z["batter_id"].tolist(), z["batter_n"].tolist()))
w4 = compare("pitcher_n", t_pn, "pitcher_id", "n")
w5 = compare("pitcher_s", t_ps, "pitcher_id", "s")
w6 = compare("batter_n", t_bn, "batter_id", "n")

print("=" * 70)
ws = {"history.json pitcher_n": w1, "history.json pitcher_s": w2,
      "history.json batter_n": w3, "trees.npz pitcher_n": w4,
      "trees.npz pitcher_s": w5, "trees.npz batter_n": w6}
for k, v in ws.items():
    flag = "" if v == 2024 else "   <- 2024 가 아니다"
    print(f"  {k:26s} <= {v}{flag}")
if all(v == 2024 for v in ws.values()):
    print("\n  전부 2019~2024 전체다. 이 축은 깨끗하다.")
else:
    print("\n  **2024 가 아닌 표가 있다.** 2025 인시즌 복원이 틀린다 —")
    print("  pn_cur 이 '빠진 시즌 + 2025' 합산이 되고 p_is_succ 이 통째로 어긋난다.")
