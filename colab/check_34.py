# -*- coding: utf-8 -*-
"""submit_34 (f2b, 4단계 체제, 시드3) 를 submit_30 과 같은 행에서 대조한다.

확인하는 것
    1. 새 추론 경로가 실제로 도는가 (4단계 체제 열 + 시드 3개 평균)
    2. 체제 열을 1.0 으로 넣으면 얼마나 달라지는가 — 안 고쳤을 때의 피해 크기
    3. submit_30 대비 퓨처스 예측이 얼마나 움직이는가
    4. 규칙 4 — 순서 섞기 / 부분집합

test.csv 는 5행짜리 형식 견본이라 2024 행으로 대신한다. 배치 모델에겐 인샘플이라
점수는 못 믿지만, **두 패키지가 같은 행에서 얼마나 다른가**는 볼 수 있다.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "open (1)", "data")
N_SUB = 20000

cols = list(pd.read_csv(os.path.join(DATA, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), N_SUB, replace=False))
sub = pool.iloc[pick][cols].copy()
y = pool.iloc[pick]["control_success"].to_numpy(np.float64)
isf = sub["game_type"].astype(str).to_numpy() == "F"
print(f"  2024 행 {len(sub):,} 표본   퓨처스 {int(isf.sum()):,} ({isf.mean()*100:.1f}%)")


def run(pkg, force_regime=None):
    """그 패키지의 TabM 라우팅 예측을 낸다. force_regime 이 주어지면 체제 열을 덮는다."""
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, pkg))
    sys.path.insert(0, os.getcwd())
    spec = importlib.util.spec_from_file_location(f"pkg_{pkg}", "script.py")
    S = importlib.util.module_from_spec(spec)
    S.__dict__["__name__"] = f"pkg_{pkg}"
    try:
        spec.loader.exec_module(S)
    except SystemExit:
        pass
    import preprocess as PPF
    with open("model/history.json", encoding="utf-8") as f:
        hist = PPF.deserialize_history(json.load(f))
    ordered = PPF.sort_by_row_id(sub)
    Xf = PPF.build_inference_features(ordered, hist)
    isf_o = ordered["game_type"].astype(str).to_numpy() == "F"
    meta_a, arc_a = S.load_bundle("model/all_tabm_seed_42.npz")
    p = S.predict_frame(Xf, meta_a, arc_a, chunk_size=4096)
    rr = np.flatnonzero(~isf_o)
    if len(rr):
        mt, ar = S.load_bundle("model/regular_tabm_seed_42.npz")
        p[rr] = 0.6 * p[rr] + 0.4 * S.predict_frame(Xf.iloc[rr], mt, ar,
                                                    chunk_size=4096)
    rf = np.flatnonzero(isf_o)
    files = ([("f2b_all_s%d" % s, "f2b_futures_s%d" % s) for s in (42, 1, 777)]
             if pkg == "submit_34" else
             [("fbregime_all_seed42", "fbregime_futures_seed42")])
    reg = (np.where(isf_o, 2.0, 3.0) if pkg == "submit_34"
           else np.ones(len(Xf)))
    if force_regime is not None:
        reg = np.full(len(Xf), float(force_regime))
    Xv = np.c_[Xf.to_numpy(dtype=np.float64), reg]
    acc = {"all": [], "futures": []}
    for fa, ff in files:
        for br, fn in (("all", fa), ("futures", ff)):
            z = np.load(f"model/{fn}.npz", allow_pickle=False)
            mt = json.loads(str(z["meta"].item()))
            mt["cat_cardinalities"] = mt["cards"]
            Tn, Tc = S._fm_prep(Xv[rf], z)
            acc[br].append(S._tabm_forward(Tn, Tc, mt, z, 4096))
    p[rf] = 0.6 * np.mean(acc["all"], 0) + 0.4 * np.mean(acc["futures"], 0)
    p = np.clip(p, 0.0, 1.0)
    tm = dict(zip(ordered[S.ID_COL].tolist(), p))
    out = np.array([tm[r] for r in sub[S.ID_COL].tolist()], np.float64)
    sys.path.pop(0)
    os.chdir(cwd)
    return out


import features44 as F                                          # noqa: E402
r = y.mean()


def sc(p, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return F.best_shift(p[m], y[m])[0]


p30 = run("submit_30")
p34 = run("submit_34")
p34w = run("submit_34", force_regime=1.0)      # 추론을 안 고쳤을 때
print(f"\n  {'구성':28s} {'평균':>9s} {'퓨처스평균':>11s} {'퓨처스점수':>11s}")
for nm, p in (("submit_30 (기존)", p30), ("submit_34 (고친 추론)", p34),
              ("submit_34 + 체제=1.0 (안 고침)", p34w)):
    print(f"  {nm:28s} {p.mean():9.5f} {p[isf].mean():11.5f} {sc(p, isf):11.1f}")
print(f"\n  퓨처스 행에서 30 vs 34   상관 {np.corrcoef(p30[isf], p34[isf])[0,1]:.5f}  "
      f"최대차 {np.abs(p30[isf]-p34[isf]).max():.5f}")
print(f"  34 고침 vs 안고침        상관 {np.corrcoef(p34[isf], p34w[isf])[0,1]:.5f}  "
      f"최대차 {np.abs(p34[isf]-p34w[isf]).max():.5f}")
print(f"  1군 행은 두 패키지가 같아야 한다  최대차 "
      f"{np.abs(p30[~isf]-p34[~isf]).max():.2e}")

# 규칙 4
per = rng.permutation(len(sub))
sub2 = sub.iloc[per].reset_index(drop=True)
g = globals()
sub_bak = sub
sub = sub2
p34b = run("submit_34")
sub = sub_bak
d1 = float(np.abs(p34b - p34[per]).max())
print(f"\n  규칙 4 — 순서 섞기 최대차 {d1:.3e}  "
      f"{'통과' if d1 < 1e-9 else '**위반**'}")
