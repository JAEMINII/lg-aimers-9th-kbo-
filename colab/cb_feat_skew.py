# -*- coding: utf-8 -*-
"""script.py 의 CatBoost 피처 생성기를 다른 두 구현과 대조한다.

왜 여기인가
    plat_dev 누출(LB +11)은 **두 구현을 대조해서** 찾았다. 관문이 아니라.
    그런데 지금까지 대조한 건 전부 TabM 쪽이었다.

        확인함   지인 preprocess 학습경로 vs 추론경로     44열 최대차 0
        확인함   TabM PyTorch 모델 vs npz 재구현          최대차 1.19e-07
        안 봄    **script.py 의 CatBoost 피처 생성기**

    CatBoost 는 script.py 안에 자기만의 build_features 를 갖고 있고
    trees.npz 의 룩업(pitcher_n/s, batter_n/s, cm_lg, cm_rel, plat_*, marcel)
    으로 44열을 따로 만든다. 혼합의 0.20~0.30 을 차지한다.

무엇을 하나
    같은 행을 세 경로로 통과시켜 열 단위로 대조한다.
        A  script.py build_features        (trees.npz 룩업)
        B  지인 transform_features          (history.json)
        C  features44 build                 (as-of)

    A 와 B 가 다르면 CatBoost 가 TabM 과 **다른 피처를 보고 있다**는 뜻이고,
    그게 의도된 것인지(다양성) 실수인지 가려야 한다.
    A 와 C 의 차이는 plat_dev 처럼 as-of 문제일 수 있다.

    2025 시험 행이 없으므로 2024 행을 시험셋처럼 넣는다. 배치에서 2025 가
    겪는 위치와 같다.
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
VS = 2024
N = 60000

sys.path.insert(0, PKG)
import features44 as F                                          # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

# script.py 를 모듈로 불러 build_features / load_model 을 쓴다
import importlib.util                                            # noqa: E402
spec = importlib.util.spec_from_file_location(
    "pkgscript", os.path.join(PKG, "script.py"))
S = importlib.util.module_from_spec(spec)
S.__dict__["__name__"] = "pkgscript"
os.environ.setdefault("SKIP_MAIN", "1")
try:
    spec.loader.exec_module(S)
except SystemExit:
    pass

tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
tr = PP.sort_by_row_id(tr)
sel = tr["season"].to_numpy() == VS
te = tr.loc[sel].drop(columns=[PP.TARGET]).reset_index(drop=True).head(N)
print(f"  {VS} 행 {len(te):,}개를 세 경로로 통과시킨다\n")

# ---- A: script.py 의 CatBoost 경로
os.chdir(PKG)
hist_cb, z, shift = S.load_model()
A = S.build_features(te, hist_cb)
os.chdir("..")
print(f"  A script.py build_features   {A.shape}")

# ---- B: 지인 transform_features (history 는 전체 구간 = 배치와 같게)
hist_pp = PP.fit_history_tables(tr)
B = PP.build_inference_features(PP.sort_by_row_id(te), hist_pp)
print(f"  B 지인 build_inference        {B.shape}")

# ---- C: features44 (as-of)
d = F.build(DATA, VS=2025, return_frame=True)
F44 = list(d["F44"])
rid = d["frame"]["row_id"].to_numpy()
idx = pd.Series(np.arange(len(rid)), index=rid).reindex(
    te["row_id"].to_numpy()).to_numpy()
C = pd.DataFrame(d["X44"][idx], columns=F44)
print(f"  C features44 (as-of)          {C.shape}\n")


def cmp(name, X, Y, cols):
    rows = []
    for c in cols:
        x = pd.to_numeric(X[c], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(Y[c], errors="coerce").to_numpy(np.float64)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 100:
            continue
        r = (np.corrcoef(x[ok], y[ok])[0, 1]
             if x[ok].std() > 1e-12 and y[ok].std() > 1e-12 else 1.0)
        rows.append((c, r, float(np.abs(x[ok] - y[ok]).max())))
    rows.sort(key=lambda t: (1.0 if np.isnan(t[1]) else t[1]))
    bad = [c for c, r, m in rows if np.isnan(r) or r < 0.9999 or m > 1e-6]
    print(f"  [{name}]  공통 {len(rows)}열   다른 열 {len(bad)}개")
    for c, r, m in rows:
        if c in bad:
            print(f"    {c:28s} 상관 {r:9.5f}   최대차 {m:10.5f}   <- 다름")
    if not bad:
        print("    전부 일치")
    print()
    return bad


common_ab = [c for c in A.columns if c in B.columns]
common_ac = [c for c in A.columns if c in C.columns]
print(f"  A 열 {len(A.columns)}  B 열 {len(B.columns)}  C 열 {len(C.columns)}")
print(f"  A∩B {len(common_ab)}   A∩C {len(common_ac)}")
print(f"  A 에만: {sorted(set(A.columns) - set(B.columns))}")
print(f"  B 에만: {sorted(set(B.columns) - set(A.columns))}\n")

bad_ab = cmp("A(CatBoost) vs B(TabM)", A, B, common_ab)
bad_ac = cmp("A(CatBoost) vs C(as-of)", A, C, common_ac)

print("=" * 72)
print(f"  CatBoost 가 TabM 과 다르게 보는 열: {bad_ab}")
print(f"  CatBoost 가 as-of 와 다르게 보는 열: {bad_ac}")
print("\n  plat_dev 는 이미 아는 문제다 (CatBoost 는 일부러 누출판을 쓴다 —")
print("  둘 다 고치면 -3.7 이었다). **그것 말고 다른 게 나오는지**가 관건이다.")
