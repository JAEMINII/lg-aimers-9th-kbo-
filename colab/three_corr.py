# -*- coding: utf-8 -*-
"""MNCA / CatBoost / TabM 세 성분의 상관을 **배치 구성에서** 잰다.

왜 배치 구성이어야 하나
    기억에 두 값이 섞여 있다 — MNCA 상관이 관문 0.954 인데 배치 스모크에서는
    0.696 이었다. 그걸 무시하고 냈다가 1069 -> 1045 였다.
    검색 기반 모델은 분포가 이동하면 이웃이 달라지므로 상관이 무너진다.
    그러니 이 질문의 답은 반드시 **2025 시험 행**에서 나와야 한다.

무엇을 재나
    submit_33 의 추론 경로를 그대로 복제한다.
        CatBoost   build_features -> trees.npz numpy 순회
        TabM       build_inference_features -> 1군 0.6all+0.4reg / 퓨처스 fbregime
        MNCA       같은 피처 + abs_regime=1 -> 후보 16,384개에 softmax 거리 가중
    혼합은 0.2 CB + 0.2 MNCA + 0.6 TabM 이었다.

표본
    2025 시험 24.6만 행 중 2만을 뽑는다. 상관 표준오차가 0.005 수준이라 충분하고,
    MNCA 가 CPU 에서 후보 16,384개와 거리를 재야 해서 전수는 느리다.
    행마다 독립이라(규칙 4 감사 완료) 부분집합으로 재도 값이 안 변한다.
"""
import importlib.util
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "submit_33")
DATA = os.path.join(ROOT, "open (1)", "data")
N_SUB = 20000

# test.csv 는 5행짜리 형식 견본이다. 실제 24.6만 행은 평가 서버에만 있다.
# 그래서 분포가 가장 가까운 **2024 행**을 배치 모델에 통과시킨다.
# '배치 스모크 0.696' 을 만든 방식과 같다. 이 행들은 배치 모델에겐 인샘플이라
# 점수는 못 믿지만, 두 모델이 서로 얼마나 다른 실수를 하는지는 볼 수 있다.
cols = list(pd.read_csv(os.path.join(DATA, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
sub = pool.iloc[np.sort(rng.choice(len(pool), N_SUB, replace=False))][cols].copy()
test = sub
print(f"  2024 행 {len(pool):,}개 중 {len(sub):,}행 표본   열 {len(cols)}개")

os.chdir(PKG)
sys.path.insert(0, PKG)
spec = importlib.util.spec_from_file_location("pkg33", "script.py")
S = importlib.util.module_from_spec(spec)
S.__dict__["__name__"] = "pkg33"
try:
    spec.loader.exec_module(S)
except SystemExit:
    pass

# CUDA 고정을 CPU 로 바꿔 끼운다. 로컬 torch 는 CPU 판이다.
src = inspect.getsource(S._mnca_predict).replace('device("cuda")', 'device("cpu")')
exec(src, S.__dict__)

hist, ztrees, shift = S.load_model()
ID = S.ID_COL

# ---------------------------------------------------------------- CatBoost
Xcb = S.build_features(sub, hist)
p_cb = np.asarray(S.predict_numpy(Xcb.to_numpy(dtype=np.float64), ztrees,
                                  chunk=8192), np.float64).ravel()
print(f"  CatBoost  평균 {p_cb.mean():.5f}  SD {p_cb.std():.5f}")

# ---------------------------------------------------------------- TabM
import preprocess as PPF                                        # noqa: E402
with open("model/history.json", encoding="utf-8") as f:
    hist_f = PPF.deserialize_history(json.load(f))
ordered = PPF.sort_by_row_id(sub)
Xf = PPF.build_inference_features(ordered, hist_f)
isf_o = ordered["game_type"].astype(str).to_numpy() == "F"

meta_a, arc_a = S.load_bundle("model/all_tabm_seed_42.npz")
p_ord = S.predict_frame(Xf, meta_a, arc_a, chunk_size=4096)
rows_r = np.flatnonzero(~isf_o)
if len(rows_r):
    mt, ar = S.load_bundle("model/regular_tabm_seed_42.npz")
    pr = S.predict_frame(Xf.iloc[rows_r], mt, ar, chunk_size=4096)
    p_ord[rows_r] = 0.6 * p_ord[rows_r] + 0.4 * pr
rows_f = np.flatnonzero(isf_o)
Xv = np.c_[Xf.to_numpy(dtype=np.float64), np.ones(len(Xf))]
if len(rows_f):
    acc = {}
    for br in ("all", "futures"):
        z2 = np.load(f"model/fbregime_{br}_seed42.npz", allow_pickle=False)
        mt2 = json.loads(str(z2["meta"].item()))
        mt2["cat_cardinalities"] = mt2["cards"]
        Tn, Tc = S._fm_prep(Xv[rows_f], z2)
        acc[br] = S._tabm_forward(Tn, Tc, mt2, z2, 4096)
    p_ord[rows_f] = 0.6 * acc["all"] + 0.4 * acc["futures"]
p_ord = np.clip(p_ord, 0.0, 1.0)
tm = dict(zip(ordered[ID].tolist(), p_ord))
p_tabm = np.array([tm[r] for r in sub[ID].tolist()], np.float64)
print(f"  TabM      평균 {p_tabm.mean():.5f}  SD {p_tabm.std():.5f}"
      f"   퓨처스 {int(isf_o.sum()):,}행 ({isf_o.mean()*100:.1f}%)")

# ---------------------------------------------------------------- MNCA
o_mn = S._mnca_predict(Xv, S.MNCA_SEEDS)
tm2 = dict(zip(ordered[ID].tolist(), np.asarray(o_mn).ravel().tolist()))
p_mn = np.array([tm2[r] for r in sub[ID].tolist()], np.float64)
print(f"  MNCA      평균 {p_mn.mean():.5f}  SD {p_mn.std():.5f}")

os.chdir(ROOT)
P = {"CatBoost": p_cb, "MNCA": p_mn, "TabM": p_tabm}
names = ["CatBoost", "MNCA", "TabM"]
print("\n" + "=" * 58)
print("  피어슨 상관 — 2025 시험 행 (배치 구성)")
print("=" * 58)
print("           " + "".join(f"{n:>12s}" for n in names))
for a in names:
    print(f"  {a:9s}" + "".join(
        f"{np.corrcoef(P[a], P[b])[0,1]:12.4f}" for b in names))
isf_t = sub["game_type"].astype(str).to_numpy() == "F"
print("\n  구간별 (MNCA x TabM 이 문제였던 축)")
for nm, m in (("1군", ~isf_t), ("퓨처스", isf_t)):
    if m.sum() < 100:
        continue
    print(f"    {nm:6s} {m.sum():6,}행   "
          f"CB-TabM {np.corrcoef(p_cb[m], p_tabm[m])[0,1]:.4f}   "
          f"MNCA-TabM {np.corrcoef(p_mn[m], p_tabm[m])[0,1]:.4f}   "
          f"CB-MNCA {np.corrcoef(p_cb[m], p_mn[m])[0,1]:.4f}")
print("\n  참고 — 관문에서는 MNCA-TabM 이 0.954 였다. 배치 스모크는 0.696.")
print("  다양성 조건은 상관 < 0.97 이고 단독 > 880 이다.")
