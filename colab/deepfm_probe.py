# -*- coding: utf-8 -*-
"""submit_42 의 DeepFM 로짓 혼합이 '새 정보' 인지 '온도 재조정' 인지 가른다.

구조
    submit_42   preds = sigmoid(0.8*bl + 0.2*fm)      <- 보정 직전
                이후 apply_calibration 이 T=1.0279, shift=-0.007 을 그대로 건다

가르는 법 — 점수가 아니라 구조를 본다 (2024 는 인샘플이라 점수는 못 믿는다)
    fm 을 base 로짓에 회귀시킨다.   fm = a + b*bl + resid
    그러면 혼합은
        0.8*bl + 0.2*(a + b*bl + resid) = (0.8 + 0.2b)*bl + 0.2a + 0.2*resid
    즉 **유효 온도 (0.8+0.2b), 유효 시프트 0.2a, 그리고 새 정보 0.2*resid** 다.
    앞의 둘은 T/shift 로 공짜로 얻을 수 있는 것이고 이미 보정단이 담당한다.
    셋 중 예측 변화를 누가 설명하는지 본다.

왜 중요한가
    기억: "온도는 유지해야 한다 — T 를 빼니 1069->1062, 관문이 17배 과소평가"
          "모델 바꾸면 시프트도 다시 잡아라 — 그대로 둬서 10~20점"
    로짓 축척을 건드리고 고정 T 를 그대로 두는 건 정확히 그 사고 패턴이다.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "open (1)", "data")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

cols = list(pd.read_csv(os.path.join(DATA, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), N, replace=False))
sub = pool.iloc[pick][cols].reset_index(drop=True)
y = pool.iloc[pick]["control_success"].to_numpy(np.float64)
isf = sub["game_type"].astype(str).to_numpy() == "F"

os.chdir(os.path.join(ROOT, "submit_42"))
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("p42", "script.py")
S = importlib.util.module_from_spec(spec); S.__dict__["__name__"] = "p42"
spec.loader.exec_module(S)
import preprocess as PPF
history, z, shift = S.load_model()
with open("model/history.json", encoding="utf-8") as f:
    hp = PPF.deserialize_history(json.load(f))

X = S.build_features(sub, history)
p_new = S.predict_numpy(X.values.astype(np.float64), z)
p_old = S.predict_numpy(X[history["_base_features"]].values.astype(np.float64),
                        history["_base_z"])
p_cb = S.W_CB_NEW * p_new + (1 - S.W_CB_NEW) * p_old

ordered = PPF.sort_by_row_id(sub)
Xf = PPF.build_inference_features(ordered, hp)
isf_o = ordered["game_type"].astype(str).to_numpy() == "F"
Xv = np.c_[Xf.to_numpy(dtype=np.float64), np.where(isf_o, 2.0, 3.0)]


def f2b(br, rows):
    zz = np.load(S.resolve(f"model/f2b_{br}_s42.npz"), allow_pickle=False)
    mt = json.loads(str(zz["meta"].item())); mt["cat_cardinalities"] = mt["cards"]
    Tn, Tc = S._fm_prep(Xv[rows], zz)
    return S._tabm_forward(Tn, Tc, mt, zz, 4096)


pt = f2b("all", np.arange(len(Xf)))
rr = np.flatnonzero(~isf_o)
pt[rr] = 0.6 * pt[rr] + 0.4 * f2b("regular", rr)
rf = np.flatnonzero(isf_o)
pt[rf] = 0.6 * pt[rf] + 0.4 * f2b("futures", rf)
tm = dict(zip(ordered[S.ID_COL].tolist(), np.clip(pt, 0, 1)))
p_tabm = np.array([tm[r] for r in sub[S.ID_COL].tolist()], np.float64)

fm = np.mean([S._deepfm_predict(Xf, np.load(S.resolve(f"model/deepfm_s{s}.npz"),
                                            allow_pickle=False))
              for s in (42, 1, 777)], 0)
fmm = dict(zip(ordered[S.ID_COL].tolist(), fm))
fm_logit = np.array([fmm[r] for r in sub[S.ID_COL].tolist()], np.float64)

base = S.W_CB * p_cb + S.W_TABM * p_tabm
bl = S._logit(base)
A = float(S.W_DEEPFM)

b, a = np.polyfit(bl, fm_logit, 1)
resid = fm_logit - (a + b * bl)
print(f"  {N:,}행\n")
print(f"  base 로짓   평균 {bl.mean():+.4f}  sd {bl.std():.4f}")
print(f"  fm   로짓   평균 {fm_logit.mean():+.4f}  sd {fm_logit.std():.4f}  "
      f"base 와 상관 {np.corrcoef(bl, fm_logit)[0,1]:.4f}")
print(f"\n  fm = {a:+.4f} {b:+.4f}*bl + resid     resid sd {resid.std():.4f}  "
      f"(fm 분산의 {resid.var()/fm_logit.var()*100:.1f}%)")
print(f"\n  혼합 = ({1-A:.2f} + {A:.2f}x{b:.4f})*bl + {A:.2f}x{a:+.4f} + {A:.2f}*resid")
print(f"       = {1-A+A*b:.4f}*bl {A*a:+.4f} + {A:.2f}*resid")
print(f"    유효 온도  {1-A+A*b:.4f}    (배치 보정 T = {S.CALIB_T})")
print(f"    유효 시프트 {A*a:+.4f}      (배치 보정 shift = {shift:+.4f})")

mix = (1 - A + A * b) * bl + A * a + A * resid
affine = (1 - A + A * b) * bl + A * a
d_tot = mix - bl
d_aff = affine - bl
print(f"\n  예측 로짓 변화량 분해")
print(f"    전체 변화 sd            {d_tot.std():.5f}")
print(f"    그중 온도·시프트분 sd   {d_aff.std():.5f}  "
      f"({d_aff.var()/d_tot.var()*100:5.1f}%)")
print(f"    그중 새 정보분 sd       {(A*resid).std():.5f}  "
      f"({(A*resid).var()/d_tot.var()*100:5.1f}%)")

pA = S.apply_calibration(base, shift)
pB = S.apply_calibration(1/(1+np.exp(-np.clip(bl + A*(fm_logit-bl), -60, 60))), shift)
print(f"\n  최종 예측   41식 평균 {pA.mean():.5f}   42식 평균 {pB.mean():.5f}   "
      f"실제 {y.mean():.5f}")
print(f"              sd {pA.std():.5f} -> {pB.std():.5f}   "
      f"상관 {np.corrcoef(pA,pB)[0,1]:.5f}   최대차 {np.abs(pA-pB).max():.4f}")
print(f"\n  참고: 예측SD 가 줄면 수축이다. 배치 T=1.0279 는 **키우는** 방향으로 맞춰둔 값이다.")
