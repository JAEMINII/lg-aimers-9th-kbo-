# -*- coding: utf-8 -*-
"""submit_44 의 수준(평균)이 41 과 얼마나 다른지 재고, 맞는 시프트를 계산한다.

왜
    44 = 0.75*(0.30 CatBoost + 0.70 TabM) + 0.25*DIN 인데 보정은 41용
    shift=-0.007 을 그대로 쓴다. DIN 예측 평균이 높으면 수준이 어긋난다.
    "모델 바꾸면 시프트도 다시 잡아라" — 예측평균 +0.005 방치로 10~20점 잃은 전례.
    시프트 곡률은 25000 (리더보드 실측), 분해능 ±0.009.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "open (1)", "data")
N = 20000
cols = list(pd.read_csv(os.path.join(DATA, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), N, replace=False))
sub = pool.iloc[pick][cols].reset_index(drop=True)
y = pool.iloc[pick]["control_success"].to_numpy(np.float64)

os.chdir(os.path.join(ROOT, "submit_44"))
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("p44", "script.py")
S = importlib.util.module_from_spec(spec); S.__dict__["__name__"] = "p44"
spec.loader.exec_module(S)
import preprocess as PPF
history, z, shift = S.load_model()
X = S.build_features(sub, history)
p_new = S.predict_numpy(X.values.astype(np.float64), z)
p_old = S.predict_numpy(X[history["_base_features"]].values.astype(np.float64),
                        history["_base_z"])
p_cb = S.W_CB_NEW * p_new + (1 - S.W_CB_NEW) * p_old

with open("model/history.json", encoding="utf-8") as f:
    hp = PPF.deserialize_history(json.load(f))
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
rr = np.flatnonzero(~isf_o); pt[rr] = 0.6*pt[rr] + 0.4*f2b("regular", rr)
rf = np.flatnonzero(isf_o);  pt[rf] = 0.6*pt[rf] + 0.4*f2b("futures", rf)
tm = dict(zip(ordered[S.ID_COL].tolist(), np.clip(pt, 0, 1)))
p_tabm = np.array([tm[r] for r in sub[S.ID_COL].tolist()], np.float64)

DM = np.load(S.resolve("model/din_meta.npz"), allow_pickle=False)
X45 = Xv
dXn, dXc = S.din_prep(X45, DM)
dS, dC, dMk, dcur = S.din_seq(ordered["pitcher_id"].to_numpy(),
                              ordered["balls_before"].to_numpy(),
                              ordered["strikes_before"].to_numpy(),
                              ordered["batter_hand"].to_numpy(), DM)
acc = {br: np.mean([S.din_forward(dXn, dXc, dS, dC, dMk, dcur,
                                  np.load(S.resolve(f"model/din_{br}_s{sd}.npz"),
                                          allow_pickle=False))
                    for sd in S.DIN_SEEDS], 0)
       for br in ("all", "regular", "futures")}
pd_ord = np.where(isf_o, 0.6*acc["all"] + 0.4*acc["futures"],
                  0.6*acc["all"] + 0.4*acc["regular"])
dmap = dict(zip(ordered[S.ID_COL].tolist(), pd_ord))
p_din = np.array([dmap[r] for r in sub[S.ID_COL].tolist()], np.float64)

base = S.W_CB * p_cb + S.W_TABM * p_tabm            # 41 혼합
for W in (0.15, 0.25):
    mix = (1 - W) * base + W * p_din
    dz = float(np.mean(S._logit(mix)) - np.mean(S._logit(base)))
    T = S.CALIB_T
    rec = shift - T * dz
    print(f"  w={W:.2f}  혼합평균 {mix.mean():.5f} (기존 {base.mean():.5f}, "
          f"Δp {mix.mean()-base.mean():+.5f})")
    print(f"          로짓수준차 Δz {dz:+.5f}   현행 shift {shift:+.4f}  "
          f"권장 {rec:+.4f}   방치비용 ≈ {25000*(T*dz)**2:.1f}점")
pa = S.apply_calibration((1-0.25)*base + 0.25*p_din, shift)
pb = S.apply_calibration(base, shift)
print(f"\n  실제 {y.mean():.5f}   41식 보정후 {pb.mean():.5f} "
      f"({pb.mean()-y.mean():+.5f})   44식 보정후 {pa.mean():.5f} "
      f"({pa.mean()-y.mean():+.5f})")
print(f"  DIN 평균 {p_din.mean():.5f}   base 와 상관 {np.corrcoef(base, p_din)[0,1]:.4f}")
