# -*- coding: utf-8 -*-
"""submit_41 의 두 가지 위험을 정량화한다.

1. 시간   구성요소별로 나눠 잰다. 모델 적재는 1회, 나머지는 행수 비례다.
2. 미스율 2024 로 재면 0% 다 — 표가 2024 를 포함해 만들어졌으니 당연하다.
          2025 는 처음 보는 시즌이다. 표를 <2024 로 만들고 2024 를 조회해 대신 잰다.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "open (1)", "data")
PKG = "submit_41"
N = 20000

cols = list(pd.read_csv(os.path.join(DATA, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
sub = pool.iloc[np.sort(rng.choice(len(pool), N, replace=False))][cols].reset_index(drop=True)

cwd = os.getcwd()
os.chdir(os.path.join(ROOT, PKG))
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("p41", "script.py")
S = importlib.util.module_from_spec(spec); S.__dict__["__name__"] = "p41"
spec.loader.exec_module(S)
import preprocess as PPF

R = 245789 / N
t = {}
t0 = time.time(); history, z, shift = S.load_model(); t["모델적재(1회)"] = time.time()-t0
t0 = time.time()
with open("model/history.json", encoding="utf-8") as f:
    hist_pkg = PPF.deserialize_history(json.load(f))
t["history 역직렬화(1회)"] = time.time()-t0
t0 = time.time(); X = S.build_features(sub, history); t["CatBoost 피처"] = time.time()-t0
t0 = time.time(); p_new = S.predict_numpy(X.values.astype(np.float64), z); t["CatBoost 신50 순회"] = time.time()-t0
t0 = time.time()
p_old = S.predict_numpy(X[history["_base_features"]].values.astype(np.float64), history["_base_z"])
t["CatBoost 구44 순회"] = time.time()-t0
t0 = time.time()
ordered = PPF.sort_by_row_id(sub); Xf = PPF.build_inference_features(ordered, hist_pkg)
t["TabM 전처리"] = time.time()-t0
isf_o = ordered["game_type"].astype(str).to_numpy() == "F"
Xv = np.c_[Xf.to_numpy(dtype=np.float64), np.where(isf_o, 2.0, 3.0)]
zc = {b: np.load(S.resolve(f"model/f2b_{b}_s42.npz"), allow_pickle=False)
      for b in ("all", "regular", "futures")}
def fwd(b, rows):
    mt = json.loads(str(zc[b]["meta"].item())); mt["cat_cardinalities"] = mt["cards"]
    Tn, Tc = S._fm_prep(Xv[rows], zc[b]); return S._tabm_forward(Tn, Tc, mt, zc[b], 4096)
t0 = time.time(); fwd("all", np.arange(len(Xf))); t["TabM all 순전파"] = time.time()-t0
t0 = time.time(); fwd("regular", np.flatnonzero(~isf_o)); t["TabM regular"] = time.time()-t0
t0 = time.time(); fwd("futures", np.flatnonzero(isf_o)); t["TabM futures"] = time.time()-t0

print(f"  구성요소별 시간   표본 {N:,}행 -> 245,789행 환산 (x{R:.1f})\n")
fixed = sum(v for k, v in t.items() if "1회" in k)
scaled = sum(v for k, v in t.items() if "1회" not in k)
for k, v in t.items():
    ex = v if "1회" in k else v * R
    print(f"    {k:22s} {v:7.2f}초  ->  {ex/60:5.2f}분")
tot = fixed + scaled * R
print(f"    {'-'*22} {'':7s}      {'-'*11}")
print(f"    {'합계':22s} {'':7s}      {tot/60:5.2f}분   예산 10분  "
      f"{'**초과**' if tot > 600 else '여유 ' + format((600-tot)/60, '.2f') + '분'}")

# ------------------------------------------------- 처음 보는 시즌의 미스율
print("\n  처음 보는 시즌에서의 조회 미스 — 표를 <2024 로 만들고 2024 를 조회한다")
d = tr.copy()
d["cnt12"] = d.balls_before.astype("int64")*3 + d.strikes_before.astype("int64")
d["pcmh"] = d.cnt12*2 + d.batter_hand.astype("int64")
hist_rows, new_rows = d[d.season < 2024], d[d.season == 2024]
for tag, kcol, mul in (("pc_key  투수x카운트", "cnt12", 16),
                       ("ph_key  투수x카운트x손", "pcmh", 32)):
    ks = np.sort(np.unique(hist_rows.pitcher_id.to_numpy(np.int64)*mul
                           + hist_rows[kcol].to_numpy(np.int64)))
    q = new_rows.pitcher_id.to_numpy(np.int64)*mul + new_rows[kcol].to_numpy(np.int64)
    pos = np.searchsorted(ks, q); safe = np.minimum(pos, len(ks)-1)
    true_hit = (pos < len(ks)) & (ks[safe] == q)
    wrong = (pos < len(ks)) & ~true_hit
    print(f"    {tag:24s} 표 {len(ks):>7,}  미스 {int((~true_hit).sum()):>7,} "
          f"({(~true_hit).mean()*100:5.2f}%)   그중 **이웃 투수의 값을 읽는 행** "
          f"{int(wrong.sum()):>7,} ({wrong.mean()*100:5.2f}%)")
pid_h = np.sort(np.unique(hist_rows.pitcher_id.to_numpy(np.int64)))
q = new_rows.pitcher_id.to_numpy(np.int64)
pos = np.searchsorted(pid_h, q); safe = np.minimum(pos, len(pid_h)-1)
th = (pos < len(pid_h)) & (pid_h[safe] == q)
print(f"    {'투수 자체가 처음':24s} 표 {len(pid_h):>7,}  미스 {int((~th).sum()):>7,} "
      f"({(~th).mean()*100:5.2f}%)")
sys.path.pop(0); os.chdir(cwd)
