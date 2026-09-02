# -*- coding: utf-8 -*-
"""submit_41 (LB 1083) 을 동료에게 넘기기 전에 검증한다.

무엇을 보나
    1. 실제로 도는가 — 2024 행을 test.csv 대신 먹인다 (test.csv 는 5행 견본이다)
    2. 규칙 4 — 순서를 섞어도, 부분집합만 줘도 각 행의 값이 같아야 한다
    3. 조회표 정합 — script.py 의 searchsorted 가 **동등성 검사를 안 한다**.
       키가 표에 없으면 이웃 칸의 n/s 를 읽는다. 그 비율을 잰다.
    4. 시간 — 24.6만 행 환산

2024 행은 배치 모델에게 인샘플이라 점수는 못 믿는다. 여기서 보는 건 정합성이다.
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
PKG = sys.argv[1] if len(sys.argv) > 1 else "submit_41"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

cols = list(pd.read_csv(os.path.join(DATA, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), N, replace=False))
sub = pool.iloc[pick][cols].reset_index(drop=True)
y = pool.iloc[pick]["control_success"].to_numpy(np.float64)
isf = sub["game_type"].astype(str).to_numpy() == "F"
print(f"  {PKG}   2024 행 {len(sub):,} 표본   퓨처스 {int(isf.sum()):,} "
      f"({isf.mean()*100:.1f}%)\n")

cwd = os.getcwd()
os.chdir(os.path.join(ROOT, PKG))
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location(f"pkg_{PKG}", "script.py")
S = importlib.util.module_from_spec(spec)
S.__dict__["__name__"] = f"pkg_{PKG}"
spec.loader.exec_module(S)
history, z, shift = S.load_model()
print(f"  모델 {int(z['n_models'])}  그룹당 {int(z['n_per_group'])}  "
      f"피처 {len(history['features'])}  shift {shift:+.4f}  "
      f"W_CB_NEW {getattr(S, 'W_CB_NEW', float('nan')):.2f}")


def full(frame):
    t0 = time.time()
    X = S.build_features(frame, history)
    p_cb = S.predict_numpy(X.values.astype(np.float64), z)
    if "_base_z" in history:
        p_old = S.predict_numpy(
            X[history["_base_features"]].values.astype(np.float64),
            history["_base_z"])
        p_cb = S.W_CB_NEW * p_cb + (1.0 - S.W_CB_NEW) * p_old
    import preprocess as PPF
    with open("model/history.json", encoding="utf-8") as f:
        hist_pkg = PPF.deserialize_history(json.load(f))
    ordered = PPF.sort_by_row_id(frame)
    Xf = PPF.build_inference_features(ordered, hist_pkg)
    isf_o = ordered["game_type"].astype(str).to_numpy() == "F"
    c4 = np.where(isf_o, 2.0, 3.0)
    Xv = np.c_[Xf.to_numpy(dtype=np.float64), c4]

    def f2b(branch, rows):
        outs = []
        for sd in S.F2B_SEEDS:
            z2 = np.load(S.resolve(f"model/f2b_{branch}_s{sd}.npz"),
                         allow_pickle=False)
            mt = json.loads(str(z2["meta"].item()))
            mt["cat_cardinalities"] = mt["cards"]
            Tn, Tc = S._fm_prep(Xv[rows], z2)
            outs.append(S._tabm_forward(Tn, Tc, mt, z2, 4096))
        return np.mean(outs, axis=0)

    p = f2b("all", np.arange(len(Xf)))
    rr = np.flatnonzero(~isf_o)
    if len(rr):
        p[rr] = 0.6 * p[rr] + 0.4 * f2b("regular", rr)
    rf = np.flatnonzero(isf_o)
    if len(rf):
        p[rf] = 0.6 * p[rf] + 0.4 * f2b("futures", rf)
    tm = dict(zip(ordered[S.ID_COL].tolist(), np.clip(p, 0, 1)))
    p_tabm = np.array([tm[r] for r in frame[S.ID_COL].tolist()], np.float64)
    out = S.apply_calibration(S.W_CB * p_cb + S.W_TABM * p_tabm, shift)
    return out, time.time() - t0, p_cb, p_tabm


p41, sec, p_cb, p_tabm = full(sub)
print(f"\n  예측 평균 {p41.mean():.5f}  실제 {y.mean():.5f}  "
      f"CatBoost {p_cb.mean():.5f}  TabM {p_tabm.mean():.5f}  "
      f"상관 {np.corrcoef(p_cb, p_tabm)[0,1]:.4f}")
print(f"  {N:,}행 {sec:.1f}초  ->  245,789행 환산 {sec*245789/N/60:.1f}분 "
      f"(예산 10분)")

# ---------------------------------------------------------------- 규칙 4
per = rng.permutation(len(sub))
p_perm, _, _, _ = full(sub.iloc[per].reset_index(drop=True))
d1 = float(np.abs(p_perm - p41[per]).max())
half = np.arange(0, len(sub), 5)
p_half, _, _, _ = full(sub.iloc[half].reset_index(drop=True))
d2 = float(np.abs(p_half - p41[half]).max())
print(f"\n  규칙 4  순서 섞기 최대차 {d1:.3e}  {'통과' if d1 < 1e-9 else '**위반**'}")
print(f"          20% 부분집합  최대차 {d2:.3e}  {'통과' if d2 < 1e-9 else '**위반**'}")

# ---------------------------------------------------------------- 조회표 정합
print("\n  조회표 — script.py 의 searchsorted 는 동등성 검사를 안 한다")
X = S.build_features(sub, history)
cnt = (sub["balls_before"].astype("int64") * 3
       + sub["strikes_before"].astype("int64")).to_numpy()
pid = sub["pitcher_id"].astype("int64").to_numpy()
hand = sub["batter_hand"].astype("int64").to_numpy()
for tag, key, tkey in (("pc_key (투수x카운트)", pid * 16 + cnt, "pc_key"),
                       ("pc_pid (투수)", pid, "pc_pid"),
                       ("ph_key (투수x카운트x손)", pid * 32 + (cnt * 2 + hand), "ph_key"),
                       ("ph_pid (투수)", pid, "ph_pid")):
    ks = np.asarray(history[tkey], np.int64)
    pos = np.searchsorted(ks, key)
    safe = np.minimum(pos, len(ks) - 1)
    coded_hit = pos < len(ks)                 # script.py 가 쓰는 판정
    true_hit = (pos < len(ks)) & (ks[safe] == key)   # 올바른 판정
    wrong = coded_hit & ~true_hit
    print(f"    {tag:26s} 표 {len(ks):>7,}  실제 미스 {int((~true_hit).sum()):>6,} "
          f"({(~true_hit).mean()*100:5.2f}%)  그중 이웃값을 읽은 행 "
          f"{int(wrong.sum()):>6,} ({wrong.mean()*100:5.2f}%)")
sys.path.pop(0)
os.chdir(cwd)
