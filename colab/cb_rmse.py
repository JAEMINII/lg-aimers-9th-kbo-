# -*- coding: utf-8 -*-
"""CB-RMSE — CatBoostRegressor 로 0/1 표적을 제곱손실 직접 최적화. VS=2024.

품질 개선(감쇠 강화)은 혼합 기여 0 이었다. 이건 품질이 아니라 오차 구조를
바꾸는 방향 — 같은 트리가 '얼마나 성공하나' 대신 '어떻게 실패하나'를 배운다.
배치 CB 와 동일: 50열, 3분기(all/regular/futures) 0.6/0.4 라우팅, 감쇠 2.0, 시드 3.
저장: cbr_2024.npy (시드 스택)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
sys.path.insert(0, ROOT + "/nn_experiments")
os.environ.setdefault("AIMERS_ROOT", "/root")
os.environ.setdefault("AUDIT_MODE", "all")
os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
import multistate_softmax as M                                  # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
from catboost import CatBoostRegressor                          # noqa: E402


def recover_state6(df):
    s = M.recover_state(df).copy()
    pid = df.pitcher_id.to_numpy()
    n = df.asof_pitcher_n.fillna(0.).to_numpy(float)
    nxt = (pid[1:] == pid[:-1]) & np.isclose(np.diff(n), 1., atol=1e-8)
    src = np.flatnonzero(nxt) + 1
    dst = src - 1
    cum = df["asof_pitcher_ball_rate"].fillna(0.).to_numpy(float) * n
    inc = cum[src] - cum[dst]
    lab = np.rint(inc)
    good = (np.abs(inc - lab) < .25) & ((lab == 0) | (lab == 1))
    ball = np.full(len(df), np.nan)
    ball[dst[good]] = lab[good]
    out = s.copy()
    m3 = s == 3
    out[m3 & (ball == 1)] = 3
    out[m3 & (ball == 0)] = 4
    out[m3 & ~np.isfinite(ball)] = -1
    return out


raw = PP.sort_by_row_id(pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig"))
aux_map = dict(zip(raw.row_id.tolist(), recover_state6(raw).tolist()))

built = F.build(DATA, VS=2024, return_frame=True)
fr = built["frame"]
base_f = list(built["F44"])
fr, _ = T.add_c12(fr, return_tables=True)
fr, _ = T.add_cmh(fr, return_tables=True)
feats = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                  "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
X = fr[feats].to_numpy(np.float32)
y = fr["control_success"].to_numpy(np.float64)
aux = np.array([aux_map.get(r, -1) for r in fr["row_id"].tolist()], np.int64)
season = fr["season"].to_numpy()
gt = fr["game_type"].to_numpy()
tr_m, te_m = season < 2024, season == 2024
isf_t = gt[te_m] == 1
yv = y[te_m]
print(f"라벨 커버 {np.mean(aux[tr_m] >= 0):.4f}", flush=True)
hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0",
          loss_function="RMSE")
masks = {"all": np.ones(int(tr_m.sum()), bool),
         "regular": gt[tr_m] == 0, "futures": gt[tr_m] == 1}
sc = lambda p, m=None: F.best_shift(p if m is None else p[m],
                                    yv if m is None else yv[m])[0]
w_all = 2.0 ** (season[tr_m].astype(np.float64) - 2019.0)
aux_tr = aux[tr_m]
ok = aux_tr >= 0
ps = []
for sd in (1, 42, 777):
    t0 = time.time()
    pg = {}
    for g, m in masks.items():
        mo = CatBoostRegressor(random_seed=sd, **hp)
        mo.fit(X[tr_m][m], y[tr_m][m], sample_weight=w_all[m])
        pg[g] = np.clip(mo.predict(X[te_m]), 1e-6, 1 - 1e-6)
    p = np.where(isf_t, .6 * pg["all"] + .4 * pg["futures"],
                 .6 * pg["all"] + .4 * pg["regular"])
    ps.append(p)
    print(f"  cbr s{sd}  {time.time()-t0:.0f}s  단독 {sc(p):.1f}  "
          f"1군 {sc(p, ~isf_t):.1f}  퓨처스 {sc(p, isf_t):.1f}", flush=True)
np.save(DL + "/cbr_2024.npy", np.asarray(ps))
p = np.mean(ps, 0)
print(f"ARM cbr 시드평균 단독 {sc(p):8.1f}  1군 {sc(p, ~isf_t):8.1f}  "
      f"퓨처스 {sc(p, isf_t):8.1f}", flush=True)
