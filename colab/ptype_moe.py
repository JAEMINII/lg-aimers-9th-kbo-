# -*- coding: utf-8 -*-
"""구종별 expert marginalization — p(x) = Σ_k P(K=k|x)·f_k(x). VS 는 PMOE_VS.

전문가 f_k: 구종그룹별 CatBoost (매칭 1군 행에서 학습, K 라벨 = lupi_match2 구종).
게이트 P(K|투수×카운트×손): 사전계산 백오프 표 (tm_pk_{cut}.csv) — y 를 안 본다.
판정은 로컬 기부자 G-검사. 저장: pmoe_{VS}.npy (비커버 행 NaN).
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
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
VS = int(os.environ.get("PMOE_VS", "2024"))
CUT = VS - 1
import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402

built = F.build(DATA, VS=VS, return_frame=True)
fr = built["frame"]
base_f = list(built["F44"])
fr, _ = T.add_c12(fr, return_tables=True)
fr, _ = T.add_cmh(fr, return_tables=True)
feats = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                  "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
X = fr[feats].to_numpy(np.float32)
y = fr["control_success"].to_numpy(np.float64)
season = fr["season"].to_numpy()
gt = fr["game_type"].to_numpy()
rid = fr["row_id"].to_numpy()

# K 라벨 (매칭 행만)
mpm = pd.read_csv(DL + "/lupi_match2.csv.gz", usecols=["row_id", "pitch_type_group"])
kmap = dict(zip(mpm.row_id.tolist(), mpm.pitch_type_group.tolist()))
KNAMES = ("fastball", "breaking", "offspeed")
kl = np.array([KNAMES.index(kmap[r]) if kmap.get(r) in KNAMES else -1
               for r in rid.tolist()], np.int64)

# 게이트 표
cellt = pd.read_csv(DL + f"/tm_pk_{CUT}.csv")
lgt = pd.read_csv(DL + f"/tm_pk_league_{CUT}.csv")
gate_cell = {(int(r.pitcher_id), int(r.cg), int(r.bh)): (r.fb_cell, r.br_cell, r.os_cell)
             for r in cellt.itertuples()}
gate_lg = {(int(r.cg), int(r.bh)): (r.fb_lg, r.br_lg, r.os_lg) for r in lgt.itertuples()}
mapped_pids = set(cellt.pitcher_id.tolist())

# 행별 게이트 확률
bhv = fr["batter_hand"].to_numpy()
cgv = np.where(fr["strikes_before"].to_numpy() == 2, 2,
               np.where(fr["balls_before"].to_numpy() >= 2, 1, 0))
pidv = fr["pitcher_id"].to_numpy()
PK = np.full((len(fr), 3), np.nan)
for i in range(len(fr)):
    pid_i = int(pidv[i])
    if pid_i not in mapped_pids:
        continue
    v = gate_cell.get((pid_i, int(cgv[i]), int(bhv[i])))
    if v is None:
        v = gate_lg.get((int(cgv[i]), int(bhv[i])))
    PK[i] = v
tr_m = season < VS
te_m = season == VS
isf_t = gt[te_m] == 1
yv = y[te_m]
w_dec = 2.0 ** (season.astype(np.float64) - 2019.0)
hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
sc = lambda p, m: F.best_shift(p[m], yv[m])[0]

ps = []
for sd in (1, 42, 777):
    t0 = time.time()
    fk = np.zeros((int(te_m.sum()), 3))
    for k in range(3):
        m = tr_m & (kl == k)
        mo = CatBoostClassifier(random_seed=sd, **hp)
        mo.fit(X[m], y[m], sample_weight=w_dec[m])
        fk[:, k] = mo.predict_proba(X[te_m])[:, 1]
    p = np.nansum(PK[te_m] * fk, 1)
    p[~np.isfinite(PK[te_m][:, 0])] = np.nan
    ps.append(p)
    okr = np.isfinite(p) & ~isf_t
    print(f"  pmoe s{sd}  {time.time()-t0:.0f}s  커버1군 {okr.mean():.3f}  "
          f"단독(커버) {sc(np.where(np.isfinite(p), p, 0.5), okr):.1f}", flush=True)
np.save(DL + f"/pmoe_{VS}.npy", np.asarray(ps))
pm = np.nanmean(np.asarray(ps), 0)
okr = np.isfinite(pm) & ~isf_t
print(f"ARM pmoe VS{VS} 시드평균 단독(커버1군) {sc(np.where(np.isfinite(pm), pm, 0.5), okr):8.1f}  "
      f"커버 {okr.sum():,}", flush=True)
