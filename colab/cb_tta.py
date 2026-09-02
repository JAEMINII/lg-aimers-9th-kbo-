# -*- coding: utf-8 -*-
"""CB 전용 반대칭 TTA — rate 류만 불확실도 비례 섭동 (x, x+cσ, x-cσ).

σ_r = sqrt(r(1-r)/(n+50)). 평균 보존형: p = .5 p(x) + .25 p(x+) + .25 p(x-).
트리 hard split 경계 스무딩이 목적. id/카운트/손/체제 등 이산열은 불변.
폴드 CB(50열, 감쇠 2.0, 3브랜치x3시드)를 재학습해 base/TTA 쌍으로 저장:
cbt_{VS}_{arm}.npy (arm: base, tta05, tta10)."""
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
import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402

KAP = 50.0
hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")

for VS in (2024, 2022, 2023):
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
    tr_m, te_m = season < VS, season == VS
    isf_t = (gt[te_m] == 1) | (gt[te_m] == "F")
    w = 2.0 ** (season[tr_m].astype(np.float64) - 2019.0)

    # (rate열, n열) 쌍 — 존재하는 것만
    pairs = []
    for rc, nc in (
            ("asof_pitcher_success_rate", "asof_pitcher_n"),
            ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
            ("asof_pitcher_middle_rate", "asof_pitcher_n"),
            ("asof_pitcher_ball_rate", "asof_pitcher_n"),
            ("asof_pitcher_strike_rate", "asof_pitcher_n"),
            ("asof_batter_success_rate", "asof_batter_n"),
            ("asof_batter_middle_rate", "asof_batter_n"),
            ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
            ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
            ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
            ("pc_c12_rate", "pc_c12_n"),
            ("pc_cmh_rate", "pc_cmh_n")):
        if rc in feats and nc in feats:
            pairs.append((feats.index(rc), feats.index(nc)))
    print(f"VS{VS} 섭동쌍 {len(pairs)}", flush=True)

    Xte = X[te_m]
    def perturbed(sign, c):
        Xp = Xte.copy()
        for ri, ni in pairs:
            r = np.clip(Xp[:, ri], 0, 1)
            nn_ = np.clip(Xp[:, ni], 0, None)
            sd = np.sqrt(r * (1 - r) / (nn_ + KAP))
            Xp[:, ri] = np.clip(r + sign * c * sd, 0, 1)
        return Xp

    masks = {"all": np.ones(int(tr_m.sum()), bool),
             "regular": ~((gt[tr_m] == 1) | (gt[tr_m] == "F")),
             "futures": (gt[tr_m] == 1) | (gt[tr_m] == "F")}
    acc = {a: [] for a in ("base", "tta05", "tta10")}
    for sd_ in (1, 42, 777):
        pg = {}
        for g, mk in masks.items():
            t0 = time.time()
            mo = CatBoostClassifier(random_seed=sd_, **hp)
            mo.fit(X[tr_m][mk], y[tr_m][mk], sample_weight=w[mk])
            pr = {"0": mo.predict_proba(Xte)[:, 1]}
            for c, tag in ((0.5, "05"), (1.0, "10")):
                pr["p" + tag] = mo.predict_proba(perturbed(+1, c))[:, 1]
                pr["m" + tag] = mo.predict_proba(perturbed(-1, c))[:, 1]
            pg[g] = pr
            print(f"  VS{VS} s{sd_} {g}  {time.time()-t0:.0f}s", flush=True)

        def route(key):
            return np.where(isf_t,
                            .6 * pg["all"][key] + .4 * pg["futures"][key],
                            .6 * pg["all"][key] + .4 * pg["regular"][key])
        acc["base"].append(route("0"))
        for tag in ("05", "10"):
            acc["tta" + tag].append(.5 * route("0") + .25 * route("p" + tag)
                                    + .25 * route("m" + tag))
    for a, ps in acc.items():
        np.save(DL + f"/cbt_{VS}_{a}.npy", np.mean(ps, 0))
print("CB TTA 끝", flush=True)
