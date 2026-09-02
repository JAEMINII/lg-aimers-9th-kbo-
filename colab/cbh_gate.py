# -*- coding: utf-8 -*-
"""cbhybrid 검증 — 준혁 CB(폴드 재학습) + candidate58(기존50+신규8) 쌍대조 3폴드.

저장: cbh_{VS}_{base|cand}.npy (3시드 평균, 0.6/0.4 라우팅), cbh_{VS}_jh.npy
판정은 로컬 믹스 채점에서."""
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
import current_catboost_features as CF                          # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402

raw = pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig")
JH = dict(iterations=230, learning_rate=0.05, l2_leaf_reg=3,
          loss_function="Logloss", random_seed=42, verbose=0,
          random_strength=1, allow_writing_files=False,
          bootstrap_type="MVS", subsample=0.8)
hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")

for VS in (2024, 2022, 2023):
    # ---- 준혁 CB 폴드 재학습 (그들 파이프라인 그대로, 무가중)
    t0 = time.time()
    hist = raw[raw.season < VS]
    prior = float(hist["control_success"].mean())
    profiles = CF.build_matchup_profiles(hist)
    tr_f = CF.transform(hist, prior, profiles)
    feats, cats = CF.feature_names(tr_f)
    tr_f = CF.prepare_categoricals(tr_f, cats)
    va_raw = raw[raw.season == VS]
    va_f = CF.prepare_categoricals(CF.transform(va_raw, prior, profiles), cats)
    mo = CatBoostClassifier(cat_features=cats, **JH)
    mo.fit(tr_f[feats], hist["control_success"].to_numpy())
    jh = mo.predict_proba(va_f[feats])[:, 1]
    np.save(DL + f"/cbh_{VS}_jh.npy", jh)
    print(f"VS{VS} 준혁CB {time.time()-t0:.0f}s  평균 {jh.mean():.4f}", flush=True)

    # ---- candidate58 vs base50 쌍대조
    built = F.build(DATA, VS=VS, return_frame=True)
    fr = built["frame"]
    base_f = list(built["F44"])
    fr, _ = T.add_c12(fr, return_tables=True)
    fr, _ = T.add_cmh(fr, return_tables=True)
    feats50 = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                        "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
    for ph in (1, 2):
        for bh in (1, 2):
            fr[f"hand_match_{ph}_{bh}"] = (
                (fr["pitcher_hand"] == ph) & (fr["batter_hand"] == bh)
            ).astype(np.float32)
    career = fr["asof_pitcher_success_rate"]
    fr["pitcher_prev1_success_dev"] = (
        fr["asof_pitcher_prev1_game_success_rate"] - career).astype(np.float32)
    fr["pitcher_prev3_success_dev"] = (
        fr["asof_pitcher_prev3_game_success_rate"] - career).astype(np.float32)
    fr["pitcher_success_trend_1v5"] = (
        fr["asof_pitcher_prev1_game_success_rate"]
        - fr["asof_pitcher_prev5_game_success_rate"]).astype(np.float32)
    fr["pitcher_middle_trend_1v5"] = (
        fr["asof_pitcher_prev1_game_middle_rate"]
        - fr["asof_pitcher_prev5_game_middle_rate"]).astype(np.float32)
    feats58 = feats50 + [f"hand_match_{p}_{b}" for p in (1, 2) for b in (1, 2)] \
        + ["pitcher_prev1_success_dev", "pitcher_prev3_success_dev",
           "pitcher_success_trend_1v5", "pitcher_middle_trend_1v5"]
    y = fr["control_success"].to_numpy(np.float64)
    season = fr["season"].to_numpy()
    gt = fr["game_type"].to_numpy()
    tr_m, te_m = season < VS, season == VS
    isf_t = (gt[te_m] == 1) | (gt[te_m] == "F")
    isf_tr = (gt[tr_m] == 1) | (gt[tr_m] == "F")
    w = 2.0 ** (season[tr_m].astype(np.float64) - 2019.0)
    masks = {"all": np.ones(int(tr_m.sum()), bool),
             "regular": ~isf_tr, "futures": isf_tr}
    for tag, cols in (("base", feats50), ("cand", feats58)):
        X = fr[cols].to_numpy(np.float32)
        ps = []
        for sd_ in (1, 42, 777):
            pg = {}
            for g, mk in masks.items():
                mo = CatBoostClassifier(random_seed=sd_, **hp)
                mo.fit(X[tr_m][mk], y[tr_m][mk], sample_weight=w[mk])
                pg[g] = mo.predict_proba(X[te_m])[:, 1]
            ps.append(np.where(isf_t, .6 * pg["all"] + .4 * pg["futures"],
                               .6 * pg["all"] + .4 * pg["regular"]))
        np.save(DL + f"/cbh_{VS}_{tag}.npy", np.mean(ps, 0))
        print(f"VS{VS} {tag} 완료 {time.time()-t0:.0f}s", flush=True)
print("cbh 끝", flush=True)
