# -*- coding: utf-8 -*-
"""팀 전년우위 절제 — b58 + teamadv2 (전년도 팀 관여우위 p/b). 3폴드.
저장: cbh6_{VS}_ta2.npy"""
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
os.environ.setdefault("C12_OUT", "/root/_c12logs")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402

hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
raw_sorted = None

for VS in (2024, 2022, 2023):
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
    car = fr["asof_pitcher_success_rate"]
    fr["pitcher_prev1_success_dev"] = (
        fr["asof_pitcher_prev1_game_success_rate"] - car).astype(np.float32)
    fr["pitcher_prev3_success_dev"] = (
        fr["asof_pitcher_prev3_game_success_rate"] - car).astype(np.float32)
    fr["pitcher_success_trend_1v5"] = (
        fr["asof_pitcher_prev1_game_success_rate"]
        - fr["asof_pitcher_prev5_game_success_rate"]).astype(np.float32)
    fr["pitcher_middle_trend_1v5"] = (
        fr["asof_pitcher_prev1_game_middle_rate"]
        - fr["asof_pitcher_prev5_game_middle_rate"]).astype(np.float32)
    feats58 = feats50 + [f"hand_match_{p}_{b}" for p in (1, 2) for b in (1, 2)] \
        + ["pitcher_prev1_success_dev", "pitcher_prev3_success_dev",
           "pitcher_success_trend_1v5", "pitcher_middle_trend_1v5"]

    # --- 현시즌 세부율 4종: 통산누적 - 학습말 상수 (train<VS 의 투수별 최종 누적)
    season = fr["season"].to_numpy()
    tr_m = season < VS
    sub = fr[tr_m]
    n_tr = sub["asof_pitcher_n"].fillna(0.).to_numpy(float)
    consts = {}
    idx = sub.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    last = fr.loc[idx]
    cn = last["asof_pitcher_n"].fillna(0.).to_numpy(float)
    base_tbl = {"n": dict(zip(last["pitcher_id"], cn))}
    for c in ("reverse", "middle", "ball", "strike"):
        cnt = (last[f"asof_pitcher_{c}_rate"].fillna(0.).to_numpy(float) * cn)
        base_tbl[c] = dict(zip(last["pitcher_id"], cnt))
    pid = fr["pitcher_id"].to_numpy()
    nall = fr["asof_pitcher_n"].fillna(0.).to_numpy(float)
    n0 = np.array([base_tbl["n"].get(p, 0.0) for p in pid])
    cur_n = np.clip(nall - n0, 0, None)
    fr["cs_n"] = np.log1p(cur_n).astype(np.float32)
    for c in ("reverse", "middle", "ball", "strike"):
        tot = fr[f"asof_pitcher_{c}_rate"].fillna(0.).to_numpy(float) * nall
        c0 = np.array([base_tbl[c].get(p, 0.0) for p in pid])
        cur = np.clip(tot - c0, 0, None)
        rate = np.where(cur_n >= 10, cur / np.maximum(cur_n, 1), np.nan)
        fr[f"cs_{c}_rate"] = rate.astype(np.float32)
    cs_cols = ["cs_n"] + [f"cs_{c}_rate" for c in ("reverse", "middle", "ball", "strike")]

    # --- team13 전환 지시자 (train 통계로 독립 검증됨)
    inv = ((fr["pitcher_team_id"] == 13) | (fr["batter_team_id"] == 13))
    trans = inv & ((fr["season"] > 2023)
                   | ((fr["season"] == 2023)
                      & ((fr["game_month"] >= 5)
                         | (fr["game_type"].astype(str).isin(["F", "1"])))))
    fr["t13_inv"] = inv.astype(np.float32)
    fr["t13_trans"] = trans.astype(np.float32)
    t13_cols = ["t13_inv", "t13_trans"]

    y = fr["control_success"].to_numpy(np.float64)
    gt = fr["game_type"].to_numpy()
    te_m = season == VS
    isf_t = (gt[te_m] == 1) | (gt[te_m] == "F")
    isf_tr = (gt[tr_m] == 1) | (gt[tr_m] == "F")
    w = 2.0 ** (season[tr_m].astype(np.float64) - 2019.0)
    masks = {"all": np.ones(int(tr_m.sum()), bool),
             "regular": ~isf_tr, "futures": isf_tr}
    ta = np.load(DL + "/teamadv.npy").astype(np.float32)
    assert len(ta) == len(fr)
    fr["teamadv_p"] = ta[:, 0]
    fr["teamadv_b"] = ta[:, 1]
    for tag, cols in (("ta2", feats58 + ["teamadv_p", "teamadv_b"]),):
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
        np.save(DL + f"/cbh6_{VS}_{tag}.npy", np.mean(ps, 0))
        print(f"VS{VS} {tag} 완료", flush=True)
print("cbh6 끝", flush=True)
