# -*- coding: utf-8 -*-
"""t13 강화 검증 — 전환 후 데이터를 학습에 포함하는 롤링 분할 2개.
A: 학습 <2024 + 2024 상반기(월<=6) -> 2024 하반기 채점
B: 학습 <2023 + 2023 월<=8      -> 2023 9~10월 채점 (1군만)
팔: b58 vs t13 (쌍시드). 저장: cbh7_{A|B}_{b58|t13}.npy + 채점 즉석 출력."""
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


def bss(p, t):
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


SPLITS = {
    "A": dict(VS=2024, tr_extra_mo=6, va_mo=7, r_only=False),
    "B": dict(VS=2023, tr_extra_mo=8, va_mo=9, r_only=True),
}
for tag, cfg in SPLITS.items():
    VS = cfg["VS"]
    built = F.build(DATA, VS=VS + 1, return_frame=True)   # 룩업창 <=VS 로
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
    inv = ((fr["pitcher_team_id"] == 13) | (fr["batter_team_id"] == 13))
    trans = inv & ((fr["season"] > 2023)
                   | ((fr["season"] == 2023)
                      & ((fr["game_month"] >= 5)
                         | (fr["game_type"].astype(str).isin(["F", "1"])))))
    fr["t13_inv"] = inv.astype(np.float32)
    fr["t13_trans"] = trans.astype(np.float32)

    season = fr["season"].to_numpy()
    month = fr["game_month"].to_numpy()
    gt = fr["game_type"].to_numpy()
    y = fr["control_success"].to_numpy(np.float64)
    tr_m = (season < VS) | ((season == VS) & (month <= cfg["tr_extra_mo"]))
    va_m = (season == VS) & (month >= cfg["va_mo"])
    if cfg["r_only"]:
        va_m = va_m & ~((gt == 1) | (gt == "F"))
    isf_t = ((gt == 1) | (gt == "F"))[va_m]
    isf_tr = ((gt == 1) | (gt == "F"))[tr_m]
    w = 2.0 ** (season[tr_m].astype(np.float64) - 2019.0)
    masks = {"all": np.ones(int(tr_m.sum()), bool),
             "regular": ~isf_tr, "futures": isf_tr}
    yv = y[va_m]
    res = {}
    for arm, cols in (("b58", feats58),
                      ("t13", feats58 + ["t13_inv", "t13_trans"])):
        X = fr[cols].to_numpy(np.float32)
        ps = []
        for sd_ in (1, 42, 777):
            pg = {}
            for g, mk in masks.items():
                mo = CatBoostClassifier(random_seed=sd_, **hp)
                mo.fit(X[tr_m][mk], y[tr_m][mk], sample_weight=w[mk])
                pg[g] = mo.predict_proba(X[va_m])[:, 1]
            ps.append(np.where(isf_t, .6 * pg["all"] + .4 * pg["futures"],
                               .6 * pg["all"] + .4 * pg["regular"]))
        p = np.mean(ps, 0)
        np.save(DL + f"/cbh7_{tag}_{arm}.npy", p)
        res[arm] = p
        print(f"분할{tag} {arm} 완료  n_va={int(va_m.sum())}", flush=True)
    print(f"분할{tag} CB단독 Δ(t13-b58) "
          f"{bss(res['t13'], yv) - bss(res['b58'], yv):+.2f}", flush=True)
    np.save(DL + f"/cbh7_{tag}_meta.npy",
            np.column_stack([np.flatnonzero(va_m)]))
print("cbh7 끝", flush=True)
