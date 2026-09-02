# -*- coding: utf-8 -*-
"""t13-R 체제 수술 — 전환전(2023.5 이전) t13 R행 가중 절하/제거/전용브랜치.
팔: w0 (가중0) / w05 (가중0.5) / t13br (t13-post 전용 브랜치, 2024+롤링만).
대조: 기존 cbh4_{VS}_b58.npy, cbh7_A_b58.npy 재사용. 시드 (1,42,777)."""
import os
import sys

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


def build_frame(VS_lookup):
    built = F.build(DATA, VS=VS_lookup, return_frame=True)
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
    return fr, feats58


def run(fr, feats58, tr_m, va_m, arms, out_prefix, yv):
    season = fr["season"].to_numpy()
    month = fr["game_month"].to_numpy()
    gtr = fr["game_type"].to_numpy()
    isr = ~((gtr == 1) | (gtr == "1") | (gtr == "F"))
    inv = ((fr["pitcher_team_id"] == 13) | (fr["batter_team_id"] == 13)).to_numpy()
    pre13 = inv & isr & ((season < 2023) | ((season == 2023) & (month < 5)))
    post13 = inv & isr & ~((season < 2023) | ((season == 2023) & (month < 5)))
    isf_t = ~isr[va_m]
    isf_tr = ~isr[tr_m]
    w_base = 2.0 ** (season[tr_m].astype(np.float64) - 2019.0)
    X = fr[feats58].to_numpy(np.float32)
    y = fr["control_success"].to_numpy(np.float64)
    t13_va = (inv & isr)[va_m]
    res = {}
    for arm in arms:
        ps = []
        for sd_ in (1, 42, 777):
            w = w_base.copy()
            if arm == "w0":
                w[pre13[tr_m]] = 0.0
            elif arm == "w05":
                w[pre13[tr_m]] *= 0.5
            masks = {"all": np.ones(int(tr_m.sum()), bool),
                     "regular": ~isf_tr, "futures": isf_tr}
            if arm == "t13br":
                masks["t13"] = post13[tr_m]
            pg = {}
            for g, mk in masks.items():
                keep = mk & (w > 0)
                mo_ = CatBoostClassifier(random_seed=sd_, **hp)
                mo_.fit(X[tr_m][keep], y[tr_m][keep], sample_weight=w[keep])
                pg[g] = mo_.predict_proba(X[va_m])[:, 1]
            p = np.where(isf_t, .6 * pg["all"] + .4 * pg["futures"],
                         .6 * pg["all"] + .4 * pg["regular"])
            if arm == "t13br":
                p = np.where(t13_va, .6 * pg["all"] + .4 * pg["t13"], p)
            ps.append(p)
        p = np.mean(ps, 0)
        np.save(DL + f"/{out_prefix}_{arm}.npy", p)
        res[arm] = p
        print(f"{out_prefix} {arm} 완료", flush=True)
    return res, t13_va, isf_t


# --- 3폴드 (대조 cbh4_{VS}_b58)
for VS in (2024, 2022, 2023):
    fr, feats58 = build_frame(VS)
    season = fr["season"].to_numpy()
    tr_m = season < VS
    va_m = season == VS
    y = fr["control_success"].to_numpy(np.float64)
    yv = y[va_m]
    arms = ("w0", "w05") if VS != 2024 else ("w0", "w05", "t13br")
    res, t13_va, isf_t = run(fr, feats58, tr_m, va_m, arms, f"cbh9_{VS}", yv)
    b = np.load(DL + f"/cbh4_{VS}_b58.npy")
    m = ~isf_t if VS == 2023 else np.ones(len(yv), bool)
    for arm, p in res.items():
        d = bss(p[m], yv[m]) - bss(b[m], yv[m])
        mt = t13_va & m
        d13 = bss(p[mt], yv[mt]) - bss(b[mt], yv[mt])
        print(f"VS{VS} {arm}  전체 {d:+.2f}  t13R조각 {d13:+.2f}", flush=True)

# --- 롤링 A (대조 cbh7_A_b58): 학습 <2024 + 2024 월<=6 -> 2024 7+
fr, feats58 = build_frame(2025)
season = fr["season"].to_numpy()
month = fr["game_month"].to_numpy()
tr_m = (season < 2024) | ((season == 2024) & (month <= 6))
va_m = (season == 2024) & (month >= 7)
y = fr["control_success"].to_numpy(np.float64)
yv = y[va_m]
res, t13_va, isf_t = run(fr, feats58, tr_m, va_m, ("w0", "w05", "t13br"),
                         "cbh9_A", yv)
b = np.load(DL + "/cbh7_A_b58.npy")
m = np.ones(len(yv), bool)
for arm, p in res.items():
    d = bss(p, yv) - bss(b, yv)
    mt = t13_va
    d13 = bss(p[mt], yv[mt]) - bss(b[mt], yv[mt])
    print(f"롤링A {arm}  전체 {d:+.2f}  t13R조각 {d13:+.2f}", flush=True)
print("cbh9 끝", flush=True)
