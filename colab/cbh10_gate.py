# -*- coding: utf-8 -*-
"""체제-오프셋 모델 검증 (조언 라운드: regime offset + centered current-state).
그룹 5분할: 0=R비13, 1=R13전환전, 2=R13전환후, 3=F구체제(<=2022), 4=F신체제(2023+).
팔: ro (baseline=logit(mu_g)만) / roc (+ 성공률 4피처 상대화).
측정: VS2024 폴드 + 롤링A. 대조 cbh4_2024_b58 / cbh7_A_b58. 시드 (1,42,777)."""
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
from catboost import CatBoostClassifier, Pool                   # noqa: E402

hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
CENTER = ["asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
          "asof_pitcher_prev3_game_success_rate",
          "asof_pitcher_prev5_game_success_rate"]


def bss(p, t):
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


def lgt(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


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


def groups_of(fr):
    season = fr["season"].to_numpy()
    month = fr["game_month"].to_numpy()
    gtr = fr["game_type"].to_numpy()
    isf = (gtr == 1) | (gtr == "1") | (gtr == "F")
    inv = ((fr["pitcher_team_id"] == 13)
           | (fr["batter_team_id"] == 13)).to_numpy()
    pre = (season < 2023) | ((season == 2023) & (month < 5))
    g = np.zeros(len(fr), np.int64)          # 0: R비13
    g[~isf & inv & pre] = 1                  # R13 전환전
    g[~isf & inv & ~pre] = 2                 # R13 전환후
    g[isf & (season <= 2022)] = 3            # F 구체제
    g[isf & (season >= 2023)] = 4            # F 신체제
    return g, isf, inv


def run(fr, feats58, tr_m, va_m, out_prefix):
    g, isf, inv = groups_of(fr)
    y = fr["control_success"].to_numpy(np.float64)
    season = fr["season"].to_numpy()
    w_all = 2.0 ** (season.astype(np.float64) - 2019.0)
    mu = np.zeros(5)
    for k in range(5):
        mk = tr_m & (g == k)
        mu[k] = np.average(y[mk], weights=w_all[mk]) if mk.sum() else y[tr_m].mean()
    print(out_prefix, "mu_g", np.round(mu, 4), flush=True)
    bl = lgt(mu[g])
    isf_t = isf[va_m]
    isf_tr = isf[tr_m]
    w = w_all[tr_m]
    yv = y[va_m]
    res = {}
    for arm in ("ro", "roc"):
        Xf = fr[feats58].copy()
        if arm == "roc":
            for c in CENTER:
                v = Xf[c].to_numpy(np.float64)
                Xf[c] = np.where(np.isnan(v), np.nan,
                                 lgt(v) - lgt(mu[g])).astype(np.float32)
        X = Xf.to_numpy(np.float32)
        ps = []
        for sd_ in (1, 42, 777):
            pg = {}
            for br, mk in (("all", np.ones(int(tr_m.sum()), bool)),
                           ("regular", ~isf_tr), ("futures", isf_tr)):
                tp = Pool(X[tr_m][mk], y[tr_m][mk], weight=w[mk],
                          baseline=bl[tr_m][mk])
                vp = Pool(X[va_m], baseline=bl[va_m])
                mo_ = CatBoostClassifier(random_seed=sd_, **hp)
                mo_.fit(tp)
                pg[br] = mo_.predict_proba(vp)[:, 1]
            ps.append(np.where(isf_t, .6 * pg["all"] + .4 * pg["futures"],
                               .6 * pg["all"] + .4 * pg["regular"]))
        p = np.mean(ps, 0)
        np.save(DL + f"/{out_prefix}_{arm}.npy", p)
        res[arm] = p
        print(f"{out_prefix} {arm} 완료", flush=True)
    return res, yv, isf_t, (inv & ~isf)[va_m]


# --- VS2024 폴드
fr, feats58 = build_frame(2024)
season = fr["season"].to_numpy()
tr_m = season < 2024
va_m = season == 2024
res, yv, isf_t, t13r = run(fr, feats58, tr_m, va_m, "cbh10_2024")
b = np.load(DL + "/cbh4_2024_b58.npy")
for arm, p in res.items():
    for tag, mm in (("전체", np.ones(len(yv), bool)), ("R", ~isf_t),
                    ("t13R", t13r), ("F", isf_t)):
        d = bss(p[mm], yv[mm]) - bss(b[mm], yv[mm])
        print(f"VS2024 {arm} {tag} {d:+.2f}", flush=True)

# --- 롤링 A
fr, feats58 = build_frame(2025)
season = fr["season"].to_numpy()
month = fr["game_month"].to_numpy()
tr_m = (season < 2024) | ((season == 2024) & (month <= 6))
va_m = (season == 2024) & (month >= 7)
res, yv, isf_t, t13r = run(fr, feats58, tr_m, va_m, "cbh10_A")
b = np.load(DL + "/cbh7_A_b58.npy")
for arm, p in res.items():
    for tag, mm in (("전체", np.ones(len(yv), bool)), ("R", ~isf_t),
                    ("t13R", t13r), ("F", isf_t)):
        d = bss(p[mm], yv[mm]) - bss(b[mm], yv[mm])
        print(f"롤링A {arm} {tag} {d:+.2f}", flush=True)
print("cbh10 끝", flush=True)
