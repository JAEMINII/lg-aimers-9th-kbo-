# -*- coding: utf-8 -*-
"""모델 용량(depth) 과 볼카운트 축 — 미포착 신호 +52점의 정체를 찾는다.

진단에서 나온 것 (관문 2024, 1군)
    축                    셀    실제SD   예측SD    미포착
    볼카운트x좌우 (48)     48   0.0221   0.0174   +75.2   <- 표본잡음 걷어내도 +52
    볼카운트 (12)          12   0.0109   0.0088   +16.6
    이닝                  12   0.0078   0.0068    +5.9
    li 10분위             10   0.0067   0.0068    -0.4
  카운트별 잔차에 방향성이 있다. 수준이동(+0.0084)을 빼면
    3-0 -0.0198 / 2-2 -0.0063 / 1-2 -0.0054  vs  1-0 +0.0063 / 2-1 +0.0061
  스트라이크가 몰린 카운트를 과소평가한다. 볼카운트 효과를 덜 배우고 있다.

가설 두 개
  A  depth=4 대칭트리는 트리당 분할이 4개뿐이라 볼/스트라이크/좌우만 써도 소진된다.
     SOLUTION_958 의 '얕을수록 좋다'(잎 63->679, 15->744) 는 비대칭 트리인
     HistGB 기준이라 CatBoost 대칭트리에 그대로 적용되지 않는다.
  B  볼카운트 조합이 명시적 피처로 없다. balls/strikes 를 따로 주고 있고
     cm 은 lg_cm_eff/cm_rel 로 요약해서만 들어간다. cm(48) 을 직접 주면
     트리가 한 번의 분할로 카운트 그룹을 나눌 수 있다.

관문: 2019~2023 학습 -> 2024, 1군 채점, shift 0.  기준 850.6 (시드3개)
비교를 빠르게 하려고 시드 1개로 돌린다. 시드효과(+15 내외)는 모든 조건에 공통이다.
"""
import os, sys, time, gc, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "gbdt_depth_progress.txt")
TGT = "control_success"
ALPHA, ALPHA_PLAT = 50.0, 300.0
VS, BLEND = 2024, 0.6


def log(s):
    print(s, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
tr["_r"] = tr.row_id.str.slice(6).astype("int32")
tr = tr.sort_values("_r").reset_index(drop=True)
IS_F = (tr.game_type.values == "F")
for c in tr.columns:
    if tr[c].dtype == "float64":
        tr[c] = tr[c].astype("float32")
PN, BN, MX = "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"
tr["c_p"] = (tr.asof_pitcher_success_rate.fillna(0) * tr[PN]).round()
tr["c_b"] = (tr.asof_batter_success_rate.fillna(0) * tr[BN]).round()
f1 = tr.groupby(["pitcher_id", "season"], sort=False).head(1)[["pitcher_id", "season", PN, "c_p"]]
f1.columns = ["pitcher_id", "season", "bn_", "bs_"]
tr = tr.merge(f1, on=["pitcher_id", "season"], how="left")
f2 = tr.groupby(["batter_id", "season"], sort=False).head(1)[["batter_id", "season", BN, "c_b"]]
f2.columns = ["batter_id", "season", "bbn_", "bbs_"]
tr = tr.merge(f2, on=["batter_id", "season"], how="left")
tr["pn_cur"] = (tr[PN] - tr.bn_).clip(lower=0).astype("float32")
tr["p_is_succ"] = (((tr.c_p - tr.bs_).clip(lower=0) + ALPHA * .5) / (tr.pn_cur + ALPHA)).astype("float32")
tr["b_is_succ"] = (((tr.c_b - tr.bbs_).clip(lower=0) + ALPHA * .5)
                   / ((tr[BN] - tr.bbn_).clip(lower=0) + ALPHA)).astype("float32")
tr.drop(columns=["c_p", "c_b", "bn_", "bs_", "bbn_", "bbs_"], inplace=True)
gc.collect()

PC = tr.groupby(["pitcher_id", "season", "batter_hand"])[TGT].agg(["size", "sum"]).unstack(fill_value=0)
PC.columns = [f"{x}{int(h)}" for x, h in PC.columns]
for h in (1, 2):
    for x in ("size", "sum"):
        if f"{x}{h}" not in PC.columns:
            PC[f"{x}{h}"] = 0
PCC = PC.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.)
PCC["n_all"] = PCC.size1 + PCC.size2
PCC["s_all"] = PCC.sum1 + PCC.sum2
PH = tr.groupby("pitcher_id").pitcher_hand.first()
BHAND = tr.batter_hand.values.copy()
for c, m in {"top_bottom": {"T": 0, "B": 1}, "game_type": {"R": 0, "F": 1},
             "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                            "12_": 4, "1_3": 5, "_23": 6, "123": 7}}.items():
    tr[c] = tr[c].map(m).fillna(-1).astype("int8")
tr["cnt12"] = (tr.balls_before * 3 + tr.strikes_before).astype("int16")
tr["cm"] = (tr.cnt12 * 4 + tr.pitcher_hand * 2 + tr.batter_hand).astype("int16")

m_va = (tr.season.values == VS)
i_gt = np.where(m_va)[0]
fv = IS_F[m_va]
yv = tr.loc[m_va, TGT].values.astype(np.float64)
mm_ = ~fv


def bss(p, yy):
    r = yy.mean()
    return 100000 * (1 - ((p - yy) ** 2).mean() / (r * (1 - r)))


def slope(x, y):
    v = x.var()
    return np.cov(x, y)[0, 1] / v if v > 1e-12 else np.nan


test_cols = pd.read_csv(os.path.join(D, "test.csv"), encoding="utf-8-sig", nrows=0).columns
BASE = [c for c in test_cols if c != "row_id"]
DUP = [MX, "away_win_expectancy", "run_total_before", "score_diff_home",
       "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b"]
F44 = ([c for c in BASE if c not in DUP + [PN, BN]]
       + ["p_is_succ", "pn_cur", "b_is_succ", "lg_cm_eff", "cm_rel", "p_adj_cm", "plat_dev"])

# ---- 룩업은 학습 구간(2019~2023)에서만 만든다
yrs = [2019, 2020, 2021, 2022, 2023]
m_tr = np.isin(tr.season.values, yrs)
hist = tr[m_tr]
gm_ = float(hist[TGT].mean())
lgph = hist.groupby(["pitcher_hand", "batter_hand"])[TGT].mean()
lgp = hist.groupby("pitcher_hand")[TGT].mean()
lg48 = hist.groupby("cm")[TGT].mean() - gm_
h2 = tr[m_tr & tr.p_is_succ.notna().values]
rel48 = (h2.groupby("cm").apply(lambda d: slope(d.p_is_succ.values, d[TGT].values),
                                include_groups=False)
         / slope(h2.p_is_succ.values, h2[TGT].values)).clip(.2, 1.8)
del hist, h2
gc.collect()
tr["lg_cm_eff"] = tr.cm.map(lg48).fillna(0.).astype("float32")
tr["cm_rel"] = tr.cm.map(rel48).fillna(1.).astype("float32")
tr["p_adj_cm"] = (gm_ + (tr.p_is_succ - gm_) * tr.cm_rel).astype("float32")
ph_of = PH.reindex(PCC.index.get_level_values("pitcher_id")).values
Pd = {}
for h in (1, 2):
    pr = np.array([lgph.get((p_, h), gm_) for p_ in ph_of])
    Pd[h] = (PCC[f"sum{h}"].values + ALPHA_PLAT * pr) / (PCC[f"size{h}"].values + ALPHA_PLAT)
pa = ((PCC.s_all.values + ALPHA_PLAT * np.array([lgp.get(p_, gm_) for p_ in ph_of]))
      / (PCC.n_all.values + ALPHA_PLAT))
lut = pd.DataFrame({"p1": Pd[1], "p2": Pd[2], "pa": pa}, index=PCC.index).reindex(
    pd.MultiIndex.from_arrays([tr.pitcher_id.values, tr.season.values]))
dv = np.where(BHAND == 1, lut.p1.values, lut.p2.values) - lut.pa.values
tr["plat_dev"] = np.where(np.isnan(dv), 0., dv).astype("float32")

ytr = tr.loc[m_tr, TGT].values
ftr = IS_F[m_tr]
log(f"학습 {int(m_tr.sum()):,}행  관문 2024 {len(i_gt):,} (1군 {int(mm_.sum()):,})")
log("기준: 시드3개 depth4 = 850.6 (shift 0).  아래는 시드1개라 그보다 낮게 나온다\n")
log(f"  {'구성':<34s}{'점수':>9s}{'depth4 대비':>12s}{'시간':>8s}")

CASES = [
    ("depth 4  (현행)",              F44, dict(depth=4, iterations=400)),
    ("depth 6",                      F44, dict(depth=6, iterations=400)),
    ("depth 8",                      F44, dict(depth=8, iterations=400)),
    ("depth 6 + 1200트리",           F44, dict(depth=6, iterations=1200, learning_rate=0.03)),
    ("depth 4 + cm(48) 피처",        F44 + ["cm"], dict(depth=4, iterations=400)),
    ("depth 6 + cm(48) 피처",        F44 + ["cm"], dict(depth=6, iterations=400)),
    ("depth 8 + cm(48) 피처",        F44 + ["cm"], dict(depth=8, iterations=400)),
]
base = None
for name, feats, hp in CASES:
    t0 = time.time()
    kw = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
              verbose=0, allow_writing_files=False, thread_count=4, random_seed=1)
    kw.update(hp)
    Xtr = tr.loc[m_tr, feats].to_numpy(dtype=np.float32)
    Xva = tr.loc[m_va, feats].to_numpy(dtype=np.float32)
    c = CatBoostClassifier(**kw); c.fit(Xtr, ytr)
    pf = c.predict_proba(Xva)[:, 1]; del c; gc.collect()
    c = CatBoostClassifier(**kw); c.fit(Xtr[~ftr], ytr[~ftr])
    pr_ = c.predict_proba(Xva)[:, 1]; del c; gc.collect()
    del Xtr, Xva; gc.collect()
    p = BLEND * pf + (1 - BLEND) * pr_
    np.save(os.path.join(SC, f"depth_{name.split()[1]}_{'cm' if 'cm' in feats else 'x'}.npy"), p)
    s = bss(p[mm_], yv[mm_])
    if base is None:
        base = s
    log(f"  {name:<34s}{s:9.1f}{s-base:+12.1f}{time.time()-t0:7.0f}s")
log("\n판독: depth 를 올려 크게 오르면 대칭트리 표현력이 병목이었다는 뜻.")
log("      cm(48) 추가로 오르면 볼카운트 조합을 명시적으로 줘야 한다는 뜻.")
