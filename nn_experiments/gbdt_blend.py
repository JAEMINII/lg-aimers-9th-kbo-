# -*- coding: utf-8 -*-
"""시즌가중 2.0 위에서 라우팅 비율과 MLP 비중을 다시 고른다.

왜
  0.6 전체 + 0.4 R단독 이라는 비율은 시즌가중이 없던 시절에 고른 값이다.
  R단독 라우팅이 하는 일 중 하나가 '퓨처스 편향 제거 = 수준 보정' 인데
  시즌가중도 드리프트를 줄이므로 둘이 겹칠 수 있다. 겹치면 0.4 는 과하다.
  (같은 이유로 shift -0.05 가 과보정이었던 전례가 있다)

  MLP 비중 0.20 도 균등가중 CatBoost 기준으로 고른 값이다.
  가중 CatBoost 는 MLP 와 상관이 달라졌을 수 있어 다시 잰다.

방식
  전체모델과 R단독모델의 예측을 따로 저장해두면 비율 스윕은 학습 없이 끝난다.
  MLP 는 관문용 예측(season_2023만.npy)을 그대로 쓴다.

관문 2019~2023 -> 2024, 1군 채점. 시드 3개.
현행(0.6 / MLP 0.20) 대비 얼마나 달라지는지만 본다.
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "gbdt_blend_progress.txt")
TGT = "control_success"
ALPHA, ALPHA_PLAT = 50.0, 300.0
VS, DECAY = 2024, 2.0
SEEDS = (1, 42, 777)


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
tr["cm"] = ((tr.balls_before * 3 + tr.strikes_before) * 4
            + tr.pitcher_hand * 2 + tr.batter_hand).astype("int16")

m_va = (tr.season.values == VS)
m_tr = (tr.season.values < VS)
fv = IS_F[m_va]
yv = tr.loc[m_va, TGT].values.astype(np.float64)
mm_ = ~fv


def bss(p):
    r = yv[mm_].mean()
    return 100000 * (1 - ((p[mm_] - yv[mm_]) ** 2).mean() / (r * (1 - r)))


def sh(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))


def best(p):
    r = minimize_scalar(lambda c: -bss(sh(p, c)), bounds=(-0.3, 0.3), method="bounded")
    return -r.fun, r.x


def slope(x, y):
    v = x.var()
    return np.cov(x, y)[0, 1] / v if v > 1e-12 else np.nan


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

test_cols = pd.read_csv(os.path.join(D, "test.csv"), encoding="utf-8-sig", nrows=0).columns
BASE = [c for c in test_cols if c != "row_id"]
DUP = [MX, "away_win_expectancy", "run_total_before", "score_diff_home",
       "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b"]
F44 = ([c for c in BASE if c not in DUP + [PN, BN]]
       + ["p_is_succ", "pn_cur", "b_is_succ", "lg_cm_eff", "cm_rel", "p_adj_cm", "plat_dev"])

ytr = tr.loc[m_tr, TGT].values
ftr = IS_F[m_tr]
w = (DECAY ** (tr.loc[m_tr, "season"].values - 2019)).astype(np.float64)
Xtr = tr.loc[m_tr, F44].to_numpy(dtype=np.float32)
Xva = tr.loc[m_va, F44].to_numpy(dtype=np.float32)
HP = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
          verbose=0, allow_writing_files=False, thread_count=6)

log(f"학습 {len(ytr):,} -> 관문 2024 1군 {int(mm_.sum()):,}  시즌가중 {DECAY}  시드 {len(SEEDS)}개")
t0 = time.time()
acc_f, acc_r = [], []
for sd in SEEDS:
    c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr, ytr, sample_weight=w)
    acc_f.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
    c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr[~ftr], ytr[~ftr], sample_weight=w[~ftr])
    acc_r.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
pf, pr_ = np.mean(acc_f, 0), np.mean(acc_r, 0)
np.save(os.path.join(SC, "wgt_full.npy"), pf)
np.save(os.path.join(SC, "wgt_r.npy"), pr_)
log(f"학습 완료 {time.time()-t0:.0f}s   전체단독 {bss(pf):.1f}  R단독 {bss(pr_):.1f}\n")

mlp = np.load(os.path.join(SC, "season_2023만.npy")).astype(np.float64)

log("라우팅 비율 (MLP 없이).  현행 0.6")
log(f"  {'전체비중':>8s}{'shift0':>10s}{'최적시프트':>11s}{'판별력':>9s}")
for b in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
    p = b * pf + (1 - b) * pr_
    s2, c2 = best(p)
    log(f"  {b:8.2f}{bss(p):10.1f}{c2:+11.4f}{s2:9.1f}")

log("\nMLP 비중 (라우팅 0.6 고정).  현행 0.20")
log(f"  {'MLP비중':>8s}{'shift0':>10s}{'최적시프트':>11s}{'판별력':>9s}")
cb = 0.6 * pf + 0.4 * pr_
for wm in (0.0, 0.15, 0.20, 0.25, 0.30, 0.40):
    p = (1 - wm) * cb + wm * mlp
    s2, c2 = best(p)
    log(f"  {wm:8.2f}{bss(p):10.1f}{c2:+11.4f}{s2:9.1f}")

log(f"\n상관: 가중CatBoost(0.6/0.4) ~ MLP = {np.corrcoef(cb[mm_], mlp[mm_])[0,1]:.4f}")
log("      (균등가중 시절엔 0.8851 이었다. 올랐으면 MLP 비중을 줄여야 한다)")

log("\n둘을 같이 고르면")
bb = (-1e9, None)
for b in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
    for wm in (0.15, 0.20, 0.25, 0.30):
        p = (1 - wm) * (b * pf + (1 - b) * pr_) + wm * mlp
        s2, _ = best(p)
        if s2 > bb[0]:
            bb = (s2, (b, wm))
log(f"  최고 판별력 {bb[0]:.1f}  라우팅 {bb[1][0]:.2f} / MLP {bb[1][1]:.2f}")
cur, _ = best((1 - 0.20) * cb + 0.20 * mlp)
log(f"  현행(0.6 / 0.20) {cur:.1f}   차이 {bb[0]-cur:+.1f}")
log("\n판독: 차이가 +5 이내면 현행 유지. 관문에서 두 값을 동시에 고르면 과적합이다.")
