# -*- coding: utf-8 -*-
"""시즌가중 조건에서 정규화 상수를 다시 고른다.

놓친 것
  시즌가중 2.0 을 넣으면서 **유효표본이 147만 -> 72만(49%)** 으로 줄었다.
  그런데 정규화 관련 상수는 전부 균등가중 시절 값 그대로다.

      depth 4          균등가중에서 측정 (gbdt_depth.py). depth6 -3, depth8 -21
      iterations 400   균등가중 시절
      learning_rate .05 균등가중 시절
      l2_leaf_reg 1.0  균등가중 시절

  표본이 절반이면 최적 정규화가 달라진다. 표본이 줄면 보통 **더 강한 정규화**가
  맞지만, 시즌가중은 '유효표본은 줄되 분포는 평가 시점에 가까워지는' 것이라
  방향을 단정할 수 없다. 재는 수밖에 없다.

  반대로 트리 수를 늘려야 할 수도 있다. 가중치가 붙으면 유효 학습신호가 달라져
  400 그루로 수렴이 덜 됐을 수 있다.

관문 2019~2023 -> 2024, 1군, 시즌가중 2.0, 라우팅 0.6, 시드 3개.
기준: 현행(depth4/400/lr.05/l2=1) CB판별력 906.7, +MLP 912.6
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "gbdt_hp2_progress.txt")
TGT = "control_success"
ALPHA, ALPHA_PLAT = 50.0, 300.0
VS, BLEND, DECAY = 2024, 0.6, 2.0
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


def best(p):
    def sh(q, c):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))
    r = minimize_scalar(lambda c: -bss(sh(p, c)), bounds=(-0.3, 0.3), method="bounded")
    return -r.fun


def slope(x, y):
    v = x.var()
    return np.cov(x, y)[0, 1] / v if v > 1e-12 else np.nan


hist = tr[m_tr]
gm_ = float(hist[TGT].mean())
lgph = hist.groupby(["pitcher_hand", "batter_hand"])[TGT].mean()
lgp = hist.groupby("pitcher_hand")[TGT].mean()
lg48 = hist.groupby("cm")[TGT].mean() - gm_
h2 = tr[m_tr & tr.p_is_succ.notna().values]
rel48 = (h2.groupby("cm").apply(lambda dd: slope(dd.p_is_succ.values, dd[TGT].values),
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
mlp = np.load(os.path.join(SC, "season_2023만.npy")).astype(np.float64)
ess = w.sum() ** 2 / (w ** 2).sum()
log(f"학습 {len(ytr):,}행 (유효표본 {ess:,.0f} = {ess/len(ytr)*100:.0f}%)  시드 {len(SEEDS)}개")
log(f"기준 현행 depth4/400/lr.05/l2=1 -> CB 906.7 / +MLP 912.6\n")
log(f"  {'구성':<34s}{'CB판별력':>10s}{'대비':>8s}{'+MLP':>9s}{'대비':>8s}{'시간':>7s}")

CASES = [
    ("현행 d4 / 400 / .05 / l2=1",   dict()),
    ("d3  (더 얕게)",                dict(depth=3)),
    ("d5",                          dict(depth=5)),
    ("d6",                          dict(depth=6)),
    ("l2=5  (더 강한 정규화)",        dict(l2_leaf_reg=5.0)),
    ("l2=20",                       dict(l2_leaf_reg=20.0)),
    ("800그루 / lr .025",            dict(iterations=800, learning_rate=0.025)),
    ("1600그루 / lr .0125",          dict(iterations=1600, learning_rate=0.0125)),
    ("d5 + l2=5 + 800/.025",        dict(depth=5, l2_leaf_reg=5.0,
                                         iterations=800, learning_rate=0.025)),
]
b1 = b2 = None
for name, hp in CASES:
    t0 = time.time()
    kw = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
              verbose=0, allow_writing_files=False, thread_count=6)
    kw.update(hp)
    af, ar = [], []
    for sd in SEEDS:
        c = CatBoostClassifier(random_seed=sd, **kw); c.fit(Xtr, ytr, sample_weight=w)
        af.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
        c = CatBoostClassifier(random_seed=sd, **kw); c.fit(Xtr[~ftr], ytr[~ftr], sample_weight=w[~ftr])
        ar.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
    cb = BLEND * np.mean(af, 0) + (1 - BLEND) * np.mean(ar, 0)
    np.save(os.path.join(SC, f"hp2_{name.split()[0].replace('/', '_')}.npy"), cb)
    s1, s2 = best(cb), best(0.8 * cb + 0.2 * mlp)
    if b1 is None:
        b1, b2 = s1, s2
    log(f"  {name:<34s}{s1:10.1f}{s1-b1:+8.1f}{s2:9.1f}{s2-b2:+8.1f}{time.time()-t0:6.0f}s")
log("\n판독: 유효표본이 절반이 됐으니 더 강한 정규화(d3, l2 크게)나")
log("      더 느린 학습(트리 늘리고 lr 낮추기)이 맞을 수 있다.")
