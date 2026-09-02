# -*- coding: utf-8 -*-
"""투수 x 볼카운트 개인 편차 — 투수 축의 마지막 빈 칸.

빈 칸이 어디인가
                  리그 수준             투수 개인
    타자좌우 축     lgph                 plat_dev      <- 있음
    볼카운트 축     lg_cm_eff / cm_rel   (없음)        <- 만든다

  cm_rel 과 p_adj_cm 은 '리그 평균이 그 카운트에서 얼마나 실력을 드러내는가' 이지
  '이 투수가 그 카운트에서 유독 잘/못 던지는가' 가 아니다.
  plat_dev 의 볼카운트 버전을 만든다.

왜 될 수도 있나
  0-2 에서는 유인구를 던지고 3-0 에서는 스트라이크를 넣어야 한다.
  그 전환을 잘하는 투수와 못하는 투수가 갈릴 수 있고, 그건 개인 성향이다.
  표본도 타자 축보다 낫다 — 투수당 평균 1,860투구 / 12셀 = 155투구/셀.

교훈 반영
  타자 축(2026-08-16 노트)은 간격 없는 검증에서 +25 였다가
  간격 있는 관문에서 -4 였다. 그래서 처음부터 간격 있는 관문으로만 잰다.
  현 제출본과 같은 조건(시즌가중 2.0, 60:40, 시드3)에서 잰다.

관문 2019~2023 -> 2024, 1군 채점, shift 0 및 최적시프트.
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar

SC = os.path.dirname(os.path.abspath(__file__))
# Resolve the workspace locally; avoid a code-page-dependent hard-coded path.
ROOT = os.environ.get("AIMERS_ROOT", os.path.dirname(SC))
D = os.path.join(ROOT, "open (1)", "data")
PROG = os.path.join(SC, "pitcher_count_fold_progress.txt")
TGT = "control_success"
ALPHA, ALPHA_PLAT, ALPHA_PC = 50.0, 300.0, 200.0
VS = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
BLEND, DECAY = 0.6, 2.0
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

for c, m in {"top_bottom": {"T": 0, "B": 1}, "game_type": {"R": 0, "F": 1},
             "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                            "12_": 4, "1_3": 5, "_23": 6, "123": 7}}.items():
    tr[c] = tr[c].map(m).fillna(-1).astype("int8")
tr["cnt12"] = (tr.balls_before * 3 + tr.strikes_before).astype("int16")
tr["cm"] = (tr.cnt12 * 4 + tr.pitcher_hand * 2 + tr.batter_hand).astype("int16")
# 카운트 3그룹: 스트라이크 우세 / 균형 / 볼 우세.  셀당 표본을 4배로 늘린다
tr["cgrp"] = np.where(tr.strikes_before > tr.balls_before, 0,
                      np.where(tr.strikes_before == tr.balls_before, 1, 2)).astype("int8")
tr["nstr"] = tr.strikes_before.astype("int8")     # 0/1/2 — 가장 거친 축

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
    return -r.fun, r.x


def slope(x, y):
    v = x.var()
    return np.cov(x, y)[0, 1] / v if v > 1e-12 else np.nan


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

hist = tr[m_tr]
gm_ = float(hist[TGT].mean())
lgph = hist.groupby(["pitcher_hand", "batter_hand"])[TGT].mean()
lgp = hist.groupby("pitcher_hand")[TGT].mean()
lg48 = hist.groupby("cm")[TGT].mean() - gm_
LG = {k: hist.groupby(k)[TGT].mean().to_dict() for k in ("cnt12", "cgrp", "nstr", "cm")}
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


def pitcher_cell(keycol, tag, alpha):
    """(투수, 시즌, keycol) 을 그 시즌 시작 직전까지 누적해 개인 편차를 만든다.

    plat_dev 와 똑같은 cumsum-shift 규약이라 자기 자신이 안 들어간다.
    dev = 그 셀 성공률 - 그 투수 전체 성공률.  '이 투수가 이 카운트에서 유독' 을 잰다.
    """
    B = tr.groupby(["pitcher_id", "season", keycol])[TGT].agg(["size", "sum"]).unstack(fill_value=0)
    B.columns = [f"{x}_{int(k)}" for x, k in B.columns]
    CC = B.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.)
    keys = np.array(sorted({int(c.split("_")[1]) for c in B.columns}))
    idx = pd.MultiIndex.from_arrays([tr.pitcher_id.values, tr.season.values])
    sz = np.nan_to_num(CC[[f"size_{k}" for k in keys]].reindex(idx).values)
    sm = np.nan_to_num(CC[[f"sum_{k}" for k in keys]].reindex(idx).values)
    kv = tr[keycol].values
    pos = np.clip(np.searchsorted(keys, kv), 0, len(keys) - 1)
    r = np.arange(len(tr))
    hit = keys[pos] == kv
    n_ = np.where(hit, sz[r, pos], 0.0)
    s_ = np.where(hit, sm[r, pos], 0.0)
    prior = np.array([LG[keycol].get(int(k), gm_) for k in kv], dtype=np.float64)
    n_all, s_all = sz.sum(1), sm.sum(1)
    base = (s_all + alpha * gm_) / (n_all + alpha)
    rate = (s_ + alpha * prior) / (n_ + alpha)
    tr[f"pc_{tag}_rate"] = rate.astype("float32")
    tr[f"pc_{tag}_dev"] = (rate - base).astype("float32")
    tr[f"pc_{tag}_n"] = np.log1p(n_).astype("float32")
    med = np.median(n_[n_ > 0]) if (n_ > 0).any() else 0
    log(f"  {tag:6s} 셀 {len(keys):>2d}개  셀당 표본 중앙값 {med:,.0f}")
    return [f"pc_{tag}_rate", f"pc_{tag}_dev", f"pc_{tag}_n"]


log("투수 x 볼카운트 개인 편차 생성")
F_C12 = pitcher_cell("cnt12", "c12", ALPHA_PC)
F_CG3 = pitcher_cell("cgrp", "cg3", ALPHA_PC)
F_NST = pitcher_cell("nstr", "nst", ALPHA_PC)
F_CM48 = pitcher_cell("cm", "cm48", ALPHA_PC)

test_cols = pd.read_csv(os.path.join(D, "test.csv"), encoding="utf-8-sig", nrows=0).columns
BASE = [c for c in test_cols if c != "row_id"]
DUP = [MX, "away_win_expectancy", "run_total_before", "score_diff_home",
       "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b"]
F44 = ([c for c in BASE if c not in DUP + [PN, BN]]
       + ["p_is_succ", "pn_cur", "b_is_succ", "lg_cm_eff", "cm_rel", "p_adj_cm", "plat_dev"])

ytr = tr.loc[m_tr, TGT].values
ftr = IS_F[m_tr]
w = (DECAY ** (tr.loc[m_tr, "season"].values - 2019)).astype(np.float64)
MLPF = os.path.join(SC, "season_2023만.npy")   # 2024 를 예측한 MLP
mlp = np.load(MLPF).astype(np.float64) if VS == 2024 else None
HP = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
          verbose=0, allow_writing_files=False, thread_count=6)

log(f"\n학습 {len(ytr):,} -> 관문 2024 1군 {int(mm_.sum()):,}  시즌가중 {DECAY}  시드 {len(SEEDS)}개")
log("현 제출본과 같은 조건. MLP 0.20 을 얹은 값도 같이 본다.\n")
log(f"  {'구성':<26s}{'피처':>5s}{'CB판별력':>10s}{'대비':>8s}{'+MLP판별력':>11s}{'대비':>8s}{'시간':>7s}")

CASES = [
    ("기존 44피처",            F44),
    ("+ 투수x카운트12",        F44 + F_C12),
    ("+ 투수xcm48",           F44 + F_CM48),
    ("+ 카운트12 + cm48",      F44 + F_C12 + F_CM48),
]
b1 = b2 = None
for name, feats in CASES:
    t0 = time.time()
    Xtr = tr.loc[m_tr, feats].to_numpy(dtype=np.float32)
    Xva = tr.loc[m_va, feats].to_numpy(dtype=np.float32)
    af, ar = [], []
    for sd in SEEDS:
        c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr, ytr, sample_weight=w)
        af.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
        c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr[~ftr], ytr[~ftr], sample_weight=w[~ftr])
        ar.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
    del Xtr, Xva; gc.collect()
    cb = BLEND * np.mean(af, 0) + (1 - BLEND) * np.mean(ar, 0)
    np.save(os.path.join(SC, f"pcfold{VS}_{len(feats)}.npy"), cb)
    s1, _ = best(cb)
    s2, _ = best(0.8 * cb + 0.2 * mlp) if mlp is not None else (float("nan"), 0)
    if b1 is None:
        b1, b2 = s1, s2
    log(f"  {name:<26s}{len(feats):>5d}{s1:10.1f}{s1-b1:+8.1f}{s2:11.1f}{s2-b2:+8.1f}{time.time()-t0:6.0f}s")
log("\n판독: +MLP 판별력이 +10 이상이면 제출본에 넣을 값어치가 있다 (LB 환산 약 +10).")
log("      타자 축은 같은 자리에서 -4 였다.")
