# -*- coding: utf-8 -*-
"""손실함수와 시즌 가중치 — 아직 안 건드린 두 축.

왜 이걸 보나
  1  채점은 Brier(제곱오차)인데 CatBoost 는 기본값 Logloss 로 학습해 왔다.
     둘 다 proper scoring rule 이라 무한 데이터에선 같은 최적해지만,
     유한 데이터 + 정규화에선 다르다. 특히 확률이 전부 0.5 근처인 이 문제는
     Logloss 가 극단값에 주는 큰 기울기가 낭비일 수 있다.
     신경망에서는 Brier 직접 최적화를 시도한 기록이 있으나(SOLUTION_958:460)
     GBDT 에서는 한 번도 안 했다.

  2  '최근 시즌만 쓰기' 는 재봤고 -27 이었다(gbdt_season.py).
     하지만 '전부 쓰되 최근을 강조' 는 다른 이야기다. 표본을 잃지 않는다.
     리그 성공률이 2019 54.95% -> 2024 48.97% 로 밀리고 있으니
     옛 시즌의 가중치를 낮추는 게 맞을 수 있다.

오늘 실패한 가설들과 다른 점
  앞선 여섯 개는 전부 '모델이 무언가를 덜 배웠다' 는 전제였고 매번 틀렸다.
  이 둘은 '무엇을 배우게 할 것인가' 를 바꾸는 것이라 성격이 다르다.

관문 2019~2023 -> 2024, 1군, shift 0, CatBoost 0.6전체+0.4R단독, 시드 1개.
같은 조건 기준값 849.0 (gbdt_depth.py 의 depth4 시드1)
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "gbdt_loss_progress.txt")
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
str_ = tr.loc[m_tr, "season"].values
Xtr = tr.loc[m_tr, F44].to_numpy(dtype=np.float32)
Xva = tr.loc[m_va, F44].to_numpy(dtype=np.float32)
log(f"학습 {len(ytr):,} -> 관문 2024 1군 {int(mm_.sum()):,}   시드 1개")
log("같은 조건 기준값 849.0 (Logloss, 균등가중)\n")
log(f"  {'구성':<28s}{'점수':>9s}{'기준대비':>10s}{'예측평균':>10s}{'시간':>7s}")

HP = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
          verbose=0, allow_writing_files=False, thread_count=4, random_seed=1)


def run(loss, decay):
    w = None if decay == 1.0 else (decay ** (str_ - 2019)).astype(np.float64)
    out = []
    for msk in (slice(None), ~ftr):
        ww = None if w is None else w[msk if msk is not slice(None) else slice(None)]
        if loss == "Logloss":
            c = CatBoostClassifier(**HP)
            c.fit(Xtr[msk], ytr[msk], sample_weight=ww)
            out.append(c.predict_proba(Xva)[:, 1])
        else:
            c = CatBoostRegressor(loss_function="RMSE", **HP)
            c.fit(Xtr[msk], ytr[msk].astype(np.float64), sample_weight=ww)
            out.append(np.clip(c.predict(Xva), 1e-4, 1 - 1e-4))
        del c; gc.collect()
    return BLEND * out[0] + (1 - BLEND) * out[1]


CASES = [
    ("Logloss, 균등 (현행)",      "Logloss", 1.00),
    ("Brier(RMSE) 직접, 균등",    "RMSE",    1.00),
    ("Logloss, 시즌가중 1.15",    "Logloss", 1.15),
    ("Logloss, 시즌가중 1.30",    "Logloss", 1.30),
    ("Logloss, 시즌가중 1.60",    "Logloss", 1.60),
    ("Logloss, 시즌가중 2.00",    "Logloss", 2.00),
    ("Brier, 시즌가중 1.30",      "RMSE",    1.30),
]
base = None
for name, loss, dec in CASES:
    t0 = time.time()
    try:
        p = run(loss, dec)
    except Exception as e:
        log(f"  {name:<28s} 실패: {type(e).__name__} {str(e)[:90]}")
        continue
    np.save(os.path.join(SC, f"loss_{loss}_{dec:.2f}.npy"), p)
    s = bss(p)
    if base is None:
        base = s
    log(f"  {name:<28s}{s:9.1f}{s-base:+10.1f}{p[mm_].mean():10.4f}{time.time()-t0:6.0f}s")
log(f"\n실제 1군 성공률 {yv[mm_].mean():.4f}")
log("판독: Brier 직접이 크게 오르면 손실함수가 병목이었다는 뜻.")
log("      시즌가중이 오르면 옛 시즌을 덜 봐야 한다는 뜻 ('최근만'의 -27 과 다른 결론).")
