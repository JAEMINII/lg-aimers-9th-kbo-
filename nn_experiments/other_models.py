# -*- coding: utf-8 -*-
"""CatBoost 말고 다른 모델을 시즌가중 조건에서 다시 잰다.

왜 다시 재나
  LightGBM 과 sklearn HistGB 는 **균등가중 시절에** 기각됐다.
      LightGBM 60+40   관문 819.4   CatBoost 850.6
      LGB0.3 + CB0.7   885.4        단독(886.3)보다 낮음
  시즌가중이라는 새 조건에서는 결과가 달라질 수 있다.
  ('기각 근거는 모델에 묶여 있을 수 있다' — CatBoost 가 -539 였다가 +20.3 이 된 전례)

무엇을 보는가 — 단독 점수가 아니라 **CatBoost 와의 상관**이다
      CatBoost <-> sklearn/LGB   0.982~0.984   섞어도 이득 없었다
      CatBoost <-> MLP           0.889         이게 +9 를 만들었다
  상관이 0.95 아래로 내려가는 모델이 나오면 앙상블 재료가 된다.

계열을 일부러 흩어 놓는다
  LightGBM      부스팅, 비대칭 트리(leaf-wise).  CatBoost 는 대칭 트리다
  HistGB        부스팅, 비대칭.  sklearn 구현
  ExtraTrees    **배깅**. 부스팅과 근본적으로 다르다. 분할점도 무작위
  로지스틱회귀   **선형**. 트리와 완전히 다른 함수족. 단독은 약하겠지만 상관이 낮을 것
  ExtraTrees 와 로지스틱은 이 프로젝트에서 앙상블 재료로 시험된 적이 없다

기준: alpha_50.npy = CatBoost 시즌가중 2.0, 시드3, 60+40.  관문 shift0 865.6
관문 2019~2023 -> 2024, 1군, 시즌가중 2.0, 라우팅 0.6.
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "other_models_progress.txt")
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

ytr = tr.loc[m_tr, TGT].values.astype(np.int32)
ftr = IS_F[m_tr]
w = (DECAY ** (tr.loc[m_tr, "season"].values - 2019)).astype(np.float64)
Xtr = np.nan_to_num(tr.loc[m_tr, F44].to_numpy(dtype=np.float32), nan=0.0)
Xva = np.nan_to_num(tr.loc[m_va, F44].to_numpy(dtype=np.float32), nan=0.0)
# 선형 모델용 표준화 (학습 구간 통계만 사용)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd

CB = np.load(os.path.join(SC, "alpha_50.npy"))          # CatBoost 시즌가중 60+40
mlp = np.load(os.path.join(SC, "season_2023만.npy")).astype(np.float64)
CUR = 0.8 * CB + 0.2 * mlp                              # 현 제출본 구성
cur_s, _ = best(CUR)
log(f"학습 {len(ytr):,} -> 관문 {VS} 1군 {int(mm_.sum()):,}  시즌가중 {DECAY}")
log(f"기준: CatBoost 단독 {best(CB)[0]:.1f} / 현 제출본(+MLP0.2) {cur_s:.1f}")
log(f"참고 상관  CatBoost~MLP {np.corrcoef(CB[mm_], mlp[mm_])[0,1]:.4f}\n")


def fit_pair(make):
    """전체모델과 R단독을 시드별로 학습해 0.6:0.4 로 합친다."""
    out = []
    for msk in (slice(None), ~ftr):
        acc = []
        for sd_ in SEEDS:
            m = make(sd_)
            X = Ztr if getattr(m, "_needs_scale", False) else Xtr
            V = Zva if getattr(m, "_needs_scale", False) else Xva
            try:
                m.fit(X[msk], ytr[msk], sample_weight=w[msk])
            except TypeError:
                m.fit(X[msk], ytr[msk])
            acc.append(m.predict_proba(V)[:, 1])
            del m; gc.collect()
        out.append(np.mean(acc, 0))
    return BLEND * out[0] + (1 - BLEND) * out[1]


def mk_lgb(sd_):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                          min_child_samples=500, reg_lambda=1.0, random_state=sd_,
                          n_jobs=6, verbose=-1)


def mk_hgb(sd_):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=15, min_samples_leaf=500,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=sd_)


def mk_et(sd_):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_leaf=200,
                                max_features=0.5, random_state=sd_, n_jobs=6)


def mk_lr(sd_):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=0.1, max_iter=300, n_jobs=6)
    m._needs_scale = True
    return m


CASES = [("LightGBM", mk_lgb), ("HistGB", mk_hgb),
         ("ExtraTrees", mk_et), ("로지스틱회귀", mk_lr)]
log(f"  {'모델':<14s}{'단독':>9s}{'CB상관':>9s}{'최적비중':>9s}{'CB와섞어':>10s}"
    f"{'현제출대비':>11s}{'시간':>7s}")
for name, mk in CASES:
    t0 = time.time()
    try:
        p = fit_pair(mk)
    except Exception as e:
        log(f"  {name:<14s} 실패: {type(e).__name__} {str(e)[:80]}")
        continue
    np.save(os.path.join(SC, f"om_{name}.npy"), p)
    solo, _ = best(p)
    cor = np.corrcoef(p[mm_], CB[mm_])[0, 1]
    bw, bs = 0.0, best(CUR)[0]
    for wv in np.arange(0.05, 0.55, 0.05):
        s, _ = best((1 - wv) * CUR + wv * p)
        if s > bs:
            bs, bw = s, wv
    log(f"  {name:<14s}{solo:9.1f}{cor:9.4f}{bw:9.2f}{bs:10.1f}{bs-cur_s:+11.1f}"
        f"{time.time()-t0:6.0f}s")
log("\n판독: CB상관 0.95 아래면 앙상블 재료가 된다. 0.98 이상이면 같은 걸 배운 것이다.")
log("      '현제출대비' 가 +8 이상이어야 넣을 값어치가 있다 (LB 환산 약 +8).")
