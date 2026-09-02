# -*- coding: utf-8 -*-
"""p_is_succ 의 수축 강도 ALPHA 를 채점 지표(BSS) 기준으로 다시 고른다.

진단에서 나온 것 (관문 2024, 최적시프트 적용)
  포착률 = 예측이 잡아낸 투수간 분산 / 실제 존재하는 실력 분산 (표본잡음 제거)

    부분집합            판별력   실력SD   예측SD   포착률
    투수이력 1000~3000  1053.7   0.0519   0.0390    75%   <- 신호 최대, 포착 최저
    투수이력 200~1000    710.3   0.0472   0.0362    77%
    투수이력 3000+       832.1   0.0407   0.0400    98%   <- 사실상 완벽
    스트라이크 2개        704.3   0.0398   0.0368    92%

  판별력이 낮은 곳(2스트라이크 704, 베테랑 832)은 애초에 실력 분산이 작아서지
  우리가 못 잡는 게 아니다. 진짜 못하는 곳은 **중간 이력 투수(200~3000투구)** 이고
  전체의 43% 를 차지한다. 신호는 가장 큰데 포착률이 가장 낮다.

  베테랑은 n >> ALPHA 라 수축이 사실상 무의미하고, 중간 이력 구간은 ALPHA 가
  결과를 지배한다. 포착률이 낮은 구간과 정확히 겹친다.

왜 ALPHA 가 틀렸을 수 있나
    p_is_succ = (성공수 + ALPHA*0.5) / (투구수 + ALPHA),   현재 ALPHA = 50
  경험적 베이즈 최적값은  ALPHA* = p(1-p) / var_true = 0.25 / 0.0454^2 = 121.
  게다가 현재 50 은 SOLUTION_958 에서 **상관계수 기준**(corr +0.0779)으로 고른 값이지
  채점 지표(BSS) 기준이 아니다. 채점 기준으로 다시 잰다.

  이건 피처를 추가하는 게 아니라 기존 피처의 상수를 채점 지표에 맞추는 것이라
  오늘 성공한 시즌 가중치와 같은 계열이다.

관문 2019~2023 -> 2024, 1군, 시즌가중 2.0, 라우팅 0.6, 시드 3개.
ALPHA 는 p_is_succ / b_is_succ / cm_rel / p_adj_cm 에 모두 영향을 주므로
케이스마다 전처리를 다시 만든다.
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "gbdt_alpha_progress.txt")
TGT = "control_success"
ALPHA_PLAT = 300.0
VS, BLEND, DECAY = 2024, 0.6, 2.0
SEEDS = (1, 42, 777)
ALPHAS = [25.0, 50.0, 100.0, 150.0, 250.0, 400.0]     # 50 이 현행


def log(s):
    print(s, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


# ------------------------------------------------------------- 고정 부분
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
# 카운트는 ALPHA 와 무관하다. 미리 만들어 둔다
tr["pn_cur"] = (tr[PN] - tr.bn_).clip(lower=0).astype("float32")
tr["ps_cur"] = (tr.c_p - tr.bs_).clip(lower=0).astype("float32")
tr["bn_cur"] = (tr[BN] - tr.bbn_).clip(lower=0).astype("float32")
tr["bs_cur"] = (tr.c_b - tr.bbs_).clip(lower=0).astype("float32")
tr.drop(columns=["c_p", "c_b", "bn_", "bs_", "bbn_", "bbs_"], inplace=True)
gc.collect()

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


# 플래툰은 ALPHA_PLAT 만 쓰므로 한 번만 만든다
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
del hist
gc.collect()
tr["lg_cm_eff"] = tr.cm.map(lg48).fillna(0.).astype("float32")
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
mlp = np.load(os.path.join(SC, "season_2023만.npy")).astype(np.float64)
HP = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
          verbose=0, allow_writing_files=False, thread_count=6)

# 실력 분산으로 경험적 베이즈 최적값을 미리 계산해 둔다
g = tr[m_tr].groupby("pitcher_id")[TGT].agg(n="size", m="mean")
g = g[g.n >= 200]
wv = g.n / g.n.sum()
var_obs = (wv * (g.m - (wv * g.m).sum()) ** 2).sum()
noise = (wv * (g.m * (1 - g.m) / g.n)).sum()
var_true = max(var_obs - noise, 1e-9)
log(f"학습 {len(ytr):,} -> 관문 {VS} 1군 {int(mm_.sum()):,}  시즌가중 {DECAY}  시드 {len(SEEDS)}개")
log(f"경험적 베이즈 최적 ALPHA* = p(1-p)/var_true = {0.25/var_true:.0f}   (현행 50)\n")
log(f"  {'ALPHA':>7s}{'CB판별력':>10s}{'대비':>8s}{'+MLP판별력':>11s}{'대비':>8s}"
    f"{'최적시프트':>11s}{'시간':>7s}")

b1 = b2 = None
for A in ALPHAS:
    t0 = time.time()
    tr["p_is_succ"] = ((tr.ps_cur + A * .5) / (tr.pn_cur + A)).astype("float32")
    tr["b_is_succ"] = ((tr.bs_cur + A * .5) / (tr.bn_cur + A)).astype("float32")
    h2 = tr[m_tr]
    rel48 = (h2.groupby("cm").apply(lambda dd: slope(dd.p_is_succ.values, dd[TGT].values),
                                    include_groups=False)
             / slope(h2.p_is_succ.values, h2[TGT].values)).clip(.2, 1.8)
    del h2
    tr["cm_rel"] = tr.cm.map(rel48).fillna(1.).astype("float32")
    tr["p_adj_cm"] = (gm_ + (tr.p_is_succ - gm_) * tr.cm_rel).astype("float32")
    Xtr = tr.loc[m_tr, F44].to_numpy(dtype=np.float32)
    Xva = tr.loc[m_va, F44].to_numpy(dtype=np.float32)
    af, ar = [], []
    for sd in SEEDS:
        c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr, ytr, sample_weight=w)
        af.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
        c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr[~ftr], ytr[~ftr], sample_weight=w[~ftr])
        ar.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
    del Xtr, Xva; gc.collect()
    cb = BLEND * np.mean(af, 0) + (1 - BLEND) * np.mean(ar, 0)
    np.save(os.path.join(SC, f"alpha_{A:.0f}.npy"), cb)
    s1, _ = best(cb)
    s2, c2 = best(0.8 * cb + 0.2 * mlp)
    if b1 is None:
        b1, b2 = s1, s2
    log(f"  {A:7.0f}{s1:10.1f}{s1-b1:+8.1f}{s2:11.1f}{s2-b2:+8.1f}{c2:+11.4f}{time.time()-t0:6.0f}s")
log("\n※ 대비는 ALPHA=25 기준이다. 현행 50 과의 차이를 보려면 50 행을 기준으로 읽을 것.")
log("판독: 곡선이 단봉이고 50 이 정점이면 현행 유지. 정점이 100~150 이면 경험적 베이즈가 맞다.")
