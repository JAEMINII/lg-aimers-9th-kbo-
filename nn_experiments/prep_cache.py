# -*- coding: utf-8 -*-
"""
신경망 실험용 배열을 한 번 만들어 캐시한다.

fc_proper.py 가 매 실행마다 147만 행 CSV 를 읽고 44피처를 다시 만든다(약 90초).
PLR·TabM 실험을 여러 번 돌릴 거라 한 번만 만들어 .npz 로 둔다.
피처 생성 경로는 fc_proper.py 와 완전히 동일하다 (GBDT 실험과도 동일).
"""
import os, sys, gc
import numpy as np
import pandas as pd

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SC, "nn_cache.npz")
TGT = "control_success"
ALPHA, ALPHA_PLAT = 50.0, 300.0
VS = 2024

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

m_tr = (tr.season < VS).values
hist = tr[m_tr]
gm_ = float(hist[TGT].mean())
lgph = hist.groupby(["pitcher_hand", "batter_hand"])[TGT].mean()
lgp = hist.groupby("pitcher_hand")[TGT].mean()
lg48 = hist.groupby("cm")[TGT].mean() - gm_
h2 = tr[m_tr & tr.p_is_succ.notna().values]
def slope(x, y):
    v = x.var()
    return np.cov(x, y)[0, 1] / v if v > 1e-12 else np.nan
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
CAT = [c for c in ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
                   "base_state", "pitcher_hand", "batter_hand", "top_bottom", "game_type",
                   "balls_before", "strikes_before", "outs_before", "inning",
                   "season", "game_month", "game_dayofweek"] if c in F44]
CON = [c for c in F44 if c not in CAT]

cat_idx, cat_card = [], []
for c in CAT:
    vals = pd.unique(tr.loc[m_tr, c])
    mp = {v: i + 1 for i, v in enumerate(sorted(v for v in vals if pd.notna(v)))}
    cat_idx.append(tr[c].map(mp).fillna(0).astype("int32").values)
    cat_card.append(len(mp) + 1)
Xc = np.stack(cat_idx, 1).astype(np.int32)

Xn = tr[CON].to_numpy(dtype=np.float32)
miss = np.isnan(Xn)
has_nan = miss[m_tr].any(0)
med = np.nanmedian(Xn[m_tr], 0)
Xn = np.where(miss, med, Xn)
# PLR 은 분위수 정규화가 더 안정적이라는 보고가 있으나, 여기서는 표준화를 쓴다.
# (GBDT 실험과 피처 값 자체를 같게 두어 비교를 깨끗하게 하려는 목적)
mu, sd = Xn[m_tr].mean(0), Xn[m_tr].std(0) + 1e-6
Xn = ((Xn - mu) / sd).astype(np.float32)
n_raw_con = Xn.shape[1]
if has_nan.any():
    Xn = np.concatenate([Xn, miss[:, has_nan].astype(np.float32)], 1)
y = tr[TGT].to_numpy(dtype=np.float32)
season = tr.season.to_numpy(dtype=np.int16)

np.savez_compressed(OUT, Xc=Xc, Xn=Xn, y=y, season=season, is_f=IS_F,
                    cat_card=np.array(cat_card, np.int32),
                    n_raw_con=np.int32(n_raw_con),
                    cat_names=np.array(CAT, dtype="U32"),
                    con_names=np.array(CON, dtype="U32"))
print(f"캐시 저장: {OUT}  ({os.path.getsize(OUT)/1024/1024:.0f} MB)")
print(f"  범주 {len(CAT)}개 (카디널리티 최대 {max(cat_card)})")
print(f"  연속 {n_raw_con}개 + 결측표시자 {Xn.shape[1]-n_raw_con}개 = {Xn.shape[1]}차원")
print(f"  행 {len(y):,}  학습(<2024) {int(m_tr.sum()):,}  관문(2024) {int((season==2024).sum()):,}")
