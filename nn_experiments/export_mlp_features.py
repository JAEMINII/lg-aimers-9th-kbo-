# -*- coding: utf-8 -*-
"""MLP 입력 60열을 평가 데이터에서 만들기 위한 상수를 뽑는다 -> mlp_feat.npz

MLP 는 CatBoost 와 피처 파이프라인이 다르다.
  CatBoost   submit_jaemin_6/train_model.py 의 피처 (marcel 등 포함)
  MLP        nn_experiments/prep_cache.py 의 60열 (범주16 + 연속28 + 결측표시16)

prep_cache.py 는 train.csv 를 통째로 읽어 만들었고, 정규화 상수(중앙값/평균/표준편차)와
범주 사전을 2024 이전 구간에서 잡았다. 평가(2025) 행에도 그 상수를 그대로 써야 하므로
여기서 전부 꺼내 npz 에 담는다. 계산식은 prep_cache.py 를 그대로 옮겼다.

시즌 기준이 달라지는 표는 두 벌 만든다.
  as-of 2025   평가용. 2019~2024 전체 누적
  as-of 2024   검증용. 2019~2023 누적 -> 2024 행을 '평가 경로' 로 다시 만들어
               nn_cache 의 2024 구간과 맞는지 확인한다
"""
import os, gc, json
import numpy as np
import pandas as pd

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SC, "mlp_feat.npz")
TGT = "control_success"
ALPHA, ALPHA_PLAT = 50.0, 300.0
VS = 2024                                  # prep_cache 와 같은 값 (상수 기준 시즌)

# ==================================================== prep_cache.py 재현 (그대로)
tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
tr["_r"] = tr.row_id.str.slice(6).astype("int32")
tr = tr.sort_values("_r").reset_index(drop=True)
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

cat_maps = []
cat_idx, cat_card = [], []
for c in CAT:
    vals = pd.unique(tr.loc[m_tr, c])
    mp = {v: i + 1 for i, v in enumerate(sorted(v for v in vals if pd.notna(v)))}
    cat_maps.append(mp)
    cat_idx.append(tr[c].map(mp).fillna(0).astype("int32").values)
    cat_card.append(len(mp) + 1)
Xc = np.stack(cat_idx, 1).astype(np.int32)

Xn = tr[CON].to_numpy(dtype=np.float32)
miss = np.isnan(Xn)
has_nan = miss[m_tr].any(0)
med = np.nanmedian(Xn[m_tr], 0)
Xn = np.where(miss, med, Xn)
mu, sd = Xn[m_tr].mean(0), Xn[m_tr].std(0) + 1e-6
Xn = ((Xn - mu) / sd).astype(np.float32)
n_raw_con = Xn.shape[1]
if has_nan.any():
    Xn = np.concatenate([Xn, miss[:, has_nan].astype(np.float32)], 1)
season = tr.season.to_numpy(dtype=np.int16)
print(f"prep_cache 재현: Xc {Xc.shape}  Xn {Xn.shape}  연속원본 {n_raw_con}  "
      f"결측표시 {int(has_nan.sum())}")

# 캐시와 같은지 확인 (계산식이 정말 동일한지)
zc = np.load(os.path.join(SC, "nn_cache.npz"), allow_pickle=False)
print(f"  nn_cache 와 일치  Xc={np.array_equal(zc['Xc'], Xc)}  Xn={np.array_equal(zc['Xn'], Xn)}")
zc.close()


# =========================================== 평가 경로용 표 (as-of 기준 시즌별)
def make_tables(asof_season):
    """asof_season 시즌을 예측할 때 쓰는 선수별 표.

    prep_cache 는 (선수, 시즌) 인덱스로 '그 시즌 시작 직전 누적' 을 잡는다.
    평가 데이터에는 그 시즌 행이 train 에 없으므로 직접 누적해서 만든다.
    """
    prev = tr[tr.season < asof_season]
    g = prev.groupby("pitcher_id")[TGT]
    p_n, p_s = g.size(), g.sum()
    gb = prev.groupby("batter_id")[TGT]
    b_n, b_s = gb.size(), gb.sum()

    # 플래툰 표는 train 에 나온 투수 전원을 담는다. 직전 시즌들 기록이 없으면
    # 카운트 0 -> 리그값으로 수축된다. prep_cache 의 cumsum-shift 가 첫 시즌 행에
    # NaN -> 0 을 넣는 것과 같은 처리다. (표에 없는 투수만 plat_dev = 0 이 된다)
    cells = prev.groupby(["pitcher_id", "batter_hand"])[TGT].agg(["size", "sum"]).unstack(fill_value=0)
    cells.columns = [f"{x}{int(h)}" for x, h in cells.columns]
    for h in (1, 2):
        for x in ("size", "sum"):
            if f"{x}{h}" not in cells.columns:
                cells[f"{x}{h}"] = 0
    cells = cells.reindex(PH.index, fill_value=0).sort_index()
    pid = cells.index.values
    hand = PH.reindex(pid).values
    n_all = cells.size1.values + cells.size2.values
    s_all = cells.sum1.values + cells.sum2.values
    out = {}
    for h in (1, 2):
        pr = np.array([lgph.get((p_, h), gm_) for p_ in hand])
        out[f"p{h}"] = (cells[f"sum{h}"].values + ALPHA_PLAT * pr) / (cells[f"size{h}"].values + ALPHA_PLAT)
    pr_a = np.array([lgp.get(p_, gm_) for p_ in hand])
    out["pa"] = (s_all + ALPHA_PLAT * pr_a) / (n_all + ALPHA_PLAT)
    return dict(
        pitcher_id=p_n.index.values.astype(np.int64),
        pitcher_n=p_n.values.astype(np.float64), pitcher_s=p_s.values.astype(np.float64),
        batter_id=b_n.index.values.astype(np.int64),
        batter_n=b_n.values.astype(np.float64), batter_s=b_s.values.astype(np.float64),
        plat_id=pid.astype(np.int64),
        plat_l=out["p1"], plat_r=out["p2"], plat_a=out["pa"],
    )


T2025 = make_tables(2025)
T2024 = make_tables(2024)
print(f"as-of 2025 표: 투수 {len(T2025['pitcher_id'])}명  타자 {len(T2025['batter_id'])}명")

# ------------------------------------------------------------------ 저장
cm_key = np.array(sorted(set(lg48.index) | set(rel48.index)), dtype=np.int64)
pack = dict(
    con_names=np.array(CON, dtype="U40"),
    cat_names=np.array(CAT, dtype="U40"),
    med=med.astype(np.float64), mu=mu.astype(np.float64), sd=sd.astype(np.float64),
    has_nan=has_nan,
    gmean=np.float64(gm_),
    cm_key=cm_key,
    cm_lg=np.array([lg48.get(k, 0.0) for k in cm_key], dtype=np.float64),
    cm_rel=np.array([rel48.get(k, 1.0) for k in cm_key], dtype=np.float64),
    alpha=np.float64(ALPHA),
    drop_dup=np.array(DUP + [PN, BN], dtype="U40"),
)
for j, (c, mp) in enumerate(zip(CAT, cat_maps)):
    pack[f"catkey_{j}"] = np.array(sorted(mp.keys()), dtype=np.float64) \
        if not isinstance(next(iter(mp)), str) else np.array(sorted(mp.keys()), dtype="U16")
    pack[f"catval_{j}"] = np.array([mp[k] for k in sorted(mp.keys())], dtype=np.int64)
for tag, T in (("t25", T2025), ("t24", T2024)):
    for k, v in T.items():
        pack[f"{tag}_{k}"] = v
np.savez_compressed(OUT, **pack)
print(f"\n저장 {OUT}  ({os.path.getsize(OUT)/1024/1024:.2f} MB)")

# 검증용으로 2024 구간의 정답 배열을 따로 남긴다
np.savez_compressed(os.path.join(SC, "mlp_feat_ref2024.npz"),
                    Xn=Xn[season == 2024], Xc=Xc[season == 2024])
print("검증 기준 저장: mlp_feat_ref2024.npz")
