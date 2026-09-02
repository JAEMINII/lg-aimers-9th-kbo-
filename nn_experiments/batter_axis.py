# -*- coding: utf-8 -*-
"""타자 축 상호작용 — 연도 간격 있는 관문에서 재검증.

왜 이걸 보나
  2026-08-16_batter_interactions.md 가 batter x pitcher_hand 로 +25.1 을 얻고
  '아직 제출본에 안 넣음' 상태로 멈춰 있다. 그 노트가 짚은 대로
  현재 44피처에는 타자 쪽 상호작용 이력이 하나도 없다.

      투수 축                          타자 축
      p_is_succ    인시즌 성적          b_is_succ   인시즌 성적
      plat_dev     x 타자좌우           (없음)  <- 이게 batter x pitcher_hand
      cm_rel       x 볼카운트           (없음)
      p_adj_cm     x 볼카운트           (없음)

  축 하나가 통째로 비어 있다.

왜 다시 재나
  그 노트의 검증은 '2019~2023 + 2024 앞 80% 학습 -> 2024 뒤 20% 검증' 이라
  연도 간격이 없다. 간격이 없으면 선수 이력 피처가 과대평가된다는 게
  이 프로젝트에서 이미 겪은 교훈이다. 간격 있는 관문으로 다시 잰다.

설계
  관문   2019~2023 학습 -> 2024 검증, 1군 채점, shift 0
  모델   CatBoost 0.6 전체 + 0.4 R단독 (제출본과 같은 구조), 시드 3개
  기준   850.6

시점 정합
  학습 행에는 '그 시즌 시작 직전까지의 누적' 을 준다 (plat_dev 와 같은 cumsum-shift).
  자기 자신이 든 통계를 쓰면 누출이다.
  평가 행에는 학습 구간 전체 누적을 준다. 제출 시 test 행이 받는 것과 같은 성질이다.
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "batter_axis_progress.txt")
TGT = "control_success"
ALPHA, ALPHA_PLAT, ALPHA_B = 50.0, 300.0, 200.0
VS, BLEND = 2024, 0.6
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

m_va = (tr.season.values == VS)
m_tr = (tr.season.values < VS)
i_gt = np.where(m_va)[0]
fv = IS_F[m_va]
yv = tr.loc[m_va, TGT].values.astype(np.float64)
mm_ = ~fv


def bss(p):
    r = yv[mm_].mean()
    return 100000 * (1 - ((p[mm_] - yv[mm_]) ** 2).mean() / (r * (1 - r)))


def slope(x, y):
    v = x.var()
    return np.cov(x, y)[0, 1] / v if v > 1e-12 else np.nan


# --------------------------------------------------- 기존 투수 축 피처 (제출본과 동일)
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
lg_bh = hist.groupby("pitcher_hand")[TGT].mean()       # 타자축 스무딩용 리그값
lg_cnt = hist.groupby("cnt12")[TGT].mean()
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


# --------------------------------------------------- 타자 축 (새로 만드는 것)
def batter_cell(keycol, tag, alpha, lgmap):
    """(타자, 시즌, keycol) 누적 -> 그 시즌 시작 직전까지의 스무딩 성공률과 표본수.

    plat_dev 와 같은 cumsum-shift 방식이라 자기 자신이 안 들어간다.
    2024 행은 2019~2023 누적을 받는다 (평가 시 test 행이 받는 것과 같은 성질).
    """
    B = tr.groupby(["batter_id", "season", keycol])[TGT].agg(["size", "sum"]).unstack(fill_value=0)
    B.columns = [f"{x}_{int(k)}" for x, k in B.columns]
    BCC = B.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.)
    keys = sorted({int(c.split("_")[1]) for c in B.columns})
    idx = pd.MultiIndex.from_arrays([tr.batter_id.values, tr.season.values])
    sz = BCC[[f"size_{k}" for k in keys]].reindex(idx).values
    sm = BCC[[f"sum_{k}" for k in keys]].reindex(idx).values
    kv = tr[keycol].values
    pos = np.searchsorted(np.asarray(keys), kv)
    pos = np.clip(pos, 0, len(keys) - 1)
    r = np.arange(len(tr))
    n_ = np.where(np.asarray(keys)[pos] == kv, sz[r, pos], 0.0)
    s_ = np.where(np.asarray(keys)[pos] == kv, sm[r, pos], 0.0)
    n_ = np.nan_to_num(n_); s_ = np.nan_to_num(s_)
    prior = np.array([lgmap.get(k, gm_) for k in kv], dtype=np.float64)
    n_all = np.nan_to_num(BCC[[f"size_{k}" for k in keys]].reindex(idx).values).sum(1)
    s_all = np.nan_to_num(BCC[[f"sum_{k}" for k in keys]].reindex(idx).values).sum(1)
    base = (s_all + alpha * gm_) / (n_all + alpha)
    rate = (s_ + alpha * prior) / (n_ + alpha)
    tr[f"b_{tag}_rate"] = rate.astype("float32")
    tr[f"b_{tag}_dev"] = (rate - base).astype("float32")    # plat_dev 대칭
    tr[f"b_{tag}_n"] = np.log1p(n_).astype("float32")
    return [f"b_{tag}_rate", f"b_{tag}_dev", f"b_{tag}_n"]


t0 = time.time()
F_BH = batter_cell("pitcher_hand", "vh", ALPHA_B, lg_bh.to_dict())
F_CT = batter_cell("cnt12", "vc", ALPHA_B, lg_cnt.to_dict())
log(f"타자 축 피처 생성 {time.time()-t0:.0f}s   {F_BH + F_CT}")

test_cols = pd.read_csv(os.path.join(D, "test.csv"), encoding="utf-8-sig", nrows=0).columns
BASE = [c for c in test_cols if c != "row_id"]
DUP = [MX, "away_win_expectancy", "run_total_before", "score_diff_home",
       "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b"]
F44 = ([c for c in BASE if c not in DUP + [PN, BN]]
       + ["p_is_succ", "pn_cur", "b_is_succ", "lg_cm_eff", "cm_rel", "p_adj_cm", "plat_dev"])

ytr = tr.loc[m_tr, TGT].values
ftr = IS_F[m_tr]
log(f"\n학습 {int(m_tr.sum()):,} -> 관문 2024 1군 {int(mm_.sum()):,}   시드 {len(SEEDS)}개")
log(f"기준(제출본 구조) 850.6\n")
log(f"  {'구성':<30s}{'피처':>5s}{'점수':>9s}{'기준대비':>10s}{'시간':>7s}")

CASES = [
    ("기존 44피처",                    F44),
    ("+ 타자x투수좌우",                 F44 + F_BH),
    ("+ 타자x볼카운트",                 F44 + F_CT),
    ("+ 둘 다",                        F44 + F_BH + F_CT),
]
base = None
for name, feats in CASES:
    t0 = time.time()
    Xtr = tr.loc[m_tr, feats].to_numpy(dtype=np.float32)
    Xva = tr.loc[m_va, feats].to_numpy(dtype=np.float32)
    HP = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
              verbose=0, allow_writing_files=False, thread_count=4)
    pf, pr_ = [], []
    for sd in SEEDS:
        c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr, ytr)
        pf.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
        c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr[~ftr], ytr[~ftr])
        pr_.append(c.predict_proba(Xva)[:, 1]); del c; gc.collect()
    del Xtr, Xva; gc.collect()
    p = BLEND * np.mean(pf, 0) + (1 - BLEND) * np.mean(pr_, 0)
    np.save(os.path.join(SC, f"baxis_{len(feats)}.npy"), p)
    s = bss(p)
    if base is None:
        base = s
    log(f"  {name:<30s}{len(feats):>5d}{s:9.1f}{s-base:+10.1f}{time.time()-t0:6.0f}s")
log("\n판독: 간격 있는 관문에서도 +15 이상이면 제출본에 넣을 값어치가 있다.")
log("      +25 가 +5 로 줄면 간격 없는 검증이 부풀린 것이다.")
