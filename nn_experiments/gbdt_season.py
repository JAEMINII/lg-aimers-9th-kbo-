# -*- coding: utf-8 -*-
"""
GBDT 에도 '최근 시즌만' 이 통하는가 — 제출본에 직접 영향.

MLP 에서 나온 결과 (관문 2024, 1군 채점, 판별력만)
    2019~2023  122만행    455.3    예측평균 0.5077
    2022~2023   49.3만    522.4    예측평균 0.5075
    2023만      24.6만    771.9    예측평균 0.4924   <- 압도적
  2022 를 넣기만 해도 250점이 날아간다. 수준만이 아니라 판별력 자체가 나빠진다.

시즌별 성공률
    2019 56.47% / 2020 53.27% / 2021 53.28% / 2022 52.89% / 2023 50.00% / 2024 48.61%
  2019 와 2024 는 7.9%p 차이. 완전히 다른 리그다.

GBDT 는 이미 대응 장치가 있다
    season 을 피처로 가짐        모델이 연도를 구분할 수 있음
    p_is_succ 인시즌 복원        커리어 오염 제거
    shift 보정                  수준 이동 보정
  그래서 이득이 작을 수 있다. 하지만 안 재봤다.

측정
  현행 제출본과 같은 구조로 학습 구간만 바꾼다.
    CatBoost 0.6 x 전체 + 0.4 x R단독,  관문 2024, 1군 채점, shift 0 / −0.05
  F단독은 1군 채점에 안 쓰이므로 만들지 않는다(학습 절반 절약).

  기준: 2019~2023 전체 = 850.6 / 886.3
"""
import os, sys, time, gc, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
TGT = "control_success"
SEEDS = (1, 42, 777)
ALPHA, ALPHA_PLAT = 50.0, 300.0
VS, BLEND = 2024, 0.6
HP = dict(iterations=400, learning_rate=.05, depth=4, l2_leaf_reg=1.,
          verbose=0, allow_writing_files=False, thread_count=6)
def H(t):
    print("\n" + "=" * 92); print(t); print("=" * 92); sys.stdout.flush()

t00 = time.time()
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
f1 = tr.groupby(["pitcher_id","season"], sort=False).head(1)[["pitcher_id","season",PN,"c_p"]]
f1.columns = ["pitcher_id","season","bn_","bs_"]; tr = tr.merge(f1, on=["pitcher_id","season"], how="left")
f2 = tr.groupby(["batter_id","season"], sort=False).head(1)[["batter_id","season",BN,"c_b"]]
f2.columns = ["batter_id","season","bbn_","bbs_"]; tr = tr.merge(f2, on=["batter_id","season"], how="left")
tr["pn_cur"] = (tr[PN]-tr.bn_).clip(lower=0).astype("float32")
tr["p_is_succ"] = (((tr.c_p-tr.bs_).clip(lower=0)+ALPHA*.5)/(tr.pn_cur+ALPHA)).astype("float32")
tr["b_is_succ"] = (((tr.c_b-tr.bbs_).clip(lower=0)+ALPHA*.5)
                   /((tr[BN]-tr.bbn_).clip(lower=0)+ALPHA)).astype("float32")
tr.drop(columns=["c_p","c_b","bn_","bs_","bbn_","bbs_"], inplace=True); gc.collect()

PC = tr.groupby(["pitcher_id","season","batter_hand"])[TGT].agg(["size","sum"]).unstack(fill_value=0)
PC.columns = [f"{x}{int(h)}" for x, h in PC.columns]
for h in (1,2):
    for x in ("size","sum"):
        if f"{x}{h}" not in PC.columns: PC[f"{x}{h}"] = 0
PCC = PC.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.)
PCC["n_all"] = PCC.size1+PCC.size2; PCC["s_all"] = PCC.sum1+PCC.sum2
PH = tr.groupby("pitcher_id").pitcher_hand.first()
BHAND = tr.batter_hand.values.copy()
for c, mp in {"top_bottom":{"T":0,"B":1}, "game_type":{"R":0,"F":1},
              "base_state":{"___":0,"1__":1,"_2_":2,"__3":3,"12_":4,"1_3":5,"_23":6,"123":7}}.items():
    tr[c] = tr[c].map(mp).fillna(-1).astype("int8")
tr["cm"] = ((tr.balls_before*3+tr.strikes_before)*4+tr.pitcher_hand*2+tr.batter_hand).astype("int16")

def slope(x, y_):
    v = x.var(); return np.cov(x, y_)[0,1]/v if v > 1e-12 else np.nan

m_va = (tr.season == VS).values
i_gt = np.where(m_va)[0]
fv = IS_F[m_va]; yv = tr.loc[m_va, TGT].values
mm_ = ~fv
def bss(p, yy):
    r = yy.mean()
    return 100000*(1-((p-yy)**2).mean()/(r*(1-r)))
def sh(p, c):
    p = np.clip(p,1e-6,1-1e-6); return 1/(1+np.exp(-(np.log(p/(1-p))+c)))

test_cols = pd.read_csv(os.path.join(D,"test.csv"), encoding="utf-8-sig", nrows=0).columns
BASE = [c for c in test_cols if c != "row_id"]
DUP = [MX,"away_win_expectancy","run_total_before","score_diff_home",
       "num_runners_on","runner_on_1b","runner_on_2b","runner_on_3b"]
F44 = ([c for c in BASE if c not in DUP+[PN,BN]]
       + ["p_is_succ","pn_cur","b_is_succ","lg_cm_eff","cm_rel","p_adj_cm","plat_dev"])

CASES = [("2019~2023", [2019,2020,2021,2022,2023]),
         ("2021~2023", [2021,2022,2023]),
         ("2022~2023", [2022,2023]),
         ("2023만",    [2023])]
H(f"CatBoost 0.6전체+0.4R단독, 관문 2024 {len(i_gt):,}, 1군 채점")
print("  MLP 실측: 2019~2023 455.3 / 2022~2023 522.4 / 2023만 771.9")
print("  GBDT 기준: 2019~2023 = 850.6 (shift0) / 886.3 (shift−.05)\n")
PROG = os.path.join(SC, "scratchpad", "gseason_progress.txt")
for name, yrs in CASES:
    m_tr = np.isin(tr.season.values, yrs)
    # 룩업 테이블은 '학습에 쓰는 구간' 에서만 만든다 (평가 시즌 미사용)
    hist = tr[m_tr]; gm_ = float(hist[TGT].mean())
    lgph = hist.groupby(["pitcher_hand","batter_hand"])[TGT].mean()
    lgp = hist.groupby("pitcher_hand")[TGT].mean()
    lg48 = hist.groupby("cm")[TGT].mean() - gm_
    h2 = tr[m_tr & tr.p_is_succ.notna().values]
    rel48 = (h2.groupby("cm").apply(lambda d: slope(d.p_is_succ.values, d[TGT].values),
                                    include_groups=False)
             / slope(h2.p_is_succ.values, h2[TGT].values)).clip(.2,1.8)
    del hist, h2; gc.collect()
    tr["lg_cm_eff"] = tr.cm.map(lg48).fillna(0.).astype("float32")
    tr["cm_rel"] = tr.cm.map(rel48).fillna(1.).astype("float32")
    tr["p_adj_cm"] = (gm_+(tr.p_is_succ-gm_)*tr.cm_rel).astype("float32")
    ph_of = PH.reindex(PCC.index.get_level_values("pitcher_id")).values
    Pd = {}
    for h in (1,2):
        pr = np.array([lgph.get((p_,h), gm_) for p_ in ph_of])
        Pd[h] = (PCC[f"sum{h}"].values+ALPHA_PLAT*pr)/(PCC[f"size{h}"].values+ALPHA_PLAT)
    pa = ((PCC.s_all.values+ALPHA_PLAT*np.array([lgp.get(p_,gm_) for p_ in ph_of]))
          /(PCC.n_all.values+ALPHA_PLAT))
    lut = pd.DataFrame({"p1":Pd[1],"p2":Pd[2],"pa":pa}, index=PCC.index).reindex(
        pd.MultiIndex.from_arrays([tr.pitcher_id.values, tr.season.values]))
    dv = np.where(BHAND==1, lut.p1.values, lut.p2.values) - lut.pa.values
    tr["plat_dev"] = np.where(np.isnan(dv), 0., dv).astype("float32")

    ytr = tr.loc[m_tr, TGT].values
    ftr = IS_F[m_tr]
    Xtr = tr.loc[m_tr, F44].to_numpy(dtype=np.float32)
    Xva = tr.loc[m_va, F44].to_numpy(dtype=np.float32)
    t1 = time.time()
    pf, pr_ = [], []
    for sd in SEEDS:
        c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr, ytr)
        pf.append(c.predict_proba(Xva)[:,1]); del c; gc.collect()
        c = CatBoostClassifier(random_seed=sd, **HP); c.fit(Xtr[~ftr], ytr[~ftr])
        pr_.append(c.predict_proba(Xva)[:,1]); del c; gc.collect()
    del Xtr, Xva; gc.collect()
    p = BLEND*np.mean(pf,0) + (1-BLEND)*np.mean(pr_,0)
    np.save(os.path.join(SC, f"gseason_{name.replace('~','_')}.npy"), p)
    line = (f"  {name:10s} {int(m_tr.sum()):>9,}행   "
            f"shift0 {bss(sh(p,0.0)[mm_], yv[mm_]):8.1f}   "
            f"shift−.05 {bss(sh(p,-0.05)[mm_], yv[mm_]):8.1f}   "
            f"예측평균 {p[mm_].mean():.4f}(실제 {yv[mm_].mean():.4f})   ({time.time()-t1:.0f}s)")
    print(line, flush=True); open(PROG,"a",encoding="utf-8").write(line+"\n")
print(f"\n총 {time.time()-t00:.0f}s")
