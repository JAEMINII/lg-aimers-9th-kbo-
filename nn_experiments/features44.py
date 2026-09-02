# -*- coding: utf-8 -*-
"""44피처 생성 — 로컬에서 검증된 경로 그대로.

검증 이력
  · 이 계산식으로 만든 배열이 `nn_cache.npz` 와 비트단위 일치
    (nn_experiments/export_mlp_features.py 에서 확인)
  · 지인 TRAINING_SPEC 의 44피처 목록과 **순서까지 동일** 함을 확인

누출 방지 규약 (이걸 어기면 관문 점수가 통째로 거짓이 된다)
  · 룩업 테이블(gm_, lgph, lgp, lg48, rel48, 플래툰)은 **학습 구간에서만** 만든다
  · 선수 이력은 (선수, 시즌) 누적을 cumsum -> shift(1) 해서
    '그 시즌 시작 직전까지' 만 본다. 자기 자신이 안 들어간다
  · VS 인자가 그 경계를 정한다. VS=2024 면 2019~2023 이 학습 구간
"""
import gc
import os

import numpy as np
import pandas as pd

TGT = "control_success"
ALPHA, ALPHA_PLAT = 50.0, 300.0
PN, BN, MX = "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"
CAT_STR_MAPS = {
    "top_bottom": {"T": 0, "B": 1},
    "game_type": {"R": 0, "F": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}
# TabM 내부에서 범주형으로 다룰 9개 (지인 SPEC 과 동일)
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
            "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]


def _slope(x, y):
    v = x.var()
    return np.cov(x, y)[0, 1] / v if v > 1e-12 else np.nan


def build(data_dir, VS=2024, return_frame=False):
    """data_dir: train.csv / test.csv 가 있는 폴더.  VS: 검증 시즌.

    반환 dict
      X44    (n, 44) float32   F44 순서
      y      (n,)    float32
      season (n,) int16 / is_f (n,) bool
      m_tr / m_va  학습·검증 마스크 (season < VS / == VS)
      F44    피처 이름 목록,  cat_idx  범주형 열 위치
    """
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"), encoding="utf-8-sig")
    tr["_r"] = tr.row_id.str.slice(6).astype("int32")
    tr = tr.sort_values("_r").reset_index(drop=True)
    IS_F = (tr.game_type.values == "F")
    for c in tr.columns:
        if tr[c].dtype == "float64":
            tr[c] = tr[c].astype("float32")

    # ---- 인시즌 복원: asof 차분으로 '이번 시즌 성적' 을 만든다
    tr["c_p"] = (tr.asof_pitcher_success_rate.fillna(0) * tr[PN]).round()
    tr["c_b"] = (tr.asof_batter_success_rate.fillna(0) * tr[BN]).round()
    f1 = tr.groupby(["pitcher_id", "season"], sort=False).head(1)[
        ["pitcher_id", "season", PN, "c_p"]]
    f1.columns = ["pitcher_id", "season", "bn_", "bs_"]
    tr = tr.merge(f1, on=["pitcher_id", "season"], how="left")
    f2 = tr.groupby(["batter_id", "season"], sort=False).head(1)[
        ["batter_id", "season", BN, "c_b"]]
    f2.columns = ["batter_id", "season", "bbn_", "bbs_"]
    tr = tr.merge(f2, on=["batter_id", "season"], how="left")
    tr["pn_cur"] = (tr[PN] - tr.bn_).clip(lower=0).astype("float32")
    tr["p_is_succ"] = (((tr.c_p - tr.bs_).clip(lower=0) + ALPHA * .5)
                       / (tr.pn_cur + ALPHA)).astype("float32")
    tr["b_is_succ"] = (((tr.c_b - tr.bbs_).clip(lower=0) + ALPHA * .5)
                       / ((tr[BN] - tr.bbn_).clip(lower=0) + ALPHA)).astype("float32")
    tr.drop(columns=["c_p", "c_b", "bn_", "bs_", "bbn_", "bbs_"], inplace=True)
    gc.collect()

    # ---- 플래툰 셀: (투수, 시즌, 타자손) 을 그 시즌 직전까지 누적
    PC = tr.groupby(["pitcher_id", "season", "batter_hand"])[TGT] \
           .agg(["size", "sum"]).unstack(fill_value=0)
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

    for c, m in CAT_STR_MAPS.items():
        if not pd.api.types.is_numeric_dtype(tr[c]):
            tr[c] = tr[c].map(m).fillna(-1).astype("int8")
    tr["cm"] = ((tr.balls_before * 3 + tr.strikes_before) * 4
                + tr.pitcher_hand * 2 + tr.batter_hand).astype("int16")

    m_va = (tr.season.values == VS)
    m_tr = (tr.season.values < VS)

    # ---- 여기부터 룩업. 반드시 학습 구간에서만 만든다
    hist = tr[m_tr]
    gm_ = float(hist[TGT].mean())
    lgph = hist.groupby(["pitcher_hand", "batter_hand"])[TGT].mean()
    lgp = hist.groupby("pitcher_hand")[TGT].mean()
    lg48 = hist.groupby("cm")[TGT].mean() - gm_
    h2 = tr[m_tr & tr.p_is_succ.notna().values]
    rel48 = (h2.groupby("cm").apply(
        lambda dd: _slope(dd.p_is_succ.values, dd[TGT].values), include_groups=False)
        / _slope(h2.p_is_succ.values, h2[TGT].values)).clip(.2, 1.8)
    del hist, h2
    gc.collect()

    tr["lg_cm_eff"] = tr.cm.map(lg48).fillna(0.).astype("float32")
    tr["cm_rel"] = tr.cm.map(rel48).fillna(1.).astype("float32")
    tr["p_adj_cm"] = (gm_ + (tr.p_is_succ - gm_) * tr.cm_rel).astype("float32")

    ph_of = PH.reindex(PCC.index.get_level_values("pitcher_id")).values
    Pd = {}
    for h in (1, 2):
        pr = np.array([lgph.get((p_, h), gm_) for p_ in ph_of])
        Pd[h] = ((PCC[f"sum{h}"].values + ALPHA_PLAT * pr)
                 / (PCC[f"size{h}"].values + ALPHA_PLAT))
    pa = ((PCC.s_all.values + ALPHA_PLAT
           * np.array([lgp.get(p_, gm_) for p_ in ph_of]))
          / (PCC.n_all.values + ALPHA_PLAT))
    lut = pd.DataFrame({"p1": Pd[1], "p2": Pd[2], "pa": pa}, index=PCC.index).reindex(
        pd.MultiIndex.from_arrays([tr.pitcher_id.values, tr.season.values]))
    dv = np.where(BHAND == 1, lut.p1.values, lut.p2.values) - lut.pa.values
    tr["plat_dev"] = np.where(np.isnan(dv), 0., dv).astype("float32")

    # ---- F44 확정 (test.csv 열 순서에서 중복·누출 제거 + 파생 7개)
    test_cols = pd.read_csv(os.path.join(data_dir, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    BASE = [c for c in test_cols if c != "row_id"]
    DUP = [MX, "away_win_expectancy", "run_total_before", "score_diff_home",
           "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b"]
    F44 = ([c for c in BASE if c not in DUP + [PN, BN]]
           + ["p_is_succ", "pn_cur", "b_is_succ", "lg_cm_eff", "cm_rel",
              "p_adj_cm", "plat_dev"])
    assert len(F44) == 44, f"F44 가 {len(F44)}개다"

    # 피처 스크리닝은 원본 프레임이 있어야 새 열을 붙일 수 있다.
    # 기본값은 예전 그대로라 기존 호출부는 영향 없다.
    return dict(
        frame=tr if return_frame else None,
        X44=tr[F44].to_numpy(dtype=np.float32),
        y=tr[TGT].to_numpy(dtype=np.float32),
        season=tr.season.to_numpy(np.int16),
        is_f=IS_F, m_tr=m_tr, m_va=m_va, F44=F44,
        cat_idx=[F44.index(c) for c in CAT_COLS],
    )


# ---------------------------------------------------------------- 채점
def bss(p, t):
    """대회 지표.  Score = 100000 x (1 - Brier / r(1-r))"""
    r = t.mean()
    return 100000 * (1 - ((p - t) ** 2).mean() / (r * (1 - r)))


def shift(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))


def best_shift(p, t):
    """최적 시프트에서의 점수 = '판별력만'.

    고정 시프트로 비교하면 수준 보정과 판별력이 섞여 최적 비중이 움직인다.
    후보 비교는 반드시 이 값으로 한다.
    """
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda c: -bss(shift(p, c), t),
                        bounds=(-0.3, 0.3), method="bounded")
    return -r.fun, r.x
