# -*- coding: utf-8 -*-
"""평가 데이터(test.csv) -> MLP 입력 60열.

계산식은 prep_cache.py 를 그대로 옮겼다. 다른 점은 딱 하나,
'시즌 시작 직전 누적' 을 구하는 방법이다.

  학습 경로  그 (선수, 시즌) 첫 행의 asof 값
  평가 경로  train 전체(2019~2024) 누적  -> mlp_feat.npz 의 t25_* 표

둘은 의미가 같다. 2024 행을 평가 경로로 다시 만들어 학습 경로 결과와
일치하는지 verify_mlp_features.py 에서 확인한다.

정규화 상수(중앙값/평균/표준편차)와 범주 사전은 2024 이전 구간에서 잡은 것을
그대로 쓴다. 모델이 그 좌표계로 학습됐기 때문이다.
"""
import numpy as np
import pandas as pd

PN, BN = "asof_pitcher_n", "asof_batter_n"
CAT_STR_MAPS = {
    "top_bottom": {"T": 0, "B": 1},
    "game_type": {"R": 0, "F": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}


def load_feat(path):
    z = np.load(path, allow_pickle=False)
    F = {k: z[k] for k in z.files}
    F["con_names"] = [str(c) for c in F["con_names"]]
    F["cat_names"] = [str(c) for c in F["cat_names"]]
    return F


def _lookup(keys, vals, q, default):
    """정렬된 keys 에서 q 를 찾아 vals 를 준다. 없으면 default."""
    q = np.asarray(q, dtype=np.float64)
    if len(keys) == 0:
        return np.full(len(q), default, dtype=np.float64)
    pos = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    hit = keys[pos] == q
    return np.where(hit, np.asarray(vals, dtype=np.float64)[pos], default)


def build_mlp_input(df, F, tag="t25"):
    """df: test.csv 를 읽은 DataFrame. 반환 (n, 60) float32 = [수치44 | 범주16]."""
    X = df.copy()
    for c in X.columns:
        if X[c].dtype == "float64":
            X[c] = X[c].astype("float32")

    # ---- 시즌 내 성적 복원 (prep_cache 와 같은 식, 기준값만 표에서 가져온다)
    alpha = float(F["alpha"])
    c_p = (X.asof_pitcher_success_rate.fillna(0) * X[PN]).round()
    c_b = (X.asof_batter_success_rate.fillna(0) * X[BN]).round()
    bn_ = _lookup(F[f"{tag}_pitcher_id"], F[f"{tag}_pitcher_n"], X.pitcher_id.values, 0.0)
    bs_ = _lookup(F[f"{tag}_pitcher_id"], F[f"{tag}_pitcher_s"], X.pitcher_id.values, 0.0)
    bbn_ = _lookup(F[f"{tag}_batter_id"], F[f"{tag}_batter_n"], X.batter_id.values, 0.0)
    bbs_ = _lookup(F[f"{tag}_batter_id"], F[f"{tag}_batter_s"], X.batter_id.values, 0.0)
    pn_cur = np.clip(X[PN].to_numpy(dtype=np.float64) - bn_, 0, None).astype(np.float32)
    X["pn_cur"] = pn_cur
    X["p_is_succ"] = ((np.clip(c_p.to_numpy(dtype=np.float64) - bs_, 0, None) + alpha * .5)
                      / (pn_cur + alpha)).astype("float32")
    X["b_is_succ"] = ((np.clip(c_b.to_numpy(dtype=np.float64) - bbs_, 0, None) + alpha * .5)
                      / (np.clip(X[BN].to_numpy(dtype=np.float64) - bbn_, 0, None) + alpha)
                      ).astype("float32")

    # ---- 문자열 범주 -> 정수 (학습과 같은 고정 사전)
    # pandas 3 은 문자열 열을 dtype 'str' 로 읽고 2.x 는 'object' 로 읽는다.
    # 버전에 안 걸리도록 '숫자가 아니면 매핑' 으로 판정한다.
    for c, mp in CAT_STR_MAPS.items():
        if c in X.columns and not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].map(mp).fillna(-1).astype("int8")

    # ---- 볼카운트 x 좌우매치업
    cm = ((X.balls_before.to_numpy(np.int64) * 3 + X.strikes_before.to_numpy(np.int64)) * 4
          + X.pitcher_hand.to_numpy(np.int64) * 2 + X.batter_hand.to_numpy(np.int64))
    gm = float(F["gmean"])
    X["lg_cm_eff"] = _lookup(F["cm_key"], F["cm_lg"], cm, 0.0).astype("float32")
    X["cm_rel"] = _lookup(F["cm_key"], F["cm_rel"], cm, 1.0).astype("float32")
    X["p_adj_cm"] = (gm + (X.p_is_succ.to_numpy(np.float64) - gm)
                     * X.cm_rel.to_numpy(np.float64)).astype("float32")

    # ---- 플래툰 편차. 표에 없는 투수는 NaN -> 0 (prep_cache 와 같다)
    nan = np.nan
    pid = X.pitcher_id.values
    p1 = _lookup(F[f"{tag}_plat_id"], F[f"{tag}_plat_l"], pid, nan)
    p2 = _lookup(F[f"{tag}_plat_id"], F[f"{tag}_plat_r"], pid, nan)
    pa = _lookup(F[f"{tag}_plat_id"], F[f"{tag}_plat_a"], pid, nan)
    cur = np.where(X.batter_hand.values == 1, p1, p2)
    X["plat_dev"] = np.where(np.isnan(cur - pa), 0., cur - pa).astype("float32")

    # ---- 연속 28열: 중앙값 대체 -> 표준화 -> 결측표시자 16열
    Xn = X[F["con_names"]].to_numpy(dtype=np.float32)
    miss = np.isnan(Xn)
    Xn = np.where(miss, F["med"], Xn)
    Xn = ((Xn - F["mu"]) / F["sd"]).astype(np.float32)
    Xn = np.concatenate([Xn, miss[:, F["has_nan"]].astype(np.float32)], 1)

    # ---- 범주 16열: 고정 사전. 학습에 없던 값은 0
    cols = []
    for j, c in enumerate(F["cat_names"]):
        cols.append(_lookup(F[f"catkey_{j}"], F[f"catval_{j}"], X[c].values, 0.0))
    Xc = np.stack(cols, 1).astype(np.float32)

    return np.concatenate([Xn, Xc], 1).astype(np.float32)
