# -*- coding: utf-8 -*-
"""Canonical submit_jaemin_5-compatible feature pipeline.

Validation and full training import this exact module.  The inference path uses
only the current row and lookup tables fitted on official training data; it
never aggregates other test rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping

import numpy as np
import pandas as pd


ID = "row_id"
TARGET = "control_success"
ALPHA = 50.0
PRIOR_SUCCESS = 0.50
PLATOON_ALPHA = 300.0
UNKNOWN_CAT = -1

CAT_MAPS = {
    "top_bottom": {"T": 0, "B": 1},
    "game_type": {"R": 0, "F": 1},
    "base_state": {
        "___": 0, "1__": 1, "_2_": 2, "__3": 3,
        "12_": 4, "1_3": 5, "_23": 6, "123": 7,
    },
}

# Exact order exported by submit_jaemin_5.
FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "score_diff_pitcher_team",
    "base_state", "home_win_expectancy", "li", "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate", "p_is_succ",
    "pn_cur", "b_is_succ", "lg_cm_eff", "cm_rel", "p_adj_cm", "plat_dev",
]

TABM_CATEGORICAL_FEATURES = [
    "top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]


def _python_value(value):
    return value.item() if isinstance(value, np.generic) else value


def sort_by_row_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    suffix = out[ID].astype(str).str.extract(r"(\d+)$", expand=False)
    if suffix.isna().any():
        bad = out.loc[suffix.isna(), ID].head(3).tolist()
        raise ValueError(f"row_id must end with digits; examples={bad}")
    out["_rid"] = suffix.astype("int64")
    return out.sort_values("_rid", kind="mergesort").reset_index(drop=True)


def recover_counts(df: pd.DataFrame) -> dict[str, pd.Series]:
    pn = df["asof_pitcher_n"].astype("float64")
    bn = df["asof_batter_n"].astype("float64")
    return {
        "p_n": pn,
        "p_succ": (df["asof_pitcher_success_rate"].fillna(0.0) * pn).round(),
        "b_n": bn,
        "b_succ": (df["asof_batter_success_rate"].fillna(0.0) * bn).round(),
    }


def _season_start_bases(df: pd.DataFrame, cnt: Mapping[str, pd.Series]):
    pg = [df["pitcher_id"], df["season"]]
    bg = [df["batter_id"], df["season"]]
    return (
        cnt["p_n"].groupby(pg, sort=False).transform("first"),
        cnt["p_succ"].groupby(pg, sort=False).transform("first"),
        cnt["b_n"].groupby(bg, sort=False).transform("first"),
        cnt["b_succ"].groupby(bg, sort=False).transform("first"),
    )


def _add_inseason(X, cnt, base_p_n, base_p_s, base_b_n, base_b_s):
    pn = (cnt["p_n"] - base_p_n).clip(lower=0)
    ps = (cnt["p_succ"] - base_p_s).clip(lower=0)
    bn = (cnt["b_n"] - base_b_n).clip(lower=0)
    bs = (cnt["b_succ"] - base_b_s).clip(lower=0)
    X["pn_cur"] = pn
    X["bn_cur"] = bn
    X["p_is_succ"] = (ps + ALPHA * PRIOR_SUCCESS) / (pn + ALPHA)
    X["b_is_succ"] = (bs + ALPHA * PRIOR_SUCCESS) / (bn + ALPHA)
    return X


def _slope_table(key, x, y):
    tmp = pd.DataFrame({"key": key, "x": x.astype(float), "y": y.astype(float)})
    gx, gy = tmp["x"].mean(), tmp["y"].mean()
    variance = ((tmp["x"] - gx) ** 2).sum()
    covariance = ((tmp["x"] - gx) * (tmp["y"] - gy)).sum()
    global_slope = covariance / variance if variance > 1e-12 else 1.0
    tmp["xy"] = tmp["x"] * tmp["y"]
    stats = tmp.groupby("key", sort=False).agg(
        x_mean=("x", "mean"), y_mean=("y", "mean"),
        x2_mean=("x", lambda s: (s * s).mean()), xy_mean=("xy", "mean"),
    )
    variance = stats["x2_mean"] - stats["x_mean"] ** 2
    covariance = stats["xy_mean"] - stats["x_mean"] * stats["y_mean"]
    return ((covariance / variance) / global_slope).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(1.0)


def fit_history_tables(df: pd.DataFrame) -> dict[str, object]:
    """Fit all target-derived tables on the fold's training rows only."""
    df = sort_by_row_id(df)
    if TARGET not in df:
        raise ValueError(f"{TARGET!r} is required to fit history tables")
    cnt = recover_counts(df)
    y = df[TARGET].astype("float64")
    gmean = float(y.mean())

    pitcher_n = df.groupby("pitcher_id", sort=False).size().astype("float64")
    pitcher_s = y.groupby(df["pitcher_id"], sort=False).sum()
    batter_n = df.groupby("batter_id", sort=False).size().astype("float64")
    batter_s = y.groupby(df["batter_id"], sort=False).sum()

    current = df[["pitcher_id", "batter_id", "season"]].copy()
    current = _add_inseason(current, cnt, *_season_start_bases(df, cnt))
    cm = (
        (df["balls_before"].astype("int32") * 3 + df["strikes_before"].astype("int32")) * 4
        + df["pitcher_hand"].astype("int32") * 2 + df["batter_hand"].astype("int32")
    )
    cm_lg = y.groupby(cm, sort=False).mean() - gmean
    cm_rel = _slope_table(cm, current["p_is_succ"], y)

    plat = pd.DataFrame({
        "pitcher_id": df["pitcher_id"], "pitcher_hand": df["pitcher_hand"],
        "batter_hand": df["batter_hand"], "y": y,
    })
    pitcher_hand = plat.groupby("pitcher_id", sort=False)["pitcher_hand"].first()
    hand_prior = plat.groupby("pitcher_hand", sort=False)["y"].mean()
    matchup_prior = plat.groupby(["pitcher_hand", "batter_hand"], sort=False)["y"].mean()
    all_stats = plat.groupby("pitcher_id", sort=False)["y"].agg(["sum", "count"])
    all_prior = pitcher_hand.map(hand_prior).astype("float64")
    plat_a = (all_stats["sum"] + PLATOON_ALPHA * all_prior) / (
        all_stats["count"] + PLATOON_ALPHA
    )
    by_hand = plat.groupby(["pitcher_id", "batter_hand"], sort=False)["y"].agg(
        ["sum", "count"]
    )

    def smooth_matchup(batter_hand):
        values = {}
        for pitcher_id, p_hand in pitcher_hand.items():
            prior = float(matchup_prior.get((p_hand, batter_hand), gmean))
            key = (pitcher_id, batter_hand)
            if key in by_hand.index:
                row = by_hand.loc[key]
                value = (float(row["sum"]) + PLATOON_ALPHA * prior) / (
                    float(row["count"]) + PLATOON_ALPHA
                )
            else:
                value = prior
            values[_python_value(pitcher_id)] = value
        return values

    return {
        "pitcher_n": pitcher_n.to_dict(), "pitcher_s": pitcher_s.to_dict(),
        "batter_n": batter_n.to_dict(), "batter_s": batter_s.to_dict(),
        "cm_lg": cm_lg.to_dict(), "cm_rel": cm_rel.to_dict(),
        "plat_a": plat_a.to_dict(), "plat_l": smooth_matchup(1),
        "plat_r": smooth_matchup(2), "gmean": gmean, "features": list(FEATURES),
    }


def _encode_fixed_categoricals(X):
    for col, mapping in CAT_MAPS.items():
        X[col] = X[col].map(mapping).fillna(UNKNOWN_CAT).astype("int16")
    return X


def transform_features(
    df: pd.DataFrame, history: Mapping[str, object], *, train_mode: bool
) -> pd.DataFrame:
    """Transform rows using train-only history; no cross-row test aggregation."""
    ordered = sort_by_row_id(df)
    X = ordered.drop(columns=[ID, TARGET, "_rid"], errors="ignore").copy()
    cnt = recover_counts(ordered)
    if train_mode:
        bases = _season_start_bases(ordered, cnt)
    else:
        bases = (
            ordered["pitcher_id"].map(history["pitcher_n"]).fillna(0.0),
            ordered["pitcher_id"].map(history["pitcher_s"]).fillna(0.0),
            ordered["batter_id"].map(history["batter_n"]).fillna(0.0),
            ordered["batter_id"].map(history["batter_s"]).fillna(0.0),
        )
    X = _add_inseason(X, cnt, *bases)
    cm = (
        (ordered["balls_before"].astype("int32") * 3 + ordered["strikes_before"].astype("int32")) * 4
        + ordered["pitcher_hand"].astype("int32") * 2
        + ordered["batter_hand"].astype("int32")
    )
    X["lg_cm_eff"] = cm.map(history["cm_lg"]).fillna(0.0).astype("float64")
    X["cm_rel"] = cm.map(history["cm_rel"]).fillna(1.0).astype("float64")
    gmean = float(history["gmean"])
    X["p_adj_cm"] = gmean + (X["p_is_succ"] - gmean) * X["cm_rel"]
    pid = ordered["pitcher_id"]
    left, right, overall = (
        pid.map(history["plat_l"]), pid.map(history["plat_r"]), pid.map(history["plat_a"])
    )
    matchup = np.where(
        ordered["batter_hand"].to_numpy() == 1,
        left.to_numpy(dtype=float), right.to_numpy(dtype=float),
    )
    X["plat_dev"] = np.nan_to_num(
        matchup - overall.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    X = _encode_fixed_categoricals(X)
    features = list(history.get("features", FEATURES))
    missing = [col for col in features if col not in X]
    if missing:
        raise ValueError(f"missing advanced features: {missing}")
    return X[features].reset_index(drop=True)


def build_inference_features(df: pd.DataFrame, history: Mapping[str, object]):
    return transform_features(df, history, train_mode=False)


def serialize_history(history: Mapping[str, object]) -> dict:
    mapping_keys = [
        "pitcher_n", "pitcher_s", "batter_n", "batter_s", "cm_lg", "cm_rel",
        "plat_a", "plat_l", "plat_r",
    ]
    out = {
        key: [[_python_value(k), float(v)] for k, v in history[key].items()]
        for key in mapping_keys
    }
    out.update(gmean=float(history["gmean"]), features=list(history["features"]))
    return out


def deserialize_history(payload: Mapping[str, object]) -> dict:
    out = dict(payload)
    for key in [
        "pitcher_n", "pitcher_s", "batter_n", "batter_s", "cm_lg", "cm_rel",
        "plat_a", "plat_l", "plat_r",
    ]:
        out[key] = {pair[0]: float(pair[1]) for pair in payload[key]}
    return out


def feature_signature() -> str:
    payload = json.dumps(
        {"features": FEATURES, "cat_maps": CAT_MAPS, "alpha": ALPHA,
         "platoon_alpha": PLATOON_ALPHA}, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

