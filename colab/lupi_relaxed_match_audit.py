# -*- coding: utf-8 -*-
"""Audit a cautious second-layer Trackman match for rows without batter IDs.

The candidate retains a mapped pitcher, both teams, date proxy, full pitch
state, and batter handedness; it still requires a one-to-one key on both
datasets.  It is accepted nowhere automatically.  Its quality is measured on
the subset for which the independent batter map supplies a hidden answer.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AIMERS_DATA", ROOT / "open (1)" / "data"))
MAP = ROOT / "trackman_map"
AUDIT = ROOT / "colab" / "_dl" / "mapping_audit"
STATE = ["inning", "top_bottom", "balls_before", "strikes_before", "outs_before"]
BASE = ["season", "game_month", "game_dayofweek", "pt", "bt"]


def hand_tm(x):
    s = x.astype(str).str.lower()
    return np.where(s.str.startswith("l"), 1, np.where(s.str.startswith("r"), 2, -1))


def row_num(x):
    return x.astype(str).str.extract(r"(\d+)$", expand=False).astype(np.int64)


def unique_pair(left, right, keys):
    a = left.groupby(keys, sort=False).size().rename("na").reset_index()
    b = right.groupby(keys, sort=False).size().rename("nb").reset_index()
    keep = a.merge(b, on=keys)
    keep = keep[(keep.na == 1) & (keep.nb == 1)][keys]
    return left.merge(keep, on=keys).merge(right.merge(keep, on=keys), on=keys,
                                             suffixes=("_tr", "_tm"), validate="one_to_one")


def sequence_report(x):
    if not len(x):
        return {"rows": 0}
    q = x.copy()
    q["rr"] = q.groupby("trackman_game_id")["row_num"].rank(method="first")
    q["rp"] = q.groupby("trackman_game_id")["pitch_no"].rank(method="first")
    n = q.groupby("trackman_game_id").size()
    corr = pd.Series({g: z.rr.corr(z.rp) for g, z in q.groupby("trackman_game_id", sort=False)})
    corr = corr[n.reindex(corr.index) >= 5]
    return {"rows": int(len(q)), "games_5plus": int(len(corr)),
            "median_rank_corr": float(corr.median()) if len(corr) else np.nan,
            "below_0_99": int((corr < .99).sum())}


def main():
    teams = pd.read_csv(MAP / "team_map.csv", encoding="utf-8-sig")
    tm2tr = dict(zip(teams.trackman_team, teams.train_team_id))
    p = pd.read_csv(AUDIT / "pitcher_map_independent.csv")
    b = pd.read_csv(AUDIT / "batter_map_independent.csv")
    pmap, bmap = dict(zip(p.pitcher_id, p.trackman_id)), dict(zip(b.batter_id, b.trackman_id))
    safe = pd.read_csv(AUDIT / "strict_matches_independent.csv.gz", usecols=["row_id", "trackman_id"])
    safe_rows, safe_tm = set(safe.row_id), set(safe.trackman_id)

    tc = ["row_id", "season", "game_month", "game_dayofweek", "game_type", "pitcher_id", "batter_id",
          "batter_hand", "pitcher_team_id", "batter_team_id"] + STATE
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig", usecols=tc)
    tr = tr[tr.game_type.eq("R")].copy()
    tr.top_bottom = tr.top_bottom.astype(str).str[:1].str.upper()
    tr["pt"], tr["bt"] = tr.pitcher_team_id, tr.batter_team_id
    tr["pid_tm"] = tr.pitcher_id.map(pmap)
    tr["bid_expected"] = tr.batter_id.map(bmap)
    tr["bh"] = tr.batter_hand.astype(np.int8)
    tr["row_num"] = row_num(tr.row_id)
    tr = tr.dropna(subset=["pid_tm"]).copy()
    tr.pid_tm = tr.pid_tm.astype(np.int64)

    mc = ["trackman_id", "season", "game_month", "game_dayofweek", "trackman_game_id", "pitch_no",
          "pitcher_trackman_id", "batter_trackman_id", "batter_hand", "pitcher_team", "batter_team"] + STATE
    tm = pd.read_csv(DATA / "trackman_history.csv", encoding="utf-8-sig", usecols=mc)
    tm.top_bottom = tm.top_bottom.astype(str).str[:1].str.upper()
    tm["pt"], tm["bt"] = tm.pitcher_team.map(tm2tr), tm.batter_team.map(tm2tr)
    tm = tm.dropna(subset=["pt", "bt"]).copy()
    tm[["pt", "bt"]] = tm[["pt", "bt"]].astype(np.int64)
    tm["pid_tm"] = tm.pitcher_trackman_id.astype(np.int64)
    tm["bid_tm"] = tm.batter_trackman_id.astype(np.int64)
    tm["bh"] = hand_tm(tm.batter_hand).astype(np.int8)

    keys = BASE + STATE + ["pid_tm", "bh"]
    pairs = unique_pair(
        tr[keys + ["row_id", "row_num", "bid_expected"]],
        tm[keys + ["trackman_id", "trackman_game_id", "pitch_no", "bid_tm"]], keys,
    )
    pairs = pairs[~pairs.row_id.isin(safe_rows) & ~pairs.trackman_id.isin(safe_tm)].copy()
    known = pairs.bid_expected.notna()
    rate = (pairs.loc[known, "bid_expected"].astype(np.int64) == pairs.loc[known, "bid_tm"]).mean()
    print(f"relaxed candidate incremental={len(pairs):,}; batter-map proxy={known.sum():,} rows / {rate:.6f} exact")
    print("sequence", sequence_report(pairs))
    # Keep this artifact so a later decision can impose a proxy threshold without rerunning the join.
    pairs[["row_id", "trackman_id", "trackman_game_id", "pitch_no", "bid_expected", "bid_tm"]].to_csv(
        AUDIT / "relaxed_pitcher_batterhand_candidates.csv.gz", index=False, compression="gzip"
    )


if __name__ == "__main__":
    main()
