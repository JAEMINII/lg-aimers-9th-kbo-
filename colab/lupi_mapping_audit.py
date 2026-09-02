# -*- coding: utf-8 -*-
"""Independent audit of the train <-> Trackman identity and pitch matching.

This intentionally does not consume the existing pitcher/batter maps when it
creates evidence.  It first uses only shared game-state fields, then keeps a
player-ID edge only when it is mutual-best and highly pure.  The output is an
audit report, not a submission feature.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AIMERS_DATA", ROOT / "open (1)" / "data"))
MAPDIR = ROOT / "trackman_map"
OUT = ROOT / "colab" / "_dl" / "mapping_audit"
OUT.mkdir(parents=True, exist_ok=True)

STATE = ["inning", "top_bottom", "balls_before", "strikes_before", "outs_before"]
BASE = ["season", "game_month", "game_dayofweek", "pt", "bt"]
KEY = BASE + STATE


def code_hand(x: pd.Series) -> np.ndarray:
    s = x.astype(str).str.strip().str.lower()
    # This dataset's anonymized train coding is Left=1, Right=2.  It is
    # inferred below from the independent match audit; do not assume the
    # conventional numeric ordering here.
    return np.where(s.str.startswith("l"), 1, np.where(s.str.startswith("r"), 2, -1))


def numeric_row_id(x: pd.Series) -> np.ndarray:
    return x.astype(str).str.extract(r"(\d+)$", expand=False).astype(np.int64).to_numpy()


def safe_merge_unique(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Match only keys occurring exactly once on *both* sides."""
    ln = left.groupby(keys, sort=False).size().rename("_ln").reset_index()
    rn = right.groupby(keys, sort=False).size().rename("_rn").reset_index()
    keep = ln.merge(rn, on=keys, how="inner")
    keep = keep[(keep._ln == 1) & (keep._rn == 1)][keys]
    left = left.merge(keep, on=keys, how="inner")
    right = right.merge(keep, on=keys, how="inner")
    return left.merge(right, on=keys, how="inner", suffixes=("_tr", "_tm"),
                      validate="one_to_one")


def mutual_map(evidence: pd.DataFrame, left: str, right: str, min_n: int = 3,
               purity: float = 0.985) -> pd.DataFrame:
    """Create a one-to-one map from independently unique game-state evidence."""
    c = evidence.groupby([left, right], sort=False).size().rename("n").reset_index()
    c["left_total"] = c.groupby(left, sort=False)["n"].transform("sum")
    c["right_total"] = c.groupby(right, sort=False)["n"].transform("sum")
    c["left_rank"] = c.groupby(left, sort=False)["n"].rank(method="first", ascending=False)
    c["right_rank"] = c.groupby(right, sort=False)["n"].rank(method="first", ascending=False)
    c["left_purity"] = c.n / c.left_total
    c["right_purity"] = c.n / c.right_total
    out = c[(c.left_rank == 1) & (c.right_rank == 1) & (c.n >= min_n)
            & (c.left_purity >= purity) & (c.right_purity >= purity)].copy()
    return out.sort_values(["n", left], ascending=[False, True]).reset_index(drop=True)


def report_map(name: str, inferred: pd.DataFrame, old: pd.DataFrame,
               left: str, right: str) -> dict[str, float]:
    old = old.rename(columns={old.columns[0]: left, old.columns[1]: right})
    z = inferred[[left, right, "n", "left_purity", "right_purity"]].merge(
        old[[left, right]], on=[left, right], how="outer", indicator=True
    )
    disagree = z[(z._merge == "both") == False]
    out = {
        "inferred": int(len(inferred)),
        "existing": int(len(old)),
        "same_edge": int((z._merge == "both").sum()),
        "one_side_only": int(len(disagree)),
        "minimum_evidence": int(inferred.n.min()) if len(inferred) else 0,
        "minimum_purity": float(min(inferred.left_purity.min(), inferred.right_purity.min()))
        if len(inferred) else 0.0,
    }
    z.sort_values(["_merge", left, right]).to_csv(OUT / f"{name}_comparison.csv", index=False)
    print(name, json.dumps(out, ensure_ascii=False))
    return out


def leave_one_season_out(evidence: pd.DataFrame, left: str, right: str) -> list[dict[str, float]]:
    """Check ID edges on a season not used to infer them.

    This is not a target validation: the held-out row's identity comes only
    from its public game-state alignment.  It tests whether an inferred
    anonymous-ID correspondence is stable across seasons.
    """
    rows = []
    for season in sorted(evidence.season.unique()):
        inferred = mutual_map(evidence[evidence.season != season], left, right)
        held = evidence[evidence.season == season][[left, right]].merge(
            inferred[[left, right]], on=left, how="inner", suffixes=("_obs", "_map")
        )
        n = len(held)
        good = (held[f"{right}_obs"] == held[f"{right}_map"]).sum()
        rows.append({"season": int(season), "covered_evidence_rows": int(n),
                     "same_edge_rate": float(good / n) if n else np.nan,
                     "inferred_id_count": int(len(inferred))})
    return rows


def main() -> None:
    teams = pd.read_csv(MAPDIR / "team_map.csv", encoding="utf-8-sig")
    tm_to_tr = dict(zip(teams.trackman_team, teams.train_team_id))

    tr_cols = ["row_id", "season", "game_month", "game_dayofweek", "game_type",
               "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
               "pitcher_team_id", "batter_team_id"] + STATE
    tr = pd.read_csv(DATA / "train.csv", usecols=tr_cols, encoding="utf-8-sig")
    tr = tr[tr.game_type.eq("R")].copy()
    tr["top_bottom"] = tr.top_bottom.astype(str).str[:1].str.upper()
    tr["pt"], tr["bt"] = tr.pitcher_team_id, tr.batter_team_id
    tr["row_num"] = numeric_row_id(tr.row_id)
    tr = tr.rename(columns={"pitcher_id": "pid_tr", "batter_id": "bid_tr",
                            "pitcher_hand": "ph_tr", "batter_hand": "bh_tr"})

    tm_cols = ["trackman_id", "season", "game_month", "game_dayofweek",
               "trackman_game_id", "pitch_no", "pitcher_trackman_id",
               "batter_trackman_id", "pitcher_hand", "batter_hand",
               "pitcher_team", "batter_team"] + STATE
    tm = pd.read_csv(DATA / "trackman_history.csv", usecols=tm_cols, encoding="utf-8-sig")
    tm["top_bottom"] = tm.top_bottom.astype(str).str[:1].str.upper()
    tm["pt"], tm["bt"] = tm.pitcher_team.map(tm_to_tr), tm.batter_team.map(tm_to_tr)
    tm = tm.dropna(subset=["pt", "bt"]).copy()
    tm[["pt", "bt"]] = tm[["pt", "bt"]].astype(np.int64)
    tm = tm.rename(columns={"pitcher_trackman_id": "pid_tm", "batter_trackman_id": "bid_tm",
                            "pitcher_hand": "ph_tm", "batter_hand": "bh_tm"})

    print(f"regular train={len(tr):,}, trackman aligned-team={len(tm):,}")
    # Evidence uses no player mapping at all.
    e = safe_merge_unique(
        tr[KEY + ["pid_tr", "bid_tr", "ph_tr", "bh_tr", "row_num"]],
        tm[KEY + ["pid_tm", "bid_tm", "ph_tm", "bh_tm", "trackman_game_id", "pitch_no"]], KEY,
    )
    print(f"independent unique-state evidence={len(e):,}")
    e["ph_ok"] = e.ph_tr.to_numpy() == code_hand(e.ph_tm)
    e["bh_ok"] = e.bh_tr.to_numpy() == code_hand(e.bh_tm)
    print(f"evidence hand agreement pitcher={e.ph_ok.mean():.6f} batter={e.bh_ok.mean():.6f}")

    pmap = mutual_map(e[e.ph_ok], "pid_tr", "pid_tm")
    bmap = mutual_map(e[e.bh_ok], "bid_tr", "bid_tm")
    pold = pd.read_csv(MAPDIR / "pitcher_map.csv", encoding="utf-8-sig")
    bold = pd.read_csv(MAPDIR / "batter_map.csv", encoding="utf-8-sig")
    stats = {
        "pitcher": report_map("pitcher", pmap, pold, "pid_tr", "pid_tm"),
        "batter": report_map("batter", bmap, bold, "bid_tr", "bid_tm"),
        "pitcher_leave_one_season_out": leave_one_season_out(e[e.ph_ok], "pid_tr", "pid_tm"),
        "batter_leave_one_season_out": leave_one_season_out(e[e.bh_ok], "bid_tr", "bid_tm"),
    }
    pmap[["pid_tr", "pid_tm"]].rename(columns={"pid_tr": "pitcher_id", "pid_tm": "trackman_id"}).to_csv(
        OUT / "pitcher_map_independent.csv", index=False
    )
    bmap[["bid_tr", "bid_tm"]].rename(columns={"bid_tr": "batter_id", "bid_tm": "trackman_id"}).to_csv(
        OUT / "batter_map_independent.csv", index=False
    )

    # Strict current-pitch match from the independently recovered maps.
    trj = tr.merge(pmap[["pid_tr", "pid_tm"]], on="pid_tr", how="inner").merge(
        bmap[["bid_tr", "bid_tm"]], on="bid_tr", how="inner"
    )
    join = KEY + ["pid_tm", "bid_tm"]
    matched = safe_merge_unique(
        trj[join + ["row_id", "row_num", "ph_tr", "bh_tr"]],
        tm[join + ["trackman_id", "trackman_game_id", "pitch_no", "ph_tm", "bh_tm"]], join,
    )
    matched["ph_ok"] = matched.ph_tr.to_numpy() == code_hand(matched.ph_tm)
    matched["bh_ok"] = matched.bh_tr.to_numpy() == code_hand(matched.bh_tm)
    both = matched.ph_ok & matched.bh_ok
    print(f"strict current-pitch match={len(matched):,} ({len(matched)/len(tr):.2%} of regular train)")
    print(f"hand agreement both={both.mean():.6f}; mismatches={int((~both).sum()):,}")

    # A true game match preserves pitch order even after we omit unmatched pitches.
    good = matched[both].copy()
    good["rank_row"] = good.groupby("trackman_game_id")["row_num"].rank(method="first")
    good["rank_pitch"] = good.groupby("trackman_game_id")["pitch_no"].rank(method="first")
    game_n = good.groupby("trackman_game_id").size()
    corr = pd.Series({gid: gg.rank_row.corr(gg.rank_pitch)
                      for gid, gg in good.groupby("trackman_game_id", sort=False)},
                     dtype=np.float64)
    seq = {
        "matched_hand_clean": int(len(good)),
        "games_with_5plus": int((game_n >= 5).sum()),
        "median_order_correlation_5plus": float(corr[game_n.reindex(corr.index) >= 5].median()),
        "games_order_corr_below_0_99": int((corr[game_n.reindex(corr.index) >= 5] < 0.99).sum()),
    }
    print("sequence", json.dumps(seq, ensure_ascii=False))
    good[["row_id", "trackman_id", "trackman_game_id", "pitch_no"]].to_csv(
        OUT / "strict_matches_independent.csv.gz", index=False, compression="gzip"
    )

    stats.update({"evidence_rows": int(len(e)), "strict_match_rows": int(len(matched)),
                  "strict_match_hand_clean": int(len(good)), "sequence": seq})
    (OUT / "report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
