# -*- coding: utf-8 -*-
"""Materialize the hand- and sequence-audited current-pitch Trackman table.

Only train row_ids appear in the result.  The result is a training artifact;
it must never be shipped in a code-submission package.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AIMERS_DATA", ROOT / "open (1)" / "data"))
DL = ROOT / "colab" / "_dl"
AUDIT = DL / "mapping_audit"
OUT = DL / "lupi_match_v2.csv.gz"
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]


def main() -> None:
    pairs = pd.read_csv(AUDIT / "strict_matches_independent.csv.gz", encoding="utf-8-sig")
    if pairs.row_id.duplicated().any() or pairs.trackman_id.duplicated().any():
        raise ValueError("audit output must be one-to-one before it can become a target table")
    tm = pd.read_csv(DATA / "trackman_history.csv", encoding="utf-8-sig",
                     usecols=["trackman_id", "pitch_type_group"] + PHYS)
    out = pairs[["row_id", "trackman_id"]].merge(tm, on="trackman_id", how="inner",
                                                    validate="one_to_one")
    if len(out) != len(pairs):
        raise ValueError("a verified Trackman row disappeared while materializing v2")
    out = out.drop(columns="trackman_id").sort_values("row_id").reset_index(drop=True)
    if out.row_id.duplicated().any():
        raise ValueError("row_id must remain unique")

    # The old file is a strict subset produced by the previous matcher.  Exact
    # agreement on their overlap proves that v2 changes coverage, not values.
    old_path = DL / "lupi_match.csv.gz"
    if old_path.exists():
        old = pd.read_csv(old_path, encoding="utf-8-sig")
        common = out.merge(old, on="row_id", suffixes=("_v2", "_old"), validate="one_to_one")
        bad = 0
        for c in ["pitch_type_group"] + PHYS:
            a, b = common[c + "_v2"], common[c + "_old"]
            if c == "pitch_type_group":
                bad += int((a.astype(str) != b.astype(str)).sum())
            else:
                bad += int((~np.isclose(a, b, equal_nan=True)).sum())
        print(f"old overlap={len(common):,}/{len(old):,}; differing fields={bad}")
        if bad:
            raise ValueError("v2 must be identical to the existing strict matches")

    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig", usecols=["row_id", "season"])
    season = out.merge(tr, on="row_id", how="left", validate="one_to_one").season
    print(f"v2 rows={len(out):,}; by-season={season.value_counts().sort_index().to_dict()}")
    print("missing", {c: float(out[c].isna().mean()) for c in PHYS})
    out.to_csv(OUT, index=False, encoding="utf-8-sig", compression="gzip")
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
