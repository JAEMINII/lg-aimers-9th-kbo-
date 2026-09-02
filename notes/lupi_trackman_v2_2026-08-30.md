# Audited Trackman LUPI v2

## Mapping

- Rebuilt the official-train ↔ Trackman mapping from shared game-state keys,
  without using the old player map as an input.
- Mutual-best, high-purity player links: 597 pitchers and 555 batters.
- Strict current-pitch matches: 1,006,723 regular-train rows (76.63%).
- Hand labels were audited as `Left=1`, `Right=2`; only 264 strict matches
  disagreed and were excluded.
- Mapping order test: 3,827 games with at least five matches, median pitch-order
  correlation 1.0 and no game below .99.
- Leave-one-season-out map stability was at least .9995 for pitchers and at
  least .9797 for batters (2019 is the weaker batter season).
- The relaxed state-only mapping was rejected: its proxy exact-batter accuracy
  was only 45/111 = .4054.

## LUPI experiments

The submitted model never receives Trackman data.  Trackman is used only to
construct an auxiliary label while fitting the direct control-success head.

| Auxiliary task | Result |
| --- | --- |
| 8 continuous current-pitch physics regressions | Rejected: VS=2024 direct score −19.50 mean, 0/3 seeds. |
| Current pitch type, CE weight .005 | Positive but smaller. |
| Current pitch type, CE weight .02 | Selected.  The direct head alone is exported. |

For the selected `.02` arm, the actual `submit_44` mixture comparison was
positive in every seed on both fully scored folds:

| Holdout | Delta vs matching no-aux TabM |
| --- | --- |
| 2022 | +8.986 mean, 3/3 |
| 2024 | +2.957 mean, 3/3 |
| 2023 direct TabM | +40.769 mean, 2/3 (one negative seed; no 44 components were cached for exact composition) |

## Deployable candidate

`submit_jaemin_49_lupi_type.zip` is a byte-compatible `submit_44` package
except for `model/f2b_{all,regular,futures}_s42.npz`.

- The three replacement archives have `d_out=5`: direct target at index 0 plus
  four training-only pitch-type logits.  `script.py` reads index 0 only.
- No Trackman CSV, player map, row map, auxiliary label, or inference matching
  logic appears in the package.
- Zip: 21 entries (same set as 44), CRC pass, 16.92 MB.
- Smoke test passed; input permutation max difference `0.0`; three-row subset
  max difference `3.1e-9` (float32 arithmetic only).
