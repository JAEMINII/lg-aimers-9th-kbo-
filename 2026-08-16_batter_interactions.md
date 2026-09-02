# Batter interaction analysis

Validation uses 2019--2023 plus the first 80% of 2024 for training and the
last 20% of 2024 for validation. All interaction statistics use only prior
training rows for training features and only the training block for validation
features.

## Coverage screening

| interaction key | validation coverage |
|---|---:|
| batter x pitcher_hand | 99.3% |
| batter x pitcher_team | 96.6% |
| batter x pitcher | 71.3% |
| batter x pitcher_hand x game_type | 98.7% |
| batter x balls/strikes | 99.3% |

## 3-seed validation

The existing 44-feature full model scored 520.95. Candidate features are
smoothed historical success rate plus log prior sample count.

| candidate | features | score, shift 0 |
|---|---:|---:|
| batter x pitcher_hand | 46 | **546.03** |
| batter x balls/strikes | 46 | 540.29 |
| both together | 48 | 533.07 |

The direct batter x pitcher feature scored 496.96 in the first single-seed
screen and is too sparse for the current model.

## Decision

`batter x pitcher_hand` is the first batter-interaction candidate to test in a
new submission. It is not yet added to submit_6. The feature is computed as a
prior-only smoothed rate and log count; no test labels or test-side aggregates
are used.

As a first check with the current full/R/F architecture, the same feature gave
one-seed scores of 544.86 for full, 540.36 for R, and 218.50 for F; the 60:40
full-to-routed blend scored 555.30. This is promising but still requires the
same multi-seed confirmation before changing the submission package.
