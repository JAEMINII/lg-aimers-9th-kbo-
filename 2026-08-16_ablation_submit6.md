# submit_6 ablation

## Validation setup

- Train: 2019--2023 plus the first 80% of 2024 in row order.
- Validation: the last 20% of 2024, kept completely out of training.
- Feature set: the existing 44-feature set.
- Models: HistGradientBoosting, three seeds (`1, 42, 777`) for the controlled ablation.

## Results

Scores are local BSS-style scores; they are not leaderboard scores.

| configuration | shift 0 | shift -0.05 |
|---|---:|---:|
| full model | 520.95 | 464.82 |
| R/F separate only | 495.82 | 452.04 |
| full 40% + R/F 60% | 530.06 | 481.13 |
| full 50% + R/F 50% | 533.58 | 483.41 |
| full 60% + R/F 40% | **535.09** | **483.69** |
| full 70% + R/F 30% | 534.58 | 481.97 |
| full 80% + R/F 20% | 532.05 | 478.25 |

## Decision

Keep the `full 60% + routed R/F 40%` blend and calibration shift `0.0`.
R/F-only prediction is not used.

The current 44-feature set has no explicit pitcher-id x batter-id matchup
history feature. It does include handedness as separate fields, count x
pitcher/batter-hand matchup features (`lg_cm_eff`, `cm_rel`, `p_adj_cm`), and
the pitcher platoon deviation feature (`plat_dev`).
