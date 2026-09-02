# Overnight validation — 2026-08-29

제출 없이 새 서버 RTX 3060에서 walk-forward 검증만 수행했다. 현재 제출 41의 기준은 TabM 70% + CatBoost 30%, 고정 logit shift -0.007이다.

## 핵심 결과

| 후보 | VS=2022 CatBoost fixed | VS=2024 CatBoost fixed | VS=2023 |
|---|---:|---:|---:|
| c12_cmh 기준 | 739.7 | 약 883.8 (10 seed) | 805.0 |
| TrackMan relh/rels 2열 | 743.1 | 880.5 (3 seed) | — |
| TrackMan 2-seam 8열 + c12_cmh | 743.1 | 885.7 (10 seed) | 804.3 |
| 전체 TrackMan + c12_cmh | 741.8 | 875.0 | — |
| strict label-family 묶음 | 728.9 | 882.4 | — |

TrackMan 2-seam 후보는 2022/2024 CatBoost에서 모두 상승했고 2023에서는 거의 동률이다. 그러나 현재 최종 blend에 적용하면 2024 10-seed 기준:

- 기존 41 재현: 전체 869.333, regular 869.396, futures 543.072
- 후보를 모든 경기 유형에 적용: 전체 870.134, regular 869.209, futures 551.275
- 후보를 futures에만 적용하고 regular는 기존 유지: 전체 870.299, regular 869.396, futures 551.275

따라서 제출 후보로 남길 수 있는 최대 근거는 futures CatBoost 경로에만 TrackMan 2-seam 누적 8열을 추가하는 것이다. 예상 LB 상승은 내부 점수 전이 오차가 커서 보수적으로 0~3점이며, 1150점까지의 +67을 설명하지 못한다.

## 기각/보류

- strict 누수 없는 ball/reverse/middle/strike label-family: 2022에서 대부분 하락, 전체 묶음 -10점.
- TrackMan 전체 13열: 최종 blend에서 기존 c12_cmh보다 약간 열세.
- 공식 TabM k=64, Stage1 4epoch + Stage2: 최종 관문 928.2로 기존 929점대보다 낮음.
- TrackMan 2-seam을 TabM 입력에 직접 추가: 2epoch direct 837.5, 4epoch direct 826.1로 기존 direct 837.5보다 악화.
- CatBoost depth 5: depth 4보다 fixed 점수 하락.
- 고정 shift를 바꿔 얻는 2024 단일 폴드 상승은 전이 근거로 사용하지 않음.

## 구현 메모

2-seam 후보의 합법적인 입력은 tm2s_n, tm2s_fastball_rate, tm2s_breaking_rate, tm2s_offspeed_rate, tm2s_fastball_shift, tm2s_breaking_shift, tm2s_speed_delta, tm2s_relh_sd이며, 각 행의 시즌보다 이전 시즌 TrackMan만 pitch-count 가중 누적한다. 아직 submit_41 파일에는 반영하지 않았다.
