# Baseline 실험 정리

## 1. 목적과 검증 방식

`baseline.ipynb`는 야구 경기 상황에서 `control_success`를 예측하는 이진분류 실험 노트북이다.

- 전체 train 데이터: 1,475,092건, 49개 컬럼
- 학습 데이터: `season < 2024` (1,221,585건)
- 검증 데이터: `season == 2024` (253,507건)
- 타깃: `control_success`
- 평가지표: Brier score 기반 skill score

점수는 아래 방식으로 계산하며, **높을수록 예측확률 품질이 좋다.**

```python
brier = ((prediction - target) ** 2).mean()
baseline_brier = target.mean() * (1 - target.mean())
score = 100000 * (1 - brier / baseline_brier)
```

## 2. 전처리

### 기본 입력 처리

- `row_id`는 항상 입력 피처에서 제외한다.
- `pitcher_id`, `batter_id`도 제외하는 실험을 진행했다. 두 ID를 제거했을 때 성능이 소폭 상승했으며, 선수 ID를 단순 암기하는 과적합 가능성을 줄이는 방향으로 해석할 수 있다.
- 일반 모델 입력에서는 범주형 변수 `top_bottom`, `game_type`, `base_state`를 `OrdinalEncoder`로 변환한다.
  - 학습에서 보지 못한 범주는 `-1`로 처리한다.
  - 수치형 결측은 중앙값으로 대치한다.

### `preprocess_data()`

LightGBM 및 Ordinal Encoding 기반 실험에 쓰는 전처리 함수다. 학습 데이터에서 범주 목록과 제거 컬럼 정보를 `state`에 저장하고, 검증·테스트 데이터에 같은 `state`를 적용해 일관된 변환을 보장한다.

#### 생성 피처

| 구분 | 피처 | 설명 |
|---|---|---|
| 투수팀 관점 | `pitcher_is_home` | 투수가 홈팀인지 여부 |
| 투수팀 관점 | `pitcher_team_runs` | 투수팀의 현재 득점 |
| 투수팀 관점 | `opponent_team_runs` | 상대팀의 현재 득점 |
| 투수팀 관점 | `pitcher_team_win_expectancy` | 투수팀 기준 승리확률 |
| 선수 이력 | `pitcher_cold_start` | 투수 누적 기록이 없거나 성공률이 결측인지 |
| 선수 이력 | `batter_cold_start` | 타자 누적 기록이 없거나 성공률이 결측인지 |
| 선수 이력 | `pitcher_recent_missing` | 투수 최근 1·3·5경기 기록 중 결측이 있는지 |
| 경기 상황 | `is_late_inning` | 7회 이후인지 |
| 경기 상황 | `is_close_game` | 투수팀 기준 점수 차가 1점 이하인지 |
| 경기 상황 | `has_risp` | 2루 또는 3루 주자가 있는지 |
| 경기 상황 | `count_pressure` | 볼과 스트라이크 수의 합 |
| 매치업 | `same_hand_matchup` | 투수와 타자의 좌·우타가 같은지 |

#### 제거 피처

아래 8개 컬럼을 제거한다.

- `run_top_before`, `run_bot_before`, `run_total_before`
- `score_diff_home`
- `asof_pitcher_pitchmix_n`
- `base_state`
- `home_win_expectancy`, `away_win_expectancy`

제거한 팀 득점 및 승리확률 관련 원본 변수 중 일부는, 투수팀 관점으로 재구성한 파생변수로 대체된다.

### `preprocess_data_cat()`

CatBoost의 native categorical 처리용 함수다.

- 파생변수 생성 및 제거 피처는 `preprocess_data()`와 같다.
- 범주형 값은 Ordinal Encoding하지 않고 문자열로 유지한다.
- 결측 범주형 값은 `__MISSING__`으로 채운 뒤 CatBoost의 `cat_features`로 직접 전달한다.

즉, 두 함수의 차이는 주로 범주형 처리 방식이다.

| 함수 | 범주형 처리 | 사용 모델 |
|---|---|---|
| `preprocess_data()` | category 지정 후 Ordinal Encoding | LightGBM, CatBoost 비교 실험 |
| `preprocess_data_cat()` | 문자열 범주를 모델에 직접 전달 | CatBoost native categorical |

## 3. 진행한 실험

1. **기본 모델 비교**: RandomForest, LightGBM, CatBoost
2. **ID 제거**: `pitcher_id`, `batter_id`를 제외한 성능 확인
3. **피처 엔지니어링**: 위 전처리 함수의 파생변수와 원본 변수 정리 적용
4. **범주형 처리 비교**: Ordinal Encoding 방식과 native categorical 방식 비교
5. **앙상블**: LightGBM과 CatBoost 예측 확률을 0.5:0.5로 평균

## 4. 마지막 barplot 결과

마지막 barplot은 모든 실험의 2024년 검증 점수를 비교한 그래프다.

- x축: validation score
- y축: 실험명
- 색상: 모델 종류 (`randomforest`, `lightgbm`, `catboost`, `ensemble`)
- 정렬: validation score 내림차순

| 순위 | 실험 | Validation score |
|---:|---|---:|
| 1 | `best_ensemble` | **731.45** |
| 2 | `feature_engineering_ensemble` | 719.14 |
| 3 | `native_categorical_catboost` | 713.30 |
| 4 | `feature_engineering_catboost` | 694.27 |
| 5 | `feature_engineering_lightgbm` | 692.96 |
| 6 | `native_categorical_lightgbm` | 690.78 |
| 7 | `baseline_lightgbm` | 608.94 |
| 8 | `delete_ids_lightgbm` | 606.21 |
| 9 | `delete_ids_catboost` | 572.76 |
| 10 | `baseline_catboost` | 565.64 |
| 11 | `feature_engineering` (RandomForest) | 470.32 |
| 12 | `delete_ids` (RandomForest) | 428.36 |
| 13 | `baseline` (RandomForest) | 415.57 |

## 5. 결론

- RandomForest보다 LightGBM과 CatBoost가 크게 높은 성능을 보였다.
- `pitcher_id`, `batter_id` 제거는 소폭의 성능 개선을 보였다.
- 피처 엔지니어링이 성능 향상의 핵심 요인이었다.
- 가장 높은 결과는 native categorical 방식의 LightGBM과 CatBoost를 50:50으로 평균한 `best_ensemble`이다.
  - Brier score: `0.247980`
  - Validation score: **731.45**
- 최종 제출 단계에서는 전체 train 데이터로 두 모델을 다시 학습하고, 테스트 예측을 50:50 평균해 `output/submission.csv`를 생성한다.

## 6. 해석 시 유의점

모든 비교는 2024년 단일 홀드아웃 검증을 기준으로 한다. 최종 모델을 확정하기 전에는 시즌 기반 추가 검증 또는 교차검증으로 결과의 재현성을 확인하는 것이 좋다.
