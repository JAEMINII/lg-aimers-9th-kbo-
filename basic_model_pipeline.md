# `basic.ipynb` 모델 파이프라인 정리

## 1. 목적과 데이터 분할

- 입력 데이터: `./data/train.csv`
- 예측 대상: `control_success`
- 식별자: `row_id`, `batter_id`, `pitcher_id`
- 학습: `season < 2024`, 검증: `season == 2024`
- `prior_rate`는 학습 분할의 타깃 평균으로 계산하며, 최종 학습에서는 전체 데이터 평균을 사용한다.

## 2. 전처리 및 파생변수

`make_features`는 원본을 복사한 뒤 다음 변수를 추가한다.

### 경기·카운트

- `season_offset = season - 2019`
- `count_state`: `balls_before-strikes_before` 문자열
- `is_two_strikes`, `is_three_balls`, `is_full_count`
- `count_pressure`: 볼 수와 스트라이크 수의 합
- `inning_group`: `1-3`, `4-6`, `7-9`, `10+`

### 점수·주자

- `score_diff_abs`: 투수 팀 기준 점수 차이 절댓값
- `score_state_pitcher`: `trailing`, `leading`, `tie`
- `is_close_game`: 점수 차이 절댓값이 1 이하인지 여부
- `has_risp`: 2루 또는 3루 주자 존재 여부
- `li_log1p`: 0 이하를 자른 `li`에 `log1p` 적용

### 투수·타자·핸드

- `hand_matchup`: 투수·타자 핸드 결합 문자열
- `pitcher_cold_start`, `batter_cold_start`: 누적 표본이 없거나 성공률이 결측인지 여부
- `pitcher_success_smoothed`, `batter_success_smoothed`: 사전확률 기반 평활 성공률
- `pitcher_success_x_hand_1_1`, `1_2`, `2_1`, `2_2`: 투수 평활 성공률의 사전확률 대비 차이와 핸드 매치업의 상호작용

평활화 공식은 다음과 같다. `prior_strength`의 기본값은 50이다.

```text
smoothed_rate = (n * observed_rate + prior_strength * prior_rate)
                / (n + prior_strength)
```

관측 성공률이 결측이면 `prior_rate`로 대체한다.

## 3. 모델 입력에서 제외하는 컬럼

```text
season, run_top_before, run_bot_before, score_diff_home,
home_win_expectancy, away_win_expectancy, runner_on_1b,
runner_on_2b, runner_on_3b, num_runners_on,
asof_pitcher_pitchmix_n, top_bottom, game_month, game_dayofweek
```

최종 feature는 타깃·식별자·위 제외 컬럼을 제거한 나머지 컬럼으로 동적으로 생성한다. CatBoost에 전달하는 범주형 후보는 `top_bottom`, `game_type`, `game_dayofweek`, `game_month`, `pitcher_hand`, `batter_hand`, `pitcher_team_id`, `batter_team_id`, `base_state`, `count_state`, `inning_group`, `score_state_pitcher`, `hand_matchup`이다.

노트북의 결측 처리 코드는 변환 결과를 재할당하지 않는다. 실제로 문자열 변환과 결측 토큰 처리를 적용하려면 다음처럼 써야 한다.

```python
for col in active_cat_cols:
    frame[col] = frame[col].astype("string").fillna("__MISSING__").astype(str)
```

## 4. CatBoost 학습

사용 모델은 `CatBoostClassifier`이며, one-hot 인코딩 대신 `cat_features=active_cat_cols`로 범주형 컬럼을 직접 지정한다.

기본 검증 설정:

```text
loss_function=Logloss, eval_metric=BrierScore
iterations=500, learning_rate=0.05
l2_leaf_reg=3, random_strength=1, random_seed=42
use_best_model=True, od_type=Iter, od_wait=100
thread_count=-1, allow_writing_files=False
```

튜닝 시에는 `iterations=1000`, `od_wait=150`, `grow_policy=SymmetricTree`, `depth=7`, `l2_leaf_reg=1`, `bootstrap_type=MVS`, `subsample=0.8`을 사용한다.

검증 세트를 `eval_set`으로 넣고 `predict_proba(X_val)[:, 1]`을 확률 예측으로 사용한다.

## 5. 평가 지표

```text
brier = mean((prediction - target)^2)
baseline_brier = prevalence * (1 - prevalence)
validation_score = max(0, 100000 * (1 - brier / baseline_brier))
```

확률 예측의 품질을 보기 위해 Brier score를 주 지표로 사용하고, 타깃 평균만 예측하는 기준선 대비 개선 점수를 함께 출력한다.

## 6. 전체 재학습 및 저장

최종 모델은 전체 `train.csv`로 재학습한다. 최종 파라미터의 핵심 값은 `iterations=235`, `learning_rate=0.05`, `depth=7`, `l2_leaf_reg=1`, `grow_policy=SymmetricTree`, `bootstrap_type=MVS`, `subsample=0.8`, `random_seed=42`이다.

모델과 추론 메타데이터를 `./model/catboost_bundle.pkl`에 저장한다.

```python
catboost_bundle = {
    "model": model,
    "drop_cols": drop_cols,
    "features": features,
    "prior_rate": prior_rate,
}
joblib.dump(catboost_bundle, "./model/catboost_bundle.pkl", compress=3)
```

추론 시에도 저장된 `features`, `drop_cols`, `prior_rate`를 사용해 학습과 동일한 feature schema를 재현해야 한다.

## 7. 확률 보정과 오류 분석

2024 검증 데이터를 앞 70% calibration, 뒤 30% evaluation으로 나눠 Platt scaling을 적용한다. 원시 확률을 logit으로 변환한 뒤 `LogisticRegression(C=1e6)`을 학습하고, 보정 전·후 Brier score를 비교한다.

추가 분석 항목:

- Brier contribution이 큰 샘플
- 실제 0인데 확률 0.8 이상인 과신 실패
- 실제 1인데 확률 0.2 이하인 과소평가
- 확률 10구간별 예측 평균, 실제 성공률, calibration gap
- `count_state`, `base_state`, `inning_group`, `hand_matchup`, 점수 상태, 투·타자 핸드별 그룹 성능
- 표본 100개 이상 투수별 Brier score와 MAE

## 8. 전체 흐름

```text
train.csv 로드
→ 2019–2023 / 2024 시계열 분할
→ 상황·점수·핸드·누적 성공률 파생변수 생성
→ 식별자 및 제외 컬럼 제거
→ CatBoost 범주형 컬럼 지정 후 학습
→ 2024 Brier 평가 및 하이퍼파라미터 튜닝
→ 전체 데이터로 최종 재학습
→ 모델·feature 목록·제외 목록·prior_rate 저장
→ 필요 시 Platt calibration 및 그룹별 오류 분석
```
