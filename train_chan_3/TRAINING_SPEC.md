# train_chan_3 학습 설정 설명서

이 문서는 현재 `train_chan_3`에서 실제로 실행되는 TabM 학습 방식과 입력 전처리를 설명한다. 목표는 `submit_jaemin_5`의 피처 전처리를 유지하면서, 전체 경기 모델과 경기 유형별 모델을 함께 사용해 `control_success = 1` 확률을 예측하는 것이다.

## 1. 전체 흐름

```text
train.csv
  ↓ row_id 정렬
submit_jaemin_5 호환 전처리
  ↓ 44개 피처
공통 TabM 입력 전처리
  ├─ 전체 경기 모델(all)
  ├─ Futures 모델(game_type=F)
  └─ Regular 모델(game_type=R)
        ↓ 각 모델 Stage 1 학습
2024년 해당 데이터로 Stage 2 fine-tuning
        ↓
추론: 전체 모델 60% + 해당 경기 유형 모델 40%
        ↓
control_success 확률
```

최종 예측은 다음과 같다.

```text
Futures 행 = 0.6 × all 예측 + 0.4 × futures 예측
Regular 행 = 0.6 × all 예측 + 0.4 × regular 예측
```

## 2. 데이터와 타깃

- 학습 파일: `open/data/train.csv`
- 타깃: `control_success`
- 예측값: 0~1 사이의 확률
- 평가 관점: Brier Score가 낮을수록 좋음
- 학습 시 랜덤 시드: `42`
- `row_id`의 숫자 부분을 기준으로 정렬한 뒤 피처를 생성한다.

## 3. 입력 피처 처리

### 3.1 최종 입력 피처 44개

전처리 결과는 아래 순서의 44개 피처다.

```text
season, game_month, game_dayofweek, inning, top_bottom,
game_type, balls_before, strikes_before, outs_before,
run_top_before, run_bot_before, score_diff_pitcher_team,
base_state, home_win_expectancy, li, pitcher_id, batter_id,
pitcher_hand, batter_hand, pitcher_team_id, batter_team_id,
asof_pitcher_success_rate, asof_pitcher_reverse_rate,
asof_pitcher_middle_rate, asof_pitcher_ball_rate,
asof_pitcher_strike_rate, asof_pitcher_prev1_game_success_rate,
asof_pitcher_prev3_game_success_rate, asof_pitcher_prev5_game_success_rate,
asof_pitcher_prev1_game_middle_rate, asof_pitcher_prev3_game_middle_rate,
asof_pitcher_prev5_game_middle_rate, asof_batter_success_rate,
asof_batter_middle_rate, asof_pitcher_fastball_rate,
asof_pitcher_breaking_rate, asof_pitcher_offspeed_rate, p_is_succ,
pn_cur, b_is_succ, lg_cm_eff, cm_rel, p_adj_cm, plat_dev
```

### 3.2 피처 그룹

| 그룹 | 주요 내용 |
|---|---|
| 경기 상황 | 시즌, 월, 요일, 이닝, 초/말, 볼·스트라이크·아웃, 주자 상태 |
| 점수·기대값 | 양 팀 득점, 투수팀 기준 점수 차, 승리 기대값, `li` |
| 범주형 식별자 | 투수·타자·팀 ID, 투수·타자 좌우 타입, 경기 유형 |
| 투수 과거 통계 | 전체 성공률, 역방향·중간·볼·스트라이크 비율, 최근 1/3/5경기 통계 |
| 타자 과거 통계 | 타자 성공률, 중간 비율 |
| 구종 통계 | 패스트볼·브레이킹·오프스피드 비율 |
| 현재 시점 파생값 | 시즌 내 현재 누적 횟수, 스무딩 성공률, 카운트별 상대 효과, 플래툰 편차 |

### 3.3 타깃 기반 통계

`fit_history_tables()`가 학습 데이터에서 다음 lookup table을 만든다.

- 투수·타자 누적 시도 수와 성공 수
- 카운트/투수 손/타자 손 조합별 평균 및 상대 효과
- 투수의 타자 손별 플래툰 통계
- 전체 성공률

스무딩 설정은 다음과 같다.

```text
일반 성공률 스무딩: ALPHA = 50
일반 사전확률: PRIOR_SUCCESS = 0.50
플래툰 통계 스무딩: PLATOON_ALPHA = 300
```

### 3.4 학습 행과 테스트 행의 차이

학습 시에는 각 행의 `asof_*` 누적 통계에서 시즌 시작 시점의 값을 빼서 시즌 내 누적값을 계산한다.

추론 시에는 테스트 행끼리 통계를 새로 합치지 않는다. 학습 데이터로 만든 history table을 기준으로 각 테스트 행을 독립적으로 변환한다. 따라서 제출 추론에서 테스트 데이터의 미래 행이나 다른 테스트 행의 정답을 이용하지 않는다.

### 3.5 범주형·수치형 변환

고정 매핑되는 값은 다음과 같다.

```text
top_bottom: T=0, B=1
game_type: R=0, F=1
base_state: ___, 1__, _2_, __3, 12_, 1_3, _23, 123
```

TabM 내부 전처리에서 범주형으로 처리하는 9개 컬럼은 다음과 같다.

```text
top_bottom, game_type, base_state,
pitcher_id, batter_id,
pitcher_hand, batter_hand,
pitcher_team_id, batter_team_id
```

- 범주형 값은 학습 데이터의 값 목록으로 인코딩한다.
- 학습에서 보지 못한 범주형 값은 unknown 코드 `0`으로 처리한다.
- 나머지는 수치형으로 처리한다.
- 수치형 결측치는 학습 중앙값으로 대체한다.
- 수치형은 학습 평균과 표준편차로 표준화한다.
- `add_missing_indicators=True`이므로 결측이 있는 수치형 컬럼에는 결측 여부 피처도 추가한다.
- 범주형 목록과 수치형 통계는 전체 학습 데이터에서 한 번만 fit하여 세 branch가 공유한다.

## 4. TabM 모델 설정

현재 `selected_config.json`의 설정은 다음과 같다.

| 항목 | 값 | 의미 |
|---|---:|---|
| 모델 | official TabM | 공식 `tabm` 패키지 기반 구현 |
| `arch_type` | `tabm` | 기본 TabM 구조 |
| `k` | `32` | 한 행에 대해 학습하는 앙상블 멤버 수 |
| `n_blocks` | `3` | MLP backbone block 수 |
| `d_block` | `256` | 각 block의 hidden 차원 |
| `dropout` | `0.1` | backbone dropout |
| `num_embeddings` | `linear_relu` | 수치형 입력을 Linear-ReLU embedding으로 변환 |
| `d_embedding` | `16` | 수치형 변수별 embedding 차원 |
| batch size | `2048` | 학습 batch 크기 |
| 평가 batch size | `8192` | 추론·평가 batch 크기 |
| learning rate | `0.002` | Stage 1 AdamW 학습률 |
| weight decay | `0.0003` | AdamW 정규화 강도 |
| scheduler | `cosine` | cosine annealing learning-rate scheduler |
| gradient clip | `5.0` | gradient 최대 norm |
| num workers | `0` | DataLoader worker 수 |
| device | `cuda` | Colab 실행 노트북에서 지정 |

`k=32`는 서로 다른 32개의 모델 파일을 의미하지 않는다. 하나의 TabM 내부에서 32개의 ensemble member 출력을 만들고, 학습 때 member 평균 loss를 사용하며, 추론 때 32개 확률의 평균을 사용한다.

## 5. Loss 설정

현재 설정 이름은 `bce_brier`다.

```text
Brier = mean((sigmoid(logit) - y)^2)
BCE   = binary cross entropy with logits
Loss  = 0.5 × BCE + 0.5 × Brier
```

최종 대회 평가는 Brier 기반이므로 학습 로그에는 다음 두 값을 모두 기록한다.

- `objective`: 실제 최적화에 사용한 BCE+Brier 혼합 손실
- `train_brier`: 32개 member의 평균 확률을 기준으로 계산한 Brier Score

## 6. 2단계 학습 방식

### Stage 1

세 가지 모델을 각각 학습한다.

| Branch | 사용하는 행 |
|---|---|
| `all` | 전체 학습 데이터 |
| `futures` | `game_type == F` |
| `regular` | `game_type == R` |

현재 Stage 1 epoch은 `selected_config.json`의 `full_train_epochs=2`를 사용한다.

### Stage 2

Stage 1 checkpoint에서 시작하여 최신 시즌(현재 데이터에서는 2024년)의 해당 데이터만 사용한다.

| Branch | Stage 2 데이터 |
|---|---|
| `all` | 2024년 전체 경기 |
| `futures` | 2024년 Futures 경기 |
| `regular` | 2024년 Regular 경기 |

현재 fine-tuning 설정은 다음과 같다.

```text
fine-tune scope: last_block
학습 대상: 마지막 MLP block + output layer
고정 대상: embedding 및 앞쪽 MLP block
Stage 2 epochs: 1
Stage 2 learning rate: 0.0002
optimizer: AdamW
weight decay: 0.0003
```

Stage 2에서 전체 모델을 처음부터 다시 학습하는 것이 아니라, Stage 1에서 배운 표현을 유지하면서 마지막 부분만 2024년 분포에 맞춰 조정한다.

참고로 `train_conditional.py`를 옵션 없이 직접 실행하면 기본값은 `head`다. 현재 의도한 설정인 `last_block`을 사용하려면 Colab 노트북처럼 반드시 `--fine-tune-scope last_block`을 지정해야 한다.

## 7. 제출 시 추론

제출 폴더에는 다음 세 가지 NumPy 모델 파일이 들어간다.

```text
model/all_tabm_seed_42.npz
model/futures_tabm_seed_42.npz
model/regular_tabm_seed_42.npz
```

추론은 다음 순서로 수행한다.

1. `test.csv`를 읽고 `row_id` 순서로 정렬한다.
2. 학습 때 저장한 history table로 44개 피처를 만든다.
3. 모든 행을 `all` 모델에 넣는다.
4. Futures 행만 `futures` 모델에 넣고, Regular 행만 `regular` 모델에 넣는다.
5. 각 행에 대해 60:40 확률 blending을 한다.
6. 원래 `sample_submission.csv`의 row_id 순서에 맞춰 `submission.csv`를 만든다.

제출 추론에서는 NumPy 기반 실행을 사용하며, `chunk_size=1024`로 메모리를 나누어 계산한다.

## 8. Colab 실행 방법

실행 파일은 다음 노트북이다.

```text
train_chan_3/train_conditional_colab.ipynb
```

노트북의 학습 옵션은 다음과 같다.

```bash
python -u train_conditional.py \
  --data-dir /content/drive/MyDrive/aimers/open/data \
  --config /content/drive/MyDrive/aimers/train_chan_3/selected_config.json \
  --output-dir /content/drive/MyDrive/aimers/train_chan_3/artifacts_conditional \
  --device cuda \
  --stage2-epochs 1 \
  --fine-tune-scope last_block
```

학습이 끝나면 builder가 세 checkpoint를 `.npz`로 변환하고, `script.py`, `preprocess.py`, `requirements.txt`, 모델 파일을 포함한 제출 ZIP을 만든다.

## 9. 주요 출력 파일

```text
artifacts_conditional/
├── all_stage1/tabm_seed_42.pt
├── all_stage2/tabm_seed_42.pt
├── futures_stage1/tabm_seed_42.pt
├── futures_stage2/tabm_seed_42.pt
├── regular_stage1/tabm_seed_42.pt
├── regular_stage2/tabm_seed_42.pt
├── history.json
├── branch_counts.json
└── training_report.json
```

각 Stage 2 폴더의 `training_history.csv`에서 epoch별 `objective`, `train_brier`, 학습 시간을 확인할 수 있다.

## 10. 한 문장 요약

`submit_jaemin_5`의 44개 피처를 동일한 방식으로 생성하고, 이를 공식 TabM(K=32, 3 blocks, hidden 256)에 BCE와 Brier를 50:50으로 섞은 loss로 2 epoch 학습한 뒤, 전체·Futures·Regular 모델의 마지막 block과 output layer를 2024년 데이터로 1 epoch 보정하고, 추론 시 전체 모델과 경기 유형 모델을 60:40으로 결합하는 방식이다.
