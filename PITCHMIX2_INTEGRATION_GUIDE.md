# Pitchmix 2열을 최종 CatBoost에 통합하는 가장 깔끔한 방법

## 결론

`current_catboost_features.make_features()`를 변경하지 않는다. 현재 채택된 모델은 current CatBoost가 아니라 다음 경로의 candidate CatBoost이기 때문이다.

```text
features44 44열
+ c12/cmh 6열
+ candidate 8열
+ pitchmix shape 2열
= 최종 60열
```

가장 깔끔한 구조는 pitchmix 2열을 만드는 작은 공용 함수를 하나 만들고, candidate 학습과 제출 추론 양쪽에서 같은 함수를 호출하는 것이다.

```text
공용 add_pitchmix_shape_features()
        ├─ 전체 학습 코드
        └─ 제출 추론 코드
```

이렇게 하면 학습·추론 간 계산식 차이와 feature 순서 불일치를 예방할 수 있다.

## 왜 current CatBoost의 make_features를 변경하지 않는가

`current_catboost_features.py`의 `make_features()`는 기존 current CatBoost 전용이다.

```text
current_catboost_features.make_features()
→ current CatBoost bundle
```

OOF 검증 결과 current CatBoost 혼합은 최종 구성에서 제외됐다. 채택 모델의 실제 학습 경로는 다음이다.

```text
features44.build()
→ add_c12()
→ add_cmh()
→ add_candidate_features()
→ CatBoost
```

따라서 current CatBoost의 `make_features()`를 수정해도 채택 모델에는 적용되지 않는다.

## 최종 feature 구성

| 블록 | 열 수 |
|---|---:|
| features44 | 44 |
| c12/cmh | 6 |
| 기존 candidate 파생변수 | 8 |
| pitchmix entropy/max | 2 |
| 합계 | **60** |

추가할 두 열:

```text
asof_pitchmix_entropy
asof_pitchmix_max_rate
```

## 1. 공용 함수 작성

새 파일을 다음과 같이 두는 것을 권장한다.

```text
1129/submit_jaemin_54_cbhybrid_full/pitchmix_features.py
```

내용:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


PITCHMIX_SHAPE_FEATURES = [
    "asof_pitchmix_entropy",
    "asof_pitchmix_max_rate",
]


def add_pitchmix_shape_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    fast = (
        out["asof_pitcher_fastball_rate"]
        .fillna(0)
        .clip(0, 1)
        .to_numpy(np.float64)
    )
    breaking = (
        out["asof_pitcher_breaking_rate"]
        .fillna(0)
        .clip(0, 1)
        .to_numpy(np.float64)
    )
    offspeed = (
        out["asof_pitcher_offspeed_rate"]
        .fillna(0)
        .clip(0, 1)
        .to_numpy(np.float64)
    )

    # 세 비율에 포함되지 않은 잔여 구종 비율
    other = np.clip(
        1.0 - fast - breaking - offspeed,
        0.0,
        1.0,
    )

    rates = np.stack(
        [fast, breaking, offspeed, other],
        axis=1,
    )

    # 반올림 오차 또는 clip으로 합이 1이 아닐 수 있으므로 재정규화한다.
    total = rates.sum(axis=1, keepdims=True)
    rates = np.divide(
        rates,
        total,
        out=np.zeros_like(rates),
        where=total > 0,
    )

    out["asof_pitchmix_entropy"] = (
        -(rates * np.log(np.clip(rates, 1e-8, 1.0))).sum(axis=1)
    ).astype(np.float32)

    out["asof_pitchmix_max_rate"] = (
        rates.max(axis=1)
    ).astype(np.float32)

    return out
```

## 2. candidate 학습 코드에 연결

수정 대상:

```text
1129/submit_jaemin_54_cbhybrid_full/train_full_2024.py
```

상단 import에 추가한다.

```python
from pitchmix_features import (
    PITCHMIX_SHAPE_FEATURES,
    add_pitchmix_shape_features,
)
```

현재 `train_candidate()`에는 다음 코드가 있다.

```python
frame, added = add_candidate_features(frame, base)
features = base + added
x = frame[features].to_numpy(np.float32)
```

이를 다음과 같이 변경한다.

```python
frame, added = add_candidate_features(frame, base)
frame = add_pitchmix_shape_features(frame)

features = base + added + PITCHMIX_SHAPE_FEATURES
x = frame[features].to_numpy(np.float32)
```

이 방식에서는 기존 `add_candidate_features()`의 의미가 유지된다.

- 기존 함수: 검증된 candidate 8열 생성
- 신규 함수: pitchmix shape 2열 생성

서로 다른 실험 단계를 한 함수에 섞지 않으므로 재사용과 ablation이 쉽다.

## 3. 제출 추론 코드에 연결

수정 대상:

```text
1129/submit_jaemin_54_cbhybrid_full/script.py
```

제출 환경에서 별도 모듈 import가 허용된다면 다음을 import한다.

```python
from pitchmix_features import add_pitchmix_shape_features
```

`build_features()`에서 기존 candidate 8열을 모두 만든 직후 호출한다.

```python
# 기존 candidate 8열
for ph in (1, 2):
    for bh in (1, 2):
        X[f"hand_match_{ph}_{bh}"] = (
            (X["pitcher_hand"] == ph)
            & (X["batter_hand"] == bh)
        ).astype("float32")

X["pitcher_prev1_success_dev"] = ...
X["pitcher_prev3_success_dev"] = ...
X["pitcher_success_trend_1v5"] = ...
X["pitcher_middle_trend_1v5"] = ...

# 신규 pitchmix 2열
X = add_pitchmix_shape_features(X)

# 모델에 저장된 순서로 최종 정렬
feats = list(history["features"])
missing = [c for c in feats if c not in X.columns]
if missing:
    raise ValueError(f"추론 입력에 없는 피처: {missing}")

return X[feats]
```

제출물이 반드시 단일 `script.py`여야 한다면 공용 함수 내용을 `script.py` 안에도 포함해야 한다. 이 경우에도 학습 함수와 추론 함수가 동일한지 테스트로 검증한다.

## 4. current CatBoost 혼합 제거

현재 제출 코드에는 다음 혼합이 남아 있다.

```python
p_cb = 0.75 * p_cb_candidate + 0.25 * p_cb_current
```

역사 폴드 OOF 결과에 따라 current CatBoost는 사용하지 않는다.

```python
p_cb = p_cb_candidate
```

따라서 다음 코드도 제거할 수 있다.

```python
current_bundle = joblib.load(...)
p_cb_current = predict_current_catboost(test, current_bundle)
```

전체 학습 코드의 `train_current()`와 `current_catboost_bundle.pkl` 생성도 최종 제출에 필요하지 않다.

## 5. 전체 모델 재학습

feature 수가 58열에서 60열로 바뀌므로 기존 CatBoost tree에 두 열을 사후 추가할 수 없다. 2019~2024 전체 데이터로 다시 학습하고 tree를 재-export해야 한다.

예시:

```powershell
.\aimers9\Scripts\python.exe `
  .\1129\submit_jaemin_54_cbhybrid_full\train_full_2024.py `
  --seeds 1,42,777 `
  --task-type GPU `
  --overwrite
```

생성되는 `trees.npz`에는 다음 정보가 반영돼야 한다.

```text
feature 개수: 60
마지막 두 feature:
  asof_pitchmix_entropy
  asof_pitchmix_max_rate
```

## 6. 학습·추론 동일성 테스트

재학습 전에 동일한 입력 행을 학습 함수와 추론 함수에 넣어 두 열이 정확히 일치하는지 확인한다.

```python
train_side = add_pitchmix_shape_features(sample.copy())
infer_side = add_pitchmix_shape_features(sample.copy())

np.testing.assert_allclose(
    train_side[PITCHMIX_SHAPE_FEATURES].to_numpy(),
    infer_side[PITCHMIX_SHAPE_FEATURES].to_numpy(),
    rtol=0,
    atol=1e-7,
)
```

추가 범위 검사:

```python
entropy = train_side["asof_pitchmix_entropy"]
max_rate = train_side["asof_pitchmix_max_rate"]

assert entropy.between(0.0, np.log(4) + 1e-6).all()
assert max_rate.between(0.0, 1.0).all()
```

구종 정보가 없는 행은 다음 값이 된다.

```text
entropy = 0
max_rate = 0
```

기존 `asof_pitcher_pitchmix_n=0`과 함께 사용되므로 CatBoost가 실제 단일 구종 의존과 cold-start를 구분할 수 있다.

## 7. 최종 체크리스트

- [ ] 공용 `add_pitchmix_shape_features()` 작성
- [ ] 전체 학습에서 candidate 8열 생성 후 공용 함수 호출
- [ ] 제출 추론에서 candidate 8열 생성 후 같은 함수 호출
- [ ] feature 목록 끝에 entropy, max rate 순서로 추가
- [ ] 최종 feature 개수 60 확인
- [ ] 학습·추론 출력 `assert_allclose` 통과
- [ ] entropy 범위 `0~log(4)` 확인
- [ ] max rate 범위 `0~1` 확인
- [ ] current CatBoost 25% 혼합 제거
- [ ] 2019~2024 전체 CatBoost 재학습
- [ ] 새 `trees.npz` export
- [ ] numpy tree 추론과 CatBoost `predict_proba` 일치 확인
- [ ] 최종 submission 행 수·row_id·확률 범위 확인

## 관련 검증 결과

pitchmix 2열을 추가한 최종 앙상블의 candidate-only 대비 변화:

| 폴드 | raw BSS | best-shift BSS |
|---|---:|---:|
| 2022 전체 | +1.449 | +1.522 |
| 2023 Regular | +2.045 | +2.146 |
| 2024 전체 | +1.418 | +1.577 |
| pooled OOF | **+1.618** | **+1.610** |

세 폴드에서 모두 개선됐고 CatBoost 슬롯을 pitchmix2 모델 100%로 교체하는 것이 최적이었다.

