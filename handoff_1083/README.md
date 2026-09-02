# submit_41 — 리더보드 1083 재현 문서

`submit_jaemin_41.zip` 은 **`submit_35`(1076) 에서 CatBoost 가지 하나만 갈아끼운 것**이다.
TabM 쪽은 바이트 단위로 손대지 않았다. 그 교체 하나로 +7.

이 문서는 두 가지를 다룬다.

```
A. 그대로 내기      zip 을 그대로 제출한다. 1083 이 나온 그 파일이다
B. 다시 만들기      CatBoost 경로를 소스에서 재학습한다 (바이트 재현은 불가, 이유는 5절)
```

읽는 순서는 1 → 4 면 충분하다. 개조할 사람만 5절 이후를 본다.

---

## 1. 무엇이 바뀌었나 — 파일 단위 대조

`submit_35` 와 `submit_41` 의 모든 파일을 md5 로 비교한 결과다.

```
preprocess.py              동일     3e3f36c5fbfaed0d28e408eebbb8c452
model/f2b_all_s42.npz      동일     TabM all 브랜치
model/f2b_regular_s42.npz  동일     TabM 1군 브랜치
model/f2b_futures_s42.npz  동일     TabM 퓨처스 브랜치
model/hgb.npz              동일     (비중 0.0 이라 안 쓰인다)
model/history.json         동일     TabM 전처리용 as-of 표

model/trees_base.npz       신규 파일이지만 내용은 submit_35/model/trees.npz 와
                           **바이트 동일** (ea10351241e4728205417dab2ddc6e58).
                           이름만 바꿔 살려둔 것이다
model/trees.npz            새로 학습. 44열 -> 50열, 30모델(10시드 x 3그룹)
script.py                  피처 함수 2개 추가 + CatBoost 이중 경로
model/config.json          설명 갱신 (코드가 읽지 않는다. 순수 메모다)
```

즉 **변경점은 CatBoost 경로 하나**다.

---

## 2. 새 피처 6개

기존 44열 뒤에 6열을 붙여 50열로 만든다. 두 묶음 모두 같은 모양이다 —
"이 투수가 이 상황에서 얼마나 잘했나" 를 리그 사전값 쪽으로 수축시킨 값,
그리고 그 투수의 전체 평균과의 차이, 그리고 표본 크기.

```
pc_c12_*      키 = 투수 x count12                     count12 = balls*3 + strikes  (0..11)
pc_cmh_*      키 = 투수 x (count12 x 타자손)           cell    = count12*2 + hand   (batter_hand 는 1/2)
```

각각 세 열이다.

```
rate = (s + ALPHA * prior) / (n + ALPHA)          ALPHA = 200
dev  = rate - base
base = (s_all + ALPHA * g) / (n_all + ALPHA)
n    = log1p(n)

s, n        그 (투수, 키) 칸의 성공수 / 시행수
s_all, n_all 그 투수 전체의 성공수 / 시행수
prior       그 키의 리그 평균     g = 전체 평균
```

**학습과 추론에서 집계 범위가 다르다. 이건 의도된 것이다.**

```
학습 시 (train_c12_submit.py)   시즌 단위 cumsum -> shift(1).
                                 S 시즌 행은 S 미만 시즌만 본다. 자기 시즌 누출 없음
추론 시 (script.py)              **전 학습시즌 합산표**를 조회한다.
                                 2025 행 입장에서 2019~2024 는 전부 과거다
```

표는 `model/trees.npz` 안에 같이 들어 있다
(`pc_key/pc_n/pc_s/pc_pid/pc_total_n/pc_total_s/pc_prior/pc_gmean/pc_alpha`,
`ph_*` 동일 구조 + `ph_prior_keys`).

### 규칙 4 관련

조회는 **그 행의 값만으로 키를 만들어 학습 데이터 표를 한 번 읽는 것**이다.
평가 데이터의 다른 행이나 전체 분포를 쓰지 않는다. 실측으로도 확인했다 (4절).

---

## 3. 예측이 조립되는 전체 경로

```
최종 = sigmoid( 1.0279 * logit(p) - 0.007 )
p    = 0.30 * CatBoost + 0.70 * TabM            (HistGB 는 0.0)

CatBoost = 0.70 * 신50 + 0.30 * 구44            <- 41에서 추가된 층
  각 경로 안에서:
    시드 10개 확률 평균 -> 그룹별로 만든 뒤
    0.6 * all + 0.4 * (1군이면 regular / 퓨처스면 futures)

TabM (submit_35 그대로)
    0.6 * f2b_all + 0.4 * (1군이면 f2b_regular / 퓨처스면 f2b_futures)
    45열 = 44열 + abs_regime(0/1/2/3), 시드는 42 하나
```

`W_CB_NEW = 0.70` 은 `script.py:116` 에 있다. 신50 단독이 아니라 구44 를
0.30 남긴 이유는 예측 다양성이다 (주석에 walk-forward 로 70/30 이 최적이라고 적혀 있다).

---

## 4. A. 그대로 내기

```
파일     handoff_1083/submit_jaemin_41.zip
크기     13.16 MB   (압축해제 13.4 MB, 제한 10GB)
sha256   1e1ac85f5c0b67a20e4998e498025a2067857531a75cc3a17b46664c61011c6b
구조     평면. preprocess.py / requirements.txt / script.py / model/*
```

`features44.py` 는 추론에 안 쓰이므로 zip 에 없다 (submit_35 도 같다).

### 내가 실제로 돌려 확인한 것

`test.csv` 는 5행짜리 형식 견본이라 2024 행 2만 개를 대신 먹여 돌렸다.

```
규칙 4  순서 섞기      최대차 0.000e+00   통과
        20% 부분집합    최대차 0.000e+00   통과
예측    평균 0.48548 (실제 0.48380)   CatBoost 0.48275   TabM 0.48963   상관 0.9029
```

재현: `py verify/verify_41.py submit_41 20000`

---

## 5. B. 다시 만들기 — 그리고 왜 바이트 재현이 안 되는가

```bash
bash build/run_build.sh
```

기본값은 `CB_SEEDS=1..10`, `INCLUDE_CMH=1` 이다.

**실린 `trees.npz` 는 CatBoost GPU 로 학습됐다.** 확인 방법과 결과:

```
같은 자료 / 같은 하이퍼파라미터로 CPU 에서 seed 1, 42 를 학습해
실린 첫 모델의 400그루와 대조 ->  분할피처 일치율 4.8% (우연 수준)
```

즉 CPU 로는 다른 모델이 나온다. GPU 라도 장치가 다르면 달라진다.
그 파일은 이제 없는 vast.ai RTX 3060 인스턴스에서 만들어졌다.

**그래서 1083 을 그대로 받으려면 4절의 zip 을 쓰고, 이 스크립트는 개조용으로만 써라.**

재현: `py verify/seed_probe.py`

### 하이퍼파라미터 (train_c12_submit.py)

```
iterations 400   learning_rate 0.05   depth 4   l2_leaf_reg 1.0
task_type  GPU   (C12_CPU=1 로 끌 수 있으나 다른 모델이 된다)
sample_weight = 2.0 ** (season - 2019)          시즌 가중, DECAY=2.0
그룹    all(전체) / regular(game_type==R) / futures(game_type==F)
시드    10개  ->  그룹당 10모델, 총 30모델, 총 12,000그루
export  대칭트리를 numpy 로 펴서 trees_c12.npz 로 저장 (제출본은 catboost 불필요)
```

시드 집합만은 실측으로 못 박았다 — GPU 빌드라 대조가 불가능하다.
`n_per_group=10` 이라는 사실과, 이 저장소의 모든 야간 스크립트가
`CB_SEEDS=1,2,...,10` 을 쓴다는 관례에서 나온 값이다.

---

## 6. 알려진 결함 두 개 — **고치지 않았다**

1083 은 이 상태에서 나온 점수다. 그대로 두는 게 재현의 조건이라 손대지 않았고,
대신 정확히 어디가 문제인지 적어둔다.

### 6-1. 추론 시간이 예산에 붙어 있다

내 PC(8논리코어, GPU 없음)에서 재고 245,789행으로 환산했다.
**표본 크기마다 값이 흔들리니 범위로 읽어라.**

```
표본  5,000행  전체 왕복        ->   8.1분
표본 20,000행  전체 왕복        ->  10.4분
표본 20,000행  구성요소 합산    ->  14.0분

구성요소 내역 (20,000행 -> 환산)
    TabM all 순전파         6.29분
    TabM regular            4.70분
    CatBoost 신50 순회      1.32분
    CatBoost 구44 순회      1.21분      <- 41에서 새로 생긴 비용
    TabM futures            0.44분
    피처/전처리/적재        0.04분
```

평가 서버에서는 실제로 통과했다 — 1083 이 나왔으니 10분 안에 끝났다는 뜻이고,
그쪽이 이 PC 보다 빠르다는 뜻이다. 그래도 **여유가 크지 않다**는 건 사실이고,
41은 35보다 CatBoost 순회를 한 번 더 한다. 비중을 늘리거나 시드를 얹기 전에
시간부터 재라.

시간의 대부분이 numpy 순전파다. 평가 서버에는 L4 GPU 가 있는데 이 패키지는
numpy 전용이라 그걸 안 쓴다. 여기에 큰 여유가 있다.

재현: `py verify/verify_41b.py`

### 6-2. 조회표 검색이 동등성을 검사하지 않는다

`script.py` 의 `add_pitcher_count12` / `add_pitcher_count_hand` 는 이렇게 찾는다.

```python
pos = np.searchsorted(pair_keys, key)
hit = (pos < len(pair_keys))            # <- 값이 같은지는 안 본다
safe = np.minimum(pos, len(pair_keys) - 1)
n = np.where(hit, pair_n[safe], 0.0)
```

키가 표에 없으면 `searchsorted` 는 **삽입 위치**를 돌려주고, 그 자리는 다음으로
큰 키 — 대개 **다른 투수의 칸**이다. `hit` 이 그걸 못 걸러서 남의 값을 읽는다.

표를 2024 미만으로 만들고 2024 를 조회해 "처음 보는 시즌" 을 흉내내 재봤다.

```
                        미스          그중 남의 칸을 읽는 행
pc_key  투수x카운트     19.92%        1.30%
ph_key  투수x카운트x손  20.14%        1.52%
투수 자체가 처음        19.86%
```

미스의 대부분은 신인 투수인데 `pitcher_id` 가 기존 최댓값보다 커서
`pos == len` 이 되고, 그러면 `hit=False` 라 정상적으로 0 이 된다. 문제가 되는 건
키가 **중간에 끼는** 1.3~1.5% 다. 24.6만 행이면 3,200~3,700행이다.

한 줄이면 고쳐진다.

```python
hit = (pos < len(pair_keys)) & (pair_keys[safe] == key)
```

**학습 코드는 이미 올바르게 하고 있다** (`train_c12_submit.py` 의 `add_cmh` 는
`hit = keys[pos] == kv`). 추론 쪽만 다르다. 즉 학습/추론 규약 불일치다.

고치면 점수가 오를지는 **안 재봤다.** 재려면 제출 한 장이 든다.

---

## 7. 건드리면 안 되는 것

```
TabM npz 3개 / history.json / preprocess.py     submit_35 와 동일해야 한다
trees_base.npz                                   submit_35 의 CatBoost 다. 지우면 W_CB_NEW 층이 사라진다
CALIB_LOGIT_SHIFT = -0.007                       리더보드 2점으로 확정된 값. 제공데이터 외삽(-0.024)은 빗나갔다
CALIB_T = 1.0279                                 이걸 빼면 1069 -> 1062 였다
F2B_SEEDS = (42,)                                시드 평균은 관문 +8 인데 리더보드 -2 였다. 시간도 3배다
```

`model/config.json` 은 **코드가 읽지 않는다.** 사람이 보라고 있는 메모다.
바꿔도 예측은 안 변한다.

---

## 8. 파일 안내

```
handoff_1083/
  README.md                     이 문서
  submit_jaemin_41.zip          1083 을 낸 제출본 그대로
  build/
    run_build.sh                재학습 명령 (기본 CB_SEEDS=1..10, INCLUDE_CMH=1)
    train_c12_submit.py         피처 생성 + CatBoost 학습 + numpy export
    features44.py               위 스크립트가 import 한다 (원본 저장소엔 colab/ 에 있다)
  verify/
    verify_41.py                규칙 4 + 시간 + 조회 미스 (2024 표본)
    verify_41b.py               구성요소별 시간 + 처음 보는 시즌 미스율
    seed_probe.py               실린 트리와 재학습 트리 대조 (GPU/CPU 판별)
```

검증 스크립트는 저장소 루트에서 `open (1)/data/` 를 찾는다.
`py verify/verify_41.py submit_41 20000` 처럼 루트에서 실행하면 된다.
