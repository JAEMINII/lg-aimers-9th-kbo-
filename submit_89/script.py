# -*- coding: utf-8 -*-
"""
script.py — 평가 서버가 실행하는 추론 스크립트.

최종 예측 = 0.10 x CatBoost + 0.30 x flatMLP + 0.60 x TabM,  로짓 시프트 -0.0145

  CatBoost   2019~2024 학습, 시즌가중 2.0**(season-2019).
             0.6 x 전체 + 0.4 x 정규리그(R)단독 라우팅 블렌딩
  flatMLP    전체 시즌 균등 학습, 6epoch, 시드 3개 평균. PLR 임베딩 + 3층 MLP
  TabM       BatchEnsemble(k=32). 2epoch 후 마지막 시즌으로 미세조정,
             전체/퓨처스/정규 3브랜치를 0.6:0.4 로 합침

시즌가중은 2026-08-20 에 추가했다. 리그 성공률이 2019 54.95% -> 2024 48.97% 로
계속 밀리는데 옛 시즌을 같은 무게로 보고 있었다. 관문에서 판별력 886.9 -> 908.9.
'최근 시즌만 쓰기' 는 이미 재봤고 손해였다(849.0 -> 823.3). 표본을 버리기 때문이다.
가중치는 표본을 하나도 안 버리면서 드리프트만 줄인다.

의존성은 numpy 와 pandas 뿐이다. sklearn / joblib / torch 를 쓰지 않는다.
  -> 학습된 트리와 신경망 가중치를 순수 numpy 배열(model/*.npz)로 내보내고
     추론도 numpy 로 한다. 피클 버전 불일치(sklearn 버전, numpy BitGenerator 등)로
     로드가 깨질 여지를 없앴다.
사전학습 가중치는 하나도 쓰지 않는다. model/ 아래 배열은 전부 제공된 train.csv
로만 학습한 결과다.

── 핵심 아이디어 ────────────────────────────────────────────────────────────
asof_* 컬럼은 '커리어 누적'이며 시즌마다 리셋되지 않는다(train 에서 100% 검증).
또한 asof_*_rate x asof_*_n 은 반올림하면 정확한 정수 카운트로 복원된다.

따라서 평가 시즌(2025) 행에 대해
    2025시즌 투구수 = asof_pitcher_n       - (그 투수의 train 총 투구수)
    2025시즌 성공수 = round(rate x asof_n) - (그 투수의 train 총 성공수)
로 '평가 시즌 한정 성적'을 각 행마다 독립적으로 복원할 수 있다.

빼는 값(train 총계)은 학습 때 얼려 model/ 에 담아둔 표에서 가져온다. 평가 데이터를
훑어서 구하지 않는다 — 지인 preprocess 쪽도 마찬가지로 train_mode 일 때만
프레임 집계를 쓰고 추론 경로는 얼린 표만 쓴다.

따라서 각 행의 예측은 그 행의 입력 변수와 train.csv 만으로 결정된다.

규칙 4 실측 (2026-08-21, 6만 행)
    행 순서를 섞고 재실행         최대차 0.000e+00
    20% (1.2만 행)만 넣고 재실행  최대차 0.000e+00
부분 검사가 핵심이다 — 순서만 섞으면 전체 평균 같은 걸 써도 값이 안 변한다.
──────────────────────────────────────────────────────────────────────────
"""
import json
import os

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

# =====================================================================
# 전처리 상수 — 학습과 추론이 반드시 공유해야 하는 값
# =====================================================================

# 베이지안 스무딩 강도. train 에서 0/50/100/200/400 스윕 결과 50 이 최적(corr +0.0779).
ALPHA = 50.0
PRIOR_SUCCESS = 0.50

# 범주형은 '고정 사전'으로 인코딩한다.
# pandas category codes 를 쓰면 평가 데이터에 특정 범주가 없을 때 코드가 밀려
# 학습과 다른 값이 되므로 절대 쓰면 안 된다.
CAT_MAPS = {
    "top_bottom": {"T": 0, "B": 1},
    "game_type": {"R": 0, "F": 1},
    "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                   "12_": 4, "1_3": 5, "_23": 6, "123": 7},
}
UNKNOWN_CAT = -1

# 시프트도 같이 바꾼다. flatMLP 은 예측 평균을 올리는 모델이라 빼면 혼합 평균이
# 0.4978 -> 0.4954 로 내려간다(관문 기준). 리더보드 두 점에서 역산한 관계
#     최적 시프트 = -4 x (예측평균 - 실제평균)
# 로는 0.0024 x 4 = 0.0096 만큼 덜 내려야 한다. -0.0145 를 그대로 두면 과보정이라
# flatMLP 의 값어치가 아니라 보정 오류를 재게 된다.
CALIB_LOGIT_SHIFT = -0.007
# T 재적합 (2026-08-30): 새 4원 혼합의 최적 T 가 두 폴드에서 합의 —
# VS=2024: 1.05, VS=2022: 1.04 (기존 1.0279 는 옛 2원 혼합에서 맞춘 값).
# 온도의 관문->LB 증폭은 2~4배 실측 (T 제거시 관문 -1.8~-3.2 <-> LB -7).
CALIB_T = 1.04

# MLP-PLR 앙상블 비중. 최종 확률 = (1-w) x CatBoost + w x MLP
#
# MLP 는 단독으로는 약한데(관문 773.3 vs CatBoost 884.5) 섞으면 이긴다.
# 학습 데이터를 다르게 준 것이 원천이다 — CatBoost 는 2019~2024 전체,
# MLP 는 최근 한 시즌만. 시즌을 자르면 MLP 가 다른 것을 배운다.
#   예측 상관  CatBoost <-> sklearn HistGB   0.982  (둘 다 GBDT, 전체 학습)
#              CatBoost <-> MLP(전체 학습)   0.935
#              CatBoost <-> MLP(한 시즌)     0.888   <- 채택
# GBDT 는 반대로 시즌을 자르면 손해다(884.5 -> 864.0). season 피처, p_is_succ
# 인시즌 복원이 이미 드리프트를 흡수하고 있어서다. 그래서 학습 구간을 달리 준다.
#
# 관문(2019~2023 학습 -> 2024 검증, 1군 채점) 에서 잰 비중별 점수.
# 시프트를 최적값(-0.034)으로 맞춰 '수준 보정' 효과를 걷어낸 판별력만의 수치다.
#   w      0.00   0.15   0.20   0.25   0.30   0.40
#   점수  887.3  901.1  903.3  904.3  904.1  900.2
# 0.15~0.30 이 평평하다. 최고점은 0.25 지만 차이가 1점이라 안전하게 0.20 을 쓴다.
# 빗나가도 손해가 1점 안쪽이다.
# 3원 앙상블 비중.  최종 = W_CB x CatBoost + W_HGB x HistGB + W_TABM x TabM
#
# 2026-08-23: flatMLP 을 뺀다. 그 값어치를 리더보드로 직접 재는 제출이다.
#
# flatMLP 은 관문 단독 802.9 로 셋 중 최약인데(TabM 911, CatBoost 903) 비중이
# 0.30 이었다. 그 비중의 근거가 관문인데 관문이 이 모델을 과대평가한다 —
#     flatMLP 관문 기여          +45.5
#     안 고른 후보 12개 평균 기여  +18.8
#     차이 약 +22 가 '후보 10개 중 관문 최고를 고른' 선택 편향
# 게다가 예측 평균이 실제보다 1.7%p 높다. 관문은 후보마다 최적 시프트로 채점해
# 그 편향을 공짜로 고쳐 주는데 배치엔 그 서비스가 없다.
#
# CatBoost 와 TabM 의 비율(1:6)은 그대로 두고 flatMLP 몫만 나눠 가진다.
W_CB, W_HGB, W_TABM = 0.3, 0.0, 0.7
# The new 50-feature route is the primary CatBoost model.  A small amount of
# the untouched 44-feature incumbent is retained for prediction diversity;
# walk-forward validation peaked around 70% new / 30% incumbent.
W_CB_NEW = 0.70
# DIN-lite 를 확률 공간에서 섞는다. 관문(colab/din_gate.py)이 잰 것과 같은 형태다.
#   VS=2022  w=0.10 +11.9 / w=0.15 +16.7 / w=0.20 +20.8   전부 3/3
#   VS=2024  w=0.10  +7.0 / w=0.15  +9.6 / w=0.20 +11.7   전부 3/3
# 관문은 가중을 올릴수록 계속 좋아진다. 그런데 조합 변경의 전이율은 0.08 이고
# MNCA(단독 897.5, 상관 0.954, 관문 +8.2)는 배치에서 1069->1045 였다.
# 그래서 중간값을 쓴다. 0 으로 두면 submit_41 과 정확히 같아진다.
# w=0.15 로 낸 submit_43 이 1083 -> 1088 (+5) 였다. 관문 +9.6 -> LB +5, 전이율 0.5.
# 관문은 0.25 까지 단조 증가하고 두 폴드 모두 3/3 이다 (2022 +24.3 / 2024 +13.2).
# 전이율 0.5 를 적용하면 +7 쯤이다. 곡선의 끝을 보려고 0.25 로 올린다.
# 4원 혼합 (submit_46). 폴드교차: 2022에서 고른 (cb.35/tab.15/din.35/ms.15) 이
# 2024 에서 +11.9 (최적시프트 기준). 세 폴드 전부 양수 (2022 +1.3 / 2023 +3.9 / 2024 +11.7).
# MS 수준은 연도별로 요동치므로 (2022 +0.052 / 2023 -0.0105 / 2024 -0.065)
# 학습 폴드 셋의 평균으로 중심화한다. 남는 연도 리스크는 w=0.15 에서 약 +-2점.
# 학습량 실험(msa_datasize)이 2022 폴드의 약함을 데이터 부족으로 확정했다:
#   MS 단독  1.22M행 958.3 / 733k행 904.5 / 428k행 822.9 (2024 폴드)
#   733k = 2022 폴드의 학습량이고, 그때 MS 는 구성원들 아래로 떨어진다.
# 배치는 148만 행이라 곡선의 오른쪽 끝이다. 2022 폴드의 w 캡은 배고픈 MS 의
# 캡이므로 배치에 적용하지 않고, 2024 관문(w0.25 에서 +5.7 추가)을 따라 올린다.
W4_CB, W4_TAB, W4_DIN, W4_MS = 0.3928, 0.0600, 0.1665, 0.3808
# 67: 가중 a=0.35 전진 + b_cnt 편입. 관문(54 대비) +7.22/+1.95/+10.32 (전폴드 양수)
# 65(공격판): 가중을 QP-2024 최적 방향으로 a=0.20 이동. 관문 +5.2/+0.6/+6.7 (pcnt+pout 포함)
MS_CENTER = -0.008          # (-0.065 + 0.052 - 0.0105)/3, 학습 폴드에서만 유도
MS6_CENTER = -0.018         # state6 1군: (-0.049 + 0.010 - 0.0155)/3, 폴드 유도
CW_CB, CW_TAB, CW_DIN, CW_MS = 0.45, 0.10, 0.25, 0.20   # cold-투수 행 가중
# 채점시간 실측 보정 (44 = 4분21초 -> 로컬/서버 배율 2.52):
# 3시드판(48)은 서버 9.9분으로 턱걸이 초과였다. 2시드로 서버 ~7.7분, 여유 2분.
# 시드 평균 손실은 소폭 (시드곡선 실측: 3->4->5->6 시드가 +1.5/+0.1/-0.7 잡음 수준)
# 52: MS 슬롯을 감쇠+S2(마지막시즌 미세조정) 3시드로, w 0.25->0.30.
# 관문: S2 슬롯 w=0.30 이 두 폴드 양수 (2024 +5.4 / 2022 +1.1), 3시드가 필수
# (2시드는 이득 절반). 시간: 51+1순전파 = 서버 ~7.7분.
# 3시드는 서버 9.6분(실측 눈금)으로 시간 불합격 -> 2시드로 조정.
# 2시드 w=0.30 관문: 2024 +3.2 / 2022 +2.1 — 두 폴드 양수 유지.
MS_SEEDS = (42, 1)
PCNT_BETA = 0.12
POUT_BETA = -0.10
BCNT_BETA = 0.05
CSF_BETA = 0.10
COMPREV_BETA = -0.45           # 복원 reverse 성분 (투수x타자손x카운트) 편차 lookup, 관문 +3.36/+1.14/-0.37             # 퓨처스 행 현시즌 편차 보정 (표준 F+8.0 / 롤링 H2-F +4.2)            # 타자x카운트 상대효과 (2024-양수 성분)
SHARP_Z0 = +0.095136          # 학습 전체 평균의 로짓 (행 단위 상수)
SHARP_TH = 0.08             # 4계열 로짓의 행내 중앙값 이탈 최대치
SHARP_G = 1.05              # 합의 행 샤픈 배율 — 관문 +0.75/+2.71/+1.31 (3/3)           # 투수x아웃 역보정 (믹스가 과반영하는 축의 디바이어스)            # 투수x카운트 상대효과 보정 세기 (관문 고원 중앙)
W_DIN = 0.25                # W4_MS=0 폴백일 때만 쓰인다 (44 와 동일 경로)
DIN_SEEDS = (42, 1, 777)
# 퓨처스 경로 f2b 모델의 시드.
# 3개 평균도 만들어 두었지만 **쓰지 않는다**. 이유 둘 —
#   시간   순전파가 2회 -> 6회가 되어 24.6만 행 환산 +111초. 예산 10분에 너무 붙는다
#          (이 패키지는 numpy 전용이라 GPU 가속 경로가 없다)
#   검증   관문에서 잰 건 Stage2 e4 하나다. 시드 평균은 안 쟀고,
#          과거에 시드 8개 평균이 관문 +8.0 인데 리더보드 -2 였다
F2B_SEEDS = (42,)

HERE = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# 경로 해석 — 평가 서버의 작업 디렉터리가 무엇이든 동작하도록
# =====================================================================

def resolve(rel):
    cands = [os.path.join(os.getcwd(), rel), os.path.join(HERE, rel),
             os.path.join(HERE, "..", rel), rel]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"'{rel}' 를 찾을 수 없음.\n"
        f"  cwd      = {os.getcwd()}\n"
        f"  __file__ = {os.path.abspath(__file__)}\n"
        "  탐색 경로:\n" + "\n".join(f"    - {c}" for c in cands) +
        f"\n  cwd 내용  = {sorted(os.listdir(os.getcwd()))[:40]}"
        f"\n  HERE 내용 = {sorted(os.listdir(HERE))[:40]}")


# =====================================================================
# 전처리 — 학습/추론 공용
# =====================================================================

def encode_categoricals(X):
    for col, mapping in CAT_MAPS.items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(UNKNOWN_CAT).astype("int16")
    return X


def recover_counts(X):
    """asof_*_rate x asof_*_n -> 정확한 정수 누적 카운트로 복원.
    최대 반올림오차가 0.0077 (< 0.5) 이므로 round 로 원래 정수가 정확히 복원된다."""
    pn = X["asof_pitcher_n"].astype("float64")
    bn = X["asof_batter_n"].astype("float64")
    return {"p_n": pn, "p_succ": (X["asof_pitcher_success_rate"].fillna(0.0) * pn).round(),
            "b_n": bn, "b_succ": (X["asof_batter_success_rate"].fillna(0.0) * bn).round()}


def add_inseason(X, cnt, base_p_n, base_p_succ, base_b_n, base_b_succ):
    """'시즌 시작 시점' 누적값을 빼서 in-season 피처를 만든다.

    base_* 의 출처만 학습/추론에서 다르고(아래) 이후 계산은 완전히 동일하다.
      학습: 해당 (투수, 시즌) 첫 행의 asof 값
      추론: 그 투수의 train(2019~2024) 전체 누적값
    둘 다 의미는 '이번 시즌 시작 직전까지의 누적'으로 같다. (실측 차이 0.0)
    """
    pn_cur = (cnt["p_n"] - base_p_n).clip(lower=0)
    ps_cur = (cnt["p_succ"] - base_p_succ).clip(lower=0)
    bn_cur = (cnt["b_n"] - base_b_n).clip(lower=0)
    bs_cur = (cnt["b_succ"] - base_b_succ).clip(lower=0)
    X["pn_cur"] = pn_cur
    X["bn_cur"] = bn_cur
    X["p_is_succ"] = (ps_cur + ALPHA * PRIOR_SUCCESS) / (pn_cur + ALPHA)
    X["b_is_succ"] = (bs_cur + ALPHA * PRIOR_SUCCESS) / (bn_cur + ALPHA)
    return X


def add_missing_flags(X, src):
    X["na_p"] = src["asof_pitcher_success_rate"].isna().astype("int8")
    X["na_b"] = src["asof_batter_success_rate"].isna().astype("int8")
    X["na_prev"] = src["asof_pitcher_prev1_game_success_rate"].isna().astype("int8")
    return X


def add_count_matchup(X, tables):
    """볼카운트 x 좌우매치업 축.

    같은 투수라도 상황에 따라 '실력이 드러나는 정도' 가 다르다.
      - 0-2 처럼 유인구를 던지는 카운트에서는 잘 던지는 투수나 못 던지는 투수나
        결과가 비슷해진다 (실력 판별력 0.72 배).
      - 우투 vs 좌타 매치업도 판별력이 낮다 (0.93 배).
    그래서 그 셀의 '실력 신뢰도' 만큼만 투수 실력 신호를 살리고 나머지는
    리그 평균 쪽으로 수축시킨다.

      cm        = 볼카운트(12) x 매치업(4) = 최대 48 셀
      lg_cm_eff = 그 셀의 리그 평균 편차
      cm_rel    = 그 셀에서 (실제 성공 ~ p_is_succ) 회귀 기울기 / 전체 기울기
      p_adj_cm  = 리그평균 + (p_is_succ - 리그평균) x cm_rel

    val=2024 실측: 12피처 791.1 -> 15피처 845.4 (+54.3)
    """
    cnt = X["balls_before"].astype("int32") * 3 + X["strikes_before"].astype("int32")
    mu = X["pitcher_hand"].astype("int32") * 2 + X["batter_hand"].astype("int32")
    cm = cnt * 4 + mu
    X["lg_cm_eff"] = cm.map(tables["cm_lg"]).astype("float64").fillna(0.0)
    X["cm_rel"] = cm.map(tables["cm_rel"]).astype("float64").fillna(1.0)
    g = float(tables["gmean"])
    X["p_adj_cm"] = g + (X["p_is_succ"] - g) * X["cm_rel"]
    return X


def add_platoon(X, tables):
    """플래툰 편차 — 그 투수의 '현재 타자 손' 상대 성공률 − 그 투수 전체 성공률.

    같은 투수라도 좌타/우타 상대 제구가 다르다. 리그 전체로는 방향이 정반대다.
      좌투 vs 좌타 49.09% / vs 우타 53.75%   -> 좌타 상대 -4.66%p
      우투 vs 좌타 53.07% / vs 우타 52.21%   -> 좌타 상대 +0.85%p
    투수 개인의 격차를 '그 투수손의 리그 격차' 쪽으로 수축시켜 만든다.
    이력이 없으면 리그값이 되어 편차가 0 이 된다 (= 평범한 투수와 구분되지 않음).

    lookup 은 pitcher_id 단위이므로 평가 데이터의 다른 행을 참조하지 않는다.
    """
    pid = X["pitcher_id"]
    l_ = pid.map(tables["plat_l"]).astype("float64")
    r_ = pid.map(tables["plat_r"]).astype("float64")
    a_ = pid.map(tables["plat_a"]).astype("float64")
    cur = np.where(X["batter_hand"].values == 1, l_.values, r_.values)
    X["plat_dev"] = np.nan_to_num(cur - a_.values, nan=0.0)
    return X


def add_marcel(X, tables):
    """Marcel Projection — 직전 3시즌 가중평균으로 추정한 '원래 실력'.

    시즌 안에서 z정규화(평균50/표준편차10)하므로 리그 수준 하락에 오염되지 않는다.
    작년x5 / 2년전x4 / 3년전x3 가중, 투구수가 적을수록 리그평균으로 수축.
    역변환 기준은 예측 대상 시즌의 1년 전 리그 평균/표준편차 (미래 정보 미사용).

    p_is_succ 가 '지금 이 시즌의 단기 실적' 이라면 이건 '장기 추정' 이다.
    """
    X["marcel"] = (X["pitcher_id"].map(tables["marcel"])
                   .fillna(float(tables["gmean"])).astype("float64"))
    return X


def add_pitcher_count12(X, history):
    """Add the exported all-training pitcher x count12 prior features.

    The table is fitted only on official training rows.  This lookup touches
    only the current row and therefore remains valid under the independent
    test-row rule.
    """
    cnt = (X["balls_before"].astype("int64") * 3
           + X["strikes_before"].astype("int64")).to_numpy()
    pid = X["pitcher_id"].astype("int64").to_numpy()
    key = pid * 16 + cnt

    pair_keys = np.asarray(history["pc_key"], dtype=np.int64)
    pair_n = np.asarray(history["pc_n"], dtype=np.float64)
    pair_s = np.asarray(history["pc_s"], dtype=np.float64)
    pos = np.searchsorted(pair_keys, key)
    hit = (pos < len(pair_keys))
    safe = np.minimum(pos, max(len(pair_keys) - 1, 0))
    n = np.where(hit, pair_n[safe], 0.0)
    s = np.where(hit, pair_s[safe], 0.0)

    pids = np.asarray(history["pc_pid"], dtype=np.int64)
    pn = np.asarray(history["pc_total_n"], dtype=np.float64)
    ps = np.asarray(history["pc_total_s"], dtype=np.float64)
    pp = np.searchsorted(pids, pid)
    phit = (pp < len(pids))
    psafe = np.minimum(pp, max(len(pids) - 1, 0))
    n_all = np.where(phit, pn[psafe], 0.0)
    s_all = np.where(phit, ps[psafe], 0.0)

    alpha = float(history.get("pc_alpha", 200.0))
    g = float(history.get("pc_gmean", history["gmean"]))
    prior_lut = np.asarray(history["pc_prior"], dtype=np.float64)
    prior = prior_lut[np.clip(cnt, 0, len(prior_lut) - 1)]
    base = (s_all + alpha * g) / (n_all + alpha)
    rate = (s + alpha * prior) / (n + alpha)
    X["pc_c12_rate"] = rate
    X["pc_c12_dev"] = rate - base
    X["pc_c12_n"] = np.log1p(n)
    return X


def add_pitcher_count_hand(X, history):
    """Add prior pitcher x (count12 x batter-hand) features.

    The exported table contains only training rows.  The lookup is per test
    row, with no aggregation over test rows, so it remains independent-row
    compliant.
    """
    cnt = (X["balls_before"].astype("int64") * 3
           + X["strikes_before"].astype("int64")).to_numpy()
    hand = X["batter_hand"].astype("int64").to_numpy()
    pid = X["pitcher_id"].astype("int64").to_numpy()
    cell = cnt * 2 + hand
    key = pid * 32 + cell
    pair_keys = np.asarray(history["ph_key"], dtype=np.int64)
    pair_n = np.asarray(history["ph_n"], dtype=np.float64)
    pair_s = np.asarray(history["ph_s"], dtype=np.float64)
    pos = np.searchsorted(pair_keys, key)
    hit = (pos < len(pair_keys))
    safe = np.minimum(pos, max(len(pair_keys) - 1, 0))
    n = np.where(hit, pair_n[safe], 0.0)
    s = np.where(hit, pair_s[safe], 0.0)

    pids = np.asarray(history["ph_pid"], dtype=np.int64)
    pn = np.asarray(history["ph_total_n"], dtype=np.float64)
    ps = np.asarray(history["ph_total_s"], dtype=np.float64)
    pp = np.searchsorted(pids, pid)
    phit = (pp < len(pids))
    psafe = np.minimum(pp, max(len(pids) - 1, 0))
    n_all = np.where(phit, pn[psafe], 0.0)
    s_all = np.where(phit, ps[psafe], 0.0)

    alpha = float(history.get("ph_alpha", 200.0))
    g = float(history.get("ph_gmean", history["gmean"]))
    prior_keys = np.asarray(history.get("ph_prior_keys", np.arange(1, 25)), dtype=np.int64)
    prior_lut = np.asarray(history["ph_prior"], dtype=np.float64)
    pp0 = np.searchsorted(prior_keys, cell)
    h0 = pp0 < len(prior_keys)
    s0 = np.minimum(pp0, max(len(prior_keys) - 1, 0))
    prior = np.where(h0, prior_lut[s0], g)
    base = (s_all + alpha * g) / (n_all + alpha)
    rate = (s + alpha * prior) / (n + alpha)
    X["pc_cmh_rate"] = rate
    X["pc_cmh_dev"] = rate - base
    X["pc_cmh_n"] = np.log1p(n)
    return X


def build_features(df, history):
    """평가 데이터 -> 모델 입력 (추론 경로).

    history: {"pitcher_n","pitcher_s","batter_n","batter_s","features"} (모두 순수 dict/list)
    """
    X = df.drop(columns=[ID_COL], errors="ignore").copy()
    cnt = recover_counts(X)

    # train 에 없는 선수(= 평가 시즌 신인)는 0 -> 커리어 전체가 곧 in-season
    base_p_n = X["pitcher_id"].map(history["pitcher_n"]).fillna(0.0).astype("float64")
    base_p_s = X["pitcher_id"].map(history["pitcher_s"]).fillna(0.0).astype("float64")
    base_b_n = X["batter_id"].map(history["batter_n"]).fillna(0.0).astype("float64")
    base_b_s = X["batter_id"].map(history["batter_s"]).fillna(0.0).astype("float64")

    X = add_inseason(X, cnt, base_p_n, base_p_s, base_b_n, base_b_s)
    X = add_missing_flags(X, df)
    X = add_count_matchup(X, history)
    X = add_platoon(X, history)
    X = add_marcel(X, history)
    if "pc_key" in history:
        X = add_pitcher_count12(X, history)
    if "ph_key" in history:
        X = add_pitcher_count_hand(X, history)
    X = encode_categoricals(X)

    # 2024 holdout에서 선택된 Candidate CatBoost의 추가 8피처.
    for ph in (1, 2):
        for bh in (1, 2):
            X[f"hand_match_{ph}_{bh}"] = (
                (X["pitcher_hand"] == ph) & (X["batter_hand"] == bh)
            ).astype("float32")
    X["pitcher_prev1_success_dev"] = (
        X["asof_pitcher_prev1_game_success_rate"] - X["p_is_succ"]
    ).astype("float32")
    X["pitcher_prev3_success_dev"] = (
        X["asof_pitcher_prev3_game_success_rate"] - X["p_is_succ"]
    ).astype("float32")
    X["pitcher_success_trend_1v5"] = (
        X["asof_pitcher_prev1_game_success_rate"]
        - X["asof_pitcher_prev5_game_success_rate"]
    ).astype("float32")
    X["pitcher_middle_trend_1v5"] = (
        X["asof_pitcher_prev1_game_middle_rate"]
        - X["asof_pitcher_prev5_game_middle_rate"]
    ).astype("float32")

    feats = list(history["features"])
    missing = [c for c in feats if c not in X.columns]
    if missing:
        raise ValueError(f"추론 입력에 없는 피처: {missing}")
    return X[feats]


# =====================================================================
# 추론 — 순수 numpy 로 구현한 히스토그램 부스팅 트리 예측
# =====================================================================

def predict_numpy(X, z, chunk=4096):
    """X: (n, n_features) float64 배열. z: trees.npz.

    CatBoost 의 대칭 트리(oblivious tree) 예측을 그대로 재현한다.

    대칭 트리는 깊이 d 면 잎이 2^d 개이고, 모든 잎이 같은 분할 조건 d 개를 공유한다.
    따라서 노드를 하나씩 따라갈 필요 없이 조건 d 개를 한 번에 평가하고
    비트로 모으면 잎 인덱스가 바로 나온다.

        잎 인덱스 = sum_k (x[feat_k] > thr_k) << k
        raw       = sum(트리별 leaf_values[잎 인덱스])
        proba     = sigmoid(raw)          (scale=1, bias=0 은 내보낼 때 검증)
        최종      = 모델(시드) 간 확률 평균

    결측: CatBoost 의 피처별 nan_value_treatment 를 따른다.
          cb_nan_left[f] 가 1 이면 NaN 을 모든 경계보다 작게 취급 -> 조건이 False.
          0 이면 크게 취급 -> True.

    성능: 조건 평가가 (chunk, T, d) 브로드캐스트 한 방이라
          sklearn 순회 방식보다 단순하고 빠르다. 분기가 없어 벡터화가 깨지지 않는다.
    """
    cb_leaf = z["cb_leaf"]                       # (T, 2^d)
    nan_left = z["cb_nan_left"].astype(bool)     # (n_features,)
    depth = int(z["cb_depth"])
    n_models = int(z["n_models"])
    T = cb_leaf.shape[0]
    per = T // n_models                          # 모델당 트리 수 (모델 순 정렬)
    # 경기유형 라우팅 블렌딩. 모델은 [전체 x10, R단독 x10, F단독 x10] 순으로 이어져 있다.
    n_grp = int(z["n_per_group"]) if "n_per_group" in z.files else n_models
    blend_full = float(z["blend_full"]) if "blend_full" in z.files else 1.0
    gt_col = int(z["gt_col"]) if "gt_col" in z.files else -1
    is_f = None
    if gt_col >= 0 and n_models == 3 * n_grp:
        is_f = (np.asarray(X)[:, gt_col] > 0.5)   # game_type: R=0, F=1

    # 고유 분할만 평가한다. 같은 (피처, 경계) 가 여러 트리에 반복되므로
    # T*d 번이 아니라 고유개수만큼만 비교하면 된다.
    sp_feat = z["sp_feat"].astype(np.int64)      # (U,)
    sp_thr = z["sp_thr"]                         # (U,) float32
    sp_idx = z["sp_idx"].astype(np.int64)        # (T, d) -> U 인덱스
    sp_nan_right = ~nan_left[sp_feat]            # NaN 일 때 오른쪽으로 갈 분할

    n = X.shape[0]
    # CatBoost 는 피처값을 float32 로 비교한다. float64 로 두면 경계에
    # 아주 가까운 행에서 판정이 갈린다(실측 최대차 1.9e-4).
    Xa = np.ascontiguousarray(X, dtype=np.float32)
    out = np.zeros(n, np.float64)
    flat_leaf = cb_leaf.ravel()
    # 잎 인덱스는 0~15, 트리 오프셋 최대 64,000 이라 int32 로 충분하다.
    # int64 로 두면 (chunk, 4000) 배열이 500MB 를 넘어 메모리 대역폭에 묶인다.
    tree_ofs = (np.arange(T, dtype=np.int32) * (1 << depth))
    sp_cols = [sp_idx[:, k] for k in range(depth)]

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        v = Xa[s:e][:, sp_feat]                      # (m, U)  U << T*d
        cond = v > sp_thr
        nanm = np.isnan(v)
        if nanm.any():
            cond = np.where(nanm, sp_nan_right, cond)
        cu = cond.view(np.uint8)                     # bool 은 1바이트라 그대로 본다
        # 트리별 잎 인덱스를 비트로 모은다. 깊이만큼만 도는 짧은 루프다.
        leaf_idx = cu[:, sp_cols[0]]
        for k in range(1, depth):
            leaf_idx = leaf_idx | (cu[:, sp_cols[k]] << k)
        raw = flat_leaf[leaf_idx.astype(np.int32) + tree_ofs]        # (m, T)
        raw = raw.reshape(-1, n_models, per).sum(axis=2)             # (m, n_models)
        p = 1.0 / (1.0 + np.exp(-raw))                               # (m, n_models)
        if is_f is None:
            out[s:e] = p.mean(axis=1)
        else:
            p_full = p[:, :n_grp].mean(axis=1)
            p_r = p[:, n_grp:2*n_grp].mean(axis=1)
            p_f = p[:, 2*n_grp:].mean(axis=1)
            routed = np.where(is_f[s:e], p_f, p_r)
            out[s:e] = blend_full * p_full + (1.0 - blend_full) * routed
    return out


def apply_calibration(p, shift=CALIB_LOGIT_SHIFT, T=None):
    """온도 + 절편.  p_out = sigmoid(T x logit(p) + shift)

    T 는 관문(2019~2023 학습 -> 2024)에서 이 배치 구성 그대로 맞춘 값이다.
    1 에 거의 붙어 있고 이득도 +0.6 이라 사실상 중립이다. 크게 잡지 않는 이유는
    다년 전이 시험 때문이다 — 다른 폴드에서 맞춘 T 를 가져다 쓰면 2022 에서
    -744, 2024 에서 -144 였다. 온도는 그 해에만 맞는 값이다.

    행 단위 단조 변환이라 순위는 보존되고 규칙 4 도 지킨다.
    """
    T = CALIB_T if T is None else T
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    z = T * np.log(p / (1.0 - p)) + shift
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

def load_model():
    """trees.npz 를 읽어 (history, z, shift) 를 돌려준다."""
    z = np.load(resolve("model/trees.npz"), allow_pickle=False)
    # 구버전 npz(플래툰/marcel 없음) 와도 호환. 없으면 빈 표 -> 피처가 중립값이 되고,
    # features 목록에도 없으므로 최종 입력에서 그대로 빠진다.
    keys = set(z.files)
    pl = z["plat_id"].tolist() if "plat_id" in keys else []
    mk = z["marcel_id"].tolist() if "marcel_id" in keys else []
    gv = lambda k: z[k].tolist() if k in keys else []
    history = {
        "pitcher_n": dict(zip(z["pitcher_id"].tolist(), z["pitcher_n"].tolist())),
        "pitcher_s": dict(zip(z["pitcher_id"].tolist(), z["pitcher_s"].tolist())),
        "batter_n": dict(zip(z["batter_id"].tolist(), z["batter_n"].tolist())),
        "batter_s": dict(zip(z["batter_id"].tolist(), z["batter_s"].tolist())),
        "cm_lg": dict(zip(z["cm_key"].tolist(), z["cm_lg"].tolist())),
        "cm_rel": dict(zip(z["cm_key"].tolist(), z["cm_rel"].tolist())),
        "gmean": float(z["gmean"]),
        "plat_l": dict(zip(pl, gv("plat_l"))),
        "plat_r": dict(zip(pl, gv("plat_r"))),
        "plat_a": dict(zip(pl, gv("plat_a"))),
        "marcel": dict(zip(mk, gv("marcel_v"))),
        "features": [str(c) for c in z["features"].tolist()],
    }
    if "pc_key" in keys:
        history.update({
            "pc_key": z["pc_key"], "pc_n": z["pc_n"], "pc_s": z["pc_s"],
            "pc_pid": z["pc_pid"], "pc_total_n": z["pc_total_n"],
            "pc_total_s": z["pc_total_s"], "pc_prior": z["pc_prior"],
            "pc_gmean": float(z["pc_gmean"]),
            "pc_alpha": float(z["pc_alpha"]),
        })
    if "ph_key" in keys:
        history.update({
            "ph_key": z["ph_key"], "ph_n": z["ph_n"], "ph_s": z["ph_s"],
            "ph_pid": z["ph_pid"], "ph_total_n": z["ph_total_n"],
            "ph_total_s": z["ph_total_s"], "ph_prior": z["ph_prior"],
            "ph_prior_keys": z["ph_prior_keys"],
            "ph_gmean": float(z["ph_gmean"]),
            "ph_alpha": float(z["ph_alpha"]),
        })
    base_path = os.path.join(os.path.dirname(resolve("model/trees.npz")), "trees_base.npz")
    if os.path.exists(base_path):
        zb = np.load(base_path, allow_pickle=False)
        history["_base_z"] = zb
        history["_base_features"] = [str(c) for c in zb["features"].tolist()]
    # npz 에 저장된 값이 아니라 이 파일의 상수를 쓴다.
    # shift 는 재학습 없이 바꿀 수 있어야 하는데, npz 에서 읽으면 export 를 다시
    # 돌려야 한다. 2026-08-18 에 -0.05 -> 0.0 으로 바꾸면서 상수 기준으로 통일했다.
    return history, z, float(CALIB_LOGIT_SHIFT)


# =====================================================================
# flatMLP — 전체 시즌 균등 학습, 시드 3개 평균 (순수 numpy)
# =====================================================================
#
# 기존 MLP 는 '2024 한 시즌만' 학습이었는데 그게 나쁜 선택이었다.
# 전체 시즌을 균등하게 보는 쪽이 CatBoost 와 멀어져 앙상블이 훨씬 좋아진다.
#     관문 전체채점   CatBoost 단독 903.2
#                    + 기존MLP 0.20   911.2
#                    + flatMLP 0.35   937.2
# 시즌가중을 주면 오히려 나빠진다(929.6 -> 921.1). CatBoost 가 이미 시즌가중
# 2.0 을 쓰므로 같은 보정을 두 번 하면 상관이 올라가고 다양성이 죽는다.
#
# 시드 3개 평균이다. 개별 시드는 편차가 커서(928.2 / 920.6 / 931.1) 하나를
# 고르면 운을 고르는 셈이고, 평균이 개별 최고치보다도 높다(937.2).
#
# torch 원본과 20만 행 최대차이 6.7e-08 로 일치 확인.
def _plr(xn, per_w, lin_w, lin_b):
    """(n, F) -> (n, F*d).  피처마다 독립적인 주기임베딩 + 선형 + ReLU."""
    n, nf = xn.shape
    d = lin_w.shape[-1] if lin_w.ndim == 3 else lin_w.shape[0]
    out = np.empty((n, nf, d), np.float32)
    for f in range(nf):
        z = (2.0 * np.pi * per_w[f])[None, :] * xn[:, f:f + 1]
        p = np.concatenate([np.cos(z), np.sin(z)], 1).astype(np.float32)
        out[:, f, :] = p @ lin_w[f] + lin_b[f]
    np.maximum(out, 0, out=out)   # 위치인수 3개는 numpy 2.4 에서 경고
    return out.reshape(n, nf * d)


def _fm_predict_one(Xn, Xc, z, chunk=16384):
    """z: np.load 한 시드 하나의 npz.  반환 (n,) 확률."""
    per_w = z["num_emb__periodic__weight"]
    lin_w = z["num_emb__linear__weight"]
    lin_b = z["num_emb__linear__bias"]
    cat_w = [z[f"cat_embs__{j}__weight"] for j in range(9)]
    W = [(z["body__0__weight"].T, z["body__0__bias"]),
         (z["body__2__weight"].T, z["body__2__bias"]),
         (z["body__4__weight"].T, z["body__4__bias"])]
    hw, hb = z["head__weight"].T, z["head__bias"]
    out = np.empty(len(Xn), np.float64)
    for a in range(0, len(Xn), chunk):
        b = min(a + chunk, len(Xn))
        h = _plr(Xn[a:b], per_w, lin_w, lin_b)
        parts = [h] + [cat_w[j][Xc[a:b, j]] for j in range(9)]
        x = np.concatenate(parts, 1)
        for w, bb in W:
            x = np.maximum(x @ w + bb, 0)
        lg = (x @ hw + hb).ravel()
        out[a:b] = 1.0 / (1.0 + np.exp(-lg.astype(np.float64)))
    return out


def _fm_predict(Xn, Xc, archives, chunk=16384):
    """시드별 확률의 평균. 개별 시드를 고르면 운을 고르는 것이라 평균이 맞다."""
    return np.mean([_fm_predict_one(Xn, Xc, z, chunk) for z in archives], 0)


def _fm_prep(X, stats):
    """학습 때 저장한 통계로 전처리. 통계는 전부 학습 구간에서 뽑은 것이다."""
    ci = np.asarray(stats["cat_idx"])
    ni = np.asarray([j for j in range(X.shape[1]) if j not in set(ci.tolist())])
    Xc = np.zeros((len(X), len(ci)), np.int64)
    for a in range(len(ci)):
        keys = stats[f"catkey_{a}"]
        vals = stats[f"catval_{a}"]
        q = X[:, ci[a]].astype(np.float64)
        pos = np.clip(np.searchsorted(keys, q), 0, max(len(keys) - 1, 0))
        hit = keys[pos] == q if len(keys) else np.zeros(len(X), bool)
        Xc[:, a] = np.where(hit, vals[pos], 0)
    Xn = X[:, ni].astype(np.float64)
    miss = np.isnan(Xn)
    Xn = np.where(miss, stats["med"], Xn)
    Xn = ((Xn - stats["mu"]) / stats["sd"]).astype(np.float32)
    hn = stats["has_nan"]
    if hn.any():
        Xn = np.concatenate([Xn, miss[:, hn].astype(np.float32)], 1)
    return Xn, Xc


# =====================================================================
# TabM — 지인 파이프라인 (리더보드 1041 실측)
# =====================================================================
#
# official tabm 을 numpy 로 뽑은 것. Stage1 2epoch + Stage2 2024 fine-tune,
# all/futures/regular 3브랜치를 0.6:0.4 로 합친다.
# epoch 을 4 로 늘렸더니 980 으로 떨어졌다 — 학습 구간 마지막 시즌에 과적합해서
# 퓨처스가 무너진다(관문 전체채점 870.8 -> 832.8). 2 를 유지한다.
#
# 피처는 지인 preprocess 로 만든다. 우리 것과 plat_dev 가 달라 섞으면 안 된다.
def load_bundle(path: str):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata"].item()))
    return metadata, archive


def _tabm_transform(X: pd.DataFrame, metadata: dict):
    expected = list(metadata["feature_names"])
    missing = [col for col in expected if col not in X.columns]
    if missing:
        raise ValueError(f"missing exported features: {missing}")
    X = X[expected]

    cat_cols = list(metadata["cat_cols"])
    num_cols = list(metadata["num_cols"])
    cats = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
    for j, col in enumerate(cat_cols):
        mapping = {value: i + 1 for i, value in enumerate(metadata["cat_values"][col])}
        cats[:, j] = X[col].map(mapping).fillna(0).astype("int64").to_numpy()

    numeric = X[num_cols].apply(pd.to_numeric, errors="coerce")
    parts = []
    for col in num_cols:
        values = numeric[col].astype("float64").fillna(float(metadata["medians"][col]))
        values = (values - float(metadata["means"][col])) / float(metadata["stds"][col])
        parts.append(values.to_numpy(dtype="float32"))
    for col in metadata["missing_cols"]:
        parts.append(numeric[col].isna().to_numpy(dtype="float32"))
    nums = np.column_stack(parts).astype("float32", copy=False)
    return nums, cats


def _one_hot(cats: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    blocks = []
    rows = np.arange(len(cats))
    for j, cardinality in enumerate(cardinalities):
        block = np.zeros((len(cats), cardinality), dtype="float32")
        values = np.clip(cats[:, j], 0, cardinality - 1)
        block[rows, values] = 1.0
        blocks.append(block)
    return np.column_stack(blocks) if blocks else np.empty((len(cats), 0), dtype="float32")


def _tabm_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def predict_frame(X: pd.DataFrame, metadata: dict, archive, chunk_size: int = 1024):
    """Predict already-transformed 44-feature rows independently."""
    nums, cats = _tabm_transform(X, metadata)
    if len(nums) == 0:
        return np.empty(0, dtype="float32")
    if metadata["num_embeddings"] == "linear_relu":
        weight = archive["num_embedding_weight"]
        bias = archive["num_embedding_bias"]
        num_repr = np.maximum(nums[:, :, None] * weight[None, :, :] + bias[None, :, :], 0.0)
        num_repr = num_repr.reshape(len(nums), -1)
    elif metadata["num_embeddings"] == "none":
        num_repr = nums
    else:
        raise ValueError(f"unsupported exported embedding: {metadata['num_embeddings']}")

    k = int(metadata["k"])
    cardinalities = list(metadata["cat_cardinalities"])
    num_dim = num_repr.shape[1]
    cat_offsets = np.cumsum([0] + cardinalities[:-1]) + num_dim
    if metadata["backbone_kind"] != "batch_ensemble":
        raise ValueError("optimized exporter currently requires batch_ensemble TabM")
    predictions = []
    for start in range(0, len(num_repr), chunk_size):
        end = start + chunk_size
        chunk_nums = num_repr[start:end]
        chunk_cats = cats[start:end]
        h = np.empty((len(chunk_nums), k, int(metadata["d_block"])), dtype="float32")
        for block in range(int(metadata["n_blocks"])):
            weight = archive[f"block_{block}_weight"]
            r = archive[f"block_{block}_r"]
            s = archive[f"block_{block}_s"]
            bias_block = archive[f"block_{block}_bias"]
            if block == 0:
                # Exact equivalent of one-hot -> BatchEnsemble -> Linear, without
                # materializing the 2,489-dimensional one-hot matrix.
                h = np.matmul(
                    chunk_nums[:, None, :] * r[None, :, :num_dim],
                    weight[:, :num_dim].T,
                )
                for j, cardinality in enumerate(cardinalities):
                    offset = int(cat_offsets[j])
                    indices = chunk_cats[:, j]
                    selected_r = r[:, offset:offset + cardinality][:, indices].T
                    selected_w = weight[:, offset:offset + cardinality][:, indices].T
                    h += selected_r[:, :, None] * selected_w[:, None, :]
            else:
                h = np.matmul(h * r[None, :, :], weight.T)
            h = h * s[None, :, :] + bias_block[None, :, :]
            h = np.maximum(h, 0.0)

        logits = np.einsum(
            "bki,kio->bko", h, archive["output_weight"]
        ) + archive["output_bias"][None, :, :]
        predictions.append(_tabm_sigmoid(logits[:, :, 0]).mean(axis=1))
    return np.concatenate(predictions) if predictions else np.empty(0, dtype="float32")


# =====================================================================
# 데이터 로드 / 제출 파일
# =====================================================================

def _logit(p):
    q = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(q / (1.0 - q))


def _tabm_forward(nums, cats, metadata: dict, archive, chunk_size: int = 8192,
                  _cache={}):
    """이미 전처리된 배열로 TabM 순전파 (접힌 가중치 최적화판).

    원본은 block0 에서 (B, k, D) 임시배열을 만들었다 — 4096행이면 7,550만 원소.
    r 은 입력축 elementwise 라 (x * r_k) @ W.T = x @ (r_k[:,None] * W.T) 이고,
    RW[n, k*o] 를 모델당 한 번 접어두면 2D GEMM 한 방이 된다. 2.1배 빠르고
    6개 npz 전부에서 최대차 2.4e-07 (float32 반올림) 로 검증됐다.
    범주형 항은 원본과 같이 one-hot 을 펴지 않고 게더로 처리한다.
    """
    if len(nums) == 0:
        return np.empty(0, dtype="float32")
    if metadata["backbone_kind"] != "batch_ensemble":
        raise ValueError("optimized exporter currently requires batch_ensemble TabM")
    key = id(archive)
    if key not in _cache:
        k = int(metadata["k"])
        db = int(metadata["d_block"])
        nb = int(metadata["n_blocks"])
        cards = list(metadata["cat_cardinalities"])
        if metadata["num_embeddings"] != "linear_relu":
            raise ValueError(f"unsupported exported embedding: {metadata['num_embeddings']}")
        ew = archive["num_embedding_weight"].astype(np.float32)
        eb = archive["num_embedding_bias"].astype(np.float32)
        D = ew.shape[0] * ew.shape[1]
        W0 = archive["block_0_weight"].astype(np.float32)
        r0 = archive["block_0_r"].astype(np.float32)
        RW = (r0[:, None, :D] * W0[None, :, :D]).transpose(2, 0, 1).reshape(D, k * db)
        offs = np.cumsum([0] + cards[:-1]) + D
        catW = []
        for j, c in enumerate(cards):
            o = int(offs[j])
            rr, ww = r0[:, o:o + c], W0[:, o:o + c]
            catW.append(np.ascontiguousarray(
                (rr[:, None, :] * ww[None, :, :]).transpose(2, 0, 1)))
        later = []
        for b in range(1, nb):
            Wb = archive[f"block_{b}_weight"].astype(np.float32)
            rb = archive[f"block_{b}_r"].astype(np.float32)
            later.append(np.ascontiguousarray(rb[:, :, None] * Wb.T[None, :, :]))
        _cache[key] = dict(
            archive=archive, ew=ew, eb=eb, D=D, RW=np.ascontiguousarray(RW),
            catW=catW, later=later,
            sb=[archive[f"block_{b}_s"].astype(np.float32) for b in range(nb)],
            bb=[archive[f"block_{b}_bias"].astype(np.float32) for b in range(nb)],
            ow=np.ascontiguousarray(
                archive["output_weight"].astype(np.float32)[:, :, :1]),
            ob=np.ascontiguousarray(
                archive["output_bias"].astype(np.float32)[:, :1]),
            k=k, db=db, nb=nb)
    C = _cache[key]
    k, db, nb, D = C["k"], C["db"], C["nb"], C["D"]
    out = np.empty(len(nums), np.float32)
    for a in range(0, len(nums), chunk_size):
        e = min(a + chunk_size, len(nums))
        B = e - a
        emb = np.maximum(nums[a:e][:, :, None] * C["ew"][None] + C["eb"][None], 0.0)
        h = (emb.reshape(B, D) @ C["RW"]).reshape(B, k, db)
        for j, cw in enumerate(C["catW"]):
            h += cw[cats[a:e, j]]
        h = np.maximum(h * C["sb"][0][None] + C["bb"][0][None], 0.0)
        for b in range(1, nb):
            h = np.matmul(h.transpose(1, 0, 2), C["later"][b - 1]).transpose(1, 0, 2)
            h = np.maximum(h * C["sb"][b][None] + C["bb"][b][None], 0.0)
        lg = np.einsum("bki,kio->bko", h, C["ow"]) + C["ob"][None]
        out[a:e] = _tabm_sigmoid(lg[:, :, 0]).mean(axis=1)
    return out


# ===== DIN-lite (numpy 전용) =====
def din_prep(X45, M):
    """X45: (n, 45) float64 — PP.transform_features 44열 + abs_regime."""
    ci, ni = M["ci"].astype(np.int64), M["ni"].astype(np.int64)
    Xc = np.zeros((len(X45), len(ci)), np.int64)
    for a, j in enumerate(ci):
        u = np.asarray(M[f"uval_{a}"], np.float64)
        col = X45[:, j]
        p = np.clip(np.searchsorted(u, col), 0, max(len(u) - 1, 0))
        Xc[:, a] = np.where((p < len(u)) & (u[p] == col), p + 1, 0)
    Xn = X45[:, ni].astype(np.float64)
    miss = np.isnan(Xn)
    Xn = np.where(miss, M["med"], Xn)
    Xn = ((Xn - M["mu"]) / M["sd"]).astype(np.float32)
    hn = M["hn"].astype(bool)
    if hn.any():
        Xn = np.concatenate([Xn, miss[:, hn].astype(np.float32)], 1)
    return Xn, Xc


def din_seq(pitcher_id, balls, strikes, batter_hand, M):
    """(투수 -> 2025 슬라이스) 조회 + 현재 행의 상황 질의."""
    pids = M["pids"].astype(np.int64)
    q = np.asarray(pitcher_id, np.int64)
    p = np.clip(np.searchsorted(pids, q), 0, max(len(pids) - 1, 0))
    hit = (p < len(pids)) & (pids[p] == q)
    idx = np.where(hit, p, 0)
    S = M["dep_state"][idx].astype(np.int64)
    C = M["dep_cell"][idx].astype(np.int64)
    Mk = M["dep_mask"][idx].astype(bool)
    # 처음 보는 투수는 이력 없음으로 둔다 (패딩 한 칸만 열어 둔다)
    S[~hit] = 0; C[~hit] = 0
    Mk[~hit] = False; Mk[~hit, 0] = True
    cell = np.clip(np.asarray(balls, np.int64) * 3
                   + np.asarray(strikes, np.int64), 0, 11) * 2 \
        + np.asarray(batter_hand, np.int64)
    cur = np.clip(cell, 0, 24) + 1
    return S, C, Mk, cur


def _relu(x):
    return np.maximum(x, 0.0)


def din_forward(Xn, Xc, S, C, Mk, cur, W, chunk=8192):
    out = np.empty(len(Xn), np.float64)
    ne = sum(1 for k in W.files if k.startswith("emb__"))
    for a in range(0, len(Xn), chunk):
        b = min(a + chunk, len(Xn))
        e = np.concatenate([W[f"emb__{i}__weight"][Xc[a:b, i]]
                            for i in range(ne)], 1)                  # (B,160)
        s = np.concatenate([W["se_state__weight"][S[a:b]],
                            W["se_cell__weight"][C[a:b]]], 2)        # (B,K,32)
        q = W["q_cell__weight"][cur[a:b]]                            # (B,16)
        K = s.shape[1]
        qe = np.repeat(q[:, None, :], K, 1)                          # (B,K,16)
        q2 = np.concatenate([qe, qe], 2)[:, :, :s.shape[2]]          # (B,K,32)
        h = np.concatenate([s, qe, s * q2], 2)                       # (B,K,80)
        h = _relu(h @ W["att__0__weight"].T + W["att__0__bias"])
        sc = h @ W["att__2__weight"].T + W["att__2__bias"]           # (B,K,1)
        sc = np.where(Mk[a:b][:, :, None], sc, -1e9)
        sc = sc - sc.max(1, keepdims=True)
        ex = np.exp(sc)
        att = ex / ex.sum(1, keepdims=True)
        pooled = (att * s).sum(1)                                    # (B,32)
        z = np.concatenate([Xn[a:b], e, pooled], 1)
        z = _relu(z @ W["mlp__0__weight"].T + W["mlp__0__bias"])
        z = _relu(z @ W["mlp__3__weight"].T + W["mlp__3__bias"])
        z = (z @ W["mlp__6__weight"].T + W["mlp__6__bias"]).ravel()
        out[a:b] = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    return out


# ===== MultiState 감사판 featurization (58열) =====
def msa_featurize(Xf, ordered, feat_names, plat_z, cm_z=None):
    """패키지 Xf(44열) + 원자료 ordered 에서 MS 감사판 58열을 만든다.

    feat_names: 학습이 저장한 열 순서 (44 + extras 12 + abs_regime + hand_pair)
    plat_z:     ms_plat_prior.npz (features44 규약의 2024까지 플래툰 편차)
    전부 그 행 안의 값만 쓴다. 다른 평가 행을 참조하지 않는다.
    """
    X44 = Xf[feat_names[:44]].to_numpy(dtype=np.float64).copy()
    # plat_dev 를 features44 규약(2024까지)으로 덮어쓴다
    j_plat = feat_names.index("plat_dev")
    pids = np.asarray(plat_z["pitcher_id"], np.int64)
    dev = np.asarray(plat_z["dev"], np.float64)
    pid = ordered["pitcher_id"].to_numpy(np.int64)
    hand = ordered["batter_hand"].to_numpy(np.int64)
    if len(pids):
        pos = np.searchsorted(pids, pid)
        safe = np.minimum(pos, len(pids) - 1)
        hit = (pos < len(pids)) & (pids[safe] == pid)
        X44[:, j_plat] = np.where(hit, np.where(hand == 1, dev[safe, 0],
                                                dev[safe, 1]), 0.0)
    # cm 3열도 features44 규약으로 덮어쓴다 (48칸 표 + 전역평균, 행 내부 연산)
    if cm_z is not None:
        cm = ((ordered["balls_before"].to_numpy(np.int64) * 3
               + ordered["strikes_before"].to_numpy(np.int64)) * 4
              + ordered["pitcher_hand"].to_numpy(np.int64) * 2
              + ordered["batter_hand"].to_numpy(np.int64))
        ck = np.asarray(cm_z["cm"], np.int64)
        cpos = np.searchsorted(ck, cm)
        csafe = np.minimum(cpos, len(ck) - 1)
        chit = (cpos < len(ck)) & (ck[csafe] == cm)
        lg = np.where(chit, np.asarray(cm_z["lg"], np.float64)[csafe], 0.0)
        rel = np.where(chit, np.asarray(cm_z["rel"], np.float64)[csafe], 1.0)
        gm = float(cm_z["gm"])
        X44[:, feat_names.index("lg_cm_eff")] = lg
        X44[:, feat_names.index("cm_rel")] = rel
        X44[:, feat_names.index("p_adj_cm")] =             gm + (X44[:, feat_names.index("p_is_succ")] - gm) * rel
    # 감사 extras — 학습(multistate_auditfeat.audit_features)과 같은 식
    p_n = ordered["asof_pitcher_n"].fillna(0.0).to_numpy(np.float64)
    b_n = ordered["asof_batter_n"].fillna(0.0).to_numpy(np.float64)
    mix_n = ordered["asof_pitcher_pitchmix_n"].fillna(0.0).to_numpy(np.float64)
    p_rate = ordered["asof_pitcher_success_rate"].fillna(.5).to_numpy(np.float64)
    b_rate = ordered["asof_batter_success_rate"].fillna(.5).to_numpy(np.float64)
    li = ordered["li"].fillna(1.0).to_numpy(np.float64)
    balls = ordered["balls_before"].to_numpy(np.float64)
    strikes = ordered["strikes_before"].to_numpy(np.float64)
    count12 = balls * 3.0 + strikes
    runners = ordered["base_state"].astype(str).str.count(r"[^_]") \
        .to_numpy(np.float64)
    p_is = X44[:, feat_names.index("p_is_succ")]
    hp = ordered["pitcher_hand"].fillna(0).to_numpy(np.int64)
    hb = ordered["batter_hand"].fillna(0).to_numpy(np.int64)
    hand_pair = np.where((hp >= 1) & (hp <= 2) & (hb >= 1) & (hb <= 2),
                         (hp - 1) * 2 + (hb - 1), -1).astype(np.float64)
    lp, lb, lm = np.log1p(p_n), np.log1p(b_n), np.log1p(mix_n)
    extras = {
        "log_p_n": lp, "log_b_n": lb, "log_mix_n": lm,
        "p_rate_x_log_pn": p_rate * lp, "b_rate_x_log_bn": b_rate * lb,
        "li_x_count12": li * count12, "li_x_runners": li * runners,
        "li_x_p_is": li * p_is,
        "li_x_hand00": li * (hand_pair == 0), "li_x_hand01": li * (hand_pair == 1),
        "li_x_hand10": li * (hand_pair == 2), "li_x_hand11": li * (hand_pair == 3),
    }
    isf = ordered["game_type"].astype(str).to_numpy() == "F"
    cols = [X44]
    for nm in feat_names[44:]:
        if nm == "abs_regime":
            cols.append(np.where(isf, 2.0, 3.0)[:, None])
        elif nm == "hand_pair":
            cols.append(hand_pair[:, None])
        else:
            cols.append(extras[nm][:, None])
    return np.concatenate(cols, axis=1)


def predict_hgb(X, z, chunk=4096):
    """HistGB 트리(npz)를 numpy 로 순회해 브랜치별 확률 (n, 3) 을 돌려준다.

    sklearn HistGradientBoostingClassifier 의 예측을 그대로 재현한다.
      분기    값 <= num_threshold 면 왼쪽
      결측    missing_go_to_left 플래그
      raw     baseline + 잎값 합,  proba = sigmoid(raw)
    내보낼 때 sklearn 원본과 최대차 5e-16 으로 일치하는 것을 확인했다.

    노드 배열을 1차원으로 펴서 1200그루를 한 번에 순회하고, 매 단계
    '아직 잎에 안 닿은' 위치만 남긴다. submit_jaemin_2(958) 에서 쓰던 방식이다.
    """
    T, M = z["feat"].shape
    fl = z["feat"].ravel().astype(np.int64)
    tl = z["thr"].ravel()
    ll = z["left"].ravel().astype(np.int64)
    rl = z["right"].ravel().astype(np.int64)
    lfl = z["leaf"].ravel().astype(bool)
    mgl_ = z["mgl"].ravel().astype(bool)
    vl = z["val"].ravel()
    base = z["baselines"]
    depth = int(z["max_depth"])
    n_models = int(z["n_models"])
    per = T // n_models
    root = np.arange(T, dtype=np.int64) * M

    n, Fw = X.shape
    Xf = np.ascontiguousarray(X, dtype=np.float64).ravel()
    out = np.zeros((n, n_models), np.float64)
    off0 = np.repeat(np.arange(chunk, dtype=np.int64) * Fw, T)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        m = e - s
        idx = np.tile(root, m)
        off = off0[:m * T] + s * Fw
        act = np.arange(m * T, dtype=np.int64)
        for _ in range(depth + 1):
            nd = idx[act]
            keep = ~lfl[nd]
            if not keep.any():
                break
            act = act[keep]
            nd = nd[keep]
            v = Xf[off[act] + fl[nd]]
            go_left = np.where(np.isnan(v), mgl_[nd], v <= tl[nd])
            idx[act] = np.where(go_left, ll[nd], rl[nd])
        raw = vl[idx].reshape(m, n_models, per).sum(axis=2)
        out[s:e] = 1.0 / (1.0 + np.exp(-(raw + base[None, :])))
    return out


def route_hgb(pb, is_f, w_branch=0.4):
    """0.6 x all + 0.4 x (경기유형별). 리더보드 989 를 만든 라우팅이다."""
    return (1.0 - w_branch) * pb[:, 0] + w_branch * np.where(is_f, pb[:, 1],
                                                             pb[:, 2])


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


def merge_predictions(sub, ids, preds):
    """sample_submission 의 row_id 순서에 맞춰 예측 병합."""
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =====================================================================
# main
# =====================================================================

def main():
    print(f"[env] numpy={np.__version__}  pandas={pd.__version__}")
    print(f"[env] cwd={os.getcwd()}")
    print(f"[env] script={os.path.abspath(__file__)}")
    print(f"[env] cwd 내용={sorted(os.listdir(os.getcwd()))[:30]}")

    TEST_PATH = resolve("data/test.csv")
    SAMPLE_SUB_PATH = resolve("data/sample_submission.csv")
    OUT_PATH = os.path.join(os.getcwd(), "output", "submission.csv")

    print("Load model...")
    history, z, shift = load_model()
    print(f" OK. models={int(z['n_models'])}  trees={z['cb_leaf'].shape[0]}  "
          f"features={len(history['features'])}  shift={shift:+.4f}")
    print(f" 앙상블 비중  CatBoost {W_CB:.2f} / HistGB {W_HGB:.2f} / TabM {W_TABM:.2f}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, history)
    print(f" features={X.shape[1]}  rows={X.shape[0]}")

    print("Inference...")
    if len(X):
        p_cb_new = predict_numpy(X.values.astype(np.float64), z)
        p_cb = p_cb_new
        if "_base_z" in history:
            base_z = history["_base_z"]
            base_features = history["_base_features"]
            p_cb_old = predict_numpy(X[base_features].values.astype(np.float64), base_z)
            p_cb = W_CB_NEW * p_cb_new + (1.0 - W_CB_NEW) * p_cb_old
            print(f" CatBoost new50={p_cb_new.mean():.4f} incumbent44={p_cb_old.mean():.4f} "
                  f"mix={p_cb.mean():.4f} ({W_CB_NEW:.2f}/{1.0-W_CB_NEW:.2f})")
        else:
            print(f" CatBoost mean={p_cb.mean():.4f}")

        # ---------------- HistGB (CatBoost 와 같은 44열 행렬을 쓴다)
        #
        # 리더보드에서 HistGB 60+40 = 989, CatBoost 60+40 = 987 로 동점이었다.
        # 둘은 같은 계열이라 서로 대체재에 가깝지만, TabM 과의 상관이
        # HistGB 0.9031 < CatBoost 0.9492 로 더 떨어져 있어 자리가 있다.
        # 시즌가중은 2.0 을 쓴다 — 단독은 1.0 이 높지만(869.9 vs 854.1)
        # 2.0 쪽이 TabM 과 덜 겹쳐 혼합에서는 더 낫다(851.5 vs 848.9).
        gt = test["game_type"].astype(str).to_numpy()
        is_f_t = gt == "F"
        if W_HGB > 0.0:
            zh = np.load(resolve("model/hgb.npz"), allow_pickle=False)
            p_hgb = route_hgb(predict_hgb(X.values.astype(np.float64), zh),
                              is_f_t)
            print(f" HistGB   mean={p_hgb.mean():.4f}  "
                  f"상관={np.corrcoef(p_cb, p_hgb)[0, 1]:.4f}")
        else:
            # 비중이 0 이면 트리 12,000개 순회를 통째로 건너뛴다.
            # 예산이 10분인데 submit_25 가 9분 26초였다.
            p_hgb = np.zeros_like(p_cb)
            print(" HistGB   건너뜀 (비중 0)")

        # ---------------- TabM — 지인 44피처
        #
        # 라우팅을 행 종류로 나눈다.
        #   1군  행: base 3브랜치  0.6 x all + 0.4 x regular   (submit_18 그대로)
        #   퓨처스행: regime 판본  0.6 x all + 0.4 x futures
        #
        # 체제 처리는 퓨처스를 크게 올리고 1군을 내린다(관문 4시드 짝비교:
        # 퓨처스 +157.6 t=12.4 4/4, 1군 -18.8 0/4). 퓨처스가 11.8% 뿐이라
        # 전체에 걸면 희석돼 +4.7 로 무의미해진다. 그래서 퓨처스에만 건다.
        # 관문에서 TabM 864.0 -> 879.2, 퓨처스 행만 492.5 -> 630.1.
        import preprocess as PPF
        with open(resolve("model/history.json"), encoding="utf-8") as f:
            hist_f = PPF.deserialize_history(json.load(f))
        ordered = PPF.sort_by_row_id(test)
        Xf = PPF.build_inference_features(ordered, hist_f)
        isf_o = ordered["game_type"].astype(str).to_numpy() == "F"

        # ---- TabM: 세 브랜치 모두 **같은 45열(c4 포함)** 위에서 돈다.
        #
        # submit_34 까지는 1군 경로가 지인 코드 산출물(44열, 체제 열 없음)이었고
        # 퓨처스 경로만 45열이었다. 즉 all 모델을 두 벌 썼다.
        # 그런데 우리 관문은 **늘 1군 브랜치에도 c4 를 붙여** 재왔다 —
        # 88.2% 가 검증된 적 없는 구성으로 돌고 있었다는 뜻이다.
        #
        # c4_1gun.py 로 그 차이를 직접 쟀다 (배치 구성 vs 관문 구성, 3시드 짝비교)
        #     VS=2022  1군 +8.0 +-1.3  t=6.32  3/3     전체 +6.8  t=5.41  3/3
        #     VS=2024  1군 +22.9 +-7.5 t=3.04  3/3     전체 +20.4 t=3.09  3/3
        # 퓨처스 행 불변량은 두 폴드 모두 0.00e+00 이라 라우팅은 정상이었다.
        #
        # 퓨처스 경로에서 같은 불일치를 고쳤을 때 관문 +2.0 -> 리더보드 +3 이었다
        # (submit_34, 1069 -> 1072). 전이율 1.5.
        #
        # 부수 효과로 순전파가 하나 줄어든다.
        #     전:  all_tabm(전체) + regular_tabm(1군) + f2b_all(퓨처스) + f2b_fut(퓨처스)
        #     후:  f2b_all(전체)  + f2b_regular(1군)  + f2b_fut(퓨처스)
        c4 = np.where(isf_o, 2.0, 3.0)
        Xv = np.c_[Xf.to_numpy(dtype=np.float64), c4]

        def _f2b(branch, rows):
            """f2b 모델 하나를 rows 에 대해 돌린다. 시드는 F2B_SEEDS."""
            outs = []
            for sd in F2B_SEEDS:
                z2 = np.load(resolve(f"model/f2b_{branch}_s{sd}.npz"),
                             allow_pickle=False)
                mt2 = json.loads(str(z2["meta"].item()))
                mt2["cat_cardinalities"] = mt2["cards"]
                want = [str(c) for c in mt2["features"]]
                if want[:-1] != list(Xf.columns) or want[-1] != "abs_regime":
                    raise ValueError(f"f2b_{branch} 피처가 preprocess 와 어긋난다")
                # 이름만 보면 못 잡는다 — 이진(0/1) 판본도 같은 이름을 쓴다.
                kk = f"catkey_{len(mt2['cards']) - 1}"
                got = sorted(float(v) for v in z2[kk])
                if got != [0.0, 1.0, 2.0, 3.0]:
                    raise ValueError(f"체제 열 후보값이 4단계가 아니다: {got}")
                Tn, Tc = _fm_prep(Xv[rows], z2)
                outs.append(_tabm_forward(Tn, Tc, mt2, z2, 4096))
            return np.mean(outs, axis=0)

        all_rows = np.arange(len(Xf))
        p_ord = _f2b("all", all_rows)
        rows_r = np.flatnonzero(~isf_o)
        if len(rows_r):
            p_ord[rows_r] = 0.6 * p_ord[rows_r] + 0.4 * _f2b("regular", rows_r)
        rows_f = np.flatnonzero(isf_o)
        if len(rows_f):
            p_ord[rows_f] = 0.6 * p_ord[rows_f] + 0.4 * _f2b("futures", rows_f)

        p_ord = np.clip(p_ord, 0.0, 1.0)
        tm = dict(zip(ordered[ID_COL].tolist(), p_ord))
        p_tabm = np.array([tm[r] for r in test[ID_COL].tolist()], dtype=np.float64)
        print(f" TabM     mean={p_tabm.mean():.4f}  퓨처스 {int(is_f_t.sum()):,}행  "
              f"상관={np.corrcoef(p_cb, p_tabm)[0, 1]:.4f}")

        preds = W_CB * p_cb + W_HGB * p_hgb + W_TABM * p_tabm
        print(f" blended  mean={preds.mean():.4f}  "
              f"({W_CB:.2f}/{W_HGB:.2f}/{W_TABM:.2f})")

        # ---------------- DIN-lite (확률 공간, 관문과 같은 형태)
        if W_DIN > 0.0:
            DM = np.load(resolve("model/din_meta.npz"), allow_pickle=False)
            want = [str(c) for c in DM["cols"]]
            if want[:-1] != list(Xf.columns) or want[-1] != "abs_regime":
                raise ValueError("DIN 피처가 preprocess 와 어긋난다")
            X45 = np.c_[Xf.to_numpy(dtype=np.float64),
                        np.where(isf_o, 2.0, 3.0)]
            dXn, dXc = din_prep(X45, DM)
            dS, dC, dMk, dcur = din_seq(
                ordered["pitcher_id"].to_numpy(),
                ordered["balls_before"].to_numpy(),
                ordered["strikes_before"].to_numpy(),
                ordered["batter_hand"].to_numpy(), DM)
            acc = {}
            for br in ("all", "regular", "futures"):
                acc[br] = np.mean([
                    din_forward(dXn, dXc, dS, dC, dMk, dcur,
                                np.load(resolve(f"model/din_{br}_s{sd}.npz"),
                                        allow_pickle=False))
                    for sd in DIN_SEEDS], axis=0)
            p_din_ord = np.where(isf_o, 0.6*acc["all"] + 0.4*acc["futures"],
                                 0.6*acc["all"] + 0.4*acc["regular"])
            dmap = dict(zip(ordered[ID_COL].tolist(), p_din_ord))
            p_din = np.array([dmap[r] for r in test[ID_COL].tolist()], np.float64)
            print(f" DIN      mean={p_din.mean():.4f}  "
                  f"상관={np.corrcoef(preds, p_din)[0,1]:.4f}  w={W_DIN:.2f}")
            preds = (1.0 - W_DIN) * preds + W_DIN * p_din
            print(f" +DIN     mean={preds.mean():.4f}")

        # ---------------- MultiState 감사판 (4원 혼합으로 재구성)
        # 53: 1군 행은 state6(보조 5범주: other-실패 ball/strike 분할), 퓨처스 행은
        # ms_s2 그대로. 행마다 한 계열만 통과하므로 추론량은 52 와 같다.
        # 관문: 2024 +4.5 / 2023(1군) +6.0 / 2022 -1.6 (단독1군 3/3 비열세).
        if W4_MS > 0.0:
            z1 = np.load(resolve("model/msa_s42.npz"), allow_pickle=False)
            feat_names = [str(c) for c in json.loads(str(z1["meta"].item()))["features"]]
            plat_z = np.load(resolve("model/ms_plat_prior.npz"), allow_pickle=False)
            cm_z = np.load(resolve("model/msa_cm48.npz"), allow_pickle=False)
            X58 = msa_featurize(Xf, ordered, feat_names, plat_z, cm_z)
            p_ms_ord = np.empty(len(ordered), np.float64)
            for rows, fam, center in (
                    (np.flatnonzero(~isf_o), "msa6", MS6_CENTER),
                    (np.flatnonzero(isf_o), "msa", MS_CENTER)):
                if len(rows) == 0:
                    continue
                acc_ms = []
                Tn = Tc = None
                for sd in MS_SEEDS:
                    zz = np.load(resolve(f"model/{fam}_s{sd}.npz"),
                                 allow_pickle=False)
                    mt = json.loads(str(zz["meta"].item()))
                    if Tn is None:      # 시드끼리 전처리 통계가 같다 (같은 학습)
                        Tn, Tc = _fm_prep(X58[rows], zz)
                    acc_ms.append(_tabm_forward(Tn, Tc, mt, zz, 4096))
                # 계열별 수준 중심화 상수 (학습 폴드 유도, 행 단위 상수 연산)
                q = np.clip(np.mean(acc_ms, axis=0), 1e-6, 1 - 1e-6)
                p_ms_ord[rows] = 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q))
                                                       + center)))
            mmap = dict(zip(ordered[ID_COL].tolist(), p_ms_ord))
            p_ms = np.array([mmap[r] for r in test[ID_COL].tolist()], np.float64)
            print(f" MS       mean={p_ms.mean():.4f}  "
                  f"상관={np.corrcoef(preds, p_ms)[0,1]:.4f}")
            # 54: cold-투수(학습에 없던 pitcher_id) 행은 cb 강화 가중.
            # ID 임베딩 암기가 무효한 행 — 관문 3폴드 +1.0/+2.8/+0.7 (전부 양수).
            wz = np.load(resolve("model/warm_ids.npz"), allow_pickle=False)
            warm_set = set(int(v) for v in wz["pitcher_id"])
            is_cold = np.array([int(v) not in warm_set
                                for v in test["pitcher_id"].tolist()])
            p_warm = (W4_CB * p_cb + W4_TAB * p_tabm + W4_DIN * p_din
                      + W4_MS * p_ms)
            p_cold = (CW_CB * p_cb + CW_TAB * p_tabm + CW_DIN * p_din
                      + CW_MS * p_ms)
            preds = np.where(is_cold, p_cold, p_warm)
            print(f" 4원 혼합  mean={preds.mean():.4f}  cold행 {is_cold.mean():.3f}  "
                  f"({W4_CB:.2f}/{W4_TAB:.2f}/{W4_DIN:.2f}/{W4_MS:.2f})")
            # 63: 투수x카운트 상대효과 보정 — Δ = logit(TE(투수,카운트)) - logit(TE(투수)),
            # 계층 수축 κ=300/300, 학습 동결 lookup (행 단위, 규칙4 안전).
            # 유일하게 교차연도 부호 일관인 잔차 신호 (스크린 +0.0035/+0.0044/+0.0056).
            # 관문 β스윕 고원: κ300 β0.12 ≈ +1.5/+1.8/+2.1 (3/3), 이웃 전부 양수.
            pz = np.load(resolve("model/pcnt.npz"), allow_pickle=False)
            pcnt_map = {(int(a), int(b)): float(c) for a, b, c in
                        zip(pz["pitcher_id"], pz["cnt"], pz["delta"])}
            cnt_t = (test["balls_before"].to_numpy(int) * 3
                     + test["strikes_before"].to_numpy(int))
            dv = np.array([pcnt_map.get((int(pv), int(cv)), 0.0)
                           for pv, cv in zip(test["pitcher_id"].tolist(),
                                             cnt_t.tolist())])
            po_map = {(int(a), int(b)): float(c) for a, b, c in
                      zip(pz["po_pitcher_id"], pz["po_outs"], pz["po_delta"])}
            dv2 = np.array([po_map.get((int(pv), int(ov)), 0.0)
                            for pv, ov in zip(test["pitcher_id"].tolist(),
                                              test["outs_before"].tolist())])
            bc_map = {(int(a), int(b)): float(c) for a, b, c in
                      zip(pz["bc_batter_id"], pz["bc_cnt"], pz["bc_delta"])}
            dv3 = np.array([bc_map.get((int(bv), int(cv)), 0.0)
                            for bv, cv in zip(test["batter_id"].tolist(),
                                              cnt_t.tolist())])
            cz = np.load(resolve("model/csf.npz"), allow_pickle=False)
            _mu = float(cz["mu"][0])
            _n0 = dict(zip(cz["pitcher_id"].tolist(), cz["n_end"].tolist()))
            _s0 = dict(zip(cz["pitcher_id"].tolist(), cz["s_end"].tolist()))
            _na = test["asof_pitcher_n"].fillna(0.0).to_numpy(np.float64)
            _ra = test["asof_pitcher_success_rate"].fillna(_mu).to_numpy(np.float64)
            _pid = test["pitcher_id"].tolist()
            _b0 = np.array([_n0.get(int(v), 0.0) for v in _pid])
            _c0 = np.array([_s0.get(int(v), 0.0) for v in _pid])
            _cn = np.clip(_na - _b0, 0, None)
            _cs = np.clip(_na * _ra - _c0, 0, None)
            _car = (_na * _ra + 300.0 * _mu) / (_na + 300.0)
            _csr = (_cs + 50.0 * _car) / (_cn + 50.0)
            def _lgx(a):
                a = np.clip(a, 1e-4, 1 - 1e-4)
                return np.log(a / (1 - a))
            dv4 = (_cn / (_cn + 100.0)) * (_lgx(_csr) - _lgx(_car))
            isf_t = test["game_type"].astype(str).to_numpy() == "F"
            _t13p = (test["pitcher_team_id"].to_numpy() == 13) & ~isf_t
            _gm = test["game_month"].to_numpy()
            _t13b = (test["batter_team_id"].to_numpy() == 13) & ~isf_t
            _cz = np.load(resolve("model/comp_rev.npz"), allow_pickle=False)
            _cmap = dict(zip(_cz["keys"].tolist(), _cz["dev"].astype(float).tolist()))
            _ck = (test["pitcher_id"].to_numpy().astype("int64") * 1000
                   + test["batter_hand"].to_numpy().astype("int64") * 100
                   + test["balls_before"].to_numpy().astype("int64") * 3
                   + test["strikes_before"].to_numpy().astype("int64"))
            dv5 = np.array([_cmap.get(int(k), 0.0) for k in _ck])
            zc = np.clip(preds, 1e-6, 1 - 1e-6)
            zmix = (np.log(zc / (1 - zc)) + PCNT_BETA * dv
                    + POUT_BETA * dv2 + BCNT_BETA * dv3
                    + CSF_BETA * np.where(isf_t, dv4, 0.0)
                    + COMPREV_BETA * dv5
                    + np.where(_t13p & (_gm <= 4), +0.105, 0.0)
                    + np.where(_t13p & (_gm >= 5) & (_gm <= 6), +0.08, 0.0)
                    + np.where(_t13p & (_gm >= 7), +0.055, 0.0)
                    + np.where(_t13b, +0.06, 0.0))
            # 66: 합의-샤픈 — 4계열 로짓이 행 내부 중앙값에서 SHARP_TH 미만으로
            # 모이면(합의) 로짓 편차를 SHARP_G 배. 행 내부 연산 + 학습 상수만
            # 사용 (규칙4 안전). 관문(65 위) +0.76/+2.50/+0.95, 이웃 18칸 전부 양수.
            def _lg(a):
                a = np.clip(a, 1e-6, 1 - 1e-6)
                return np.log(a / (1 - a))
            Z4 = np.column_stack([_lg(p_cb), _lg(p_tabm), _lg(p_din),
                                  _lg(p_ms)])
            iso = np.abs(Z4 - np.median(Z4, axis=1, keepdims=True)).max(1)
            zmix = np.where(iso < SHARP_TH,
                            SHARP_Z0 + SHARP_G * (zmix - SHARP_Z0), zmix)
            preds = 1.0 / (1.0 + np.exp(-zmix))
            print(f" pcnt보정  커버 {np.mean(dv != 0):.3f}  β={PCNT_BETA}  "
                  f"pout커버 {np.mean(dv2 != 0):.3f}  β2={POUT_BETA}")
        preds = apply_calibration(preds, shift)
        print(f" calibrated mean={preds.mean():.4f}  T={CALIB_T}")
    else:
        preds = []

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")

    # 작업 디렉터리와 script.py 위치가 다를 경우를 대비해 양쪽에 남긴다.
    alt = os.path.join(HERE, "output", "submission.csv")
    if os.path.abspath(alt) != os.path.abspath(OUT_PATH):
        save_submission(alt, sub)
        print(f"Saved(alt): {alt}")


if __name__ == "__main__":
    main()
