#!/bin/bash
# 제출용 TabM — 2019~2024 전체 학습(Stage1 4ep) + 2024 fine-tune(Stage2 1ep)
#
# 왜 ep4 인가
#   관문(2019~2023 -> 2024)에서 Stage1 epoch 을 바꿔가며 쟀다.
#       ep2(지인 기본)  CB상관 0.9708  앙상블 +1.3
#       ep4             CB상관 0.9226  앙상블 +18.1   <- 채택
#       ep8             CB상관 0.8396  앙상블 +16.2
#       ep16            CB상관 0.7515  단독 붕괴
#   2 epoch 은 underfit 이라 가장 강한 신호(투수 실력)만 배우는데 그건 CatBoost 도
#   배우는 것이다. 더 학습해야 신경망 고유의 것을 배우고 그게 다양성이 된다.
#   지인은 단독 성능 기준으로 2 를 골랐지만 앙상블 재료의 기준은 다르다.
#
# 나머지는 지인 파이프라인 그대로다 (검증된 경로: 리더보드 1047).
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/aimers
export PYTHONPATH=/workspace/aimers

echo "### 학습시작 $(date +%H:%M:%S)"
python -u code/train_conditional.py \
  --data-dir  /workspace/aimers/data \
  --config    /workspace/aimers/code/selected_config.json \
  --output-dir /workspace/aimers/artifacts \
  --device cuda \
  --stage1-epochs 4 \
  --stage2-epochs 1 \
  --fine-tune-scope last_block

echo "### 학습완료 $(date +%H:%M:%S)"
python -u code/build_submission.py \
  --artifacts-dir /workspace/aimers/artifacts \
  --data-dir      /workspace/aimers/data \
  --output-dir    /workspace/aimers/submit_tabm

echo "### 추출완료 $(date +%H:%M:%S)"
ls -la /workspace/aimers/submit_tabm/model/
du -sh /workspace/aimers/submit_tabm
