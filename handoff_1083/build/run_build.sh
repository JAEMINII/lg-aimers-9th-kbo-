#!/bin/bash
# submit_41 의 CatBoost 경로(model/trees.npz)를 다시 만든다.
#
# 주의 — 이 명령으로 **바이트 동일한 파일은 안 나온다.**
#   실린 trees.npz 는 CatBoost GPU(task_type="GPU") 로 학습됐고,
#   CatBoost GPU 는 장치가 다르면 트리가 달라진다. CPU 로 돌리면 아예 다르다
#   (실측: CPU seed 1/42 로 재현 시 분할피처 일치율 4.8% = 우연 수준).
#   1083 을 그대로 받으려면 handoff_1083/submit_jaemin_41.zip 을 그냥 내라.
#   이 스크립트는 **개조하거나 다시 학습할 때** 쓰는 것이다.
#
# 필요한 것   NVIDIA GPU + catboost, open (1)/data/train.csv, features44.py
# 걸리는 시간 RTX 3060 기준 30모델(10시드 x 3그룹) 약 20~30분
set -e
cd "$(dirname "$0")"
export AIMERS_ROOT="${AIMERS_ROOT:-$(cd ../.. && pwd)}"
export C12_OUT="${C12_OUT:-$AIMERS_ROOT/c12_out}"
export CB_SEEDS="${CB_SEEDS:-1,2,3,4,5,6,7,8,9,10}"
export INCLUDE_CMH=1          # pc_cmh_* 3열을 켠다. 이게 없으면 47열 모델이 된다
# export C12_CPU=1            # GPU 가 없을 때만. 다른 모델이 나온다
python -u train_c12_submit.py
echo
echo "산출물  $C12_OUT/trees_c12.npz"
echo "설치    cp \$C12_OUT/trees_c12.npz <패키지>/model/trees.npz"
echo "        (기존 44열 CatBoost 는 model/trees_base.npz 로 남겨둬야 한다)"
