#!/bin/bash
# 서버 환경 복구. vast.ai 인스턴스를 껐다 켜면 pip 설치분이 날아간다.
# /workspace 는 볼륨이라 data/ 와 out/ 은 남지만 패키지와 코드는 다시 올려야 한다.
#
# 사용법 (로컬에서)
#   bash colab/restore_server.sh <포트> <호스트>
# 예)
#   bash colab/restore_server.sh 17030 91.150.160.38
set -e
PORT="${1:?포트를 넣어라}"
HOST="${2:?호스트를 넣어라}"
SSH="ssh -F /dev/null -o StrictHostKeyChecking=no -p $PORT root@$HOST"
SCP="scp -F /dev/null -o StrictHostKeyChecking=no -P $PORT"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 접속 확인 =="
$SSH "hostname; nvidia-smi --query-gpu=name --format=csv,noheader"

echo "== 패키지 =="
# torch 는 드라이버에 맞춰 cu121. tabm/rtdl 은 학습에만 필요하다(제출본은 numpy 뿐).
$SSH "pip install --break-system-packages -q numpy pandas scipy scikit-learn && \
      pip install --break-system-packages -q torch --index-url https://download.pytorch.org/whl/cu121 && \
      pip install --break-system-packages -q tabm rtdl_num_embeddings && \
      python3 -c 'import torch,pandas,numpy; from tabm import TabM; \
                  print(\"OK\", torch.__version__, torch.cuda.is_available())'"

echo "== 코드 =="
$SSH "mkdir -p /workspace/aimers/colab /workspace/aimers/out /workspace/aimers/data"
cd "$ROOT"
tar -cf /tmp/aimers_code.tar colab train_chan_3
$SCP /tmp/aimers_code.tar "root@$HOST:/workspace/aimers/"
$SSH "cd /workspace/aimers && tar -xf aimers_code.tar && ls colab | wc -l"

echo "== 데이터 =="
# /workspace 볼륨이 살아 있으면 건너뛴다. 349MB+369MB 라 매번 올리면 느리다.
if $SSH "test -f /workspace/aimers/data/train.csv"; then
  echo "  data/ 이미 있음"
else
  echo "  data/ 업로드 (수 분 걸림)"
  $SCP "$ROOT/open (1)/data/train.csv" "$ROOT/open (1)/data/test.csv" \
       "$ROOT/open (1)/data/sample_submission.csv" "root@$HOST:/workspace/aimers/data/"
fi

echo "== 관문에 필요한 예측 파일 =="
# cb_gate.npy / mlp_gate.npy 는 colab/ 안에 있어 코드와 같이 올라간다.
# out/mlpnew_flat_s3.npy 는 out/ 에 있어야 한다. 없으면 로컬 사본에서 올린다.
if ! $SSH "test -f /workspace/aimers/out/mlpnew_flat_s3.npy"; then
  if [ -f "$ROOT/colab/_dl/mlpnew_flat_s3.npy" ]; then
    $SCP "$ROOT/colab/_dl/mlpnew_flat_s3.npy" "root@$HOST:/workspace/aimers/out/"
  else
    echo "  !! out/mlpnew_flat_s3.npy 없음 — flatMLP 관문 예측을 다시 만들어야 한다"
  fi
fi

echo "== 확인 =="
$SSH "cd /workspace/aimers && ls data/ && ls out/*.npy 2>/dev/null | wc -l && \
      python3 -c 'import sys; sys.path.insert(0,\"colab\"); import features44; print(\"features44 OK\")'"
echo "복구 완료"
