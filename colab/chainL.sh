#!/bin/bash
# dropped_cols 완료 플래그를 기다린다. 프로세스 유무로 보면 chainK 가 다음
# 작업을 띄우기 전 30초 공백에 끼어들어 3060 에서 OOM 이 난다.
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f /workspace/aimers/out/DP_DONE ]; do sleep 30; done
while pgrep -f "[d]ropped_cols.py" >/dev/null; do sleep 20; done
/venv/main/bin/python -u pcdev_gate.py > /workspace/aimers/out/pcdev.log 2>&1
echo DONE > /workspace/aimers/out/PCD_DONE
