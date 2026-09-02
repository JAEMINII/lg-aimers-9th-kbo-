#!/bin/bash
# 야간 대기열. cm_gate 완료 플래그를 기다린 뒤 스윕 셋을 순차 실행한다.
# 한 스크립트 안에서 이어 돌려야 사이에 GPU 가 비지 않는다.
# 하나가 죽어도 다음으로 넘어간다 (|| true).
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python
O=/workspace/aimers/out

while [ ! -f $O/CM_DONE ]; do sleep 30; done
while pgrep -f "[c]m_gate.py" >/dev/null; do sleep 20; done

$PY -u sweep_runner.py oldw > $O/oldw.log 2>&1 || true
echo DONE > $O/OLDW_DONE
$PY -u sweep_runner.py loss > $O/loss.log 2>&1 || true
echo DONE > $O/LOSS_DONE
$PY -u sweep_runner.py bslr > $O/bslr.log 2>&1 || true
echo DONE > $O/BSLR_DONE
echo ALL > $O/NIGHT_DONE
