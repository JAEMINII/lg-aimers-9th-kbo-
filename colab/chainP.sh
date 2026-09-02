#!/bin/bash
# 야간 3단. emb_freq 가 끝난 뒤 용량(k x d_block) 스윕.
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/aimers/out
while [ ! -f $O/EMB_DONE ]; do sleep 30; done
/venv/main/bin/python -u cap_sweep.py > $O/cap.log 2>&1 || true
echo DONE > $O/CAP_DONE
