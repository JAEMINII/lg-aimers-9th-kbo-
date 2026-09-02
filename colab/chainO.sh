#!/bin/bash
# 야간 2단. chainN 이 끝난 뒤 임베딩 sigma 스윕을 돌린다.
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/aimers/out
while [ ! -f $O/NIGHT_DONE ]; do sleep 30; done
/venv/main/bin/python -u emb_freq.py > $O/embfreq.log 2>&1 || true
echo DONE > $O/EMB_DONE
