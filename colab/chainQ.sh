#!/bin/bash
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/aimers/out
while [ ! -f $O/CAP_DONE ]; do sleep 30; done
/venv/main/bin/python -u fut_stage2.py > $O/futs2.log 2>&1 || true
echo DONE > $O/FS2_DONE
