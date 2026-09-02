#!/bin/bash
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/aimers/out
/venv/main/bin/python -u reg_weight.py > $O/regw.log 2>&1 || true
echo DONE > $O/RW_DONE
