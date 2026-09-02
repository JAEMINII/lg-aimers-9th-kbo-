#!/bin/bash
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/aimers/out
/venv/main/bin/python -u aux_target.py > $O/auxt.log 2>&1 || true
echo DONE > $O/AT_DONE
