#!/bin/bash
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/aimers/out
/venv/main/bin/python -u reg_stage2.py > $O/regs2.log 2>&1 || true
echo DONE > $O/RG_DONE
