#!/bin/bash
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/aimers/out
/venv/main/bin/python -u fut_stage2b.py > $O/futs2b.log 2>&1 || true
echo DONE > $O/FS2B_DONE
