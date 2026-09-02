#!/bin/bash
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/aimers/out
while [ ! -f $O/FS2B_DONE ]; do sleep 30; done
while pgrep -f "[f]ut_stage2b.py" >/dev/null; do sleep 20; done
/venv/main/bin/python -u fut_residual.py > $O/futres.log 2>&1 || true
echo DONE > $O/FR_DONE
