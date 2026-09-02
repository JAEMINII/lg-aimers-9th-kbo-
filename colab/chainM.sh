#!/bin/bash
cd /workspace/aimers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "[p]cdev_gate.py" >/dev/null; do sleep 30; done
/venv/main/bin/python -u cm_gate.py > /workspace/aimers/out/cm.log 2>&1
echo DONE > /workspace/aimers/out/CM_DONE
