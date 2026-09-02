#!/bin/bash
# Finite GPU job wrapper for the remote Vast instance.
# The supervisor configuration keeps the process visible and its output durable.
set -euo pipefail

source /venv/main/bin/activate
cd /workspace/aimers
export PYTHONPATH=/workspace/aimers:/workspace/aimers/colab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u colab/reg_weight.py 2>&1 | tee -a /workspace/aimers/out/regw.log
exit ${PIPESTATUS[0]}
