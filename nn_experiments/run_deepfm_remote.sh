#!/usr/bin/env bash
set -u

# Wait for the currently running TabM-packed check, then run both temporal folds.
while pgrep -x -f '/venv/main/bin/python -u /root/arch_sweep.py' >/dev/null 2>&1; do
  sleep 20
done

mkdir -p /root/modelcheck/deepfm/vs2022 /root/modelcheck/deepfm/vs2024
for V in 2022 2024; do
  VS="$V" \
  AIMERS_DATA="/root/open (1)/data" \
  AIMERS_OUT="/root/modelcheck/deepfm/vs${V}" \
  /venv/main/bin/python -u /root/deepfm_resid.py \
    > "/root/modelcheck/deepfm_vs${V}.log" 2>&1 || true
done
echo "DEEPFM_DONE $(date -Is)"
