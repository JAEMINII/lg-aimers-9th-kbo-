#!/usr/bin/env bash
set -u
cd /root
source /venv/main/bin/activate
LOG=/root/overnight_nontm.log
ROOT=/root
CASES="base_hist,c12_cmh_hist,c12_cmh_log,c12_co,c12_ct,c12_cb,c12_cmh_co,c12_cmh_cg,c12_cmh_ci"
mkdir -p /root/overnight_cb_nontm
exec >> "$LOG" 2>&1
echo "===== non-TrackMan queue start $(date -Is) ====="

for vs in 2021 2022 2023 2024; do
  out="/root/overnight_cb_nontm/more/vs$vs"
  mkdir -p "$out"
  echo "--- MORE cases VS=$vs $(date -Is) ---"
  timeout 2400s env AIMERS_ROOT="$ROOT" SAVE_DIR="$out" MORE=1 ONLY="$CASES" \
    CB_SEEDS=1,2,3,4,5,6,7,8,9,10 \
    python /root/pc_gpu_sweep_nontm.py "$vs" || \
    echo "MORE cases VS=$vs ended with status $?"
done

echo "===== non-TrackMan queue finished $(date -Is) ====="
