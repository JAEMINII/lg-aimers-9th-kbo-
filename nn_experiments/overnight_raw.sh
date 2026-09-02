#!/usr/bin/env bash
set -u
cd /root
source /venv/main/bin/activate
LOG=/root/overnight_raw.log
ROOT=/root
CASES="raw_n,raw_context,raw_all,c12_cmh_raw_n,c12_cmh_raw_all"
mkdir -p /root/overnight_cb_raw
exec >> "$LOG" 2>&1
echo "===== raw-feature queue start $(date -Is) ====="

for vs in 2021 2022 2023 2024; do
  out="/root/overnight_cb_raw/vs$vs"
  mkdir -p "$out"
  echo "--- RAW cases VS=$vs $(date -Is) ---"
  timeout 2400s env AIMERS_ROOT="$ROOT" SAVE_DIR="$out" RAW=1 ONLY="$CASES" \
    CB_SEEDS=1,2,3,4,5,6,7,8,9,10 \
    python /root/pc_gpu_sweep_raw.py "$vs" || \
    echo "RAW cases VS=$vs ended with status $?"
done

echo "===== raw-feature queue finished $(date -Is) ====="
