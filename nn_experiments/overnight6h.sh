#!/usr/bin/env bash
set -u
cd /root
source /venv/main/bin/activate
LOG=/root/overnight6h.log
ROOT=/root
TM=/root/trackman_pitcher_season.csv
mkdir -p /root/overnight_cb /root/overnight_tabm
exec >> "$LOG" 2>&1
echo "===== overnight queue start $(date -Is) ====="

run_cb() {
  local tag="$1"
  local vs="$2"
  shift 2
  local out="/root/overnight_cb/$tag/vs$vs"
  mkdir -p "$out"
  echo "--- CB $tag VS=$vs $(date -Is) ---"
  timeout 1800s env AIMERS_ROOT="$ROOT" TM_PATH="$TM" TM_2S=1 \
    SAVE_DIR="$out" ONLY=c12_cmh_tm_2s "$@" \
    python /root/pc_gpu_sweep_overnight.py "$vs" || \
    echo "CB $tag VS=$vs ended with status $?"
}

for vs in 2021 2022 2023 2024; do
  run_cb s01_30 "$vs" CB_SEEDS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
done

for setting in l2_5 l2_05 depth3 depth5 iter800; do
  case "$setting" in
    l2_5)   opts=(CB_L2=5.0) ;;
    l2_05)  opts=(CB_L2=0.5) ;;
    depth3) opts=(CB_DEPTH=3) ;;
    depth5) opts=(CB_DEPTH=5) ;;
    iter800) opts=(CB_ITERS=800 CB_LR=0.025) ;;
  esac
  for vs in 2021 2022 2023 2024; do
    run_cb "$setting" "$vs" CB_SEEDS=1,2,3,4,5,6,7,8,9,10 "${opts[@]}"
  done
done

for subset in tm2s_n tm2s_rates tm2s_shifts; do
  case "$subset" in
    tm2s_n)      cols=tm2s_n ;;
    tm2s_rates)  cols=tm2s_fastball_rate,tm2s_breaking_rate,tm2s_offspeed_rate ;;
    tm2s_shifts) cols=tm2s_fastball_shift,tm2s_breaking_shift,tm2s_speed_delta,tm2s_relh_sd ;;
  esac
  for vs in 2021 2022 2023 2024; do
    run_cb "$subset" "$vs" CB_SEEDS=1,2,3,4,5,6,7,8,9,10 TM2S_SET="$cols"
  done
done

for vs in 2021 2022 2023 2024; do
  echo "--- CB baseline VS=$vs $(date -Is) ---"
  out="/root/overnight_cb/baseline/vs$vs"
  mkdir -p "$out"
  timeout 1800s env AIMERS_ROOT="$ROOT" SAVE_DIR="$out" ONLY=c12_cmh \
    CB_SEEDS=1,2,3,4,5,6,7,8,9,10 \
    python /root/pc_gpu_sweep_overnight.py "$vs" || echo "baseline VS=$vs status $?"
done

for case in ep3 ep4 ep6 ep8 ep4_k64; do
  echo "--- TabM TM2S $case $(date -Is) ---"
  out="/root/overnight_tabm/$case"
  mkdir -p "$out"
  timeout 2400s env AIMERS_DATA="/root/open (1)/data" AIMERS_OUT="$out" \
    TM_PATH="$TM" VS=2024 python /root/tabm_tm2s_gate.py "$case" || \
    echo "TabM $case status $?"
done

echo "===== overnight queue finished $(date -Is) ====="

