#!/usr/bin/env bash
set -u
cd /root
source /venv/main/bin/activate
LOG=/root/overnight6h_extra.log
ROOT=/root
TM=/root/trackman_pitcher_season.csv
# Wait for the first queue (PID recorded when it was launched) so the GPU is
# never shared by two training processes.
while ps -p 75362 >/dev/null 2>&1; do sleep 30; done
exec >> "$LOG" 2>&1
echo "===== extra queue start $(date -Is) ====="

run_cb() {
  local tag="$1"; local vs="$2"; shift 2
  local out="/root/overnight_cb_extra/$tag/vs$vs"
  mkdir -p "$out"
  echo "--- CB $tag VS=$vs $(date -Is) ---"
  timeout 1800s env AIMERS_ROOT="$ROOT" TM_PATH="$TM" TM_2S=1     SAVE_DIR="$out" ONLY=c12_cmh_tm_2s "$@"     python /root/pc_gpu_sweep_overnight.py "$vs" || echo "CB $tag VS=$vs status $?"
}

# More independent seeds to estimate variance and reduce winner's-curse risk.
for vs in 2021 2022 2023 2024; do
  run_cb s31_60 "$vs" CB_SEEDS=31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
done

# Additional CatBoost regularization and boosting-length checks.
for setting in l2_2 l2_10 depth2 depth6 iter200 iter1200 alpha100 alpha400; do
  case "$setting" in
    l2_2)    opts=(CB_L2=2.0) ;;
    l2_10)   opts=(CB_L2=10.0) ;;
    depth2)  opts=(CB_DEPTH=2) ;;
    depth6)  opts=(CB_DEPTH=6) ;;
    iter200) opts=(CB_ITERS=200 CB_LR=0.1) ;;
    iter1200) opts=(CB_ITERS=1200 CB_LR=0.0166667) ;;
    alpha100) opts=(PC_ALPHA=100) ;;
    alpha400) opts=(PC_ALPHA=400) ;;
  esac
  for vs in 2021 2022 2023 2024; do
    run_cb "$setting" "$vs" CB_SEEDS=1,2,3,4,5,6,7,8,9,10 "${opts[@]}"
  done
done

# Cross-check combinations of TrackMan reliability, rates, and movement.
for subset in tm2s_rate_shift tm2s_rate_n tm2s_shift_n; do
  case "$subset" in
    tm2s_rate_shift) cols=tm2s_fastball_rate,tm2s_breaking_rate,tm2s_offspeed_rate,tm2s_fastball_shift,tm2s_breaking_shift,tm2s_speed_delta,tm2s_relh_sd ;;
    tm2s_rate_n) cols=tm2s_n,tm2s_fastball_rate,tm2s_breaking_rate,tm2s_offspeed_rate ;;
    tm2s_shift_n) cols=tm2s_n,tm2s_fastball_shift,tm2s_breaking_shift,tm2s_speed_delta,tm2s_relh_sd ;;
  esac
  for vs in 2021 2022 2023 2024; do
    run_cb "$subset" "$vs" CB_SEEDS=1,2,3,4,5,6,7,8,9,10 TM2S_SET="$cols"
  done
done

echo "===== extra queue finished $(date -Is) ====="

