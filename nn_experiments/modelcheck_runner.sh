#!/usr/bin/env bash
set -u

# The raw non-TrackMan queue keeps PID 193173 (it execs overnight_raw.sh).
# Wait for that queue to finish, then run the two requested model checks.
WAIT_PID=193173
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

cd /root
source /venv/main/bin/activate
export AIMERS_DATA='/root/open (1)/data'
export AIMERS_OUT=/root/modelcheck
mkdir -p "$AIMERS_OUT"

echo "MODEL queue start $(date -Is)"
for VS in 2022 2024; do
  export VS
  export AIMERS_OUT="/root/modelcheck/vs${VS}/arch"
  mkdir -p "$AIMERS_OUT"
  export ARCH_ONLY=tabm,tabm-mini
  echo "ARCH VS=$VS start $(date -Is)"
  timeout 10800s python /root/arch_sweep.py
  echo "arch VS=$VS exit=$? $(date -Is)"

  export AIMERS_OUT="/root/modelcheck/vs${VS}/tabr"
  mkdir -p "$AIMERS_OUT"
  unset ARCH_ONLY
  echo "TABR VS=$VS start $(date -Is)"
  timeout 21600s python /root/tabr_fair.py
  echo "tabr VS=$VS exit=$? $(date -Is)"
done
echo "MODEL queue done $(date -Is)"
