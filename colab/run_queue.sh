#!/bin/bash
# 서버 실험 대기열. GPU 가 하나뿐이라 순서대로 돌린다.
#
# 파일로 두는 이유: 대기 조건을 인라인으로 쓰면 pgrep -f 가 자기 자신의
# 명령줄을 잡아 영원히 기다린다. 전에 그걸로 여섯 시간을 날렸다.
cd /workspace/aimers || exit 1
source /venv/main/bin/activate
: > out/queue.log
say() { echo "[$(date +%H:%M:%S)] $*" >> out/queue.log; }

say "fb_arms 대기"
while pgrep -f 'fb_arms\.py' > /dev/null; do sleep 30; done
say "fb_arms 종료 확인"

say "mkcbgate 시작"
python colab/mkcbgate.py > out/mkcbgate.log 2>&1
say "mkcbgate 종료 rc=$?"

for V in 2022 2023 2024; do
  say "tempscale VS=$V 시작"
  VS=$V python colab/tempscale.py > "out/ts$V.log" 2>&1
  say "tempscale VS=$V 종료 rc=$?"
done

say "보고서"
python colab/tempscale.py --report > out/ts_report.log 2>&1
say "전부 종료"
touch out/queue.done
