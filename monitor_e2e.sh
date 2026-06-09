#!/usr/bin/env bash
# Monitor e2e: eval mỗi checkpoint KÈM --eval-continue --rl-postmerge (đo trọn hành trình: merge +
# sống-sót post-merge + nhường + sóng lùi). Port 1002 (không đụng train ở 1001).
set -u
RUN="${1:-yt40_e2e}"
PORT=1002
WIN="${2:-200}"
LOG="outputs/logs/${RUN}_monitor.log"
CKDIR="outputs/models/checkpoints/${RUN}"
echo "[monitor-e2e] start $(date '+%H:%M:%S') run=$RUN win=$WIN" > "$LOG"
bash restart_headless.sh $PORT >> "$LOG" 2>&1
for ck in 50000 100000 150000 200000 250000 300000 350000 400000; do
  M="${CKDIR}/${RUN}_${ck}_steps.zip"
  waited=0
  while [ ! -f "$M" ]; do
    sleep 30; waited=$((waited+30))
    if [ $waited -gt 2400 ]; then echo "[monitor] timeout $ck" >> "$LOG"; break; fi
  done
  [ ! -f "$M" ] && continue
  sleep 15
  echo "=== ckpt ${ck} @ $(date '+%H:%M:%S') ===" >> "$LOG"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe rl/evaluate_marl.py --algo ppo --model "$M" --episodes 10 --seed 0 --port $PORT --max-episode-steps 400 --eval-continue --postmerge-window $WIN --rl-postmerge 2>&1 | grep -iE "merging_0  \| success|VERIFY goal-2" >> "$LOG"
done
echo "[monitor-e2e] done $(date '+%H:%M:%S')" >> "$LOG"
