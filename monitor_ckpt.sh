#!/usr/bin/env bash
# Monitor: eval mỗi checkpoint của yt38_long800k NGAY khi xuất hiện (port 1002, không đụng train ở 1001).
# Ghi trend success + %nhường-merger vào log để theo dõi tiến triển / bắt collapse sớm.
set -u
RUN="${1:-yt38_long800k}"
PORT=1002
LOG="outputs/logs/${RUN}_monitor.log"
CKDIR="outputs/models/checkpoints/${RUN}"
echo "[monitor] start $(date '+%H:%M:%S') run=$RUN port=$PORT" > "$LOG"
bash restart_headless.sh $PORT >> "$LOG" 2>&1
for ck in 50000 100000 150000 200000 250000 300000 350000 400000 450000 500000 550000 600000 650000 700000 750000 800000; do
  M="${CKDIR}/${RUN}_${ck}_steps.zip"
  # chờ checkpoint xuất hiện (tối đa ~25 phút mỗi cái)
  waited=0
  while [ ! -f "$M" ]; do
    sleep 30; waited=$((waited+30))
    if [ $waited -gt 1800 ]; then echo "[monitor] timeout chờ $ck" >> "$LOG"; break; fi
  done
  [ ! -f "$M" ] && continue
  sleep 15   # đảm bảo ghi xong file
  echo "=== ckpt ${ck} @ $(date '+%H:%M:%S') ===" >> "$LOG"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe rl/evaluate_marl.py --algo ppo --model "$M" --episodes 10 --seed 0 --port $PORT --max-episode-steps 300 2>&1 | grep -iE "merging_0  \| success|VERIFY goal-2" >> "$LOG"
done
echo "[monitor] done $(date '+%H:%M:%S')" >> "$LOG"
