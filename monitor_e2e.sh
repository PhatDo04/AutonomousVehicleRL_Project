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
  CSV="outputs/logs/${RUN}_ck${ck}.csv"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe rl/evaluate_marl.py --algo ppo --model "$M" --episodes 10 --seed 0 --port $PORT --max-episode-steps 400 --eval-continue --postmerge-window $WIN --rl-postmerge --out "$CSV" 2>&1 | grep -iE "merging_0  \| success|VERIFY goal-2" >> "$LOG"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -c "import csv; r=list(csv.DictReader(open('$CSV',encoding='utf-8'))); v=[float(x['mean_speed']) for x in r if x.get('mean_speed','') not in ('','None')]; print('  MERGER mean_speed = %.3f'%(sum(v)/len(v)) if v else '  mean_speed n/a')" >> "$LOG" 2>&1
done
echo "[monitor-e2e] done $(date '+%H:%M:%S')" >> "$LOG"
