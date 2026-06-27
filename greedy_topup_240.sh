#!/usr/bin/env bash
# Chạy bù Greedy baseline seed 1,2 cho ma trận GĐ1 240m (seed0 đã có).
# Cùng điều kiện các ô đã chạy: no-shield (enable_safety_shield=false mặc định), terminate-at-merge (500 steps).
set -u
PORT="${1:-1001}"
PY=./.venv/Scripts/python.exe
PROG="outputs/logs/greedy_topup_240.log"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROG"; }

say "===== GREEDY TOP-UP 240m (seed 1,2 × low/medium/high) ====="
for SC in low medium high; do
  say "patch density '$SC' vào GAML ..."
  PYTHONIOENCODING=utf-8 $PY -c "from rl.scenario_utils import apply_scenario_all_gaml; apply_scenario_all_gaml('$SC')" >> "$PROG" 2>&1
  say "restart GAMA cho '$SC' ..."
  bash restart_headless.sh "$PORT" >> "$PROG" 2>&1
  for S in 1 2; do
    OUT="outputs/logs/g1_240_${SC}_on/greedy_baseline_seed${S}.csv"
    say "  greedy $SC seed$S → $OUT"
    PYTHONIOENCODING=utf-8 $PY rl/baselines.py --policy greedy --episodes 50 --seed "$S" \
      --host localhost --port "$PORT" --max-episode-steps 500 --scenario "$SC" --out "$OUT" \
      >> "outputs/logs/greedy_topup_${SC}_seed${S}.log" 2>&1
    say "  done $SC seed$S (exit $?)"
  done
done
say "khôi phục GAML về committed (git checkout) ..."
PYTHONIOENCODING=utf-8 $PY -c "from rl.scenario_utils import restore_gaml_backup; restore_gaml_backup()" >> "$PROG" 2>&1
say "===== XONG GREEDY TOP-UP ====="
