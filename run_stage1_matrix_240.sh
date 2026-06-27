#!/usr/bin/env bash
# ============================================================================
# Chạy lại MA TRẬN GIAI ĐOẠN 1 trên env 240m (3 hạt giống) — thay data 200m cũ.
#   6 ô = {low, medium, high} × {shield on, off}, preset thesis (3 seed × 200k × PPO+A2C + Greedy/Random).
#   • restart GAMA headless trước MỖI ô (chống suy hao qua nhiều giờ).
#   • run_experiments.py tự vá density theo --scenario + set khiên theo --shield, restore GAML cuối ô.
#   • mỗi ô ghi vào run-tag riêng: plots/thesis/g1_240_<sc>_<sh>/comparison_table.csv
# Chạy:    bash run_stage1_matrix_240.sh [PORT]
# Theo dõi: tail -f outputs/logs/g1_240_progress.log
# ============================================================================
set -u
PORT="${1:-1001}"
TAG="g1_240"
PY=./.venv/Scripts/python.exe
PROG="outputs/logs/${TAG}_progress.log"
mkdir -p outputs/logs

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }

guard(){
  if ! grep -qE "road_length +<- +240" models/Main_Traffic.gaml; then
    say "LỖI: GAML KHÔNG phải 240m — DỪNG để khỏi chạy 33h ra số sai."; exit 1
  fi
}

run_one(){ # scenario shield
  local SC="$1" SH="$2" RT="${TAG}_${1}_${2}"
  guard
  say "restart GAMA (port $PORT) trước ô $SC/$SH ..."
  bash restart_headless.sh "$PORT" >> "$PROG" 2>&1
  say ">>> START  $SC / shield-$SH  (run-tag $RT)"
  PYTHONIOENCODING=utf-8 $PY rl/run_experiments.py --preset thesis \
    --scenario "$SC" --shield "$SH" --port "$PORT" --run-tag "$RT" \
    > "outputs/logs/${RT}_run.log" 2>&1
  say "<<< DONE   $SC / shield-$SH  (exit $?)"
}

say "===== BẮT ĐẦU MA TRẬN GĐ1 240m — 3 seed (6 ô) ====="
for SH in on off; do
  for SC in low medium high; do
    run_one "$SC" "$SH"
  done
done

say "===== TỔNG HỢP (success% | collision% theo ô) ====="
PYTHONIOENCODING=utf-8 $PY - <<'PYEOF' 2>&1 | tee -a "$PROG"
import csv
for sh in ("on","off"):
    for sc in ("low","medium","high"):
        f=f"outputs/plots/thesis/g1_240_{sc}_{sh}/comparison_table.csv"
        try:
            rows=list(csv.DictReader(open(f,encoding="utf-8")))
            print(f"--- {sc} / shield-{sh} ---")
            for r in rows:
                print(f"  {r['algorithm']:10s} succ={float(r['success_rate_mean'])*100:5.1f}%  coll={float(r['collision_rate_mean'])*100:5.1f}%  (n={r['n_episodes']})")
        except FileNotFoundError:
            print(f"--- {sc} / shield-{sh}: THIẾU comparison_table ---")
PYEOF
say "===== XONG MA TRẬN GĐ1 240m ====="
