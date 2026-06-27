#!/usr/bin/env bash
# Clean 3-seed shockwave eval — re-evaluate EXISTING best checkpoints (no retrain).
# Matches headline protocol: e2e high-84, shield ON, PPO=argmax, A2C=stochastic.
# Restarts GAMA clean before each policy block (per user request).
set -u
cd "d:/DoAn_RL/GAMA/temp/AutonomousVehicleRL_Project" || exit 1
PORT=1001
PY=./.venv/Scripts/python.exe
CK=outputs/models/checkpoints
OUT=outputs/logs/sw_eval
mkdir -p "$OUT"
E2E="--max-episode-steps 800 --eval-continue --postmerge-window 400 --rl-postmerge"
PROG="$OUT/_progress.log"
: > "$PROG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROG"; }

# best ckpt per night (from thesis_night_summary_*.md: PPO 150k/150k/100k, A2C 300k/200k/300k)
PPO1="$CK/night_ppo_s2_20260611_225051/night_ppo_s2_20260611_225051_150000_steps.zip"
PPO2="$CK/night_ppo_s2_20260612_135116/night_ppo_s2_20260612_135116_150000_steps.zip"
PPO3="$CK/night_ppo_s2_20260612_232535/night_ppo_s2_20260612_232535_100000_steps.zip"
A2C1="$CK/night_a2c_s2_20260611_225051/night_a2c_s2_20260611_225051_300000_steps.zip"
A2C2="$CK/night_a2c_s2_20260612_135116/night_a2c_s2_20260612_135116_200000_steps.zip"
A2C3="$CK/night_a2c_s2_20260612_232535/night_a2c_s2_20260612_232535_300000_steps.zip"

# sanity: all model files present
for m in "$PPO1" "$PPO2" "$PPO3" "$A2C1" "$A2C2" "$A2C3"; do
  [ -f "$m" ] || { say "MISSING MODEL: $m"; exit 2; }
done
say "All 6 models present. Starting clean 3-seed shockwave eval."

eval_one() { # eval_one <algo> <model> <tag> <extra...>
  local A="$1" M="$2" T="$3"; shift 3
  say "EVAL $T ..."
  PYTHONIOENCODING=utf-8 $PY rl/evaluate_marl.py --algo "$A" --model "$M" --episodes 10 \
    --port "$PORT" --out "$OUT/${T}.csv" $E2E "$@" > "$OUT/${T}.log" 2>&1
  say "EVAL $T done."
}

# ---- PPO block (argmax) ----
bash restart_headless.sh "$PORT" >> "$PROG" 2>&1
for n in 1 2 3; do
  eval "M=\$PPO$n"
  for S in 0 1 2; do eval_one ppo "$M" "ppo_n${n}_seed${S}" --seed "$S"; done
done

# ---- A2C block (stochastic) ----
bash restart_headless.sh "$PORT" >> "$PROG" 2>&1
for n in 1 2 3; do
  eval "M=\$A2C$n"
  for S in 0 1 2; do eval_one a2c "$M" "a2c_n${n}_seed${S}" --seed "$S" --stochastic; done
done

# ---- Greedy block (3 eval seeds) ----
bash restart_headless.sh "$PORT" >> "$PROG" 2>&1
for S in 0 1 2; do
  say "EVAL greedy_seed${S} ..."
  PYTHONIOENCODING=utf-8 $PY rl/baselines.py --policy greedy --episodes 10 --seed "$S" \
    --port "$PORT" --out "$OUT/greedy_seed${S}.csv" $E2E > "$OUT/greedy_seed${S}.log" 2>&1
  say "EVAL greedy_seed${S} done."
done

say "===== ALL SHOCKWAVE EVAL DONE ====="
