#!/usr/bin/env bash
# ============================================================================
# RUN THESIS OVERNIGHT — MỘT LỆNH chạy trọn bộ so sánh thuật toán qua đêm.
#
# CÔNG BẰNG (nguyên tắc khóa cứng):
#   • Môi trường ĐỒNG NHẤT cho mọi policy: cùng GAML (road 240, nb_cars_max 84,
#     cùng lưới khiên), cùng seed train, cùng curriculum 2-stage, cùng budget/stage.
#   • TUẦN TỰ trên 1 port duy nhất — không tiến trình nào đè tiến trình nào.
#   • restart_headless trước MỌI block (chống GAMA suy hao qua đêm).
#   • Mỗi thuật toán chạy thêm giai đoạn đặc thù của nó (A2C: mài nhọn lr 7e-4 +
#     entropy thấp) — phần CHÊNH LỆCH BƯỚC được ghi rõ trong summary (minh bạch).
#   • Eval cùng giao thức: gate S1 + sweep S2 + ma trận (chế độ × khiên × 3 seed).
#
# Pipeline:  PPO  S1 300k → S2 300k                        (600k)
#            A2C  S1 300k → sharpen 100k → S2 300k → sharpen 100k   (800k, ghi chú)
#            Greedy: không train (luật tĩnh)
#            EVAL: gates → sweep S2 → ma trận best-ckpt → greedy → summary + plots
#
# Chạy:  bash run_thesis_overnight.sh [PORT] [SEED] [TAG]
# Theo dõi: tail -f outputs/logs/night_<TAG>_progress.log
# Kết quả sáng ra: outputs/logs/thesis_night_summary_<TAG>.md + outputs/plots/
# ============================================================================
set -u
PORT="${1:-1001}"
SEED="${2:-0}"
TAG="${3:-$(date +%Y%m%d_%H%M%S)}"
PY=./.venv/Scripts/python.exe
PROG="outputs/logs/night_${TAG}_progress.log"
SUM="outputs/logs/thesis_night_summary_${TAG}.md"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROG"; }

guard_density() {
  if ! grep -qE "nb_cars_max          <- 84;" models/Main_Traffic.gaml; then
    say "LỖI: nb_cars_max != 84 — GAML bị patch density. DỪNG để giữ công bằng."
    exit 1
  fi
}

fresh() { bash restart_headless.sh "$PORT" >> "$PROG" 2>&1; }

train() { # train <algo> <steps> <run_name> <extra args...>
  local A="$1" STEPS="$2" NAME="$3"; shift 3
  guard_density; fresh
  say "TRAIN [$A] $NAME ($STEPS steps) ..."
  PYTHONIOENCODING=utf-8 $PY rl/train_marl.py --algo "$A" --timesteps "$STEPS" --seed "$SEED" \
    --port "$PORT" --run-name "$NAME" --checkpoint-interval 50000 "$@" \
    > "outputs/logs/${NAME}_train.log" 2>&1
  say "TRAIN [$A] $NAME XONG."
}

evalm() { # evalm <algo> <model> <label> <extra args...>
  local A="$1" M="$2" L="$3"; shift 3
  echo "##### $L" >> "$SUM"
  PYTHONIOENCODING=utf-8 $PY rl/evaluate_marl.py --algo "$A" --model "$M" --episodes 10 \
    --port "$PORT" "$@" 2>&1 | grep -iE "merging_0  \| success|VERIFY goal-2|VERIFY ỷ" >> "$SUM"
  say "EVAL $L xong."
}

mkdir -p outputs/logs
echo "# TỔNG KẾT THESIS OVERNIGHT — tag $TAG (seed train $SEED)" > "$SUM"
echo "" >> "$SUM"
echo "Môi trường đồng nhất: road 240, nb_cars_max 84, cùng lưới khiên, curriculum 2-stage." >> "$SUM"
echo "Budget: PPO 600k | A2C 800k (gồm 2×100k mài nhọn — đặc thù thuật toán, ghi chú minh bạch)." >> "$SUM"
say "===== BẮT ĐẦU OVERNIGHT (tag $TAG) ====="

# ───────────────────────── PHẦN A: TRAIN ─────────────────────────
P1="night_ppo_s1_${TAG}";  P2="night_ppo_s2_${TAG}"
A1="night_a2c_s1_${TAG}";  A1S="night_a2c_s1sharp_${TAG}"
A2="night_a2c_s2_${TAG}";  A2S="night_a2c_s2sharp_${TAG}"
CK="outputs/models/checkpoints"

# PPO
train ppo 300000 "$P1" --max-episode-steps 300
train ppo 300000 "$P2" --max-episode-steps 600 --postmerge-window 400 --rl-postmerge \
  --resume-from "$CK/$P1/${P1}_300000_steps.zip" --resume-mode warmstart

# A2C (recipe riêng đã kiểm chứng: lr 7e-4 + mài nhọn entropy thấp sau mỗi stage)
train a2c 300000 "$A1" --max-episode-steps 300 --learning-rate 7e-4
train a2c 100000 "$A1S" --max-episode-steps 300 --learning-rate 7e-4 \
  --ent-coef 0.0005 --no-ent-anneal \
  --resume-from "$CK/$A1/${A1}_300000_steps.zip" --resume-mode warmstart
train a2c 300000 "$A2" --max-episode-steps 600 --postmerge-window 400 --rl-postmerge \
  --learning-rate 7e-4 \
  --resume-from "$CK/$A1S/${A1S}_100000_steps.zip" --resume-mode warmstart
train a2c 100000 "$A2S" --max-episode-steps 600 --postmerge-window 400 --rl-postmerge \
  --learning-rate 7e-4 --ent-coef 0.0005 --no-ent-anneal \
  --resume-from "$CK/$A2/${A2}_300000_steps.zip" --resume-mode warmstart

# ───────────────────────── PHẦN B: EVAL ─────────────────────────
guard_density; fresh
E2E="--max-episode-steps 800 --eval-continue --postmerge-window 400 --rl-postmerge"
GATE="--max-episode-steps 300"

echo "" >> "$SUM"; echo "## GATE STAGE-1 (nhập làn, 2 chế độ, seed $SEED)" >> "$SUM"
evalm ppo "$CK/$P1/${P1}_300000_steps.zip" "PPO-S1 argmax" --seed "$SEED" $GATE
evalm ppo "$CK/$P1/${P1}_300000_steps.zip" "PPO-S1 stochastic" --seed "$SEED" $GATE --stochastic
evalm a2c "$CK/$A1S/${A1S}_100000_steps.zip" "A2C-S1(sharpened) argmax" --seed "$SEED" $GATE --log-actions
evalm a2c "$CK/$A1S/${A1S}_100000_steps.zip" "A2C-S1(sharpened) stochastic" --seed "$SEED" $GATE --stochastic

echo "" >> "$SUM"; echo "## SWEEP STAGE-2 (e2e, chế độ vận hành: PPO=argmax, A2C=stochastic)" >> "$SUM"
for ck in 100000 150000 200000 250000 300000; do
  evalm ppo "$CK/$P2/${P2}_${ck}_steps.zip" "PPO-S2 ${ck} argmax" --seed "$SEED" $E2E
done
for ck in 100000 150000 200000 250000 300000; do
  evalm a2c "$CK/$A2/${A2}_${ck}_steps.zip" "A2C-S2 ${ck} stochastic" --seed "$SEED" $E2E --stochastic
done
evalm a2c "$CK/$A2S/${A2S}_100000_steps.zip" "A2C-S2-sharpened argmax" --seed "$SEED" $E2E --log-actions
evalm a2c "$CK/$A2S/${A2S}_100000_steps.zip" "A2C-S2-sharpened stochastic" --seed "$SEED" $E2E --stochastic

# Chọn best ckpt theo success trong sweep (parse summary) — chế độ vận hành mỗi bên.
PPO_BEST=$($PY - "$SUM" PPO-S2 <<'PYEOF'
import re, sys
txt = open(sys.argv[1], encoding="utf-8").read()
best, score = None, -1.0
for m in re.finditer(r"##### %s (\d+) \S+\n.*?success=([\d.]+)%%" % sys.argv[2], txt, re.S):
    ck, sc = m.group(1), float(m.group(2))
    if sc > score: best, score = ck, sc
print(best or 150000)
PYEOF
)
A2C_BEST=$($PY - "$SUM" A2C-S2 <<'PYEOF'
import re, sys
txt = open(sys.argv[1], encoding="utf-8").read()
best, score = None, -1.0
for m in re.finditer(r"##### %s (\d+) \S+\n.*?success=([\d.]+)%%" % sys.argv[2], txt, re.S):
    ck, sc = m.group(1), float(m.group(2))
    if sc > score: best, score = ck, sc
print(best or 150000)
PYEOF
)
say "Best ckpt: PPO=$PPO_BEST, A2C=$A2C_BEST"
echo "" >> "$SUM"; echo "## BEST CKPT: PPO=${PPO_BEST}, A2C=${A2C_BEST}" >> "$SUM"

echo "" >> "$SUM"; echo "## MA TRẬN BEST (3 seeds × khiên/no-shield, chế độ vận hành)" >> "$SUM"
for S in 0 1 2; do
  evalm ppo "$CK/$P2/${P2}_${PPO_BEST}_steps.zip" "PPO-best seed$S khiên" --seed "$S" $E2E
  evalm ppo "$CK/$P2/${P2}_${PPO_BEST}_steps.zip" "PPO-best seed$S NO-SHIELD" --seed "$S" $E2E --no-shield
  evalm a2c "$CK/$A2/${A2}_${A2C_BEST}_steps.zip" "A2C-best seed$S khiên (stoch)" --seed "$S" $E2E --stochastic
  evalm a2c "$CK/$A2/${A2}_${A2C_BEST}_steps.zip" "A2C-best seed$S NO-SHIELD (stoch)" --seed "$S" $E2E --stochastic --no-shield
done

echo "" >> "$SUM"; echo "## GREEDY (cùng env + cùng khiên, e2e tự lái)" >> "$SUM"
for S in 0 1 2; do
  echo "##### GREEDY seed$S khiên" >> "$SUM"
  PYTHONIOENCODING=utf-8 $PY rl/baselines.py --policy greedy --episodes 10 --seed "$S" --port "$PORT" \
    --max-episode-steps 800 --eval-continue --postmerge-window 400 --rl-postmerge 2>&1 \
    | grep -cE "outcome=success" | sed 's/^/  success_count=/' >> "$SUM"
  echo "##### GREEDY seed$S NO-SHIELD" >> "$SUM"
  PYTHONIOENCODING=utf-8 $PY rl/baselines.py --policy greedy --episodes 10 --seed "$S" --port "$PORT" \
    --max-episode-steps 800 --eval-continue --postmerge-window 400 --rl-postmerge --no-shield 2>&1 \
    | grep -cE "outcome=success" | sed 's/^/  success_count=/' >> "$SUM"
  say "GREEDY seed$S xong."
done

# ───────────────────────── PHẦN C: TỔNG HỢP ─────────────────────────
say "Sinh biểu đồ ..."
PYTHONIOENCODING=utf-8 $PY rl/make_thesis_plots.py >> "$PROG" 2>&1 || say "(plots lỗi nhẹ — xem progress log)"
say "===== OVERNIGHT XONG. Đọc: $SUM ====="
echo "" >> "$SUM"; echo "_Learning curves: outputs/logs/night_*_${TAG}_episodes.csv + tensorboard. Plots: outputs/plots/._" >> "$SUM"
