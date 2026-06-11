#!/usr/bin/env bash
# ============================================================================
# CURRICULUM TRAIN HOÀN CHỈNH — tái hiện model thesis TỪ ĐẦU trên mã nguồn
# hiện tại, KHÔNG sửa code. Dùng để hướng dẫn/tái lập (reproducibility).
#
#   Stage 1: học MERGE cơ bản   — from scratch, terminate-at-merge, 300k steps
#   Stage 2: học E2E + hợp tác  — warmstart Stage-1, chạy tới cuối đường
#            (--rl-postmerge --postmerge-window 400), 300k steps
#   Sau đó: eval sweep các checkpoint Stage-2 (50k..300k) và CHỌN điểm cân
#            bằng 3 goal (merge success + %nhường-merger + sóng lùi) — model
#            cuối thường KHÔNG phải checkpoint cuối (catastrophic forgetting).
#
# Yêu cầu: GAML giữ nguyên (road 240, nb_cars_max 84). Script tự kiểm.
# Chạy:   bash train_curriculum.sh [PORT] [SEED] [TAG]   (log: outputs/logs/curr_*.log)
#         TAG mặc định = timestamp -> mỗi lần chạy tự ra bộ tên mới, không ghi đè.
# ============================================================================
set -u
PORT="${1:-1001}"
SEED="${2:-0}"
# TAG (tham số 3, tùy chọn): mặc định TỰ SINH theo timestamp (như run_experiments
# <preset>_YYYYmmdd_HHMMSS) -> mỗi lần chạy một bộ tên riêng, KHÔNG BAO GIỜ ghi đè.
#   bash train_curriculum.sh              -> curr_stage1_merge_20260611_153000 / ...
#   bash train_curriculum.sh 1001 0 demo  -> curr_stage1_merge_demo / ... (tên tự đặt)
TAG="${3:-$(date +%Y%m%d_%H%M%S)}"
S1=curr_stage1_merge_${TAG}
S2=curr_stage2_e2e_${TAG}
echo "[curriculum] run tag: ${TAG}  (S1=$S1, S2=$S2)"
PY=./.venv/Scripts/python.exe

# Guard: mật độ phải đúng HIGH (bài học sự cố GUI-patch lẫn 54 vào commit).
if ! grep -qE "nb_cars_max          <- 84;" models/Main_Traffic.gaml; then
  echo "[curriculum] LỖI: nb_cars_max != 84 (GAML bị patch density?) — dừng để tránh train sai kịch bản."
  exit 1
fi

echo "[curriculum] ===== STAGE 1: MERGE from scratch (300k, terminate-at-merge, seed=$SEED) ====="
bash restart_headless.sh "$PORT"
PYTHONIOENCODING=utf-8 $PY rl/train_marl.py --algo ppo --timesteps 300000 --seed "$SEED" --port "$PORT" \
  --max-episode-steps 300 --run-name "$S1" --checkpoint-interval 50000 \
  > "outputs/logs/${S1}_train.log" 2>&1
echo "[curriculum] Stage 1 xong. Checkpoint: outputs/models/checkpoints/$S1/"

echo "[curriculum] ===== STAGE 2: E2E + hợp tác (300k, warmstart Stage-1, window 400) ====="
bash restart_headless.sh "$PORT"
PYTHONIOENCODING=utf-8 $PY rl/train_marl.py --algo ppo --timesteps 300000 --seed "$SEED" --port "$PORT" \
  --max-episode-steps 600 --postmerge-window 400 --rl-postmerge \
  --resume-from "outputs/models/checkpoints/$S1/${S1}_300000_steps.zip" --resume-mode warmstart \
  --run-name "$S2" --checkpoint-interval 50000 \
  > "outputs/logs/${S2}_train.log" 2>&1
echo "[curriculum] Stage 2 xong. Checkpoint: outputs/models/checkpoints/$S2/"

echo "[curriculum] ===== HƯỚNG DẪN CHỌN MODEL (làm tay hoặc script eval) ====="
echo "  Eval từng checkpoint (50k..300k) với 2-3 seed:"
echo "    $PY rl/evaluate_marl.py --algo ppo --model outputs/models/checkpoints/$S2/${S2}_<CK>_steps.zip \\"
echo "       --episodes 10 --seed <S> --port 1002 --max-episode-steps 800 \\"
echo "       --eval-continue --postmerge-window 400 --rl-postmerge"
echo "  Chọn checkpoint CÂN BẰNG: success cao + %NHƯỜNG-merger > 0 + ỷ-khiên thấp."
echo "  (Checkpoint cuối thường merger giỏi nhưng quên nhường — đừng lấy mù.)"
echo "[curriculum] DONE."
