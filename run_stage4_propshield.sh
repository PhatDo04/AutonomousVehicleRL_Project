#!/usr/bin/env bash
# ============================================================================
# STAGE-4 PROPSHIELD — fine-tune CAI KHIÊN từ model best có sẵn của một đêm.
# Mục tiêu: nâng no-shield success (lái trần) mà vẫn giữ success có-khiên ≥80%,
# tái lập được trong pipeline (thay vì mượn model di sản yt52).
#
# Cơ chế: train tiếp ~150k từ S3-yield-best của đêm <TAG>, trong env phạt khiên
# tỷ lệ k=5 (đã có sẵn) → sweep checkpoint → chọn theo NO-SHIELD success.
# KHÔNG train lại từ đầu — chỉ warmstart model sẵn nên nhanh (~40-55 phút/đêm).
#
# Chạy:  bash run_stage4_propshield.sh [PORT] [SEED] [TAG]
#   PORT mặc định 1001, SEED = train-seed của đêm, TAG = tag đêm đó.
# ============================================================================
set -u
PORT="${1:-1001}"
SEED="${2:-0}"
TAG="${3:?Cần TAG đêm (vd 20260611_225051)}"
PY=./.venv/Scripts/python.exe
SUM="outputs/logs/thesis_night_summary_${TAG}.md"
PROG="outputs/logs/stage4_${TAG}_progress.log"
CK="outputs/models/checkpoints"
S3="night_ppo_s3yield_${TAG}"
S4="night_ppo_s4prop_${TAG}"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PROG"; }
guard() { grep -qE "nb_cars_max          <- 84;" models/Main_Traffic.gaml || { say "LỖI nb_cars_max!=84"; exit 1; }; }
fresh() { bash restart_headless.sh "$PORT" >> "$PROG" 2>&1; }

# Lấy checkpoint S3-yield-best của đêm (dòng "## STAGE-3 YIELD-BEST: <ck>")
S3BEST=$(grep -oE "STAGE-3 YIELD-BEST: [0-9]+" "$SUM" | head -1 | grep -oE "[0-9]+$")
[ -z "$S3BEST" ] && { say "Không tìm thấy S3-yield-best trong $SUM"; exit 1; }
SRC="$CK/$S3/${S3}_${S3BEST}_steps.zip"
[ -f "$SRC" ] || { say "Thiếu checkpoint nguồn: $SRC"; exit 1; }
say "Stage-4 nguồn: $SRC (S3-best=$S3BEST)"

E2E="--max-episode-steps 800 --eval-continue --postmerge-window 400 --rl-postmerge"
echo "" >> "$SUM"; echo "## STAGE-4 PROPSHIELD (fine-tune cai khiên từ S3-best=$S3BEST)" >> "$SUM"

# Fine-tune 150k với phạt khiên k=5 sẵn có.
guard; fresh
say "TRAIN Stage-4 $S4 (150k, warmstart S3-best) ..."
PYTHONIOENCODING=utf-8 $PY rl/train_marl.py --algo ppo --timesteps 150000 --seed "$SEED" --port "$PORT" \
  --max-episode-steps 600 --postmerge-window 400 --rl-postmerge \
  --resume-from "$SRC" --resume-mode warmstart \
  --run-name "$S4" --checkpoint-interval 50000 \
  > "outputs/logs/${S4}_train.log" 2>&1
say "TRAIN Stage-4 xong."

# Sweep: mỗi checkpoint eval CẢ HAI (khiên + no-shield, argmax), seed train.
guard; fresh
evalm() { # evalm <model> <label> <extra>
  local M="$1" L="$2"; shift 2
  echo "##### $L" >> "$SUM"
  PYTHONIOENCODING=utf-8 $PY rl/evaluate_marl.py --algo ppo --model "$M" --episodes 10 --port "$PORT" "$@" \
    2>&1 | grep -iE "merging_0  \| success|VERIFY goal-2|VERIFY ỷ" >> "$SUM"
  say "EVAL $L xong."
}
for ck in 50000 100000 150000; do
  evalm "$CK/$S4/${S4}_${ck}_steps.zip" "S4 ${ck} khiên"     --seed "$SEED" $E2E
  evalm "$CK/$S4/${S4}_${ck}_steps.zip" "S4 ${ck} NO-SHIELD" --seed "$SEED" $E2E --no-shield
done

# Chọn checkpoint: no-shield success cao nhất, ràng buộc khiên ≥80%.
S4BEST=$($PY - "$SUM" "$S4" <<'PYEOF'
import re, sys
txt=open(sys.argv[1],encoding="utf-8").read()
def succ(label):
    m=re.search(r"##### %s\n\s*merging_0  \| success=([\d.]+)%%"%re.escape(label),txt)
    return float(m.group(1)) if m else None
best,score=None,-1.0
for ck in (50000,100000,150000):
    sh=succ("S4 %d khiên"%ck); ns=succ("S4 %d NO-SHIELD"%ck)
    if sh is not None and ns is not None and sh>=80.0 and ns>score:
        best,score=ck,ns
print(best or "NONE")
PYEOF
)
say "Stage-4 propshield-best: $S4BEST"
echo "" >> "$SUM"; echo "## STAGE-4 PROPSHIELD-BEST: ${S4BEST} (chọn theo no-shield, khiên≥80%)" >> "$SUM"
if [ "$S4BEST" != "NONE" ]; then
  cp "$CK/$S4/${S4}_${S4BEST}_steps.zip" "outputs/models/thesis_stage4_propshield_${TAG}.zip"
  for S in 0 1 2; do
    evalm "$CK/$S4/${S4}_${S4BEST}_steps.zip" "S4-best seed$S khiên"     --seed "$S" $E2E
    evalm "$CK/$S4/${S4}_${S4BEST}_steps.zip" "S4-best seed$S NO-SHIELD" --seed "$S" $E2E --no-shield
  done
fi
say "===== STAGE-4 XONG ($TAG). Đọc: $SUM ====="
