"""Tự động chạy thực nghiệm MARL: baseline Greedy/Random → MARL PPO/A2C.

Sau khi tất cả các thuật toán và seed chạy xong, script gọi plots.py và analysis.py.

Sử dụng:
    python rl/run_experiments.py --preset thesis --port 1001
    python rl/run_experiments.py --preset smoke  --port 1001
    python rl/run_experiments.py --preset short  --port 1001 --scenario medium
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import (
    LOG_DIR,
    MARL_AGENTS,
    MARL_ALGORITHMS,
    MARL_EXPERIMENT_PRESETS,
    MODEL_DIR,
    PLOT_DIR,
    SCENARIO_PRESETS,
)
from rl.gama_compat import _configure_windows_cli_io
from rl.scenario_utils import apply_scenario_all_gaml, restore_gaml_backup, set_safety_shield

_configure_windows_cli_io()


def _python() -> str:
    """Trả về đường dẫn Python interpreter đang chạy."""
    return sys.executable


def _run(cmd: list[str], step_label: str) -> int:
    """Chạy một lệnh con, in tiến trình và trả về exit code."""
    print(f"\n{'='*60}")
    print(f"  {step_label}")
    print(f"  > {' '.join(cmd)}")
    print(f"{'='*60}")
    start = time.time()
    env = os.environ.copy()
    if sys.platform == "win32":
        env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(cmd, check=False, env=env)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  [{status}] elapsed {elapsed:.1f}s")
    return result.returncode


def _default_checkpoint_interval(timesteps: int) -> int:
    """Lưu khoảng 5 checkpoint/run, đủ bắt overfit mà không làm nổ thời gian eval."""
    if timesteps <= 0:
        return 0
    return max(1_000, timesteps // 5)


def marl_run_basename(algo: str, seed: int, timesteps: int, scenario: str) -> str:
    """Tên file gốc cho 1 run MARL (KHÔNG kèm run-tag). Nguồn duy nhất để train + model_path đồng bộ."""
    base = f"marl_{algo}_seed{seed}_{timesteps // 1000}k"
    if scenario != "medium":
        base = f"{base}_{scenario}"
    return base


def run_baselines(
    *,
    episodes: int,
    seed: int,
    host: str,
    port: int,
    max_episode_steps: int,
    scenario: str,
    dry_run: bool,
    logs_dir: Path = LOG_DIR,
) -> list[Path]:
    """Chạy baseline greedy, trả về danh sách CSV đã sinh."""
    csv_files: list[Path] = []
    # Baseline NON-LEARNING làm mốc so với PPO/A2C:
    #   "greedy" = tham lam (tăng tốc + merge sớm + vượt làn; né va chạm dọc nhờ khiên môi trường).
    # "random" ĐÃ BỎ (đề cương không dùng random). base_rule cũng đã bỏ — demo GUI Heuristic dùng luật GAML.
    for policy in ("greedy",):
        out_csv = logs_dir / f"{policy}_baseline_seed{seed}.csv"
        cmd = [
            _python(), str(ROOT / "rl" / "baselines.py"),
            "--policy", policy,
            "--episodes", str(episodes),
            "--seed", str(seed),
            "--host", host,
            "--port", str(port),
            "--max-episode-steps", str(max_episode_steps),
            "--scenario", scenario,
            "--out", str(out_csv),
        ]
        if not dry_run:
            rc = _run(cmd, f"Baseline: {policy} | seed={seed} | scenario={scenario}")
            if rc != 0:
                print(f"  [WARN] Baseline {policy} kết thúc với lỗi, tiếp tục...")
        else:
            print(f"  [DRY-RUN] {' '.join(cmd)}")
        csv_files.append(out_csv)
    return csv_files


def run_training(
    *,
    algo: str,
    timesteps: int,
    seed: int,
    host: str,
    port: int,
    max_episode_steps: int,
    scenario: str,
    dry_run: bool,
    checkpoint_interval: int = 0,
    run_tag: str | None = None,
    logs_dir: Path = LOG_DIR,
    ent_anneal: bool = True,
) -> Path:
    """Huấn luyện MARL (shared policy) với một seed.

    ``run_tag`` (nếu có) biến ``--run-name`` thành "<tag>/<base>" → train_marl ghi model/CSV
    vào ``models/<tag>/`` và ``logs/<tag>/`` (tách output theo từng lần chạy).
    """
    base = marl_run_basename(algo, seed, timesteps, scenario)
    run_name = f"{run_tag}/{base}" if run_tag else base
    out_csv = logs_dir / f"{base}_episodes.csv"
    cmd = [
        _python(), str(ROOT / "rl" / "train_marl.py"),
        "--algo", algo,
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--host", host,
        "--port", str(port),
        "--max-episode-steps", str(max_episode_steps),
        "--scenario", scenario,
        "--run-name", run_name,
    ]
    if checkpoint_interval > 0:
        cmd.extend(["--checkpoint-interval", str(checkpoint_interval)])
    if not ent_anneal:
        cmd.append("--no-ent-anneal")
    if not dry_run:
        rc = _run(cmd, f"MARL Train: {algo.upper()} | seed={seed} | {timesteps} steps | scenario={scenario}")
        if rc != 0:
            print(f"  [WARN] Train {algo} seed={seed} kết thúc với lỗi, tiếp tục...")
    else:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
    return out_csv


def run_evaluate(
    *,
    algo: str,
    model_path: Path,
    episodes: int,
    seed: int,
    host: str,
    port: int,
    max_episode_steps: int,
    dry_run: bool,
    stochastic: bool = False,
    logs_dir: Path = LOG_DIR,
) -> Path:
    """Chạy evaluate deterministic sau training, trả về CSV eval.

    Metrics từ evaluate (deterministic) phân biệt rõ với metrics từ training
    (có exploration noise). Cả hai được dùng trong báo cáo nhưng cần ghi rõ nguồn.
    """
    if not model_path.exists() and not dry_run:
        print(f"  [SKIP] Model không tồn tại: {model_path}")
        return model_path.parent / "missing.csv"

    out_csv = (
        logs_dir / f"{model_path.stem}_eval_stochastic.csv"
        if stochastic
        else logs_dir / f"{model_path.stem}_eval.csv"
    )
    cmd = [
        _python(), str(ROOT / "rl" / "evaluate_marl.py"),
        "--algo", algo,
        "--model", str(model_path),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--host", host,
        "--port", str(port),
        "--max-episode-steps", str(max_episode_steps),
        "--out", str(out_csv),
        # Bat action histogram de debug policy collapse: thay vi chi xem reward/outcome,
        # in luon phan phoi action argmax cua tung agent. Neu thay action 3 (Merge) 0% ->
        # policy chua hoc duoc merge; neu action 1 (Keep) > 90% -> ket trong local minimum.
        "--log-actions",
    ]
    if stochastic:
        cmd.append("--stochastic")
    if not dry_run:
        rc = _run(cmd, f"MARL Eval: {algo.upper()} | model={model_path.name}")
        if rc != 0:
            print(f"  [WARN] Evaluate {algo} kết thúc với lỗi, tiếp tục...")
    else:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
    return out_csv


def _read_rate(path: Path, column: str) -> float:
    if not path.exists():
        return 0.0
    values: list[bool] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = str(row.get(column, "")).strip().lower()
            values.append(raw in {"1", "true", "yes"})
    if not values:
        return 0.0
    return sum(values) / len(values)


def _highway_eval_csv(eval_csv: Path) -> Path:
    """File highway metrics đi kèm output eval MARL."""
    return eval_csv.parent / eval_csv.name.replace(".csv", "_highway.csv")


def _score_eval_csv(
    eval_csv: Path,
    *,
    reject_unsafe_highway: bool = False,
) -> float:
    """Score chọn checkpoint MARL.

    Công thức: ``success - collision - timeout - 0.5 * highway_collision``.

    HW collision dùng **soft penalty** (weight 0.5) thay vì hard reject — vì shared
    policy có baseline HW collision rate ~25%, nếu reject mọi checkpoint có HW collision
    thì tất cả đều bị loại và fallback về final model, làm mất giá trị chọn best ckpt.
    Tham số ``reject_unsafe_highway`` giữ lại cho backward compatibility (không có tác dụng).
    """
    success_rate = _read_rate(eval_csv, "success")
    collision_rate = _read_rate(eval_csv, "collision")
    timeout_rate = _read_rate(eval_csv, "timeout")
    highway_csv = _highway_eval_csv(eval_csv)
    highway_collision_rate = _read_rate(highway_csv, "collision")
    return success_rate - collision_rate - timeout_rate - 0.5 * highway_collision_rate


def select_best_checkpoint(
    *,
    algo: str,
    run_name: str,
    model_path: Path,
    episodes: int,
    seed: int,
    host: str,
    port: int,
    max_episode_steps: int,
    dry_run: bool,
    reject_unsafe_highway: bool = False,
    logs_dir: Path = LOG_DIR,
) -> Path:
    """Eval các checkpoint đã lưu và copy checkpoint tốt nhất thành model chính."""
    checkpoint_dir = MODEL_DIR / "checkpoints" / run_name
    candidates = sorted(checkpoint_dir.glob("*.zip")) if checkpoint_dir.exists() else []
    if model_path.exists() or dry_run:
        candidates.append(model_path)
    if not candidates:
        print(f"  [SKIP] Không có checkpoint để chọn best cho {run_name}.")
        return model_path

    best_path: Path | None = None
    best_score = float("-inf")
    print(f"\n  Chọn best checkpoint cho {run_name} ({len(candidates)} candidates, {episodes} eval episodes/checkpoint)")
    for candidate in candidates:
        eval_csv = run_evaluate(
            algo=algo,
            model_path=candidate,
            episodes=episodes,
            seed=seed,
            host=host,
            port=port,
            max_episode_steps=max_episode_steps,
            dry_run=dry_run,
            stochastic=True,
            logs_dir=logs_dir,
        )
        score = 0.0 if dry_run else _score_eval_csv(
            eval_csv,
            reject_unsafe_highway=reject_unsafe_highway,
        )
        print(f"  checkpoint={candidate.name} score={score:.4f}")
        if best_path is None or score > best_score:
            best_score = score
            best_path = candidate

    if best_path is None or best_score == float("-inf"):
        print(f"  [BEST] Không có checkpoint an toàn/hợp lệ cho {run_name}; giữ model hiện tại.")
        return model_path

    if not dry_run and best_path != model_path:
        shutil.copy2(best_path, model_path)
        print(f"  [BEST] {best_path.name} -> {model_path.name} (score={best_score:.4f})")
    elif not dry_run:
        print(f"  [BEST] Giữ model cuối: {model_path.name} (score={best_score:.4f})")
    return model_path


def run_plots(csv_files: list[Path], out_dir: Path, dry_run: bool) -> None:
    """Sinh tất cả biểu đồ từ CSV đã thu thập."""
    existing = [str(f) for f in csv_files if f.exists()]
    if not existing:
        print("  [SKIP] Không có CSV nào để vẽ biểu đồ.")
        return
    cmd = [_python(), str(ROOT / "rl" / "plots.py")] + existing + ["--out-dir", str(out_dir)]
    if not dry_run:
        _run(cmd, f"Plots → {out_dir}")
    else:
        print(f"  [DRY-RUN] {' '.join(cmd)}")


def run_analysis(csv_files: list[Path], out_dir: Path, dry_run: bool) -> None:
    """Sinh bảng so sánh thống kê và file LaTeX."""
    existing = [str(f) for f in csv_files if f.exists()]
    if not existing:
        print("  [SKIP] Không có CSV nào để phân tích.")
        return
    cmd = [_python(), str(ROOT / "rl" / "analysis.py")] + existing + ["--out-dir", str(out_dir)]
    if not dry_run:
        _run(cmd, f"Analysis → {out_dir}")
    else:
        print(f"  [DRY-RUN] {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--preset",
        choices=list(MARL_EXPERIMENT_PRESETS.keys()),
        default="short",
        help="Preset thực nghiệm: smoke (kiểm thử), short (nhanh), thesis (đầy đủ).",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_PRESETS.keys()),
        default="low",
        help="Mật độ giao thông: low / medium / high (mặc định low để học merge ổn định).",
    )
    parser.add_argument(
        "--shield",
        choices=("on", "off"),
        default="on",
        help=(
            "Khiên an toàn GAMA (gate merge + shield M3c) cho TOÀN BỘ run. "
            "on (mặc định) = mọi policy có khiên. off = mọi policy KHÔNG khiên (A/B: đo ảnh hưởng khiên). "
            "Áp đồng đều greedy/random/PPO/A2C → so sánh công bằng trong từng run."
        ),
    )
    parser.add_argument("--host", default="localhost", help="GAMA headless host.")
    parser.add_argument("--port", type=int, default=1001, help="GAMA headless socket port.")
    parser.add_argument(
        "--skip-baselines", action="store_true", help="Bỏ qua bước chạy baseline."
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=list(MARL_ALGORITHMS),
        default=list(MARL_ALGORITHMS),
        help="Thuật toán MARL (mặc định: ppo và a2c).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In lệnh sẽ chạy mà không thực sự thực thi (dùng để kiểm tra cấu hình).",
    )
    parser.add_argument(
        "--select-best-checkpoint",
        action="store_true",
        help="Sau train, eval checkpoint định kỳ và copy checkpoint tốt nhất thành model chính.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Khoảng lưu checkpoint. 0 = tự chọn ~5 checkpoint/run khi bật --select-best-checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=10,
        help="Số episode dùng để chọn best checkpoint trước eval chính.",
    )
    parser.add_argument(
        "--allow-highway-collision-checkpoints",
        action="store_true",
        help=(
            "Cho phép MARL best-checkpoint chọn checkpoint có highway collision. "
            "Mặc định sẽ reject để ưu tiên safety mainline."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("both", "train", "eval", "full"),
        default="both",
        help=(
            "both/full: baseline + train + eval + plots (mặc định). "
            "train: chỉ huấn luyện. eval: chỉ đánh giá model có sẵn."
        ),
    )
    parser.add_argument(
        "--eval-stochastic",
        action="store_true",
        help=(
            "Ép eval dùng sampling (PPO/A2C) kể cả khi preset = short. "
            "Tránh ảo tưởng '100% success' do deterministic eval khoá cùng quỹ đạo."
        ),
    )
    parser.add_argument(
        "--no-ent-anneal",
        dest="ent_anneal",
        action="store_false",
        default=True,
        help=(
            "TẮT entropy annealing khi train (giữ entropy cố định). MẶC ĐỊNH BẬT — "
            "recipe a3_anneal2 (entropy 0.05→0.002, giữ 65%) cho policy deterministic sạch."
        ),
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Tên thư mục con tách output của LẦN CHẠY này: models/<tag>/, logs/<tag>/, plots/<preset>/<tag>/. "
            "MẶC ĐỊNH (không truyền): tự sinh theo timestamp '<preset>_YYYYmmdd_HHMMSS' — mỗi lần chạy 1 thư mục riêng. "
            "Truyền tên tuỳ ý (vd 'baocao_v1') để tự đặt. "
            "Truyền 'none' để ghi PHẲNG như cũ (models/*.zip, logs/*.csv — dùng khi muốn đè/ghi đúng bộ thesis phẳng)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preset = MARL_EXPERIMENT_PRESETS[args.preset]
    timesteps: int = preset["timesteps"]
    eval_episodes: int = preset["episodes"]
    seeds: list[int] = preset["seeds"]
    max_episode_steps: int = preset["max_episode_steps"]

    # Tách output theo TỪNG LẦN CHẠY. Mặc định (None) → tự sinh timestamp '<preset>_YYYYmmdd_HHMMSS'.
    # 'none'/'flat'/'off' → ghi phẳng như cũ (escape hatch). Tên khác → dùng nguyên (vd 'baocao_v1').
    run_tag: str | None = args.run_tag
    if run_tag is None or run_tag == "auto":
        from datetime import datetime
        run_tag = f"{args.preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif run_tag.lower() in ("none", "flat", "off"):
        run_tag = None

    models_dir = MODEL_DIR / run_tag if run_tag else MODEL_DIR
    logs_dir = LOG_DIR / run_tag if run_tag else LOG_DIR
    plot_out_dir = (PLOT_DIR / args.preset / run_tag) if run_tag else (PLOT_DIR / args.preset)

    print(f"\n{'#'*60}")
    print(f"  ĐỒ ÁN RL NHẬP LÀN CAO TỐC — THỰC NGHIỆM MARL")
    print(f"  Preset  : {args.preset}")
    print(f"  Khiên   : {args.shield.upper()} (gate merge + shield M3c cho mọi policy)")
    print(f"  Scenario: {args.scenario}")
    print(f"  Thuật toán: {args.algos}")
    print(f"  Seeds   : {seeds}")
    print(f"  MARL timesteps: {timesteps:,}  agents: {list(MARL_AGENTS)}")
    print(f"  Eval episodes: {eval_episodes}")
    if run_tag:
        print(f"  Run-tag : {run_tag}  → models/{run_tag}/, logs/{run_tag}/, plots/{args.preset}/{run_tag}/")
    else:
        print(f"  Run-tag : (none) → ghi phẳng models/, logs/, plots/{args.preset}/")
    print(f"{'#'*60}")

    # Chỉ tạo thư mục khi chạy thật (tránh dry-run sinh thư mục timestamp rỗng mỗi lần gọi).
    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

    # Sua bug #9: LUON va GAML theo scenario. Truoc day chi va khi scenario != "medium",
    # nhung default GAML co nb_cars_max=20 (de tranh gridlock GUI) - khac han config medium=45.
    # -> preset thesis + medium thuc te chay density 20 nhung CSV ghi nhan "medium" -> bao cao sai.
    # Va luon dam bao mat do thuc trung voi nhan scenario.
    _apply_gaml_scenario = True
    if _apply_gaml_scenario and not args.dry_run:
        print(f"\n  Áp dụng scenario '{args.scenario}' vào GAML (MARL + archive baseline)...")
        apply_scenario_all_gaml(args.scenario)
        # Khiên an toàn cho CẢ run (sau scenario patch; restore_gaml_backup cuối run sẽ revert về true).
        if args.shield == "off":
            print("  [shield] enable_safety_shield <- false cho TOÀN BỘ run (mọi policy KHÔNG khiên).")
            set_safety_shield(False)

    if args.preset == "thesis" and not args.select_best_checkpoint:
        args.select_best_checkpoint = True
        if args.checkpoint_interval == 0:
            args.checkpoint_interval = _default_checkpoint_interval(timesteps)

    try:
        _run_all_experiments(
            args=args,
            timesteps=timesteps,
            eval_episodes=eval_episodes,
            seeds=seeds,
            max_episode_steps=max_episode_steps,
            plot_out_dir=plot_out_dir,
            run_tag=run_tag,
            logs_dir=logs_dir,
            models_dir=models_dir,
        )
    finally:
        if _apply_gaml_scenario and not args.dry_run:
            print(f"\n  Khôi phục GAML về medium density...")
            restore_gaml_backup()

    print(f"\n{'#'*60}")
    print("  HOÀN THÀNH TOÀN BỘ THỰC NGHIỆM")
    print(f"  Biểu đồ và bảng tổng hợp: {plot_out_dir}")
    if run_tag:
        print(f"  Models : {models_dir}")
        print(f"  Logs   : {logs_dir}")
    print(f"{'#'*60}\n")


def _run_all_experiments(
    *,
    args: argparse.Namespace,
    timesteps: int,
    eval_episodes: int,
    seeds: list[int],
    max_episode_steps: int,
    plot_out_dir: Path,
    run_tag: str | None = None,
    logs_dir: Path = LOG_DIR,
    models_dir: Path = MODEL_DIR,
) -> None:
    """Thân thực nghiệm — tách riêng để main() có thể bọc trong try/finally."""
    all_csv: list[Path] = []

    run_train = args.mode in ("both", "full", "train")
    run_eval_step = args.mode in ("both", "full", "eval")
    run_baselines_step = args.mode in ("both", "full") and not args.skip_baselines

    if run_baselines_step:
        baseline_csvs = run_baselines(
            episodes=eval_episodes,
            seed=seeds[0],
            host=args.host,
            port=args.port,
            max_episode_steps=max_episode_steps,
            scenario=args.scenario,
            dry_run=args.dry_run,
            logs_dir=logs_dir,
        )
        all_csv.extend(baseline_csvs)
    elif args.mode in ("both", "full"):
        print("\n  [SKIP] Bỏ qua bước baseline theo --skip-baselines.")

    if not run_train and not run_eval_step:
        print("\n  [SKIP] Không train/eval theo --mode.")
        return

    if run_train or run_eval_step:
        print(f"\n{'='*60}")
        if run_train:
            print("  Bắt đầu huấn luyện MARL (merging_0 + highway_0/1/2, shared policy)...")
        else:
            print("  Chỉ đánh giá model MARL có sẵn (--mode eval)...")
        print(f"  Thuật toán: {args.algos}")

    for algo in args.algos:
        for seed in seeds:
            base = marl_run_basename(algo, seed, timesteps, args.scenario)
            # run_name (kèm tag nếu có) phải khớp với run_training để model_path/checkpoint trỏ đúng chỗ.
            run_name = f"{run_tag}/{base}" if run_tag else base
            marl_model_path = models_dir / f"{base}.zip"

            if run_train:
                checkpoint_interval = 0
                if args.select_best_checkpoint:
                    checkpoint_interval = args.checkpoint_interval or _default_checkpoint_interval(timesteps)
                csv_path = run_training(
                    algo=algo,
                    timesteps=timesteps,
                    seed=seed,
                    host=args.host,
                    port=args.port,
                    max_episode_steps=max_episode_steps,
                    scenario=args.scenario,
                    dry_run=args.dry_run,
                    checkpoint_interval=checkpoint_interval,
                    run_tag=run_tag,
                    logs_dir=logs_dir,
                    ent_anneal=getattr(args, "ent_anneal", True),
                )
                all_csv.append(csv_path)
                if args.select_best_checkpoint:
                    select_best_checkpoint(
                        algo=algo,
                        run_name=run_name,
                        model_path=marl_model_path,
                        episodes=args.checkpoint_eval_episodes,
                        seed=seed,
                        host=args.host,
                        port=args.port,
                        max_episode_steps=max_episode_steps,
                        dry_run=args.dry_run,
                        reject_unsafe_highway=not args.allow_highway_collision_checkpoints,
                        logs_dir=logs_dir,
                    )

            if run_eval_step:
                # Stochastic eval cho PPO/A2C: argmax deterministic dễ sập về brake/keep
                # (policy có entropy cao). Thesis dùng sampling; preset short opt-in qua flag.
                eval_stochastic = args.preset == "thesis" or args.eval_stochastic
                eval_csv = run_evaluate(
                    algo=algo,
                    model_path=marl_model_path,
                    episodes=eval_episodes,
                    seed=seed,
                    host=args.host,
                    port=args.port,
                    max_episode_steps=max_episode_steps,
                    dry_run=args.dry_run,
                    stochastic=eval_stochastic,
                    logs_dir=logs_dir,
                )
                all_csv.append(eval_csv)

    # --- Bước 3: Sinh biểu đồ + phân tích thống kê ---
    print(f"\n{'='*60}")
    print("  Sinh biểu đồ tổng hợp...")
    run_plots(all_csv, plot_out_dir, args.dry_run)

    print(f"\n{'='*60}")
    print("  Phân tích thống kê và sinh bảng LaTeX...")
    run_analysis(all_csv, plot_out_dir, args.dry_run)


if __name__ == "__main__":
    main()
