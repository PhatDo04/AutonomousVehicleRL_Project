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
from rl.scenario_utils import apply_scenario_all_gaml, restore_gaml_backup

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


def run_baselines(
    *,
    episodes: int,
    seed: int,
    host: str,
    port: int,
    max_episode_steps: int,
    scenario: str,
    dry_run: bool,
) -> list[Path]:
    """Chạy baseline greedy, trả về danh sách CSV đã sinh."""
    csv_files: list[Path] = []
    # "greedy" = heuristic rule-based (tên trong đề cương). Bỏ "heuristic" để tránh dữ liệu trùng.
    # Random baseline đã bỏ — không còn dùng trong báo cáo (rl/baselines.py vẫn hỗ trợ nếu chạy tay).
    for policy in ("greedy",):
        out_csv = LOG_DIR / f"{policy}_baseline_seed{seed}.csv"
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
) -> Path:
    """Huấn luyện MARL (shared policy) với một seed."""
    run_name = f"marl_{algo}_seed{seed}_{timesteps // 1000}k"
    if scenario != "medium":
        run_name = f"{run_name}_{scenario}"
    out_csv = LOG_DIR / f"{run_name}_episodes.csv"
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
) -> Path:
    """Chạy evaluate deterministic sau training, trả về CSV eval.

    Metrics từ evaluate (deterministic) phân biệt rõ với metrics từ training
    (có exploration noise). Cả hai được dùng trong báo cáo nhưng cần ghi rõ nguồn.
    """
    if not model_path.exists() and not dry_run:
        print(f"  [SKIP] Model không tồn tại: {model_path}")
        return model_path.parent / "missing.csv"

    out_csv = (
        LOG_DIR / f"{model_path.stem}_eval_stochastic.csv"
        if stochastic
        else LOG_DIR / f"{model_path.stem}_eval.csv"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preset = MARL_EXPERIMENT_PRESETS[args.preset]
    timesteps: int = preset["timesteps"]
    eval_episodes: int = preset["episodes"]
    seeds: list[int] = preset["seeds"]
    max_episode_steps: int = preset["max_episode_steps"]

    print(f"\n{'#'*60}")
    print(f"  ĐỒ ÁN RL NHẬP LÀN CAO TỐC — THỰC NGHIỆM MARL")
    print(f"  Preset  : {args.preset}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Thuật toán: {args.algos}")
    print(f"  Seeds   : {seeds}")
    print(f"  MARL timesteps: {timesteps:,}  agents: {list(MARL_AGENTS)}")
    print(f"  Eval episodes: {eval_episodes}")
    print(f"{'#'*60}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plot_out_dir = PLOT_DIR / args.preset

    # Sua bug #9: LUON va GAML theo scenario. Truoc day chi va khi scenario != "medium",
    # nhung default GAML co nb_cars_max=20 (de tranh gridlock GUI) - khac han config medium=45.
    # -> preset thesis + medium thuc te chay density 20 nhung CSV ghi nhan "medium" -> bao cao sai.
    # Va luon dam bao mat do thuc trung voi nhan scenario.
    _apply_gaml_scenario = True
    if _apply_gaml_scenario and not args.dry_run:
        print(f"\n  Áp dụng scenario '{args.scenario}' vào GAML (MARL + archive baseline)...")
        apply_scenario_all_gaml(args.scenario)

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
        )
    finally:
        if _apply_gaml_scenario and not args.dry_run:
            print(f"\n  Khôi phục GAML về medium density...")
            restore_gaml_backup()

    print(f"\n{'#'*60}")
    print("  HOÀN THÀNH TOÀN BỘ THỰC NGHIỆM")
    print(f"  Biểu đồ và bảng tổng hợp: {plot_out_dir}")
    print(f"{'#'*60}\n")


def _run_all_experiments(
    *,
    args: argparse.Namespace,
    timesteps: int,
    eval_episodes: int,
    seeds: list[int],
    max_episode_steps: int,
    plot_out_dir: Path,
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
            marl_prefix = f"marl_{algo}_seed{seed}_{timesteps // 1000}k"
            if args.scenario != "medium":
                marl_prefix = f"{marl_prefix}_{args.scenario}"
            marl_model_path = MODEL_DIR / f"{marl_prefix}.zip"

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
                )
                all_csv.append(csv_path)
                if args.select_best_checkpoint:
                    select_best_checkpoint(
                        algo=algo,
                        run_name=marl_prefix,
                        model_path=marl_model_path,
                        episodes=args.checkpoint_eval_episodes,
                        seed=seed,
                        host=args.host,
                        port=args.port,
                        max_episode_steps=max_episode_steps,
                        dry_run=args.dry_run,
                        reject_unsafe_highway=not args.allow_highway_collision_checkpoints,
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
