"""Kiểm tra KPI eval sau run_experiments — dùng cho vòng lặp tự động."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_comparison_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eval_ok(
    plot_dir: Path,
    *,
    min_success: float = 0.10,
    max_collision: float = 0.85,
    min_greedy_success: float = 0.05,
) -> tuple[bool, str]:
    """Trả về (ok, message) dựa trên comparison_table.csv (eval KPI)."""
    table = plot_dir / "comparison_table.csv"
    rows = _read_comparison_table(table)
    if not rows:
        return False, f"Thiếu {table}"

    by_algo = {r["algorithm"]: r for r in rows}

    def f(key: str, algo: str) -> float:
        return float(by_algo[algo].get(key, 0) or 0)

    issues: list[str] = []

    for algo in ("marl_ppo", "marl_a2c"):
        if algo not in by_algo:
            issues.append(f"thiếu {algo}")
            continue
        sr = f("success_rate_mean", algo)
        cr = f("collision_rate_mean", algo)
        if sr < min_success:
            issues.append(f"{algo} success={sr:.1%} < {min_success:.0%}")
        if cr > max_collision:
            issues.append(f"{algo} collision={cr:.1%} > {max_collision:.0%}")

    if "greedy" in by_algo:
        gs = f("success_rate_mean", "greedy")
        gc = f("collision_rate_mean", "greedy")
        if gs < min_greedy_success and gc > 0.95:
            issues.append(f"greedy success={gs:.1%} collision={gc:.1%} (môi trường/heuristic)")

    if issues:
        return False, "; ".join(issues)
    ppo_sr = f("success_rate_mean", "marl_ppo") if "marl_ppo" in by_algo else 0
    a2c_sr = f("success_rate_mean", "marl_a2c") if "marl_a2c" in by_algo else 0
    return True, f"OK — marl_ppo success={ppo_sr:.1%}, marl_a2c success={a2c_sr:.1%}"


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
    p = argparse.ArgumentParser()
    p.add_argument("--plot-dir", type=Path, default=ROOT / "outputs" / "plots" / "short")
    p.add_argument("--min-success", type=float, default=0.10)
    p.add_argument("--max-collision", type=float, default=0.85)
    args = p.parse_args()
    ok, msg = eval_ok(
        args.plot_dir,
        min_success=args.min_success,
        max_collision=args.max_collision,
    )
    _safe_print(msg)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
