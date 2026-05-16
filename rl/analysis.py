"""Phân tích thống kê kết quả thực nghiệm và sinh bảng tổng hợp cho báo cáo.

Module này đọc tất cả CSV episode metrics (từ train.py, evaluate.py, baselines.py),
tính toán các chỉ số thống kê và xuất:
  - comparison_table.csv  : bảng wide (thuật toán × metric, mean ± std)
  - latex_table.tex        : bảng LaTeX copy-paste vào báo cáo
  - learning_efficiency.csv: số episode để đạt ngưỡng success rate ổn định

Sử dụng:
    python rl/analysis.py outputs/logs/*.csv --out-dir outputs/plots/thesis
    python rl/analysis.py outputs/logs/dqn_*_episodes.csv outputs/logs/ppo_*_episodes.csv
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from rl.config import PLOT_DIR
from rl.plots import as_bool_series, ensure_metric_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_phase(filename: str) -> str:
    """Phân loại CSV theo giai đoạn thực nghiệm.

    - `*_eval.csv`                    → "eval"      (KPI thật, đánh giá policy đã train)
    - `*_episodes.csv` / `*_highway_*` → "training" (rollout giữa quá trình train,
                                          mỗi vec env step có thể là 1 "episode" giả)
    - `random_baseline_*.csv`,
      `greedy_baseline_*.csv`,
      `heuristic_baseline_*.csv`      → "baseline" (chính sách cố định, mỗi row là 1 episode thật)
    """
    name = filename.lower()
    if "_eval" in name:
        return "eval"
    if "baseline" in name:
        return "baseline"
    return "training"


def load_and_merge(paths: list[str]) -> pd.DataFrame:
    """Tải các CSV, đảm bảo cột đầy đủ và ghép thành một DataFrame duy nhất.

    Mỗi row được gắn thêm cột ``phase`` (`eval` / `training` / `baseline`) dựa trên
    tên file, để bảng tổng hợp tách bạch KPI thực (eval) khỏi rollout training
    (mỗi training step có thể được CSV xuất ra như "1 episode" trong VecEnv wrapper,
    gây sai lệch nghiêm trọng nếu bị gộp chung).
    """
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            fname = Path(path).name
            df["source_file"] = fname
            df["phase"] = _classify_phase(fname)
            frames.append(df)
        except Exception as exc:
            print(f"  [WARN] Bỏ qua {path}: {exc}")
    if not frames:
        raise SystemExit("Không có CSV nào được tải thành công.")
    combined = pd.concat(frames, ignore_index=True)
    return ensure_metric_columns(combined)


def _ms(series: pd.Series, decimals: int = 2) -> str:
    """Định dạng mean ± std."""
    return f"{series.mean():.{decimals}f} ± {series.std():.{decimals}f}"


def _ci95(series: pd.Series) -> float:
    """Tính half-width của 95% confidence interval (t-distribution, two-sided).

    Dùng scipy.stats.t.ppf để lấy t-critical value chính xác thay vì 1.96 cứng.
    Trả về 0.0 nếu n <= 1 (không đủ mẫu).
    """
    n = len(series.dropna())
    if n <= 1:
        return 0.0
    sem = float(series.std(ddof=1)) / math.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    return round(float(t_crit * sem), 4)


# ---------------------------------------------------------------------------
# Bảng so sánh tổng hợp
# ---------------------------------------------------------------------------

def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tính mean ± std cho từng metric theo thuật toán."""
    rows = []
    for algo, group in df.groupby("algorithm"):
        success = as_bool_series(group["success"])
        collision = as_bool_series(group["collision"])
        failed_merge = as_bool_series(group["failed_merge"])
        timeout = as_bool_series(group["timeout"])

        merge_steps = group.loc[success, "merge_step"]

        # Loại giá trị sentinel 999.0 (episode kết thúc trước khi đo được gap thực tế)
        # để tránh làm lệch trung bình min_front_gap / min_rear_gap trong bảng báo cáo.
        gap_front = group["min_front_gap"].replace(999.0, float("nan"))
        gap_rear  = group["min_rear_gap"].replace(999.0, float("nan"))

        rows.append({
            "algorithm": algo,
            "n_episodes": len(group),
            "success_rate_mean": round(success.mean(), 4),
            "success_rate_std":  round(success.std(), 4),
            "success_rate_ci95": _ci95(success),
            "collision_rate_mean": round(collision.mean(), 4),
            "collision_rate_std":  round(collision.std(), 4),
            "collision_rate_ci95": _ci95(collision),
            "failed_merge_rate_mean": round(failed_merge.mean(), 4),
            "failed_merge_rate_std":  round(failed_merge.std(), 4),
            "failed_merge_rate_ci95": _ci95(failed_merge),
            "timeout_rate_mean": round(timeout.mean(), 4),
            "timeout_rate_std":  round(timeout.std(), 4),
            "timeout_rate_ci95": _ci95(timeout),
            "avg_reward_mean": round(group["reward"].mean(), 2),
            "avg_reward_std":  round(group["reward"].std(), 2),
            "avg_reward_ci95": _ci95(group["reward"]),
            "avg_steps_mean": round(group["length"].mean(), 2),
            "avg_steps_std":  round(group["length"].std(), 2),
            "avg_steps_ci95": _ci95(group["length"]),
            "avg_merge_step_mean": round(merge_steps.mean(), 2) if not merge_steps.empty else float("nan"),
            "avg_merge_step_std":  round(merge_steps.std(), 2) if not merge_steps.empty else float("nan"),
            "avg_merge_step_ci95": _ci95(merge_steps) if not merge_steps.empty else float("nan"),
            "avg_mean_speed_mean": round(group["mean_speed"].mean(), 4),
            "avg_mean_speed_std":  round(group["mean_speed"].std(),  4),
            "avg_mean_speed_ci95": _ci95(group["mean_speed"]),
            "avg_min_front_gap_mean": round(gap_front.mean(), 4) if gap_front.notna().any() else float("nan"),
            "avg_min_rear_gap_mean":  round(gap_rear.mean(), 4)  if gap_rear.notna().any()  else float("nan"),
            "avg_throughput_mean": round(group["throughput"].mean(), 4) if "throughput" in group.columns else float("nan"),
            "avg_throughput_std":  round(group["throughput"].std(), 4) if "throughput" in group.columns else float("nan"),
            "avg_throughput_ci95": _ci95(group["throughput"]) if "throughput" in group.columns else float("nan"),
            "avg_shockwave_mean": round(group["shockwave_index"].mean(), 4) if "shockwave_index" in group.columns else float("nan"),
            "avg_shockwave_std":  round(group["shockwave_index"].std(), 4) if "shockwave_index" in group.columns else float("nan"),
            "avg_shockwave_ci95": _ci95(group["shockwave_index"]) if "shockwave_index" in group.columns else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("algorithm").reset_index(drop=True)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


_DEFAULT_ALGO_ORDER = [
    "random", "greedy", "heuristic",   # baselines
    "dqn", "a2c", "ppo",               # single-agent RL
    "marl_a2c", "marl_ppo",            # MARL
]


def build_latex_table(
    comp: pd.DataFrame,
    algo_order: list[str] | None = None,
) -> str:
    """Sinh chuỗi LaTeX bảng so sánh thuật toán với 95% CI.

    Dùng CI95 thay vì std vì CI phản ánh độ tin cậy ước lượng trung bình —
    phù hợp hơn cho báo cáo học thuật (std phản ánh phân tán dữ liệu).

    Parameters
    ----------
    comp:
        DataFrame từ build_comparison_table().
    algo_order:
        Danh sách tên thuật toán theo thứ tự hiển thị trong bảng.
        Mặc định: random → greedy → DQN → A2C → PPO → MARL-A2C → MARL-PPO.
        Thuật toán không có trong comp sẽ bị bỏ qua.
    """
    order = algo_order or _DEFAULT_ALGO_ORDER
    present = set(comp["algorithm"].tolist())
    # Giữ thứ tự từ order, thêm các algo còn lại (chưa có trong order) vào cuối
    sorted_algos = [a for a in order if a in present] + \
                   sorted([a for a in present if a not in order])
    comp = comp.set_index("algorithm").loc[sorted_algos].reset_index()
    algos = comp["algorithm"].tolist()
    col_spec = "l" + "r" * len(algos)
    header_cols = " & ".join(f"\\textbf{{{a.upper()}}}" for a in algos)

    def row(label: str, mean_col: str, ci_col: str, fmt: str = ".3f") -> str:
        cells = []
        for _, r in comp.iterrows():
            m = r.get(mean_col)
            ci = r.get(ci_col, float("nan"))
            if pd.isna(m):
                cells.append("—")
            elif pd.isna(ci) or ci == 0.0:
                cells.append(f"{m:{fmt}}")
            else:
                cells.append(f"{m:{fmt}} $\\pm$ {ci:{fmt}}")
        return f"  {label} & " + " & ".join(cells) + " \\\\"

    n_row = f"  $n$ & " + " & ".join(str(int(r["n_episodes"])) for _, r in comp.iterrows()) + " \\\\"

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{So sánh hiệu năng các thuật toán RL (mean $\\pm$ 95\\% CI)}",
        "  \\label{tab:algo_comparison}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    \\textbf{{Chỉ số}} & {header_cols} \\\\",
        "    \\midrule",
        n_row,
        "    \\midrule",
        row("Tỷ lệ thành công",     "success_rate_mean",     "success_rate_ci95"),
        row("Tỷ lệ va chạm",        "collision_rate_mean",    "collision_rate_ci95"),
        row("Tỷ lệ thất bại merge", "failed_merge_rate_mean", "failed_merge_rate_ci95"),
        row("Tỷ lệ hết giờ",        "timeout_rate_mean",      "timeout_rate_ci95"),
        "    \\midrule",
        row("Phần thưởng TB",       "avg_reward_mean",        "avg_reward_ci95",       ".2f"),
        row("Số bước TB",           "avg_steps_mean",         "avg_steps_ci95",        ".1f"),
        row("Bước nhập làn TB",     "avg_merge_step_mean",    "avg_merge_step_ci95",   ".1f"),
        row("Tốc độ TB",            "avg_mean_speed_mean",    "avg_mean_speed_ci95",   ".4f"),
        row("Thông lượng",          "avg_throughput_mean",    "avg_throughput_ci95",   ".3f"),
        row("Chỉ số sóng lùi",     "avg_shockwave_mean",     "avg_shockwave_ci95",    ".3f"),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Learning efficiency
# ---------------------------------------------------------------------------

def compute_learning_efficiency(
    df: pd.DataFrame,
    window: int = 10,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Tính episode đầu tiên mà rolling success rate vượt ngưỡng (ổn định).

    Chỉ áp dụng cho dữ liệu training (các thuật toán RL). Baseline không có
    đường cong học nên sẽ bị bỏ qua.
    """
    # Loại baseline không có đường cong học — tính learning efficiency cho chúng không có ý nghĩa.
    _baselines = {"random", "heuristic", "greedy"}
    rl_algos = [a for a in df["algorithm"].unique() if a.lower() not in _baselines]
    rows = []
    for algo in sorted(rl_algos):
        sub = df[df["algorithm"] == algo].copy()
        if "episode" not in sub.columns:
            continue
        sub = sub.sort_values("episode").reset_index(drop=True)
        rolling_sr = as_bool_series(sub["success"]).rolling(window, min_periods=window).mean()
        above = rolling_sr[rolling_sr >= threshold]
        first_ep = int(above.index[0]) + 1 if not above.empty else None
        rows.append({
            "algorithm": algo,
            "window": window,
            "threshold": threshold,
            "first_episode_above_threshold": first_ep if first_ep is not None else "chưa đạt",
            "final_rolling_success_rate": round(float(rolling_sr.iloc[-1]) if not rolling_sr.empty else 0.0, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", nargs="+", help="Một hoặc nhiều file CSV episode metrics.")
    parser.add_argument(
        "--out-dir",
        default=str(PLOT_DIR),
        help="Thư mục lưu kết quả phân tích.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Cửa sổ rolling để tính learning efficiency (mặc định 10).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng success rate để đánh giá learning efficiency (mặc định 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang tải {len(args.csv)} file CSV...")
    df = load_and_merge(args.csv)
    print(f"  Tổng {len(df)} row từ {df['algorithm'].nunique()} thuật toán: {sorted(df['algorithm'].unique())}")
    phase_counts = df["phase"].value_counts().to_dict()
    print(f"  Phân chia theo phase: {phase_counts}")

    # Tách 2 nhóm dữ liệu:
    # - eval + baseline → KPI THẬT để báo cáo (mỗi row = 1 episode hoàn chỉnh).
    # - training       → rollout trong quá trình train (mỗi row có thể là 1 vec env step
    #                    bị wrapper đánh dấu là "episode", KHÔNG nên gộp vào KPI chính).
    df_eval = df[df["phase"].isin(["eval", "baseline"])].copy()
    df_train = df[df["phase"] == "training"].copy()

    # 1. Bảng so sánh CHÍNH (eval + baseline) — đây là bảng dùng cho báo cáo
    if not df_eval.empty:
        comp_eval = build_comparison_table(df_eval)
        comp_eval_path = out_dir / "comparison_table.csv"
        comp_eval.to_csv(comp_eval_path, index=False)
        print(f"\n  Đã lưu: {comp_eval_path} (KPI thật từ eval + baseline)")

        print("\n--- Tóm tắt EVAL + BASELINE (KPI báo cáo) ---")
        display_cols = ["algorithm", "n_episodes", "success_rate_mean", "collision_rate_mean",
                        "timeout_rate_mean", "avg_reward_mean", "avg_merge_step_mean"]
        print(comp_eval[display_cols].to_string(index=False))

        latex_str = build_latex_table(comp_eval)
        latex_path = out_dir / "latex_table.tex"
        latex_path.write_text(latex_str, encoding="utf-8")
        print(f"\n  Đã lưu bảng LaTeX: {latex_path}")
    else:
        comp_eval = pd.DataFrame()
        print("\n  [WARN] Không có CSV phase=eval/baseline — bảng so sánh CHÍNH bị bỏ qua.")
        print("         Hãy chạy `evaluate.py` để có CSV `*_eval.csv` đánh giá đúng KPI.")

    # 2. Bảng tham khảo (training rollouts) — KHÔNG dùng cho báo cáo, chỉ để debug
    if not df_train.empty:
        comp_train = build_comparison_table(df_train)
        comp_train_path = out_dir / "comparison_table_training.csv"
        comp_train.to_csv(comp_train_path, index=False)
        print(f"\n  Đã lưu: {comp_train_path} (rollout training - CHỈ THAM KHẢO)")

        print("\n--- Tóm tắt TRAINING ROLLOUTS (CHỈ THAM KHẢO, KHÔNG dùng cho báo cáo) ---")
        display_cols = ["algorithm", "n_episodes", "success_rate_mean", "collision_rate_mean",
                        "timeout_rate_mean", "avg_reward_mean"]
        print(comp_train[display_cols].to_string(index=False))
        print("  [LƯU Ý] Training rollouts có thể có 'success' giả do VecEnv wrapper")
        print("          đánh dấu mỗi vec step thành 1 'episode'. Tin tưởng cột EVAL ở trên.")

    # 3. Learning efficiency — dùng training data (cần đường cong học theo episode)
    if not df_train.empty:
        eff = compute_learning_efficiency(df_train, window=args.window, threshold=args.threshold)
        if not eff.empty:
            eff_path = out_dir / "learning_efficiency.csv"
            eff.to_csv(eff_path, index=False)
            print(f"\n  Đã lưu learning efficiency: {eff_path}")
            print("\n--- Learning Efficiency (training rollouts) ---")
            print(eff.to_string(index=False))

    print(f"\nPhân tích hoàn tất. Kết quả tại: {out_dir}")


if __name__ == "__main__":
    main()
