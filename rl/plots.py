"""Generate report-ready plots from training/evaluation CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rl.config import PLOT_DIR


def as_bool_series(series: pd.Series) -> pd.Series:
    """Normalize bool columns that may come from CSV strings."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def parse_args() -> argparse.Namespace:
    """Read CSV paths and output directory from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="One or more episode metric CSV files.")
    parser.add_argument("--out-dir", default=str(PLOT_DIR), help="Directory for generated PNG plots.")
    return parser.parse_args()


def load_metrics(paths: list[str]) -> pd.DataFrame:
    """Load and concatenate metric CSV files into one dataframe."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source"] = Path(path).stem
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill new thesis metrics when plotting older smoke-test CSV files."""
    defaults = {
        "success": False,
        "collision": False,
        "failed_merge": False,
        "timeout": False,
        "mean_speed": 0.0,
        "min_front_gap": 999.0,
        "min_rear_gap": 999.0,
        "merge_step": 0,
        "throughput": 0.0,
        "shockwave_index": 0.0,
        "mainline_mean_speed": 0.0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    for column in ("success", "collision", "failed_merge", "timeout"):
        df[column] = as_bool_series(df[column])
    return df


def plot_reward_curve(df: pd.DataFrame, out_dir: Path) -> None:
    """Đường cong phần thưởng episode trung bình theo thuật toán."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="episode", y="reward", hue="algorithm", estimator="mean", errorbar="sd")
    plt.title("Phần thưởng episode theo thuật toán")
    plt.xlabel("Episode")
    plt.ylabel("Tổng phần thưởng")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve.png", dpi=160)
    plt.close()


def plot_episode_length(df: pd.DataFrame, out_dir: Path) -> None:
    """Độ dài episode trung bình — đo tốc độ xe quyết định nhập làn."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="episode", y="length", hue="algorithm", estimator="mean", errorbar="sd")
    plt.title("Độ dài episode theo thuật toán")
    plt.xlabel("Episode")
    plt.ylabel("Số bước")
    plt.tight_layout()
    plt.savefig(out_dir / "episode_length.png", dpi=160)
    plt.close()


def plot_rate(df: pd.DataFrame, out_dir: Path, column: str, title: str, filename: str) -> None:
    """Bar chart tỷ lệ outcome theo thuật toán, có error bar (±1 độ lệch chuẩn)."""
    # Dùng df gốc (không groupby trước) để seaborn tính errorbar đúng.
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="algorithm", y=column, estimator="mean", errorbar="sd")
    plt.title(title)
    plt.xlabel("Thuật toán")
    plt.ylabel("Tỷ lệ")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def plot_merge_step(df: pd.DataFrame, out_dir: Path) -> None:
    """Số bước trung bình để nhập làn thành công (chỉ tính episode success)."""
    successful = df[df["success"]]
    if successful.empty:
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(data=successful, x="algorithm", y="merge_step", estimator="mean", errorbar="sd")
    plt.title("Số bước nhập làn trung bình theo thuật toán")
    plt.xlabel("Thuật toán")
    plt.ylabel("Số bước đến khi nhập làn thành công")
    plt.tight_layout()
    plt.savefig(out_dir / "merge_step.png", dpi=160)
    plt.close()


def plot_shockwave(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart shockwave index (CV tốc độ mainline) theo thuật toán.

    Shockwave index = std(speed) / mean(speed) của xe mainline trong vùng merge.
    Giá trị thấp hơn = tốt hơn (ít sóng lùi hơn).
    """
    if "shockwave_index" not in df.columns:
        return
    summary = df.groupby("algorithm", as_index=False)["shockwave_index"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="algorithm", y="shockwave_index", ax=ax)
    ax.set_title("Chỉ số sóng lùi (Shockwave Index — thấp hơn là tốt hơn)")
    ax.set_xlabel("Thuật toán")
    ax.set_ylabel("Shockwave Index (std/mean tốc độ mainline)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "shockwave_index.png", dpi=160)
    plt.close(fig)


def plot_throughput(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart thông lượng (throughput) trung bình theo thuật toán.

    Throughput = tỷ lệ xe merge thành công / tổng xe đã cố merge trong simulation.
    Đây là chỉ số quan trọng để đánh giá 'không gây ùn tắc' theo đề cương.
    """
    if "throughput" not in df.columns:
        return
    summary = df.groupby("algorithm", as_index=False)["throughput"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="algorithm", y="throughput", ax=ax)
    ax.set_title("Thông lượng nhập làn trung bình (Throughput)")
    ax.set_xlabel("Thuật toán")
    ax.set_ylabel("Throughput (tỷ lệ merge thành công / tổng xe ramp)")
    # Throughput có thể >1 → không ép trần, để seaborn tự co trục.
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "throughput.png", dpi=160)
    plt.close(fig)


def write_summary_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a compact CSV table for the thesis result section.

    Khớp quy ước analysis.py: merge_step CHỈ tính trên episode success (>0), gap bỏ sentinel 999.
    """
    rows = []
    for algo, g in df.groupby("algorithm"):
        succ = as_bool_series(g["success"])
        merge_steps = g.loc[succ.values, "merge_step"]
        merge_steps = merge_steps[merge_steps > 0]
        front = g["min_front_gap"].replace(999.0, float("nan"))
        rear = g["min_rear_gap"].replace(999.0, float("nan"))
        row = {
            "algorithm": algo,
            "episodes": len(g),
            "success_rate": round(succ.mean(), 4),
            "collision_rate": round(as_bool_series(g["collision"]).mean(), 4),
            "failed_merge_rate": round(as_bool_series(g["failed_merge"]).mean(), 4),
            "timeout_rate": round(as_bool_series(g["timeout"]).mean(), 4),
            "avg_reward": round(g["reward"].mean(), 2),
            "avg_steps": round(g["length"].mean(), 2),
            "avg_merge_step": round(merge_steps.mean(), 2) if not merge_steps.empty else float("nan"),
            "avg_mean_speed": round(g["mean_speed"].mean(), 4),
            "avg_min_front_gap": round(front.mean(), 4) if front.notna().any() else float("nan"),
            "avg_min_rear_gap": round(rear.mean(), 4) if rear.notna().any() else float("nan"),
        }
        if "throughput" in g.columns:
            row["avg_throughput"] = round(g["throughput"].mean(), 4)
        if "shockwave_index" in g.columns:
            row["avg_shockwave_index"] = round(g["shockwave_index"].mean(), 4)
        rows.append(row)
    pd.DataFrame(rows).sort_values("algorithm").to_csv(out_dir / "summary_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Biểu đồ nâng cao (bổ sung cho báo cáo đồ án)
# ---------------------------------------------------------------------------

def plot_smoothed_reward(df: pd.DataFrame, out_dir: Path, window: int = 20) -> None:
    """Đường cong phần thưởng có làm mượt (rolling average) và dải độ lệch chuẩn.

    Giúp quan sát xu hướng hội tụ của từng thuật toán rõ hơn so với đường thô.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = sns.color_palette("tab10", n_colors=df["algorithm"].nunique())

    for color, (algo, group) in zip(palette, df.groupby("algorithm")):
        group_sorted = group.sort_values("episode").reset_index(drop=True)
        smoothed = group_sorted["reward"].rolling(window, min_periods=1).mean()
        std_roll = group_sorted["reward"].rolling(window, min_periods=1).std().fillna(0)
        ax.plot(group_sorted["episode"], smoothed, label=algo.upper(), color=color, linewidth=1.8)
        ax.fill_between(
            group_sorted["episode"],
            smoothed - std_roll,
            smoothed + std_roll,
            alpha=0.15,
            color=color,
        )

    ax.set_title(f"Đường cong học (rolling window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Phần thưởng (làm mượt)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "smoothed_reward_curve.png", dpi=160)
    plt.close(fig)


def plot_boxplot_final(df: pd.DataFrame, out_dir: Path, last_n: int | None = None) -> None:
    """Box plot phân phối phần thưởng mỗi thuật toán.

    last_n=None: dùng TOÀN BỘ rows được truyền (khi gọi với dữ liệu eval đã hội tụ —
    đúng nghĩa "hiệu năng sau khi ổn định"). last_n=N: chỉ lấy N episode cuối (khi
    truyền dữ liệu training để cắt giai đoạn khám phá đầu).
    """
    frames = []
    for algo, group in df.groupby("algorithm"):
        g = group.sort_values("episode")
        tail = (g.tail(last_n) if last_n else g).copy()
        tail["algo_label"] = algo.upper()
        frames.append(tail)

    if not frames:
        return

    plot_df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    order = sorted(plot_df["algo_label"].unique())
    sns.boxplot(data=plot_df, x="algo_label", y="reward", order=order, ax=ax)
    ax.set_title("Phân phối phần thưởng episode" + (f" ({last_n} cuối)" if last_n else " (đánh giá)"))
    ax.set_xlabel("Thuật toán")
    ax.set_ylabel("Tổng phần thưởng episode")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_final_reward.png", dpi=160)
    plt.close(fig)


def plot_radar(df: pd.DataFrame, out_dir: Path) -> None:
    """Biểu đồ radar (spider chart) so sánh đa chỉ số giữa các thuật toán.

    Năm trục:
      - Tỷ lệ thành công (cao hơn = tốt hơn)
      - An toàn: 1 − tỷ lệ va chạm (cao hơn = tốt hơn)
      - Tốc độ nhập làn: 1 − avg_merge_step chuẩn hóa (cao hơn = nhanh hơn)
      - Phần thưởng TB chuẩn hóa về [0, 1]
      - Giữ khoảng cách: avg_min_front_gap chuẩn hóa về [0, 1]
    """
    import math

    summary = (
        df.groupby("algorithm")
        .agg(
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            avg_reward=("reward", "mean"),
            # Chỉ tính trên episode thật sự nhập (merge_step>0); algo không nhập → NaN → fillna(_ms_max) = điểm thấp nhất.
            avg_merge_step=("merge_step", lambda s: s[s > 0].mean()),
            avg_min_front_gap=("min_front_gap", "mean"),
        )
        .reset_index()
    )

    def _norm(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - lo) / (hi - lo)

    summary["safety"]       = 1.0 - summary["collision_rate"]
    # Xử lý trường hợp tất cả avg_merge_step là NaN (không có episode success nào).
    # fillna(0) → _norm(0) = 0.5 không phản ánh thực tế; dùng giá trị max quan sát được.
    # Nếu toàn NaN, merge_speed = 0 (thuật toán không merge được → điểm thấp nhất).
    _ms = summary["avg_merge_step"]
    _ms_max = _ms.dropna().max()
    if pd.isna(_ms_max):
        summary["merge_speed"] = 0.0   # không có episode success nào trong toàn bộ dữ liệu
    else:
        summary["merge_speed"] = 1.0 - _norm(_ms.fillna(_ms_max))
    summary["reward_norm"]  = _norm(summary["avg_reward"])
    summary["gap_safety"]   = _norm(summary["avg_min_front_gap"].clip(upper=20.0))

    categories = ["Tỷ lệ\nthành công", "An toàn\n(1−va chạm)", "Tốc độ\nnhập làn", "Phần thưởng\n(chuẩn hóa)", "Khoảng cách\nan toàn"]
    metric_cols = ["success_rate", "safety", "merge_speed", "reward_norm", "gap_safety"]
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    palette = sns.color_palette("tab10", n_colors=len(summary))

    for color, (_, row) in zip(palette, summary.iterrows()):
        values = [float(row[col]) for col in metric_cols]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["algorithm"].upper(), color=color)
        ax.fill(angles, values, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7)
    ax.set_title("So sánh đa chỉ số giữa các thuật toán", size=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(out_dir / "radar_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


_BASELINE_ALGOS = {"random", "greedy", "heuristic", "base_rule"}


def _phase_of(source: str) -> str:
    """Phân loại 1 CSV theo tên file (khớp analysis.py). eval/baseline = KPI thật;
    training = rollout giữa train; highway/checkpoint = không dùng cho KPI merge."""
    n = str(source).lower()
    if "highway" in n:
        return "highway"
    if "_steps_eval" in n:       # eval checkpoint giữa chừng — không phải KPI
        return "checkpoint"
    if "_eval" in n:
        return "eval"
    if "baseline" in n:
        return "baseline"
    return "training"


def split_phases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (df_kpi, df_curve). df_kpi = eval+baseline (đã loại highway) để vẽ rate/radar/
    throughput/merge_step. df_curve = training rollouts để vẽ đường cong học. Có fallback cho
    trường hợp chọn lẻ vài CSV (Results tab) không suy được phase."""
    src = df["source"] if "source" in df.columns else pd.Series([""] * len(df), index=df.index)
    phase = src.map(_phase_of)
    algo_l = df["algorithm"].astype(str).str.lower()
    is_baseline = algo_l.isin(_BASELINE_ALGOS)
    is_highway = phase.eq("highway") | algo_l.str.contains("highway")
    df_kpi = df[(phase.isin(["eval", "baseline"]) | is_baseline) & ~is_highway].copy()
    df_curve = df[phase.eq("training") & ~is_baseline & ~is_highway].copy()
    if df_kpi.empty:                       # fallback: chọn lẻ CSV không rõ phase
        df_kpi = df[~is_highway].copy()
    if df_curve.empty:                     # không có training rollouts → dùng luôn KPI cho curve
        df_curve = df_kpi
    return df_kpi, df_curve


def main() -> None:
    """Create all plots needed for the first experimental report."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_metric_columns(load_metrics(args.csv))
    df_kpi, df_curve = split_phases(df)

    # Đường cong học: dùng dữ liệu TRAINING (tiến trình theo episode).
    plot_reward_curve(df_curve, out_dir)
    plot_episode_length(df_curve, out_dir)
    plot_smoothed_reward(df_curve, out_dir)

    # KPI tổng hợp: dùng EVAL+BASELINE (khớp comparison_table.csv), KHÔNG trộn training/highway.
    plot_rate(df_kpi, out_dir, "success",      "Tỷ lệ nhập làn thành công",    "success_rate.png")
    plot_rate(df_kpi, out_dir, "collision",    "Tỷ lệ va chạm",                 "collision_rate.png")
    plot_rate(df_kpi, out_dir, "failed_merge", "Tỷ lệ thất bại nhập làn",      "failed_merge_rate.png")
    plot_rate(df_kpi, out_dir, "timeout",      "Tỷ lệ hết giờ (timeout)",       "timeout_rate.png")
    plot_merge_step(df_kpi, out_dir)
    write_summary_table(df_kpi, out_dir)
    plot_boxplot_final(df_kpi, out_dir)   # last_n=None → toàn bộ eval (đã hội tụ)
    plot_radar(df_kpi, out_dir)
    plot_throughput(df_kpi, out_dir)
    plot_shockwave(df_kpi, out_dir)

    print(f"saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
