"""Xuất bộ biểu đồ thesis (PNG 300dpi) từ dữ liệu train/eval.

Nguồn dữ liệu:
  - Learning curve : outputs/logs/curr_stage1_merge_episodes.csv + curr_stage2_e2e_episodes.csv
                     (thesis-run curriculum 2-stage from-scratch — đọc THÔ từng episode).
  - G3 (sóng lùi)  : outputs/logs/rl150k_h84.csv / rl250k_h84.csv / greedy_h84.csv (cùng mật độ high-84).
  - Bảng so sánh   : SUMMARY — số liệu đã kiểm chứng thủ công (mỗi dòng kèm nguồn log) vì các CSV eval
                     lịch sử trộn 2 mật độ (54/84); bảng này là single-source-of-truth cho bar chart.
                     Cập nhật số mới: sửa SUMMARY bên dưới rồi chạy lại — mọi hình tự đồng bộ.

Chạy:  ./.venv/Scripts/python.exe rl/make_thesis_plots.py  [--out-dir outputs/plots]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"

# ---------------------------------------------------------------------------
# SUMMARY — số liệu kiểm chứng @ HIGH density 84, road 240, e2e (window 400).
# nguồn: tên log trong outputs/logs (giữ để truy vết khi bị hỏi "số ở đâu ra").
# ---------------------------------------------------------------------------
SUMMARY = {
    # model: (success%, collision%, nhường%, no_shield_success%, shield_mrg/ván, nguồn)
    "Curriculum\n(from scratch)": dict(success=90, collision=10, yield_pct=41, noshield=30,
                                       shield_mrg=3.4, src="s2_sweep.log + curr150k_final.log"),
    "RL yt50-150k\n(hợp tác)":     dict(success=90, collision=10, yield_pct=94, noshield=10,
                                       shield_mrg=9.9, src="true_high84.log + noshield84.log"),
    "RL yt52-150k\n(propshield)":  dict(success=100, collision=0, yield_pct=20, noshield=50,
                                       shield_mrg=0.7, src="yt52_eval.log"),
    "Greedy\n(rule-based)":        dict(success=40, collision=60, yield_pct=0, noshield=0,
                                       shield_mrg=1.5, src="true_high84.log + noshield84.log"),
}

# Tiến trình nội tâm hóa theo hệ số phạt khiên (cùng nền yt50-150k, eval no-shield seed0).
PROPSHIELD_TREND = [
    ("Binary -1\n(yt50)", 10, "noshield84.log"),
    ("Tỷ lệ k=1\n(yt51-200k)", 30, "yt51_200k_eval.log"),
    ("Tỷ lệ k=5\n(yt52-150k)", 50, "yt52_eval.log"),
]

COLORS = {"success": "#2e7d32", "collision": "#c62828", "yield": "#1565c0",
          "flow": "#6a1b9a", "shield": "#ef6c00", "noshield": "#455a64"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [thiếu] {path.name} — bỏ qua")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _merger_rows(rows: list[dict]) -> list[dict]:
    """Lọc dòng MERGER: CSV eval chứa cả dòng highway (outcome running/exited, metric=0)."""
    return [r for r in rows if r.get("outcome", "") in ("success", "collision", "failed_merge", "timeout")]


def _rolling(xs: list[float], w: int) -> list[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        seg = xs[lo:i + 1]
        out.append(sum(seg) / len(seg))
    return out


def fig_learning_curve(out: Path) -> None:
    """Hình 1: learning curve 2-stage (success-rate trượt + reward) theo bước huấn luyện."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    for col, (name, title) in enumerate([("curr_stage1_merge", "Giai đoạn 1 — học NHẬP LÀN (from scratch)"),
                                         ("curr_stage2_e2e", "Giai đoạn 2 — học E2E + hợp tác (warmstart GĐ1)")]):
        rows = _read_csv(LOGS / f"{name}_episodes.csv")
        if not rows:
            continue
        steps, succ, rew = [], [], []
        cum = 0
        for r in rows:
            cum += int(float(r["length"]))
            steps.append(cum / 1000.0)
            succ.append(1.0 if r["outcome"] == "success" else 0.0)
            rew.append(float(r["reward"]))
        w = max(5, len(rows) // 20)
        axes[0][col].plot(steps, [s * 100 for s in _rolling(succ, w)], color=COLORS["success"], lw=1.8)
        axes[0][col].set_ylabel("Tỷ lệ thành công (%)" if col == 0 else "")
        axes[0][col].set_ylim(-5, 105)
        axes[0][col].set_title(title, fontsize=11)
        axes[0][col].grid(alpha=0.3)
        axes[1][col].plot(steps, _rolling(rew, w), color=COLORS["yield"], lw=1.5)
        axes[1][col].set_ylabel("Reward/episode (trượt)" if col == 0 else "")
        # Trục x = tick môi trường tích lũy của merger; SB3 timesteps = 4× (4 agent slot / cycle).
        axes[1][col].set_xlabel("Tick môi trường tích lũy (nghìn)")
        axes[1][col].grid(alpha=0.3)
    fig.suptitle("Đường cong huấn luyện — curriculum 2 giai đoạn (from scratch, 600k bước)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig1_learning_curve.png", dpi=300)
    plt.close(fig)


def fig_g1_bars(out: Path) -> None:
    """Hình 2: Goal 1 — success/collision theo model (@ high 84, có khiên)."""
    names = list(SUMMARY)
    succ = [SUMMARY[n]["success"] for n in names]
    coll = [SUMMARY[n]["collision"] for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 0.2 for i in x], succ, 0.38, label="Thành công", color=COLORS["success"])
    ax.bar([i + 0.2 for i in x], coll, 0.38, label="Va chạm", color=COLORS["collision"])
    for i, v in enumerate(succ):
        ax.text(i - 0.2, v + 1.5, f"{v}%", ha="center", fontsize=10, fontweight="bold")
    for i, v in enumerate(coll):
        ax.text(i + 0.2, v + 1.5, f"{v}%", ha="center", fontsize=10)
    ax.set_xticks(list(x), names, fontsize=10)
    ax.set_ylabel("% episode")
    ax.set_ylim(0, 112)
    ax.set_title("Mục tiêu 1 — Nhập làn an toàn (mật độ cao 84, có khiên, seed 0)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_g1_success.png", dpi=300)
    plt.close(fig)


def fig_g2_yield(out: Path) -> None:
    """Hình 3: Goal 2 — %nhường-merger (decel có chủ đích khi merger <35m)."""
    names = list(SUMMARY)
    ys = [SUMMARY[n]["yield_pct"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, ys, 0.5, color=COLORS["yield"])
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("% lần giảm tốc hướng-merger")
    ax.set_ylim(0, 108)
    ax.set_title("Mục tiêu 2 — Highway chủ động nhường xe nhập làn")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_g2_yield.png", dpi=300)
    plt.close(fig)


def fig_g3_flow(out: Path) -> None:
    """Hình 4: Goal 3 — CV sóng lùi / tốc độ dòng chính / throughput (đọc CSV @84)."""
    series = [("RL-150k", "rl150k_h84.csv"), ("RL-250k", "rl250k_h84.csv"), ("Greedy", "greedy_h84.csv")]
    metrics = [("shockwave_index", "CV sóng lùi (thấp = tốt)"),
               ("mainline_mean_speed", "Tốc độ dòng chính"),
               ("throughput", "Throughput")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (col, title) in zip(axes, metrics):
        vals, names = [], []
        for name, fn in series:
            rows = _merger_rows(_read_csv(LOGS / fn))
            vs = [float(r[col]) for r in rows if r.get(col, "") not in ("", "None")]
            if vs:
                names.append(name)
                vals.append(sum(vs) / len(vs))
        bars = ax.bar(names, vals, 0.5, color=COLORS["flow"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Mục tiêu 3 — Dòng giao thông sau nhập làn (high 84, seed 1, e2e)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig4_g3_flow.png", dpi=300)
    plt.close(fig)


def fig_ablation_shield(out: Path) -> None:
    """Hình 5: ablation có-khiên vs không-khiên (đo nội tâm hóa an toàn)."""
    names = list(SUMMARY)
    with_s = [SUMMARY[n]["success"] for n in names]
    no_s = [SUMMARY[n]["noshield"] for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 0.2 for i in x], with_s, 0.38, label="Có khiên", color=COLORS["success"])
    ax.bar([i + 0.2 for i in x], no_s, 0.38, label="KHÔNG khiên (ablation)", color=COLORS["noshield"])
    for i, (a, b) in enumerate(zip(with_s, no_s)):
        ax.text(i - 0.2, a + 1.5, f"{a}%", ha="center", fontsize=10)
        ax.text(i + 0.2, b + 1.5, f"{b}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x), names, fontsize=10)
    ax.set_ylabel("Tỷ lệ thành công (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Ablation khiên an toàn — mức nội tâm hóa kỹ năng lái")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig5_ablation_shield.png", dpi=300)
    plt.close(fig)


def fig_propshield_trend(out: Path) -> None:
    """Hình 6: tiến trình nội tâm hóa theo thiết kế phạt khiên (binary → tỷ lệ k=1 → k=5)."""
    names = [t[0] for t in PROPSHIELD_TREND]
    vals = [t[1] for t in PROPSHIELD_TREND]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(names, vals, "o-", lw=2.2, ms=9, color=COLORS["shield"])
    for n, v in zip(names, vals):
        ax.annotate(f"{v}%", (n, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Thành công KHÔNG khiên (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Phạt can-thiệp-khiên: binary → tỷ lệ (cùng nền yt50-150k)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig6_propshield_trend.png", dpi=300)
    plt.close(fig)


def fig_shield_reliance(out: Path) -> None:
    """Hình 7: mức ỷ-khiên (số lần khiên can thiệp khẩn cấp / episode, merger)."""
    names = list(SUMMARY)
    vals = [SUMMARY[n]["shield_mrg"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, vals, 0.5, color=COLORS["shield"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Lần can thiệp khẩn / episode")
    ax.set_title("Mức lệ thuộc khiên của merger (thấp = tự lái thật)\n(greedy: trung bình thấp nhưng worst-case 155/ván khi kẹt xe)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig7_shield_reliance.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "plots"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"  # render được tiếng Việt có dấu
    for fn in (fig_learning_curve, fig_g1_bars, fig_g2_yield, fig_g3_flow,
               fig_ablation_shield, fig_propshield_trend, fig_shield_reliance):
        fn(out)
        print(f"  [ok] {fn.__name__}")
    print(f"Đã xuất biểu đồ vào: {out}")


if __name__ == "__main__":
    main()
