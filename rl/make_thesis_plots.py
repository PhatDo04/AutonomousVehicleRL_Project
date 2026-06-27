"""Xuất bộ biểu đồ thesis (PNG 300dpi) từ dữ liệu train/eval.

Nguồn dữ liệu:
  - Learning curve : outputs/logs/curr_stage{1,2}_*_episodes.csv (tên cố định của train_curriculum.sh;
                     fallback tự tìm file curr_*_stage{1,2}_*_episodes.csv mới nhất của run_thesis_overnight).
  - G3 (sóng lùi)  : outputs/logs/sw_eval/{ppo,a2c}_n{1,2,3}_seed{0,1,2}.csv + greedy_seed{0,1,2}.csv
                     (eval lại 3 ckpt tốt nhất × 3 hạt giống đánh giá, e2e high-84, cùng giao thức Bảng 4.3).
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
    # Số là TRUNG BÌNH±std qua 3 hạt giống huấn luyện (run_thesis_overnight tag 0611_225051,
    # 0612_135116, 0612_232535). PPO: success/collision = Giai đoạn 2; yield = Giai đoạn 3
    # (yield-hunt, năng lực hợp tác đạt được); noshield = Giai đoạn 4 (propshield, năng lực cai
    # khiên đạt được). Mỗi mục tiêu là checkpoint chuyên hóa riêng của cùng dòng curriculum.
    "PPO\n(argmax)":     dict(success=86, collision=14, yield_pct=89, noshield=28,
                              shield_mrg=2.3, src="thesis_night_summary_*.md (3 train-seeds)"),
    "MAA2C\n(lấy mẫu)":  dict(success=73, collision=27, yield_pct=37, noshield=52,
                              shield_mrg=2.8, src="thesis_night_summary_*.md (3 train-seeds)"),
    "Greedy\n(luật)":    dict(success=53, collision=47, yield_pct=0, noshield=0,
                              shield_mrg=6.2, src="thesis_night_summary_*.md (GREEDY, 3 eval-seeds)"),
}

# Tiến trình nội tâm hóa theo hệ số phạt khiên. Hai điểm đầu là THĂM DÒ 1 hạt giống (seed 0);
# điểm k=5 có cả thăm dò seed-0 (50%) và xác minh 3 hạt giống (28±17%, variance cao).
PROPSHIELD_TREND = [
    ("Binary -1\n(1 seed)", 10, "noshield84.log"),
    ("Tỷ lệ k=1\n(1 seed)", 30, "yt51_200k_eval.log"),
    ("Tỷ lệ k=5\n(3 seeds: 28±17)", 28, "thesis_night_summary_*.md S4"),
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

    def _curve_csv(algo: str, stage: int) -> Path | None:
        # Overnight: night_<algo>_s{stage}_<TAG>_episodes.csv (ưu tiên, mới nhất); fallback curr_<algo>_stage{stage}_*.
        for pat in (f"night_{algo}_s{stage}_*_episodes.csv", f"curr_{algo}_stage{stage}_*_episodes.csv"):
            cands = sorted([p for p in LOGS.glob(pat) if "highway" not in p.name], key=lambda p: p.stat().st_mtime)
            if cands:
                return cands[-1]
        return None

    def _series(path: Path | None):
        rows = _read_csv(path) if path else []
        steps, succ, rew, cum = [], [], [], 0
        for r in rows:
            cum += int(float(r["length"]))
            steps.append(cum / 1000.0)
            succ.append(1.0 if r["outcome"] == "success" else 0.0)
            rew.append(float(r["reward"]))
        return steps, succ, rew

    ALGOS = [("ppo", "MAPPO", "#1f77b4"), ("a2c", "MAA2C", "#d62728")]
    for col, (stage, title) in enumerate([(1, "Giai đoạn 1 — học NHẬP LÀN (from scratch)"),
                                          (2, "Giai đoạn 2 — học E2E + hợp tác (warmstart GĐ1)")]):
        for algo, label, color in ALGOS:
            steps, succ, rew = _series(_curve_csv(algo, stage))
            if not steps:
                continue
            w = max(5, len(steps) // 20)
            axes[0][col].plot(steps, [s * 100 for s in _rolling(succ, w)], color=color, lw=1.8, label=label)
            axes[1][col].plot(steps, _rolling(rew, w), color=color, lw=1.5, label=label)
        axes[0][col].set_ylabel("Tỷ lệ thành công (%)" if col == 0 else "")
        axes[0][col].set_ylim(-5, 105)
        axes[0][col].set_title(title, fontsize=11)
        axes[0][col].grid(alpha=0.3)
        axes[0][col].legend(fontsize=9, loc="lower right")
        axes[1][col].set_ylabel("Reward/episode (trượt)" if col == 0 else "")
        # Trục x = tick môi trường tích lũy của merger; SB3 timesteps = 4× (4 agent slot / cycle).
        axes[1][col].set_xlabel("Tick môi trường tích lũy (nghìn)")
        axes[1][col].grid(alpha=0.3)
        axes[1][col].legend(fontsize=9, loc="lower right")
    fig.suptitle("Đường cong huấn luyện — curriculum 2 giai đoạn: đối sánh MAPPO vs MAA2C (overnight 3 seed)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig1_learning_curve.png", dpi=300)
    plt.close(fig)


# Ma trận giai đoạn 1 (đánh giá lấy mẫu) — env 240m, cổng-nhập-làn tắt (công bằng), shield bật.
# Mỗi ô = (mean, std qua 3 HẠT GIỐNG) — std là độ lệch chuẩn tổng thể (pop-std).
# Nguồn (đã xác minh tái lập):
#   - MAPPO/MAA2C: 3 seed train trong outputs/plots/thesis/g1_240_{low,medium,high}_on/.
#   - Greedy: 3 seed eval g1_240_{sc}_on/greedy_baseline_seed{0,1,2}.csv (seed0 từ matrix,
#     seed1/2 bù bằng greedy_topup_240.sh). VD vừa: per-seed 98/96/94 → 96.0±1.6.
#     (comparison_table.csv trong dir chỉ có seed0=98% vì sinh trước khi bù seed1/2.)
G1_SUCC = {
    "thấp":  {"MAPPO": (87.3, 9.0), "MAA2C": (80.7, 6.8), "Greedy": (28.7, 1.9)},
    "vừa":   {"MAPPO": (74.7, 11.6), "MAA2C": (49.3, 7.7), "Greedy": (96.0, 1.6)},
    "cao":   {"MAPPO": (74.7, 18.4), "MAA2C": (41.3, 2.5), "Greedy": (64.7, 3.4)},
}
G1_COLL = {
    "thấp":  {"MAPPO": (12.7, 9.0), "MAA2C": (19.3, 6.8), "Greedy": (71.3, 1.9)},
    "vừa":   {"MAPPO": (25.3, 11.6), "MAA2C": (50.7, 7.7), "Greedy": (4.0, 1.6)},
    "cao":   {"MAPPO": (25.3, 18.4), "MAA2C": (58.7, 2.5), "Greedy": (35.3, 3.4)},
}


def fig_g1_bars(out: Path) -> None:
    """Hình 4.1: giai đoạn 1 — success/va chạm theo MẬT ĐỘ (3 chính sách, no-shield công bằng, 3 hạt giống)."""
    dens = ["thấp", "vừa", "cao"]
    pols = ["MAPPO", "MAA2C", "Greedy"]
    pcol = {"MAPPO": "#1565c0", "MAA2C": "#2e7d32", "Greedy": "#c62828"}
    w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (MAT, ylab) in zip(axes, [(G1_SUCC, "Tỷ lệ thành công (%)"),
                                      (G1_COLL, "Tỷ lệ va chạm (%)")]):
        for k, pol in enumerate(pols):
            xs = [i + (k - 1) * w for i in range(len(dens))]
            vals = [MAT[d][pol][0] for d in dens]
            errs = [MAT[d][pol][1] for d in dens]
            ax.bar(xs, vals, w, yerr=errs, capsize=4, label=pol, color=pcol[pol],
                   error_kw=dict(ecolor="#333", lw=1.0))
            for xx, v in zip(xs, vals):
                ax.text(xx, v + 3, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_ylim(0, 118)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel(ylab)
        ax.set_xticks(range(len(dens)))
        ax.set_xticklabels(dens)
        ax.set_xlabel("Mật độ giao thông")
    axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Giai đoạn 1 — Thành công và va chạm theo mật độ (đánh giá lấy mẫu, 3 hạt giống)", fontsize=13)
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
    """Hình 4: Goal 3 — CV sóng lùi / tốc độ dòng chính / throughput.

    Nguồn: outputs/logs/sw_eval/ — eval LẠI 3 checkpoint tốt nhất (PPO/A2C, một/đêm)
    trên 3 hạt giống đánh giá (e2e high-84, khiên ON, PPO=argmax, A2C=stochastic),
    cùng giao thức với bộ headline 86±6% (Bảng 4.3). Cột = trung bình QUA 3 HẠT GIỐNG
    HUẤN LUYỆN; thanh sai số = độ lệch chuẩn giữa 3 hạt giống; n = tổng số ván thành công.
    """
    SW = LOGS / "sw_eval"
    # mỗi policy: danh sách "nhóm hạt giống"; mỗi nhóm = các file CSV gộp lại (eval-seeds).
    series = [
        ("MAPPO", [[SW / f"ppo_n{n}_seed{s}.csv" for s in (0, 1, 2)] for n in (1, 2, 3)]),
        ("MAA2C", [[SW / f"a2c_n{n}_seed{s}.csv" for s in (0, 1, 2)] for n in (1, 2, 3)]),
        ("Greedy", [[SW / f"greedy_seed{s}.csv"] for s in (0, 1, 2)]),
    ]
    metrics = [("shockwave_index", "CV sóng lùi (thấp = tốt)"),
               ("mainline_mean_speed", "Tốc độ dòng chính"),
               ("throughput", "Throughput")]

    def seedwise(groups, col):
        """Trả (mean_qua_seed, std_qua_seed, tổng_n_success)."""
        per_seed, n_succ = [], 0
        for grp in groups:
            rows = []
            for fp in grp:
                rows += [r for r in _merger_rows(_read_csv(fp)) if r.get("outcome") == "success"]
            n_succ += len(rows)
            vs = [float(r[col]) for r in rows if r.get(col, "") not in ("", "None")]
            if vs:
                per_seed.append(sum(vs) / len(vs))
        if not per_seed:
            return None, 0.0, n_succ
        mean = sum(per_seed) / len(per_seed)
        var = sum((x - mean) ** 2 for x in per_seed) / len(per_seed)
        return mean, var ** 0.5, n_succ

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (col, title) in zip(axes, metrics):
        names, vals, errs, ns = [], [], [], []
        for name, groups in series:
            m, sd, n = seedwise(groups, col)
            if m is not None:
                names.append(name); vals.append(m); errs.append(sd); ns.append(n)
        bars = ax.bar(names, vals, 0.5, yerr=errs, capsize=5, color=COLORS["flow"],
                      error_kw=dict(ecolor="#222", lw=1.2))
        for b, v, sd, n in zip(bars, vals, errs, ns):
            ax.text(b.get_x() + b.get_width() / 2, v + sd, f"{v:.3f}±{sd:.3f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.35)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Mục tiêu 3 — Dòng giao thông sau nhập làn (high 84, e2e, 3 hạt giống; chỉ ván hoàn tất lộ trình)\n"
                 "Lưu ý: CV = std/tốc-độ — tốc độ hành trình thấp của RL (cân bằng an toàn) làm CV cao; Greedy chỉ hoàn tất ~53% số ván.",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
    ax.set_title("Phạt can-thiệp-khiên: binary → tỷ lệ (k=5: xác minh 3 hạt giống, variance cao)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig6_propshield_trend.png", dpi=300)
    plt.close(fig)


# Phân phối hành động merging_0 (% bước) — nguồn: eval --log-actions @ high 84 e2e (hist_eval.log).
MERGING_ACT_HIST = {
    "PPO (argmax)":  {"giảm": 0.0, "giữ": 0.0,  "tăng": 99.3, "nhập": 0.2, "chờ": 0.5},
    "A2C (lấy mẫu)": {"giảm": 6.0, "giữ": 26.6, "tăng": 54.5, "nhập": 4.9, "chờ": 8.0},
}


def fig_shield_reliance(out: Path) -> None:
    """Hình 4.4: histogram hành động merging_0 + số lần khiên can thiệp/episode."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    # Panel TRÁI — histogram hành động merging_0 (minh họa hành-động-hiếm-sống-còn 'nhập')
    acts = ["giảm", "giữ", "tăng", "nhập", "chờ"]
    acol = {"PPO (argmax)": "#1f77b4", "A2C (lấy mẫu)": "#d62728"}
    algos = list(MERGING_ACT_HIST)
    w = 0.38
    for k, al in enumerate(algos):
        xs = [i + (k - 0.5) * w for i in range(len(acts))]
        vals = [MERGING_ACT_HIST[al][a] for a in acts]
        axes[0].bar(xs, vals, w, label=al, color=acol[al])
        for xx, v in zip(xs, vals):
            if v > 0:
                axes[0].text(xx, v + 1.5, f"{v:g}", ha="center", fontsize=8)
    axes[0].set_xticks(range(len(acts)), acts)
    axes[0].set_ylabel("% trên TỔNG số bước")
    axes[0].set_ylim(0, 108)
    axes[0].set_title("Phân phối hành động merging_0 (% trên tổng bước lái, e2e)\n'nhập' chỉ cần 1 lần/episode đúng lúc → tỷ lệ nhỏ là ĐỦ (KHÔNG phải merger ít nhập)", fontsize=9.5)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    # Chú thích chống hiểu nhầm: merge hiếm nhưng merger vẫn nhập thành công (gate 100%).
    axes[0].annotate("PPO vẫn nhập đúng lúc\n→ gate nhập làn 100%",
                     xy=(3.19, 8), xytext=(4.5, 60), fontsize=8, ha="right", color="#1f77b4",
                     arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))
    # Panel PHẢI — số lần khiên can thiệp/episode (mức ỷ khiên)
    names = list(SUMMARY)
    vals = [SUMMARY[n]["shield_mrg"] for n in names]
    bars = axes[1].bar(names, vals, 0.5, color=COLORS["shield"])
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v}", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Lần can thiệp / episode")
    axes[1].set_ylim(0, max(vals) * 1.25)
    axes[1].set_title("Số lần khiên can thiệp / episode (thấp = tự lái thật)\n(RL ỷ khiên thấp & ổn định; Greedy cao hơn + spiky khi kẹt)", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Histogram hành động & mức lệ thuộc khiên của merger (high 84, e2e)", fontsize=13)
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
