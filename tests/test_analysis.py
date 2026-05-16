"""Unit tests for rl/analysis.py — thống kê và bảng so sánh thuật toán.

7 test cases bao phủ:
  - _ci95: n=1 → 0.0, n>1 → dương, tính đúng theo t-distribution
  - build_comparison_table: success_rate đúng, xử lý nhiều thuật toán
  - compute_learning_efficiency: tìm đúng episode đạt ngưỡng, trả "chưa đạt" khi không đủ

Chạy: python -m pytest tests/test_analysis.py -v
"""

import math

import pandas as pd
import pytest
from scipy import stats as scipy_stats

from rl.analysis import _ci95, build_comparison_table, compute_learning_efficiency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    algorithm: str,
    successes: list[bool],
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal DataFrame compatible with analysis functions."""
    n = len(successes)
    return pd.DataFrame({
        "algorithm":     [algorithm] * n,
        "seed":          [seed] * n,
        "episode":       list(range(1, n + 1)),
        "success":       successes,
        "collision":     [False] * n,
        "failed_merge":  [False] * n,
        "timeout":       [not s for s in successes],
        "reward":        [10.0 if s else -5.0 for s in successes],
        "length":        [50] * n,
        "merge_step":    [25 if s else 0 for s in successes],
        "mean_speed":    [0.7] * n,
        "min_front_gap": [2.0] * n,
        "min_rear_gap":  [2.0] * n,
        "throughput":    [0.8 if s else 0.0 for s in successes],
        "shockwave_index": [0.1] * n,
    })


# ---------------------------------------------------------------------------
# _ci95
# ---------------------------------------------------------------------------

class TestCi95:
    def test_single_sample_returns_zero(self):
        """n=1 → cannot estimate CI → return 0.0."""
        result = _ci95(pd.Series([1.0]))
        assert result == 0.0

    def test_multiple_samples_positive(self):
        """n>1 with non-zero variance → CI half-width is positive."""
        result = _ci95(pd.Series([0.0, 1.0, 0.5, 0.8]))
        assert result > 0.0

    def test_matches_manual_calculation(self):
        """Verify against scipy t.ppf calculation for 2-element series."""
        series = pd.Series([0.0, 1.0])
        n = 2
        std = series.std(ddof=1)      # = sqrt(0.5) ≈ 0.7071
        sem = std / math.sqrt(n)
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        expected = round(t_crit * sem, 4)
        assert _ci95(series) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def test_all_success_gives_rate_one(self):
        """All-success episodes → success_rate_mean == 1.0."""
        df = _make_df("dqn", [True] * 10)
        comp = build_comparison_table(df)
        row = comp[comp["algorithm"] == "dqn"].iloc[0]
        assert row["success_rate_mean"] == pytest.approx(1.0)

    def test_multiple_algorithms_produce_separate_rows(self):
        """Two algorithms → two rows in comparison table."""
        df = pd.concat([
            _make_df("dqn", [True, False, True, True]),
            _make_df("ppo", [True, True, True, True]),
        ], ignore_index=True)
        comp = build_comparison_table(df)
        assert set(comp["algorithm"]) == {"dqn", "ppo"}
        assert len(comp) == 2

    def test_mixed_outcomes_rate_fraction(self):
        """3 success out of 4 episodes → success_rate_mean ≈ 0.75."""
        df = _make_df("a2c", [True, True, True, False])
        comp = build_comparison_table(df)
        row = comp[comp["algorithm"] == "a2c"].iloc[0]
        assert row["success_rate_mean"] == pytest.approx(0.75, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_learning_efficiency
# ---------------------------------------------------------------------------

class TestLearningEfficiency:
    def test_finds_first_episode_above_threshold(self):
        """10 consecutive successes → rolling(10) hits 1.0 at episode 10."""
        df = _make_df("ppo", [True] * 10)
        eff = compute_learning_efficiency(df, window=10, threshold=0.5)
        assert len(eff) == 1
        assert eff.iloc[0]["first_episode_above_threshold"] == 10

    def test_never_reached_returns_label(self):
        """All failures → success rate never reaches threshold → 'chưa đạt'."""
        df = _make_df("dqn", [False] * 15)
        eff = compute_learning_efficiency(df, window=10, threshold=0.5)
        assert eff.iloc[0]["first_episode_above_threshold"] == "chưa đạt"

    def test_baselines_excluded_from_efficiency(self):
        """Baseline algorithms (random, greedy) should be excluded from efficiency table."""
        df = pd.concat([
            _make_df("random", [True] * 10),
            _make_df("ppo", [True] * 10),
        ], ignore_index=True)
        eff = compute_learning_efficiency(df, window=5, threshold=0.5)
        algos = list(eff["algorithm"])
        assert "random" not in algos
        assert "ppo" in algos
