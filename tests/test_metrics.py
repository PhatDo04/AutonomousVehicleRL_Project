"""Unit tests for rl/metrics.py — build_episode_metric normalization.

9 test cases bao phủ:
  - Phân loại outcome: success / collision / timeout / failed_merge
  - Pass-through: throughput, shockwave_index
  - Chuyển đổi kiểu: string "true" → bool True
  - merge_step = 0 khi không phải success

Chạy: python -m pytest tests/test_metrics.py -v
"""

import pytest

from rl.metrics import build_episode_metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(**info_kwargs) -> object:
    """Shortcut: build an EpisodeMetric with minimal required args."""
    info = {"outcome": "unknown", **info_kwargs}
    return build_episode_metric(
        algorithm="test_algo",
        seed=0,
        episode=1,
        reward=5.0,
        length=30,
        terminated=True,
        truncated=False,
        info=info,
    )


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

class TestOutcomeClassification:
    def test_success_outcome(self):
        m = _build(outcome="success")
        assert m.success is True
        assert m.collision is False
        assert m.failed_merge is False
        assert m.timeout is False
        assert m.outcome == "success"

    def test_collision_outcome(self):
        m = _build(outcome="collision")
        assert m.collision is True
        assert m.success is False
        assert m.outcome == "collision"

    def test_failed_merge_outcome(self):
        m = _build(outcome="failed_merge")
        assert m.failed_merge is True
        assert m.success is False
        assert m.outcome == "failed_merge"

    def test_timeout_from_truncated(self):
        """Episode truncated by wrapper timeout → timeout=True, outcome='timeout'."""
        m = build_episode_metric(
            algorithm="test_algo",
            seed=0,
            episode=1,
            reward=0.0,
            length=300,
            terminated=False,
            truncated=True,
            info={},
        )
        assert m.timeout is True
        assert m.outcome == "timeout"
        assert m.success is False


# ---------------------------------------------------------------------------
# Field pass-through
# ---------------------------------------------------------------------------

class TestFieldPassthrough:
    def test_throughput_passthrough(self):
        m = _build(outcome="success", throughput=0.75)
        assert m.throughput == pytest.approx(0.75)

    def test_shockwave_passthrough(self):
        m = _build(outcome="success", shockwave_index=0.42)
        assert m.shockwave_index == pytest.approx(0.42)

    def test_mean_speed_passthrough(self):
        m = _build(outcome="success", mean_speed=0.65)
        assert m.mean_speed == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------

class TestTypeConversion:
    def test_string_true_success_flag(self):
        """info["success"] as string "true" should be treated as bool True."""
        m = build_episode_metric(
            algorithm="test_algo",
            seed=0,
            episode=1,
            reward=8.0,
            length=20,
            terminated=True,
            truncated=False,
            info={"outcome": "success", "success": "true"},
        )
        assert m.success is True


# ---------------------------------------------------------------------------
# merge_step logic
# ---------------------------------------------------------------------------

class TestMergeStep:
    def test_merge_step_zero_when_not_success(self):
        """merge_step should be 0 even when GAMA sends a non-zero value for failed episodes."""
        m = _build(outcome="timeout", merge_step=42)
        assert m.merge_step == 0

    def test_merge_step_preserved_when_success(self):
        """merge_step from info should be preserved when outcome is success."""
        m = _build(outcome="success", merge_step=47)
        assert m.merge_step == 47
