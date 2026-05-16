"""Unit tests for rl/baselines.py — heuristic_action rule-based policy.

Kiểm tra 9 tình huống thực tế trong kịch bản nhập làn:
  merge khi điều kiện an toàn, decelerate khi xe sau áp sát,
  desperate merge khi urgency cao, keep_speed mặc định, v.v.

Chạy: python -m pytest tests/test_baselines.py -v
"""

import numpy as np
import pytest

from rl.baselines import heuristic_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(
    speed: float = 0.8,
    in_accel_zone: float = 0.0,
    gap_rear: float = 0.5,
    speed_rear: float = 0.3,
    gap_safe: float = 0.0,
    ramp_front_gap: float = 0.5,
    urgency: float = 0.0,
) -> np.ndarray:
    """Build a 15-D observation with named fields matching baselines.py indices."""
    obs = np.zeros(15, dtype=np.float32)
    obs[0]  = speed           # norm_speed
    obs[4]  = in_accel_zone   # in_accel_zone (threshold ≥ 0.5)
    obs[7]  = gap_rear        # norm_gap_rear
    obs[8]  = speed_rear      # norm_speed_rear
    obs[9]  = gap_safe        # gap_safe (threshold ≥ 0.5)
    obs[10] = ramp_front_gap  # ramp_front_gap
    obs[12] = urgency         # urgency
    return obs


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestHeuristicMerge:
    def test_merge_when_in_accel_zone_and_safe(self):
        """Priority 1: merge immediately when in acceleration zone and gap is safe."""
        obs = _obs(in_accel_zone=1.0, gap_safe=1.0)
        assert heuristic_action(obs) == 3  # merge_attempt

    def test_no_merge_when_gap_not_safe(self):
        """In acceleration zone but gap is NOT safe → should NOT merge."""
        obs = _obs(in_accel_zone=1.0, gap_safe=0.0, urgency=0.0, speed=0.8)
        action = heuristic_action(obs)
        assert action != 3  # must not attempt merge


class TestHeuristicDecelerate:
    def test_decelerate_when_rear_tight_and_fast(self):
        """Priority 2: rear vehicle is dangerously close and fast → decelerate."""
        obs = _obs(gap_rear=0.05, speed_rear=0.8)
        assert heuristic_action(obs) == 0  # decelerate

    def test_no_decelerate_when_rear_gap_ok(self):
        """gap_rear > 0.1 → priority-2 decelerate should NOT trigger."""
        obs = _obs(gap_rear=0.15, speed_rear=0.8, speed=0.8)
        assert heuristic_action(obs) == 1  # keep_speed (default for speed ≥ 0.7)

    def test_decelerate_when_ramp_front_blocked(self):
        """Priority 3: ramp car directly ahead → decelerate."""
        obs = _obs(ramp_front_gap=0.05)
        assert heuristic_action(obs) == 0  # decelerate


class TestHeuristicUrgency:
    def test_desperate_merge_high_urgency(self):
        """Priority 4a: urgency > 0.9 with no safe gap → desperate merge attempt."""
        obs = _obs(urgency=0.95, gap_safe=0.0, in_accel_zone=1.0, speed=0.8)
        assert heuristic_action(obs) == 3  # desperate merge

    def test_wait_moderate_urgency(self):
        """Priority 4b: urgency in (0.75, 0.9] with no safe gap → wait."""
        obs = _obs(urgency=0.80, gap_safe=0.0, in_accel_zone=0.0, speed=0.8)
        assert heuristic_action(obs) == 4  # wait


class TestHeuristicSpeedDefault:
    def test_accelerate_when_slow(self):
        """Priority 5: speed < 0.7 and no other trigger → accelerate."""
        obs = _obs(speed=0.3, urgency=0.0, in_accel_zone=0.0)
        assert heuristic_action(obs) == 2  # accelerate

    def test_keep_speed_default(self):
        """No special condition active and speed already high → keep speed."""
        obs = _obs(speed=0.8, urgency=0.0, in_accel_zone=0.0, gap_safe=0.0)
        assert heuristic_action(obs) == 1  # keep_speed


class TestHeuristicPriority:
    def test_merge_wins_over_rear_gap_alert(self):
        """Priority 1 > priority 2: merge overrides even when rear vehicle is close."""
        obs = _obs(in_accel_zone=1.0, gap_safe=1.0, gap_rear=0.05, speed_rear=0.9)
        assert heuristic_action(obs) == 3  # merge wins
