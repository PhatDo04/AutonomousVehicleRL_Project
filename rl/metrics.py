"""CSV logging utilities for training and evaluation runs."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# The callback imports SB3, which imports PyTorch TensorBoard; keep TensorFlow out of this path.
os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")

from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class EpisodeMetric:
    """One row of episode-level metrics used in plots and report tables."""

    algorithm: str
    seed: int
    episode: int
    reward: float
    length: int
    terminated: bool
    truncated: bool
    outcome: str = "unknown"
    success: bool = False
    collision: bool = False
    failed_merge: bool = False
    timeout: bool = False
    mean_speed: float = 0.0
    min_front_gap: float = 999.0
    min_rear_gap: float = 999.0
    merge_step: int = 0
    # Throughput: tỷ lệ xe merge thành công / tổng xe đã cố merge trong simulation (0–1).
    throughput: float = 0.0
    # Shockwave index = std/mean tốc độ xe mainline vùng merge (Coefficient of Variation).
    # Thấp → dòng chảy êm, không sóng lùi. Cao → ùn tắc sóng lùi.
    shockwave_index: float = 0.0


_TERMINAL_OUTCOMES: frozenset[str] = frozenset({"success", "collision", "failed_merge"})
# Tránh spam warnings khi outcome lặp lại hàng ngàn episode.
_warned_classify_unknown_outcomes: set[str] = set()


def finalize_merging_info_on_step_limit(
    info: dict[str, Any] | None,
    *,
    hit_step_limit: bool,
) -> dict[str, Any]:
    """Giữ metrics GAMA khi vòng lặp Python dừng vì max_episode_steps (agent vẫn 'running')."""
    out = dict(info or {})
    if not hit_step_limit:
        return out
    outcome = str(out.get("outcome") or "")
    if outcome in ("", "running", "unknown"):
        out["outcome"] = "timeout"
        out["timeout"] = True
    return out


def classify_episode(outcome: str, wrapper_truncated: bool = False) -> tuple[bool, bool]:
    """Phân loại episode thành (terminated, truncated) một cách nhất quán.

    Quy tắc ưu tiên:
    1. Nếu GAMA báo terminal cụ thể (success/collision/failed_merge) → terminated=True.
       Ngay cả khi wrapper cũng timeout cùng lúc, outcome GAMA có ý nghĩa học thuật hơn.
    2. Nếu không có GAMA outcome → dùng wrapper_truncated (TimeLimit hoặc timeout flag).

    Dùng chung trong EpisodeCSVCallback (train.py), MARLEpisodeCSVCallback (train_marl.py)
    và evaluate_marl.py để số liệu báo cáo nhất quán.

    Với outcome không thuộc terminal/timeout/unknown hợp lệ, dùng truncated và (tùy) ``warnings.warn``
    tối đa một lần cho mỗi chuỗi outcome lạ trong tiến trình — cân bằng log vs. debug.

    Parameters
    ----------
    outcome:
        Chuỗi outcome từ info["outcome"] của GAMA (hoặc "unknown"/"timeout" từ wrapper).
    wrapper_truncated:
        True nếu wrapper set TimeLimit.truncated hoặc timeout.

    Returns
    -------
    (terminated, truncated): chỉ một trong hai có thể True.
    """
    if outcome in _TERMINAL_OUTCOMES:
        return True, False
    if wrapper_truncated or outcome == "timeout":
        return False, True
    # Fallback: outcome không xác định — coi truncated (an toàn thống kê); cảnh báo tối đa 1 lần / outcome lạ.
    if outcome not in ("unknown", "running", "missing_agent", ""):
        if outcome not in _warned_classify_unknown_outcomes:
            _warned_classify_unknown_outcomes.add(outcome)
            import warnings
            warnings.warn(
                f"classify_episode: outcome không xác định '{outcome}' → coi là truncated.",
                stacklevel=2,
            )
    return False, True


def _as_bool(value: Any) -> bool:
    """Convert GAMA/PettingZoo info values to bools for CSV metrics."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert optional numeric info values without failing on missing GAMA keys."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    """Convert optional integer info values without failing on missing GAMA keys."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_episode_metric(
    *,
    algorithm: str,
    seed: int,
    episode: int,
    reward: float,
    length: int,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any] | None = None,
) -> EpisodeMetric:
    """Create a normalized metric row from the final environment info."""
    info = info or {}
    outcome = str(info.get("outcome") or ("timeout" if truncated else "unknown"))
    timeout = _as_bool(info.get("timeout")) or (truncated and not terminated)
    success = _as_bool(info.get("success")) or outcome == "success"
    collision = _as_bool(info.get("collision")) or outcome == "collision"
    failed_merge = _as_bool(info.get("failed_merge")) or outcome == "failed_merge"

    return EpisodeMetric(
        algorithm=algorithm,
        seed=seed,
        episode=episode,
        reward=reward,
        length=length,
        terminated=terminated,
        truncated=truncated,
        outcome=outcome,
        success=success,
        collision=collision,
        failed_merge=failed_merge,
        timeout=timeout,
        mean_speed=_as_float(info.get("mean_speed")),
        min_front_gap=_as_float(info.get("min_front_gap"), default=999.0),
        min_rear_gap=_as_float(info.get("min_rear_gap"), default=999.0),
        # merge_step: step cụ thể khi xe merge thành công (từ GAML "merge_step").
        # Khác episode_step (= tổng độ dài episode). Nếu không success thì = 0.
        merge_step=_as_int(info.get("merge_step", 0) if success else 0),
        throughput=_as_float(info.get("throughput"), default=0.0),
        shockwave_index=_as_float(info.get("shockwave_index"), default=0.0),
    )


def write_episode_metrics(path: Path, rows: Iterable[EpisodeMetric]) -> None:
    """Write episode metrics to CSV, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


class EpisodeCSVCallback(BaseCallback):
    """Ghi reward/length episode ra CSV (single-agent).

    Giả định ``locals['rewards'][0]``, ``dones[0]``, ``infos[0]`` — đúng với ``Monitor(GamaMergingEnv)`` (**một**
    vec slot). Nếu sau này bọc thêm SB3 VecEnv (nhiều replica), cần lặp hoặc chỉnh indexing.
    """

    def __init__(self, algorithm: str, seed: int, output_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.algorithm = algorithm
        self.seed = seed
        self.output_path = output_path
        self.rows: list[EpisodeMetric] = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
        """Collect per-step rewards and flush one metric row whenever an episode ends."""
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        self.current_reward += reward
        self.current_length += 1

        if done:
            gama_outcome = str(info.get("outcome", ""))
            wrapper_trunc = bool(info.get("TimeLimit.truncated", False)) or bool(info.get("timeout", False))
            is_terminated, is_truncated = classify_episode(gama_outcome, wrapper_trunc)

            self.rows.append(
                build_episode_metric(
                    algorithm=self.algorithm,
                    seed=self.seed,
                    episode=len(self.rows) + 1,
                    reward=self.current_reward,
                    length=self.current_length,
                    terminated=is_terminated,
                    truncated=is_truncated,
                    info=info,
                )
            )
            self.current_reward = 0.0
            self.current_length = 0

        return True

    def _on_training_end(self) -> None:
        """Persist metrics once training finishes."""
        write_episode_metrics(self.output_path, self.rows)
