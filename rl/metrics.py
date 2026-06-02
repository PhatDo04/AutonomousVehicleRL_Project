"""CSV logging utilities for training and evaluation runs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable



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
    # LƯU Ý: CV KHÔNG phản ánh gridlock (mọi xe chậm đều nhau → std nhỏ → CV nhỏ "êm giả").
    # Dùng làm chỉ số PHỤ; thước đo ùn tắc chính là mainline_mean_speed (thấp = kẹt) + completion.
    shockwave_index: float = 0.0
    # Tốc độ trung bình dòng chính vùng merge (space-mean-speed, từ GAML sw_mean).
    # THẤP = ùn tắc/kẹt, CAO = dòng chảy thông. Thước đo congestion robust (phân biệt được gridlock).
    mainline_mean_speed: float = 0.0


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

    Dùng chung trong MARLEpisodeCSVCallback (train_marl.py) và evaluate_marl.py
    để số liệu báo cáo nhất quán.

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
    # 4 cờ suy TỪ outcome (khớp evaluate_marl.py + train_marl.py) — tránh bug success=True khi timeout.
    success = outcome == "success"
    collision = outcome == "collision"
    failed_merge = outcome == "failed_merge"
    timeout = outcome == "timeout"

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
        mainline_mean_speed=_as_float(info.get("mainline_mean_speed"), default=0.0),
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
