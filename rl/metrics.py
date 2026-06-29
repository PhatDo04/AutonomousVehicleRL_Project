"""CSV logging utilities for training and evaluation runs.

Định nghĩa "1 dòng số liệu của 1 episode" (EpisodeMetric) và các hàm chuẩn hóa
info thô từ GAMA thành dòng đó, rồi ghi ra CSV. Mọi bảng/biểu đồ trong báo cáo
đều dựng lại từ các file CSV này → đây là nơi quyết định "đo cái gì".
"""

from __future__ import annotations

import csv  # ghi file CSV.
from dataclasses import asdict, dataclass  # dataclass: lớp dữ liệu; asdict: đổi sang dict để ghi CSV.
from pathlib import Path
from typing import Any, Iterable


@dataclass
class EpisodeMetric:
    """One row of episode-level metrics used in plots and report tables."""

    # --- Định danh ván ---
    algorithm: str   # tên chính sách (ppo/a2c/greedy).
    seed: int        # hạt giống.
    episode: int     # số thứ tự ván.
    reward: float    # tổng phần thưởng của merging_0 trong ván.
    length: int      # độ dài ván (số bước merging_0 đi).
    terminated: bool # kết thúc do sự kiện (success/collision/failed_merge).
    truncated: bool  # kết thúc do hết giờ (timeout).
    # --- Kết cục + 4 cờ suy ra từ outcome (chỉ 1 cờ True) ---
    outcome: str = "unknown"      # "success"/"collision"/"failed_merge"/"timeout".
    success: bool = False         # nhập làn (và đi hết lộ trình ở e2e) thành công.
    collision: bool = False       # va chạm.
    failed_merge: bool = False    # hết đường mà chưa nhập làn.
    timeout: bool = False         # quá số bước tối đa.
    # --- Các chỉ số phụ ---
    mean_speed: float = 0.0       # tốc độ trung bình của xe nhập làn.
    min_front_gap: float = 999.0  # khoảng trống nhỏ nhất phía trước (đo an toàn).
    min_rear_gap: float = 999.0   # khoảng trống nhỏ nhất phía sau.
    merge_step: int = 0           # bước mà xe nhập làn thành công (0 nếu không success).
    # Throughput: tỷ lệ xe merge thành công / tổng xe đã cố merge trong simulation (0–1).
    throughput: float = 0.0
    # === GOAL 3 (hạn chế SÓNG LÙI) — đo bằng CẶP CV + mean speed (đọc CÙNG nhau) ===
    # Shockwave index = std/mean tốc độ xe mainline đoạn vùng merge (Coefficient of Variation).
    # Đây là thước đo OSCILLATION = stop-and-go: CV cao = giật cục = sóng lùi; CV thấp = tốc độ đều.
    # ĐIỂM MÙ (bắt buộc đọc kèm mean speed): gridlock (mọi xe chậm ĐỀU) → std nhỏ → CV thấp "êm GIẢ".
    # Vì vậy goal-3 chỉ đạt khi CV THẤP *VÀ* mean speed CAO (vừa không giật cục, vừa không kẹt).
    shockwave_index: float = 0.0
    # Tốc độ trung bình ĐOẠN vùng nhập làn (section/regional mean speed; arithmetic mean, chuẩn hóa 0–1).
    # Vai trò kép: (a) mức ùn tắc — THẤP=kẹt, CAO=thông; (b) GUARD chống điểm-mù-CV ở trên (gridlock).
    mainline_mean_speed: float = 0.0


# Tập các outcome được coi là "kết thúc do sự kiện" (terminated, không phải timeout).
_TERMINAL_OUTCOMES: frozenset[str] = frozenset({"success", "collision", "failed_merge"})
# Tránh spam warnings khi outcome lặp lại hàng ngàn episode.
_warned_classify_unknown_outcomes: set[str] = set()


def finalize_merging_info_on_step_limit(
    info: dict[str, Any] | None,
    *,
    hit_step_limit: bool,
) -> dict[str, Any]:
    """Giữ metrics GAMA khi vòng lặp Python dừng vì max_episode_steps (agent vẫn 'running')."""
    out = dict(info or {})       # sao chép để không sửa dict gốc.
    if not hit_step_limit:
        return out               # chưa chạm trần bước → giữ nguyên.
    outcome = str(out.get("outcome") or "")
    # Nếu GAMA chưa kịp báo kết cục (vẫn "running") mà Python đã hết bước → coi là timeout.
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
        return True, False   # sự kiện rõ ràng → terminated.
    if wrapper_truncated or outcome == "timeout":
        return False, True   # hết giờ → truncated.
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


# --- 3 hàm ép kiểu an toàn: GAMA có thể trả None/chuỗi/thiếu key → không được crash ---
def _as_bool(value: Any) -> bool:
    """Convert GAMA/PettingZoo info values to bools for CSV metrics."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)                 # 0 → False, khác 0 → True.
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}  # chuỗi → bool.
    return False                            # thiếu/không hiểu → False.


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert optional numeric info values without failing on missing GAMA keys."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default                      # None hoặc không parse được → giá trị mặc định.


def _as_int(value: Any, default: int = 0) -> int:
    """Convert optional integer info values without failing on missing GAMA keys."""
    try:
        return int(float(value))            # qua float trước để chấp nhận "3.0".
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
    # Lấy outcome từ GAMA; nếu thiếu thì suy: truncated→timeout, còn lại→unknown.
    outcome = str(info.get("outcome") or ("timeout" if truncated else "unknown"))
    # 4 cờ suy TỪ outcome (khớp evaluate_marl.py + train_marl.py) — tránh bug success=True khi timeout.
    success = outcome == "success"
    collision = outcome == "collision"
    failed_merge = outcome == "failed_merge"
    timeout = outcome == "timeout"

    # Gói tất cả vào 1 EpisodeMetric, ép kiểu an toàn cho các trường số.
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
    path.parent.mkdir(parents=True, exist_ok=True)  # tạo thư mục cha nếu chưa có.
    rows = list(rows)
    if not rows:
        return                                       # không có dòng nào → bỏ qua.

    with path.open("w", newline="", encoding="utf-8") as f:
        # Tên cột = tên các trường của EpisodeMetric (lấy từ dòng đầu).
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()                         # ghi dòng tiêu đề.
        for row in rows:
            writer.writerow(asdict(row))             # mỗi EpisodeMetric → 1 dòng CSV.
