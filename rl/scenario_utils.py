"""Áp dụng kịch bản mật độ giao thông trực tiếp vào Main_Traffic.gaml.

Kỷ luật khi đổi GAML:
    Regex trông đời tên ``reflex``/khối văn bản có sẵn. Nếu đổi tên/refactor GAML nhưng quên chạy
    test ``tests/test_scenario_utils.py``, vá có thể **không** thay chỗ ([WARN] không khớp pattern).
    Sau mỗi lần sửa cấu trúc relevant reflex, kiểm tra output apply hoặc cập nhật pattern trong module này.

Lý do tồn tại:
    ScenarioConfig trong config.py chỉ ghi metadata vào CSV/log — nó KHÔNG
    tự động thay đổi GAMA. Module này dùng regex để vá ba tham số trong GAML
    trước khi khởi động GAMA headless, sau đó khôi phục về bản gốc khi xong.

Workflow (được gọi từ run_experiments.py):
    1. apply_scenario_to_gaml("high")   → tạo backup, vá GAML
    2. khởi động GAMA headless
    3. chạy thực nghiệm
    4. restore_gaml_backup()            → khôi phục từ backup và xóa `.gaml.bak`

Hai tham số được vá:
    nb_cars_max             ← int trong global block
    spawn_ramp_tick interval← ngưỡng counter để sinh xe ramp NPC

Lưu ý (sua bug #8): truoc day patch luon `balance_tick` voi cung gia tri `spawn_interval`.
Hai counter co semantics khac nhau (`balance_tick` la mainline maintenance, default 10 cycle;
`spawn_ramp_tick` la ramp NPC spawn, default 25 cycle), khong nen sync chung. Giu balance_tick
o default GAML, chi va spawn_ramp_tick theo scenario.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from rl.config import MODEL_PATH, SCENARIO_PRESETS

_BACKUP_SUFFIX = ".gaml.bak"

# Regex vá scenario — dùng chung cho apply và cho ``scenario_regex_matches()`` (test / kiểm tra tay).
# Sua bug #8: bo "balance_tick interval" — khac semantics voi spawn_ramp_tick, khong sync chung.
SCENARIO_GAML_REGEX: dict[str, str] = {
    "nb_cars_max": r"(\bnb_cars_max\s*<-\s*)\d+",
    "spawn_ramp_tick interval": r"(if\s*\(\s*spawn_ramp_tick\s*>=\s*)\d+(\s*\))",
}


def scenario_regex_matches(gaml_text: str) -> dict[str, bool]:
    """Trả về từng pattern có khớp ``gaml_text`` hay không (read-only, không vá file)."""
    return {name: bool(re.search(pat, gaml_text)) for name, pat in SCENARIO_GAML_REGEX.items()}


# Khiên an toàn GAMA (gate is_merge_gap_safe + shield M3c) — A/B per-RUN cho toàn bộ thuật toán.
_SAFETY_SHIELD_REGEX = r"(\benable_safety_shield\s*<-\s*)(?:true|false)"


def set_safety_shield(enabled: bool, *, gaml_path: Path | None = None) -> None:
    """Ép cờ ``enable_safety_shield`` trong GAML (idempotent). Gọi SAU apply_scenario, TRƯỚC khi chạy.

    ``run_experiments --shield off`` set False cho cả run (train+eval+baseline). ``restore_gaml_backup()``
    cuối run sẽ revert về true (backup tạo từ bản committed mặc định = true). Bypass phía GAML còn đòi
    ``dashboard_policy="Python/SB3 model"`` nên demo GUI Heuristic luôn giữ khiên dù cờ = false.
    """
    path = gaml_path or MODEL_PATH
    content = path.read_text(encoding="utf-8")
    value = "true" if enabled else "false"
    new_content, n = re.subn(_SAFETY_SHIELD_REGEX, rf"\g<1>{value}", content)
    if n == 0:
        print("  [WARN] set_safety_shield: không tìm thấy 'enable_safety_shield <- ...' trong GAML.")
        return
    if new_content != content:
        path.write_text(new_content, encoding="utf-8")
    print(f"  [scenario_utils] enable_safety_shield <- {value}")


def _backup_path(gaml_path: Path = MODEL_PATH) -> Path:
    return gaml_path.with_suffix(_BACKUP_SUFFIX)


def _apply_scenario_to_path(scenario_name: str, gaml_path: Path) -> None:
    """Vá một file GAML in-place với các giá trị từ ScenarioPreset."""
    if scenario_name not in SCENARIO_PRESETS:
        raise ValueError(
            f"Kịch bản '{scenario_name}' không hợp lệ. "
            f"Chọn một trong: {list(SCENARIO_PRESETS)}"
        )

    scenario = SCENARIO_PRESETS[scenario_name]
    bak_path = _backup_path(gaml_path)

    # Backup bản gốc (để restore_gaml_backup() cuối pipeline trả lại nguyên trạng).
    if not bak_path.exists():
        shutil.copy2(gaml_path, bak_path)

    # Đọc từ file hiện tại (không từ .bak) để giữ mọi chỉnh tay; an toàn vì 2 regex dưới
    # idempotent (chỉ thay con số sau '<-'/'>=', vá lại nhiều lần vẫn ra đúng giá trị scenario).
    content = gaml_path.read_text(encoding="utf-8")

    def _sub_checked(pattern: str, replacement: str, text: str, label: str) -> str:
        """re.sub với cảnh báo nếu không khớp (0 thay thế → GAML có thể đã thay đổi cú pháp)."""
        new_text, n = re.subn(pattern, replacement, text)
        if n == 0:
            print(
                f"  [WARN] scenario_utils: pattern '{label}' không khớp trong GAML. "
                f"Kiểm tra cú pháp GAML — tham số có thể chưa được vá."
            )
        return new_text

    # 1–3: dùng cùng pattern với SCENARIO_GAML_REGEX / scenario_regex_matches()
    content = _sub_checked(
        SCENARIO_GAML_REGEX["nb_cars_max"],
        rf"\g<1>{scenario.nb_cars_max}",
        content,
        "nb_cars_max",
    )

    content = _sub_checked(
        SCENARIO_GAML_REGEX["spawn_ramp_tick interval"],
        rf"\g<1>{scenario.spawn_interval}\2",
        content,
        "spawn_ramp_tick interval",
    )

    # Bug #8 da bo: KHONG patch balance_tick — giu default GAML (10 cycle, mainline maintenance).

    gaml_path.write_text(content, encoding="utf-8")
    print(
        f"  [scenario_utils] {gaml_path.name} đã vá: nb_cars_max={scenario.nb_cars_max}, "
        f"spawn_interval={scenario.spawn_interval}"
    )


def apply_scenario_to_gaml(scenario_name: str, *, gaml_path: Path | None = None) -> None:
    """Vá Main_Traffic.gaml (mặc định) hoặc file GAML chỉ định."""
    _apply_scenario_to_path(scenario_name, gaml_path or MODEL_PATH)


def apply_scenario_all_gaml(scenario_name: str) -> None:
    """Vá MARL Main_Traffic.gaml theo scenario."""
    _apply_scenario_to_path(scenario_name, MODEL_PATH)


def _restore_path(gaml_path: Path) -> None:
    bak_path = _backup_path(gaml_path)
    if not bak_path.exists():
        print(f"  [scenario_utils] Không có backup cho {gaml_path.name} — bỏ qua.")
        return
    shutil.copy2(bak_path, gaml_path)
    bak_path.unlink()
    print(f"  [scenario_utils] Đã khôi phục {gaml_path.name} về medium.")


def restore_gaml_backup() -> None:
    """Khôi phục Main_Traffic.gaml từ backup."""
    _restore_path(MODEL_PATH)
