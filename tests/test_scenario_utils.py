"""Unit tests cho scenario_utils.py — kiểm tra regex patterns và logic backup/restore."""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixture: tạo GAML content tối giản để test regex mà không cần file thật
# ---------------------------------------------------------------------------

_GAML_TEMPLATE = """\
global {
    int   nb_cars_max          <- 45;   // medium density

    int spawn_ramp_tick <- 0;
    reflex spawn_ramp_tickup {
        spawn_ramp_tick <- spawn_ramp_tick + 1;
        if (spawn_ramp_tick >= 20) {
            spawn_ramp_tick <- 0;
        }
    }

    int balance_tick <- 0;
    reflex balance_traffic_tickup {
        balance_tick <- balance_tick + 1;
        if (balance_tick >= 5) {
            balance_tick <- 0;
        }
    }
}
"""


def _make_fake_gaml(tmp_path: Path, content: str = _GAML_TEMPLATE) -> Path:
    p = tmp_path / "Main_Traffic.gaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Import sau khi biết ROOT để patch MODEL_PATH
# ---------------------------------------------------------------------------

from rl.config import SCENARIO_PRESETS


class TestRegexNbCarsMax:
    def test_nb_cars_max_replaced_low(self, tmp_path):
        """Pattern nb_cars_max thay đúng giá trị cho scenario low."""
        import rl.scenario_utils as su
        gaml = _make_fake_gaml(tmp_path)
        # Monkeypatch MODEL_PATH
        original = su.MODEL_PATH
        su.MODEL_PATH = gaml
        try:
            su.apply_scenario_to_gaml("low")
            content = gaml.read_text(encoding="utf-8")
            assert "nb_cars_max          <- 20" in content
        finally:
            su.MODEL_PATH = original
            bak = gaml.parent / (gaml.stem + ".gaml.bak")
            bak.unlink(missing_ok=True)

    def test_nb_cars_max_replaced_high(self, tmp_path):
        """Pattern nb_cars_max thay đúng giá trị cho scenario high."""
        import rl.scenario_utils as su
        gaml = _make_fake_gaml(tmp_path)
        original = su.MODEL_PATH
        su.MODEL_PATH = gaml
        try:
            su.apply_scenario_to_gaml("high")
            content = gaml.read_text(encoding="utf-8")
            assert "nb_cars_max          <- 70" in content
        finally:
            su.MODEL_PATH = original
            bak = gaml.parent / (gaml.stem + ".gaml.bak")
            bak.unlink(missing_ok=True)


class TestRegexBalanceTraffic:
    def test_spawn_interval_replaced(self, tmp_path):
        """Pattern balance_tick interval thay đúng interval."""
        import rl.scenario_utils as su
        gaml = _make_fake_gaml(tmp_path)
        original = su.MODEL_PATH
        su.MODEL_PATH = gaml
        try:
            su.apply_scenario_to_gaml("high")
            content = gaml.read_text(encoding="utf-8")
            cfg = SCENARIO_PRESETS["high"]
            assert f"balance_tick >= {cfg.spawn_interval}" in content
        finally:
            su.MODEL_PATH = original
            bak = gaml.parent / (gaml.stem + ".gaml.bak")
            bak.unlink(missing_ok=True)

    def test_spawn_ramp_interval_replaced(self, tmp_path):
        """Pattern spawn_ramp_tick interval thay đúng interval."""
        import rl.scenario_utils as su
        gaml = _make_fake_gaml(tmp_path)
        original = su.MODEL_PATH
        su.MODEL_PATH = gaml
        try:
            su.apply_scenario_to_gaml("low")
            content = gaml.read_text(encoding="utf-8")
            cfg = SCENARIO_PRESETS["low"]
            assert f"spawn_ramp_tick >= {cfg.spawn_interval}" in content
        finally:
            su.MODEL_PATH = original
            bak = gaml.parent / (gaml.stem + ".gaml.bak")
            bak.unlink(missing_ok=True)


class TestBackupRestore:
    def test_backup_created_and_restore_works(self, tmp_path):
        """Backup được tạo khi apply, khôi phục về nội dung gốc khi restore."""
        import rl.scenario_utils as su
        gaml = _make_fake_gaml(tmp_path)
        original_content = gaml.read_text(encoding="utf-8")
        original_path = su.MODEL_PATH
        su.MODEL_PATH = gaml
        try:
            su.apply_scenario_to_gaml("high")
            bak = gaml.parent / (gaml.stem + ".gaml.bak")
            assert bak.exists(), "Backup phải được tạo sau apply"
            assert "nb_cars_max          <- 70" in gaml.read_text(encoding="utf-8")

            su.restore_gaml_backup()
            assert not bak.exists(), "Backup phải bị xóa sau restore"
            assert gaml.read_text(encoding="utf-8") == original_content
        finally:
            su.MODEL_PATH = original_path
            bak = gaml.parent / (gaml.stem + ".gaml.bak")
            bak.unlink(missing_ok=True)

    def test_idempotent_apply(self, tmp_path):
        """Gọi apply 2 lần liên tiếp phải cho cùng kết quả (đọc từ backup)."""
        import rl.scenario_utils as su
        gaml = _make_fake_gaml(tmp_path)
        original_path = su.MODEL_PATH
        su.MODEL_PATH = gaml
        try:
            su.apply_scenario_to_gaml("high")
            content1 = gaml.read_text(encoding="utf-8")
            su.apply_scenario_to_gaml("high")
            content2 = gaml.read_text(encoding="utf-8")
            assert content1 == content2, "Apply idempotent phải cho cùng kết quả"
        finally:
            su.MODEL_PATH = original_path
            bak = gaml.parent / (gaml.stem + ".gaml.bak")
            bak.unlink(missing_ok=True)

    def test_invalid_scenario_raises(self, tmp_path):
        """Scenario không hợp lệ phải raise ValueError."""
        import rl.scenario_utils as su
        gaml = _make_fake_gaml(tmp_path)
        original_path = su.MODEL_PATH
        su.MODEL_PATH = gaml
        try:
            with pytest.raises(ValueError, match="không hợp lệ"):
                su.apply_scenario_to_gaml("extreme")
        finally:
            su.MODEL_PATH = original_path


class TestRegexOnRepoMainTraffic:
    """Đảm bảo 3 pattern trong scenario_utils vẫn khớp ``models/Main_Traffic.gaml`` thật (read-only)."""

    def test_repo_main_traffic_matches_all_scenario_patterns(self):
        from rl.config import MODEL_PATH
        import rl.scenario_utils as su

        if not MODEL_PATH.exists():
            pytest.skip("Không có models/Main_Traffic.gaml trong workspace")
        text = MODEL_PATH.read_text(encoding="utf-8")
        hits = su.scenario_regex_matches(text)
        missing = [k for k, ok in hits.items() if not ok]
        assert not missing, (
            f"Regex scenario không còn khớp GAML repo (cần cập nhật SCENARIO_GAML_REGEX hoặc format GAML): {missing}"
        )

    def test_archive_single_agent_matches_scenario_patterns(self):
        from rl.config import SINGLE_AGENT_GAML_PATH
        import rl.scenario_utils as su

        if not SINGLE_AGENT_GAML_PATH.exists():
            pytest.skip("Không có archive single-agent GAML")
        text = SINGLE_AGENT_GAML_PATH.read_text(encoding="utf-8")
        hits = su.scenario_regex_matches(text)
        missing = [k for k, ok in hits.items() if not ok]
        assert not missing, f"Archive GAML thiếu pattern scenario: {missing}"
