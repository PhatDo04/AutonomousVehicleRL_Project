"""Vị trí dành cho integration test có GAMA thật (không chạy trong CI mặc định).

Chạy tay khi socket GAMA đang mở:
    PowerShell: $env:RUN_GAMA_INTEGRATION = \"1\"
    pytest tests/test_gama_optional.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_GAMA_INTEGRATION"),
    reason="Cần GAMA headless (socket) — đặt RUN_GAMA_INTEGRATION=1 để kích hoạt",
)


def test_import_env_without_constructing() -> None:
    """Chỉ import module; không chứng minh kết nối cho đến khi có harness thật."""
    import rl.legacy.env_single  # noqa: F401

    assert rl.legacy.env_single.__name__ == "rl.legacy.env_single"
