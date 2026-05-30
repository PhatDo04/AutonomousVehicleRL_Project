"""Công cụ tùy chọn: quét thư mục ``.zip`` và sinh ``registry.json`` (metadata suy ra từ tên file).

**Không** được import tự động bởi train/eval — chạy tay sau khi có model (xem README *Quản lý model*).
Dùng để lập danh mục artifact cho báo cáo hoặc dashboard nội bộ.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import MODEL_DIR, OUTPUT_DIR


@dataclass
class ModelRegistryEntry:
    """Metadata for one trained model artifact."""

    id: str
    algorithm: str
    path: str
    version: str
    seed: int | None = None
    timesteps: int | None = None
    scenario: str = "default"
    notes: str = ""


def parse_model_name(path: Path) -> ModelRegistryEntry:
    """Infer lightweight metadata from common filenames.

    Định dạng chuẩn:
      marl_ppo_seed0_200k.zip     → algorithm=marl_ppo
      marl_a2c_seed3_200k.zip     → algorithm=marl_a2c
    """
    path = path.resolve()
    stem = path.stem
    parts = stem.split("_")
    # Nhận diện MARL models: prefix "marl" → ghép "marl_algo"
    if parts and parts[0].lower() == "marl" and len(parts) >= 2:
        algorithm = f"marl_{parts[1].lower()}"
    else:
        algorithm = parts[0].lower() if parts else "unknown"

    seed = None
    seed_match = re.search(r"seed(\d+)", stem)
    if seed_match:
        seed = int(seed_match.group(1))

    timesteps = None
    step_match = re.search(r"(\d+)k", stem.lower())
    if step_match:
        timesteps = int(step_match.group(1)) * 1000

    return ModelRegistryEntry(
        id=stem,
        algorithm=algorithm,
        path=str(path.relative_to(ROOT)),
        version="_".join(parts[1:]) if len(parts) > 1 else stem,
        seed=seed,
        timesteps=timesteps,
    )


def build_registry(model_dir: Path) -> list[ModelRegistryEntry]:
    """Scan a model directory for .zip files."""
    return [parse_model_name(path) for path in sorted(model_dir.glob("*.zip"))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Directory containing .zip models.")
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR / "models" / "registry.json"),
        help="Output JSON registry path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_path = Path(args.out)
    entries = build_registry(model_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(entry) for entry in entries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"registered {len(entries)} models: {output_path}")


if __name__ == "__main__":
    main()
