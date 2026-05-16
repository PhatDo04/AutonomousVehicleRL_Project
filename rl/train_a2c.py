"""Entry point tiện lợi cho huấn luyện A2C.

Không trùng lặp logic: chỉ chèn ``--algo a2c`` rồi gọi ``rl.train.main``.
Dùng khi muốn lệnh ngắn ``python rl/train_a2c.py --timesteps ...`` thay vì
``python rl/train.py --algo a2c ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.train import main


if __name__ == "__main__":
    # Inject the algorithm so users can run: python rl/train_a2c.py --timesteps 50000
    sys.argv = [sys.argv[0], "--algo", "a2c", *sys.argv[1:]]
    main()
