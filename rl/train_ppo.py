"""Entry point tiện lợi cho huấn luyện PPO — wrapper mỏng quanh ``rl.train.main`` (xem ``train_a2c.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.train import main


if __name__ == "__main__":
    # Inject the algorithm so users can run: python rl/train_ppo.py --timesteps 50000
    sys.argv = [sys.argv[0], "--algo", "ppo", *sys.argv[1:]]
    main()
