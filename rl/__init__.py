"""Gói Python huấn luyện MARL cho đồ án nhập làn cao tốc.

API công khai theo **submodule** (tránh import nặng khi ``import rl``):

- ``rl.config`` — đường dẫn, preset, ``GamaConnectionConfig``
- ``rl.centralized_policy`` — CTDE policy (actor local / critic global)
- ``rl.marl_env`` — pipeline wrapper GAMA → PettingZoo → SB3 VecEnv
- ``rl.train_marl`` — huấn luyện MAPPO/MAA2C (SB3)
- ``rl.evaluate_marl`` — đánh giá model + log action histogram
- ``rl.metrics`` — CSV episode, ``classify_episode``

Ví dụ::

    from rl.config import GamaConnectionConfig, MARL_AGENTS
    from rl.train_marl import build_marl_model
"""

__version__ = "1.0.0"
__author__ = "Đỗ Tiến Phát — 2251172447"

__all__ = ["__version__", "__author__"]
