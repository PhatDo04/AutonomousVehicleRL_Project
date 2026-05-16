"""Gói Python huấn luyện RL cho đồ án nhập làn cao tốc.

API công khai theo **submodule** (tránh import nặng khi ``import rl``):

- ``rl.config`` — đường dẫn, preset, ``GamaConnectionConfig``
- ``rl.train_marl`` — huấn luyện MARL (SB3)
- ``rl.legacy`` — single-agent archive (baseline / tái chạy cũ)
- ``rl.metrics`` — CSV episode, ``classify_episode``

Ví dụ::

    from rl.config import GamaConnectionConfig, MARL_AGENTS
    from rl.legacy.env_single import GamaMergingEnv
"""

__version__ = "1.0.0"
__author__ = "Đỗ Tiến Phát — 2251172447"

__all__ = ["__version__", "__author__"]
