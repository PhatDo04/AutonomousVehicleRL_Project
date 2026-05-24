"""Wrapper reset MARL — reload GAMA + bootstrap đủ cycle trước khi train/eval."""

from __future__ import annotations

from typing import Any

from rl.config import MARL_AGENTS

_BOOTSTRAP_STEPS = 40


def reset_marl_episode(env_ss: Any, seed: int | None) -> tuple[dict, dict]:
    """Reset và chờ đủ 4 agent (bootstrap_initial_cars chạy sau cycle>0)."""
    obs, info = env_ss.reset(seed=seed)
    noop = {a: 1 for a in MARL_AGENTS}
    for _ in range(_BOOTSTRAP_STEPS):
        missing = [a for a in MARL_AGENTS if a not in obs]
        if not missing:
            break
        obs, _, terminations, _, info = env_ss.step(noop)
        if all(terminations.get(a, False) for a in MARL_AGENTS if a in obs):
            break
    return obs, info
