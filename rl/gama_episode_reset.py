"""Wrapper reset MARL — reload GAMA + bootstrap đủ cycle trước khi train/eval."""

from __future__ import annotations

from typing import Any

from rl.config import MARL_AGENTS

_BOOTSTRAP_STEPS = 40


def _set_sim_seed(env_ss: Any, seed: int) -> None:
    """Đẩy seed xuống GAML global ``pz_sim_seed`` (sim-scope) để traffic biến thiên theo seed.

    Lý do: wrapper gama_gymnasium chạy ``seed <- X`` ở EXPERIMENT-scope sau reload → KHÔNG reseed
    RNG của simulation đang chạy → traffic giống hệt mọi episode. Set ``pz_sim_seed`` qua expression
    rồi bootstrap_initial_cars (sim-scope) ``seed <- pz_sim_seed`` ngay trước spawn → traffic đổi theo
    seed (reproducible + paired). Phải gọi NGAY SAU reset, TRƯỚC khi step sang cycle>0 (bootstrap).
    """
    node = env_ss
    for _ in range(8):
        gc = getattr(node, "gama_client", None)
        exp = getattr(node, "experiment_id", None)
        if gc is not None and exp is not None:
            try:
                gc._execute_expression(exp, f"pz_sim_seed <- {float(int(seed))};")
            except Exception:
                pass
            return
        node = getattr(node, "env", None)
        if node is None:
            return


def reset_marl_episode(env_ss: Any, seed: int | None) -> tuple[dict, dict]:
    """Reset và chờ đủ 4 agent (bootstrap_initial_cars chạy sau cycle>0)."""
    obs, info = env_ss.reset(seed=seed)
    if seed is not None:
        _set_sim_seed(env_ss, seed)
    noop = {a: 1 for a in MARL_AGENTS}
    for _ in range(_BOOTSTRAP_STEPS):
        missing = [a for a in MARL_AGENTS if a not in obs]
        if not missing:
            break
        obs, _, terminations, _, info = env_ss.step(noop)
        if all(terminations.get(a, False) for a in MARL_AGENTS if a in obs):
            break
    return obs, info
