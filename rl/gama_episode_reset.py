"""Wrapper reset MARL — reload GAMA + bootstrap đủ cycle trước khi train/eval."""

from __future__ import annotations

from typing import Any

from rl.config import MARL_AGENTS

_BOOTSTRAP_STEPS = 40


def _exec_global(env_ss: Any, expr: str) -> None:
    """Chạy 1 expression GAML ở sim-scope (set global) qua gama_client. Phải gọi NGAY SAU reset,
    TRƯỚC khi step sang cycle>0 (để bootstrap đọc giá trị mới)."""
    node = env_ss
    for _ in range(8):
        gc = getattr(node, "gama_client", None)
        exp = getattr(node, "experiment_id", None)
        if gc is not None and exp is not None:
            try:
                gc._execute_expression(exp, expr)
            except Exception:
                pass
            return
        node = getattr(node, "env", None)
        if node is None:
            return


def _set_sim_seed(env_ss: Any, seed: int) -> None:
    """Đẩy seed xuống GAML ``pz_sim_seed`` để traffic biến thiên theo seed (reproducible + paired)."""
    _exec_global(env_ss, f"pz_sim_seed <- {float(int(seed))};")


def reset_marl_episode(
    env_ss: Any,
    seed: int | None,
    eval_continue: bool = False,
    postmerge_window: float | None = None,
    rl_postmerge: bool = False,
    no_shield: bool = False,
) -> tuple[dict, dict]:
    """Reset và chờ đủ agent (bootstrap_initial_cars chạy sau cycle>0).

    ``eval_continue=True`` (EVAL): set ``pz_eval_continue=1`` → merger chạy tiếp sau merge thêm
    ``pz_postmerge_window`` tick rồi end → đo SÓNG LÙI sau merge (KHÔNG dùng khi train).
    ``postmerge_window`` (nếu set) ghi đè số tick chạy tiếp (rất lớn ≈ chạy tới cuối làn)."""
    obs, info = env_ss.reset(seed=seed)
    if seed is not None:
        _set_sim_seed(env_ss, seed)
    if eval_continue:
        _exec_global(env_ss, "pz_eval_continue <- 1.0;")
        if postmerge_window is not None:
            _exec_global(env_ss, f"pz_postmerge_window <- {float(postmerge_window)};")
    if rl_postmerge:
        _exec_global(env_ss, "pz_rl_postmerge <- 1.0;")
    if no_shield:
        # ABLATION: bỏ khiên RL (IDM-cap + gate ngang) — đo nội-tâm-hóa an toàn. EVAL-only.
        _exec_global(env_ss, "pz_no_shield <- 1.0;")
    noop = {a: 1 for a in MARL_AGENTS}
    for _ in range(_BOOTSTRAP_STEPS):
        missing = [a for a in MARL_AGENTS if a not in obs]
        if not missing:
            break
        obs, _, terminations, _, info = env_ss.step(noop)
        if all(terminations.get(a, False) for a in MARL_AGENTS if a in obs):
            break
    return obs, info
