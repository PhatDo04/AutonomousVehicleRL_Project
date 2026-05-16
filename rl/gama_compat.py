"""Compatibility helpers for the current gama-pettingzoo/gama-gymnasium stack."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np


def _configure_windows_cli_io() -> None:
    """Console Windows (cp1252) gây UnicodeEncodeError khi in tiếng Việt; ép UTF-8 khi có thể."""
    if sys.platform != "win32":
        return
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Text


def _patch_gama_parallel_env_space_cache() -> None:
    """Tránh gọi GAMA lặp cho mỗi step: GAMA 2025.6.x có thể lỗi
    ``already registered`` / synthetic ``.gaml`` khi ``get_observation_spaces`` chạy nhiều lần.
    """
    from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

    if getattr(GamaParallelEnv, "_rl_repo_space_cache_patch", False):
        return

    _orig_obs = GamaParallelEnv.observation_space
    _orig_act = GamaParallelEnv.action_space

    def observation_space(self: Any, agent: str) -> Any:
        if not hasattr(self, "_cached_obs_spaces"):
            self._cached_obs_spaces: dict[str, Any] = {}
        cache: dict[str, Any] = self._cached_obs_spaces
        if agent not in cache:
            cache[agent] = _orig_obs(self, agent)
        return cache[agent]

    def action_space(self: Any, agent: str) -> Any:
        if not hasattr(self, "_cached_act_spaces"):
            self._cached_act_spaces: dict[str, Any] = {}
        cache: dict[str, Any] = self._cached_act_spaces
        if agent not in cache:
            cache[agent] = _orig_act(self, agent)
        return cache[agent]

    GamaParallelEnv.observation_space = observation_space  # type: ignore[method-assign]
    GamaParallelEnv.action_space = action_space  # type: ignore[method-assign]
    GamaParallelEnv._rl_repo_space_cache_patch = True


def patch_gama_pettingzoo_bridge_species() -> None:
    """GAMA 2025.6.x: tên ``PetzAgent`` đụng namespace skill → ``PetzAgent[0]`` lỗi kiểu.

    Model ``Main_Traffic.gaml`` dùng species ``PzBridgeAgent``; ép wrapper socket gọi đúng tên
    (không cần sửa tay ``site-packages`` sau mỗi ``pip install``).
    """
    from gama_gymnasium.exceptions import GamaCommandError
    from gama_pettingzoo.gama_client_wrapper import GamaClientWrapperPtZ
    from gama_client.message_types import MessageTypes

    if getattr(GamaClientWrapperPtZ, "_rl_pz_bridge_patch", False):
        return

    def get_agents(self: Any, experiment_id: str) -> Any:
        return self._execute_expression(experiment_id, "pz_agents")

    def get_possible_agents(self: Any, experiment_id: str) -> Any:
        return self._execute_expression(experiment_id, "pz_possible_agents")

    def get_observation_spaces(self: Any, experiment_id: str) -> Any:
        return self._execute_expression(experiment_id, "pz_observation_spaces")

    def get_action_spaces(self: Any, experiment_id: str) -> Any:
        return self._execute_expression(experiment_id, "pz_action_spaces")

    def get_observations(self: Any, experiment_id: str) -> dict[str, Any]:
        return self._execute_expression(experiment_id, "pz_observations")

    def get_infos(self: Any, experiment_id: str) -> dict[str, Any]:
        return self._execute_expression(experiment_id, "pz_infos")

    def execute_step(self: Any, experiment_id: str, actions: Any) -> dict[str, Any]:
        self._execute_expression(
            experiment_id,
            f"pz_actions <- from_json('{actions}');",
        )
        response = self.client.step(experiment_id, sync=True)
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise GamaCommandError(f"Failed to execute step: {response}")
        # GAMA 2025.6.x can race its synthetic expression resource when data is
        # read immediately after a synchronous step over the websocket.
        time.sleep(0.005)
        return self._execute_expression(experiment_id, "pz_data")

    GamaClientWrapperPtZ.get_agents = get_agents  # type: ignore[method-assign]
    GamaClientWrapperPtZ.get_possible_agents = get_possible_agents  # type: ignore[method-assign]
    GamaClientWrapperPtZ.get_observation_spaces = get_observation_spaces  # type: ignore[method-assign]
    GamaClientWrapperPtZ.get_action_spaces = get_action_spaces  # type: ignore[method-assign]
    GamaClientWrapperPtZ.get_observations = get_observations  # type: ignore[method-assign]
    GamaClientWrapperPtZ.get_infos = get_infos  # type: ignore[method-assign]
    GamaClientWrapperPtZ.execute_step = execute_step  # type: ignore[method-assign]
    GamaClientWrapperPtZ._rl_pz_bridge_patch = True


def patch_gama_gymnasium() -> None:
    """Patch API gaps between gama-pettingzoo 0.1.0 and newer gama-gymnasium builds."""
    _configure_windows_cli_io()
    import gama_gymnasium as gama_gymnasium_module
    from gama_gymnasium.gama_client_wrapper import GamaClientWrapper
    from gama_gymnasium.space_converter import SpaceConverter

    # gama-pettingzoo imports these names from the package root, so expose them when missing.
    for name, obj in (
        ("GamaClientWrapper", GamaClientWrapper),
        ("SpaceConverter", SpaceConverter),
    ):
        if not hasattr(gama_gymnasium_module, name):
            setattr(gama_gymnasium_module, name, obj)

    # Older gama-pettingzoo calls convert_gama_to_gym_observation; newer gama-gymnasium renamed it.
    # Lưu ý: _convert_gama_to_gym_observation được gán như instance method lên class.
    # Khi gama-pettingzoo gọi converter_instance.convert_gama_to_gym_observation(space, state),
    # Python tự động truyền instance làm `self` — hoạt động đúng.
    # Nếu được gọi như unbound SpaceConverter.convert_gama_to_gym_observation(space, state)
    # thì `self` sẽ nhận `space` → TypeError. Trường hợp này chưa gặp với gama-pettingzoo 0.1.0.
    if not hasattr(SpaceConverter, "convert_gama_to_gym_observation"):
        SpaceConverter.convert_gama_to_gym_observation = _convert_gama_to_gym_observation

    _patch_gama_parallel_env_space_cache()
    patch_gama_pettingzoo_bridge_species()


def _convert_gama_to_gym_observation(self, space, state):
    """Convert raw GAMA values into Gymnasium observations with the expected dtype/shape."""
    if isinstance(space, Box):
        arr = np.asarray(state, dtype=space.dtype)
        if space.shape is not None and tuple(arr.shape) != tuple(space.shape):
            arr = arr.reshape(space.shape)
        return np.clip(arr, space.low, space.high)

    if isinstance(space, Discrete):
        return int(state)

    if isinstance(space, MultiBinary):
        return np.asarray(state, dtype=np.int8).reshape(space.shape)

    if isinstance(space, MultiDiscrete):
        return np.asarray(state, dtype=space.dtype).reshape(space.nvec.shape)

    if isinstance(space, Text):
        return str(state)

    return state
