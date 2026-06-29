"""Compatibility helpers for the current gama-pettingzoo/gama-gymnasium stack.

Đây là tầng "vá tương thích" giữa thư viện cầu nối (gama-pettingzoo/gama-gymnasium)
và phiên bản GAMA 2025.6.x. Gọi ``patch_gama_gymnasium()`` MỘT LẦN trước khi dùng
GamaParallelEnv. Các bản vá chính:
  - ép UTF-8 cho console Windows (in tiếng Việt không lỗi),
  - cache observation/action space (tránh gọi GAMA lặp gây lỗi 'already registered'),
  - bootstrap quan sát sau reset (chờ GAMA sinh đủ xe),
  - sửa tên species cầu nối (PzBridgeAgent), chống race condition socket khi step.
"""

from __future__ import annotations

import json
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

from rl.config import MARL_AGENTS

# GAMA 2025.6.x headless: `pz_action_spaces` đôi khi trả None qua socket (Discrete map).
_DEFAULT_DISCRETE_ACTION_SPACE: dict[str, Any] = {"type": "Discrete", "n": 5}
_DEFAULT_BOX_OBS_SPACE: dict[str, Any] = {
    "type": "Box",
    "low": 0.0,
    "high": 1.0,
    "shape": [15],
    "dtype": "float",
}


def _default_marl_observation_spaces(agent_ids: list[str] | tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    ids = list(agent_ids) if agent_ids else list(MARL_AGENTS)
    return {str(aid): dict(_DEFAULT_BOX_OBS_SPACE) for aid in ids}


def _default_marl_action_spaces(agent_ids: list[str] | tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    ids = list(agent_ids) if agent_ids else list(MARL_AGENTS)
    return {str(aid): dict(_DEFAULT_DISCRETE_ACTION_SPACE) for aid in ids}


def _fetch_pz_step_payload(client: Any, experiment_id: str) -> dict[str, Any]:
    """Đọc pz_data; fallback pz_observations khi bridge race (GAMA 2025.6.x synthetic .gaml)."""
    from gama_gymnasium.exceptions import GamaCommandError

    try:
        data = client._execute_expression(experiment_id, "pz_data")
        if isinstance(data, dict) and data.get("Observations") is not None:
            return data
    except GamaCommandError:
        pass
    time.sleep(0.02)
    observations = client._execute_expression(experiment_id, "pz_observations") or {}
    return {
        "Observations": observations,
        "Rewards": {a: 0.0 for a in MARL_AGENTS},
        "Terminations": {a: False for a in MARL_AGENTS},
        "Truncations": {a: False for a in MARL_AGENTS},
        "Infos": {a: {} for a in MARL_AGENTS},
    }


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


def _bootstrap_pz_observations_after_reset(env: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sau reset GAMA, ``pz_observations`` thường None cho đến khi bootstrap_initial_cars chạy."""
    client = env.gama_client
    exp = env.experiment_id
    agents = list(env.possible_agents)
    noop = json.dumps({a: 1 for a in agents})  # hành động "giữ tốc" cho mọi agent (JSON).

    def _ready(states: Any) -> bool:
        # "Sẵn sàng" = đã có quan sát (khác None) cho đủ mọi agent.
        return isinstance(states, dict) and all(
            a in states and states[a] is not None for a in agents
        )

    states = client.get_observations(exp)
    infos = client.get_infos(exp) or {}
    if _ready(states):
        return states, infos                   # may mắn đã sẵn ngay → trả luôn.

    # Chưa sẵn → bước "giữ tốc" tối đa 25 lần, chờ GAMA chạy bootstrap_initial_cars.
    for _ in range(25):
        client._execute_expression(exp, f"pz_actions <- from_json('{noop}');")  # gán hành động.
        client.client.step(exp, sync=True)     # tiến 1 bước mô phỏng (đồng bộ).
        time.sleep(0.04)                        # chờ GAMA xử lý (chống race socket).
        states = client.get_observations(exp)
        infos = client.get_infos(exp) or infos
        env.agents = client.get_agents(exp)
        if _ready(states):
            return states, infos               # đủ xe → xong.

    states = client.get_observations(exp)
    return (states if isinstance(states, dict) else {}), infos


def _patch_gama_parallel_env_reset_and_step() -> None:
    from gama_gymnasium.exceptions import GamaEnvironmentError
    from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

    if getattr(GamaParallelEnv, "_rl_reset_step_obs_patch", False):
        return

    def reset(self: Any, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._seed(seed)
        self.gama_client.reset_experiment(self.experiment_id, seed)
        self.agents = self.gama_client.get_agents(self.experiment_id)
        states, infos = _bootstrap_pz_observations_after_reset(self)
        observations: dict[str, Any] = {}
        try:
            for agent in self.possible_agents:
                state = states.get(agent) if isinstance(states, dict) else None
                space = self.observation_space(agent)
                if state is None:
                    observations[agent] = np.zeros(space.shape, dtype=space.dtype)
                else:
                    observations[agent] = self.space_converter.convert_gama_to_gym_observation(
                        space, state
                    )
        except Exception as exc:
            raise GamaEnvironmentError(f"Failed to convert observation: {exc}") from exc
        return observations, infos

    def step(self: Any, actions: dict[str, Any]):
        observations, rewards, terminated, truncated, infos = _orig_step(self, actions)
        for agent in self.possible_agents:
            if agent not in observations or observations[agent] is None:
                space = self.observation_space(agent)
                observations[agent] = np.zeros(space.shape, dtype=space.dtype)
            rewards.setdefault(agent, 0.0)
            terminated.setdefault(agent, False)
            truncated.setdefault(agent, False)
            infos.setdefault(agent, {})
        return observations, rewards, terminated, truncated, infos

    _orig_step = GamaParallelEnv.step
    GamaParallelEnv.reset = reset  # type: ignore[method-assign]
    GamaParallelEnv.step = step  # type: ignore[method-assign]
    GamaParallelEnv._rl_reset_step_obs_patch = True


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
        result = self._execute_expression(experiment_id, "pz_observation_spaces")
        if not isinstance(result, dict):
            return _default_marl_observation_spaces()
        out = _default_marl_observation_spaces()
        for aid in MARL_AGENTS:
            spec = result.get(aid)
            if spec is not None:
                out[aid] = spec
        return out

    def get_action_spaces(self: Any, experiment_id: str) -> Any:
        result = self._execute_expression(experiment_id, "pz_action_spaces")
        if result is not None:
            return result
        try:
            agents = self.get_possible_agents(experiment_id)
        except Exception:
            agents = list(MARL_AGENTS)
        return _default_marl_action_spaces(agents)

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
        time.sleep(0.015)
        return _fetch_pz_step_payload(self, experiment_id)

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
    _patch_gama_parallel_env_reset_and_step()
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
