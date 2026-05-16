"""Gymnasium wrapper around the GAMA/PettingZoo highway merging environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from rl.config import SingleAgentGamaConfig
from rl.gama_compat import patch_gama_gymnasium


patch_gama_gymnasium()

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv  # noqa: E402


class GamaMergingEnv(gym.Env[np.ndarray, int]):
    """Single-agent Gymnasium view of the PettingZoo agent `merging_0`."""

    metadata = {"render_modes": []}

    def __init__(self, config: SingleAgentGamaConfig | None = None) -> None:
        """Tạo wrapper; **bắt buộc** GAMA headless đã chạy trên socket config.

        Gọi ``parallel_env.reset()`` ngay trong ``__init__`` để lấy observation_space / action_space
        và kiểm tra agent ``merging_0`` có mặt. Hệ quả: không có live server thì không thể khởi tạo
        (pytest offline cần mock ``GamaParallelEnv`` hoặc không import/instantiate lớp này).
        """
        super().__init__()
        self.config = config or SingleAgentGamaConfig()
        self.agent_id = self.config.agent_id
        self.steps = 0
        self.episode_reward = 0.0

        # GamaParallelEnv talks to the headless GAMA socket and exposes PettingZoo parallel API.
        self.parallel_env = GamaParallelEnv(
            gaml_experiment_path=str(self.config.gaml_path),
            gaml_experiment_name=self.config.experiment_name,
            gama_ip_address=self.config.host,
            gama_port=self.config.port,
        )

        # Reset once so spaces are available and validated before SB3 starts training.
        observations, _infos = self.parallel_env.reset(seed=self.config.simulation_seed)
        self._assert_agent_available(observations)

        # Stable-Baselines3 expects standard Gymnasium spaces on the wrapper itself.
        self.observation_space = self.parallel_env.observation_space(self.agent_id)
        self.action_space = self.parallel_env.action_space(self.agent_id)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset one episode and return the initial observation for `merging_0`."""
        super().reset(seed=seed)
        self.steps = 0
        self.episode_reward = 0.0
        observations, infos = self.parallel_env.reset(seed=seed)
        self._assert_agent_available(observations)
        return self._obs_from(observations), self._info_from(infos)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Forward one discrete action to GAMA and convert PettingZoo output to Gym output."""
        self.steps += 1

        # PettingZoo parallel API requires a dict keyed by agent id, even for one agent.
        actions = {self.agent_id: int(action)}
        observations, rewards, terminations, truncations, infos = self.parallel_env.step(actions)

        terminated = bool(terminations.get(self.agent_id, False))
        truncated = bool(truncations.get(self.agent_id, False))

        # Add a wrapper-side timeout so bad policies cannot leave training stuck in long episodes.
        if self.steps >= self.config.max_episode_steps and not terminated:
            truncated = True

        reward = float(rewards.get(self.agent_id, 0.0))
        self.episode_reward += reward
        info = self._info_from(infos)
        info["action"] = int(action)
        info["episode_step"] = self.steps
        info["episode_reward"] = self.episode_reward
        if truncated and not terminated:
            info["outcome"] = "timeout"
            info["timeout"] = True
            info.setdefault("success", False)
            info.setdefault("collision", False)
            info.setdefault("failed_merge", False)
        else:
            info["timeout"] = bool(info.get("outcome") == "timeout")

        return self._obs_from(observations), reward, terminated, truncated, info

    def close(self) -> None:
        """Close the PettingZoo/GAMA client connection."""
        self.parallel_env.close()

    def _obs_from(self, observations: dict[str, Any]) -> np.ndarray:
        """Return a clipped float32 observation for the configured agent.

        Clip vào [low, high] của observation_space (Box [0,1]) để bảo vệ SB3
        khỏi giá trị ngoài khoảng do lỗi GAML hoặc mất kết nối socket.
        """
        raw_obs = observations.get(self.agent_id)
        if raw_obs is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        arr = np.asarray(raw_obs, dtype=np.float32)
        return np.clip(arr, self.observation_space.low, self.observation_space.high)

    def _info_from(self, infos: dict[str, Any]) -> dict[str, Any]:
        """Normalize PettingZoo info payload into a plain dict for SB3 callbacks."""
        raw_info = infos.get(self.agent_id, {})
        if isinstance(raw_info, dict):
            return dict(raw_info)
        if raw_info in (None, []):
            return {}
        return {"raw_info": raw_info}

    def _assert_agent_available(self, observations: dict[str, Any]) -> None:
        """Fail early if the GAML model does not expose the expected merging agent."""
        if self.agent_id not in observations:
            available = sorted(observations.keys())
            raise RuntimeError(f"Expected agent {self.agent_id!r}, got {available!r}")
