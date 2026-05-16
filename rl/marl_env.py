"""MARL environment wrapper: GAMA PettingZoo → SuperSuit → SB3 VecEnv.

Kiến trúc:
    GamaParallelEnv (PettingZoo parallel, 4 agents)
        → AgentIndicatorParallelWrapper  (one-hot ID 15→19D, **không** qua parallel↔AEC của SuperSuit)
        → MarkovVectorEnv(..., black_death=True)
        → GamaMarkovSB3VecEnv      (stable_baselines3.common.vec_env.VecEnv)

    ``concat_vec_envs_v1(..., base_class="stable_baselines3")`` của SuperSuit
    cloudpickle env để nhân bản — GAMA sau reset không pickle được; adapter
    trên bỏ pickle và cài đủ API VecEnv (get_attr, …) mà SB3 2.x yêu cầu.

Agent roles:
    merging_0   : xe nhập làn (ramp) — mục tiêu: nhập làn an toàn
    highway_0/1/2: xe cao tốc làn dưới — mục tiêu: duy trì tốc độ + nhường gap

Shared policy: tất cả agent dùng cùng 1 SB3 model; one-hot agent ID trong obs
giúp policy phân biệt vai trò mà không cần 2 model riêng biệt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Type

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Space
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import GamaConnectionConfig, MARL_AGENTS, MODEL_PATH
from rl.gama_compat import patch_gama_gymnasium

patch_gama_gymnasium()

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv  # noqa: E402
from pettingzoo import ParallelEnv  # noqa: E402

try:
    from supersuit.utils import agent_indicator as _agent_ider
    from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv
except ImportError as exc:
    raise ImportError(
        "SuperSuit không được cài đặt. Chạy: pip install supersuit>=3.9.0"
    ) from exc

from stable_baselines3.common.vec_env import VecEnv


class AgentIndicatorParallelWrapper(ParallelEnv):
    """Giống ``supersuit.agent_indicator_v0`` nhưng giữ nguyên Parallel API.

    SuperSuit bọc ParallelEnv bằng ``parallel_to_aec`` → ``aec_to_parallel``; GAMA
    khi agent động sẽ làm AEC lệch key (KeyError trên terminations[agent_selection]).
    """

    def __init__(self, env: ParallelEnv, *, type_only: bool = False) -> None:
        self.env = env
        self.metadata = dict(getattr(env, "metadata", {}) or {})
        self.render_mode = getattr(env, "render_mode", None)
        self.possible_agents = list(env.possible_agents)
        self.agents = list(getattr(env, "agents", []))
        spaces = [env.observation_space(a) for a in self.possible_agents]
        _agent_ider.check_params(spaces)
        self._indicator_map = _agent_ider.get_indicator_map(self.possible_agents, type_only)
        self._num_indicators = len(set(self._indicator_map.values()))

    def observation_space(self, agent: str) -> Space:
        return _agent_ider.change_obs_space(self.env.observation_space(agent), self._num_indicators)

    def action_space(self, agent: str) -> Space:
        return self.env.action_space(agent)

    def _patch_obs_dict(self, obs: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for agent, o in obs.items():
            base_space = self.env.observation_space(agent)
            out[agent] = _agent_ider.change_observation(
                o,
                base_space,
                (self._indicator_map[agent], self._num_indicators),
            )
        return out

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return self._patch_obs_dict(observations), infos

    def step(self, actions: dict[str, Any]):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        return self._patch_obs_dict(obs), rewards, terms, truncs, infos

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()


class GamaMarkovSB3VecEnv(VecEnv):
    """Bọc ``supersuit`` MarkovVectorEnv (Gymnasium vector) thành SB3 ``VecEnv``.

    SB3 2.x chỉ coi là VecEnv khi isinstance(..., stable_baselines3...VecEnv);
    MarkovVectorEnv kế thừa gymnasium.vector.VectorEnv nên bị ``_patch_env`` sai.
    SuperSuit ``SB3VecEnvWrapper`` cũng gọi ``venv.get_attr`` mà MarkovVectorEnv
    không có — lớp này forward tới ``par_env`` (PettingZoo / GAMA).
    """

    def __init__(self, markov_vector_env: Any) -> None:
        self._m = markov_vector_env
        super().__init__(
            num_envs=int(markov_vector_env.num_envs),
            observation_space=markov_vector_env.observation_space,
            action_space=markov_vector_env.action_space,
        )

    def _par(self) -> Any:
        return self._m.par_env

    def reset(self) -> VecEnvObs:
        seed = self._seeds[0]
        raw_opts = self._options[0] if self._options else {}
        options: dict[str, Any] | None = raw_opts if raw_opts else None
        obs, infos = self._m.reset(seed=seed, options=options)
        self.reset_infos = list(infos)
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._m.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, terms, truncs, infos = self._m.step_wait()
        dones = np.logical_or(np.asarray(terms, dtype=bool), np.asarray(truncs, dtype=bool))
        return obs, rewards, dones, infos

    def close(self) -> None:
        self._m.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        idxs = list(self._get_indices(indices))
        par = self._par()
        unwrapped = getattr(par, "unwrapped", par)
        try:
            val = getattr(unwrapped, attr_name)
        except AttributeError:
            val = None
        return [val for _ in idxs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        par = self._par()
        setattr(par, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> List[Any]:
        idxs = list(self._get_indices(indices))
        method = getattr(self._par(), method_name)
        out = method(*method_args, **method_kwargs)
        return [out for _ in idxs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        idxs = list(self._get_indices(indices))
        if hasattr(self._m, "env_is_wrapped"):
            full = self._m.env_is_wrapped(wrapper_class)
            if isinstance(full, list) and len(full) == self.num_envs:
                return [full[i] for i in idxs]
        return [False for _ in idxs]

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return tuple(None for _ in range(self.num_envs))


def make_marl_vec_env(
    config: GamaConnectionConfig | None = None,
    num_vec_envs: int = 1,
) -> VecEnv:
    """Tạo SB3-compatible VecEnv từ GAMA PettingZoo parallel env.

    Parameters
    ----------
    config:
        Cấu hình kết nối GAMA. Dùng mặc định nếu None.
    num_vec_envs:
        Số lượng vectorized copies. Giữ mặc định 1 vì GAMA chỉ có 1 server.

    Returns
    -------
    VecEnv với obs shape (19,) và action Discrete(5).
    Mỗi step trả về batch_size = num_agents * num_vec_envs samples.
    """
    cfg = config or GamaConnectionConfig()

    # Tạo PettingZoo parallel env kết nối GAMA headless
    parallel_env = GamaParallelEnv(
        gaml_experiment_path=str(MODEL_PATH),
        gaml_experiment_name=cfg.experiment_name,
        gama_ip_address=cfg.host,
        gama_port=cfg.port,
    )

    # Xác nhận 4 agent MARL đều có mặt.
    # Dùng try/finally để đóng socket GAMA nếu validation thất bại (tránh treo connection).
    try:
        obs, _ = parallel_env.reset(seed=cfg.simulation_seed)
        missing = [a for a in MARL_AGENTS if a not in obs]
        if missing:
            raise RuntimeError(
                f"GAMA không expose đủ agent MARL. Thiếu: {missing}. "
                f"Kiểm tra possible_agents trong Main_Traffic.gaml."
            )
    except Exception:
        parallel_env.close()
        raise

    # 1. One-hot agent ID (15,) → (19,) — không dùng ss.agent_indicator_v0 (AEC round-trip + GAMA → KeyError).
    env = AgentIndicatorParallelWrapper(parallel_env, type_only=False)

    # 2. MarkovVectorEnv: không dùng pettingzoo_env_to_vec_env_v1 (black_death=False).
    #    GAMA có bước par_env.agents != possible_agents (merge/respawn) → cần black_death=True.
    markov = MarkovVectorEnv(env, black_death=True)

    # 3. SB3 VecEnv — không dùng concat_vec_envs_v1 (pickle GAMA → lỗi coroutine).
    if num_vec_envs == 1:
        return GamaMarkovSB3VecEnv(markov)

    raise NotImplementedError(
        "num_vec_envs>1 cần concat_vec_envs_v1 (pickle env). GAMA bridge không pickle được; "
        "giữ num_vec_envs=1 (mặc định)."
    )
