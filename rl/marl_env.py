"""MARL environment wrapper: GAMA PettingZoo → SuperSuit → SB3 VecEnv.

Kiến trúc (CTDE — centralized critic, decentralized actor):
    GamaParallelEnv (PettingZoo parallel, 4 agents)
        → PadPossibleAgentsParallelWrapper (luôn đủ 4 obs — MarkovVectorEnv.concat_obs)
        → AgentIndicatorParallelWrapper  (one-hot ID 15→19D, **không** qua parallel↔AEC của SuperSuit)
        → GlobalStateParallelWrapper     (nối global state 60D → 79D cho centralized critic)
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


class PadPossibleAgentsParallelWrapper(ParallelEnv):
    """Giữ cố định ``possible_agents`` cho MarkovVectorEnv khi GAMA respawn/merge."""

    def __init__(self, env: ParallelEnv) -> None:
        self.env = env
        self.metadata = dict(getattr(env, "metadata", {}) or {})
        self.render_mode = getattr(env, "render_mode", None)
        self.possible_agents = list(env.possible_agents)
        self.agents = list(self.possible_agents)

    def observation_space(self, agent: str) -> Space:
        return self.env.observation_space(agent)

    def action_space(self, agent: str) -> Space:
        return self.env.action_space(agent)

    def _zero_obs(self, agent: str) -> np.ndarray:
        space = self.env.observation_space(agent)
        return np.zeros(space.shape, dtype=space.dtype)

    def _pad_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            agent: obs[agent]
            if agent in obs and obs[agent] is not None
            else self._zero_obs(agent)
            for agent in self.possible_agents
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return self._pad_obs(observations), infos

    def step(self, actions: dict[str, Any]):
        act = {a: actions.get(a, 1) for a in self.possible_agents}
        obs, rewards, terms, truncs, infos = self.env.step(act)
        self.agents = list(self.possible_agents)
        padded_obs = self._pad_obs(obs)
        return (
            padded_obs,
            {a: float(rewards.get(a, 0.0)) for a in self.possible_agents},
            {a: bool(terms.get(a, False)) for a in self.possible_agents},
            {a: bool(truncs.get(a, False)) for a in self.possible_agents},
            {a: infos.get(a, {}) for a in self.possible_agents},
        )

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()


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
        self._type_only = type_only
        self._indicator_map: dict[str, int] | None = None
        self._num_indicators = len(self.possible_agents)

    def _ensure_indicator_map(self) -> None:
        if self._indicator_map is not None:
            return
        spaces = [self.env.observation_space(a) for a in self.possible_agents]
        _agent_ider.check_params(spaces)
        self._indicator_map = _agent_ider.get_indicator_map(self.possible_agents, self._type_only)
        self._num_indicators = len(set(self._indicator_map.values()))

    def observation_space(self, agent: str) -> Space:
        self._ensure_indicator_map()
        return _agent_ider.change_obs_space(self.env.observation_space(agent), self._num_indicators)

    def action_space(self, agent: str) -> Space:
        return self.env.action_space(agent)

    def _patch_obs_dict(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ensure_indicator_map()
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


class GlobalStateParallelWrapper(ParallelEnv):
    """CTDE: nối global state vào obs mỗi agent → centralized critic.

    Global state = concat 15D base obs (KHÔNG kèm one-hot ID) của cả 4 agent theo
    thứ tự cố định ``MARL_AGENTS`` = 60D. Mỗi agent obs (19D) → 19+60 = 79D.

    Actor chỉ đọc ``obs[:19]`` (decentralized); critic đọc ``obs[19:79]`` (global).
    Đặt SAU ``AgentIndicatorParallelWrapper`` (obs đã là 19D = 15 base + 4 one-hot).
    """

    BASE_DIM = 15  # 15D sensor gốc (trước one-hot agent ID)

    def __init__(self, env: ParallelEnv) -> None:
        self.env = env
        self.metadata = dict(getattr(env, "metadata", {}) or {})
        self.render_mode = getattr(env, "render_mode", None)
        self.possible_agents = list(env.possible_agents)
        self.agents = list(getattr(env, "agents", []))
        self._global_dim = self.BASE_DIM * len(self.possible_agents)

    def observation_space(self, agent: str) -> Space:
        base = self.env.observation_space(agent)
        low = float(np.min(base.low)) if np.ndim(base.low) else float(base.low)
        high = float(np.max(base.high)) if np.ndim(base.high) else float(base.high)
        new_len = int(base.shape[0]) + self._global_dim
        return gym.spaces.Box(low=low, high=high, shape=(new_len,), dtype=base.dtype)

    def action_space(self, agent: str) -> Space:
        return self.env.action_space(agent)

    def _build_global(self, obs: dict[str, Any]) -> np.ndarray:
        """Concat 15D base của cả 4 agent theo thứ tự possible_agents (thiếu → zeros)."""
        parts: list[np.ndarray] = []
        for a in self.possible_agents:
            o = obs.get(a)
            if o is None:
                parts.append(np.zeros(self.BASE_DIM, dtype=np.float32))
            else:
                arr = np.asarray(o, dtype=np.float32)
                parts.append(arr[: self.BASE_DIM])
        return np.concatenate(parts, axis=0)

    def _augment(self, obs: dict[str, Any]) -> dict[str, Any]:
        glob = self._build_global(obs)
        out: dict[str, Any] = {}
        for a, o in obs.items():
            arr = np.asarray(o, dtype=np.float32)
            out[a] = np.concatenate([arr, glob], axis=0)
        return out

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return self._augment(observations), infos

    def step(self, actions: dict[str, Any]):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        return self._augment(obs), rewards, terms, truncs, infos

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

    def __init__(self, markov_vector_env: Any, reset_on_merging_death: bool = True,
                 postmerge_window: float | None = None, rl_postmerge: bool = False) -> None:
        self._m = markov_vector_env
        self._rl_postmerge = rl_postmerge   # =True → RL điều khiển merger post-merge (end-to-end, thesis)
        # TRAIN post-merge bounded: nếu set → mỗi reset bơm pz_eval_continue=1 + pz_postmerge_window
        # → episode train KHÔNG end tại merge mà chạy thêm `postmerge_window` tick (học lái post-merge).
        # KHÁC yt22 (chạy-tới-cuối, unbounded → sập): đây là cửa sổ CÓ CHẶN, episode vẫn kết thúc.
        self._postmerge_window = postmerge_window
        # ``MarkovVectorEnv(black_death=True)`` che ``done`` cho dead agents — SB3 PPO
        # vì thế không bao giờ thấy episode boundary khi merging_0 chết → rollout buffer
        # tràn dead-agent transitions (zero obs, 0 reward) → policy gradient nhiễu nặng.
        #
        # Khi cờ này bật, ``step_wait`` tự phát hiện merging_0 chết → force ALL slots
        # done=True + reset env → SB3 bootstrap value đúng và bắt đầu episode sạch.
        # Khớp với pipeline eval (gọi ``reset_marl_episode`` sau mỗi merging_0 termination).
        self._reset_on_merging_death = reset_on_merging_death
        super().__init__(
            num_envs=int(markov_vector_env.num_envs),
            observation_space=markov_vector_env.observation_space,
            action_space=markov_vector_env.action_space,
        )

    def _par(self) -> Any:
        return self._m.par_env

    def _inject_postmerge(self) -> None:
        """Bơm cờ post-merge xuống GAML sau reset (GAMA reload đã reset chúng về default)."""
        if self._postmerge_window is None and not self._rl_postmerge:
            return
        from rl.gama_episode_reset import _exec_global
        if self._postmerge_window is not None:
            _exec_global(self._par(), "pz_eval_continue <- 1.0;")
            _exec_global(self._par(), f"pz_postmerge_window <- {float(self._postmerge_window)};")
        if self._rl_postmerge:
            _exec_global(self._par(), "pz_rl_postmerge <- 1.0;")

    def reset(self) -> VecEnvObs:
        seed = self._seeds[0]
        raw_opts = self._options[0] if self._options else {}
        options: dict[str, Any] | None = raw_opts if raw_opts else None
        obs, infos = self._m.reset(seed=seed, options=options)
        self._inject_postmerge()
        self.reset_infos = list(infos)
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._m.step_async(actions)

    def _merging_just_died(self, infos: list, terms: Any) -> bool:
        """Phát hiện merging_0 vừa chết trong step hiện tại.

        Hai tín hiệu vì ``MarkovVectorEnv(black_death=True)`` che ``terms`` cho dead agents:
          1. ``terms[0]=True`` — trực tiếp tại step chết đầu tiên (sau đó black_death ép False).
          2. ``par_env.agents`` không có "merging_0" + ``infos[0].outcome`` là terminal — bắt
             trường hợp pipeline xử lý termination không sync với ``terms`` array.
        """
        terms_arr = np.asarray(terms, dtype=bool)
        if terms_arr.shape[0] > 0 and bool(terms_arr[0]):
            return True
        par = self._par()
        agents_now = getattr(par, "agents", None)
        if agents_now is not None and "merging_0" not in agents_now:
            info0 = infos[0] if len(infos) > 0 else {}
            outcome = str(info0.get("outcome", "")).lower() if isinstance(info0, dict) else ""
            if outcome in {"collision", "success", "failed_merge"}:
                return True
        return False

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, terms, truncs, infos = self._m.step_wait()
        dones = np.logical_or(np.asarray(terms, dtype=bool), np.asarray(truncs, dtype=bool))

        if self._reset_on_merging_death and self._merging_just_died(list(infos), terms):
            # Theo convention SB3: lưu terminal_observation trong info để PPO bootstrap
            # value đúng tại episode boundary, sau đó trả về obs sau reset như obs mới.
            infos_out = [dict(info) if isinstance(info, dict) else {} for info in infos]
            for i in range(self.num_envs):
                infos_out[i]["terminal_observation"] = np.asarray(obs[i])
                infos_out[i]["TimeLimit.truncated"] = False
            dones = np.ones(self.num_envs, dtype=bool)
            new_obs, _ = self._m.reset()
            self._inject_postmerge()
            return new_obs, np.asarray(rewards, dtype=np.float32), dones, infos_out

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
    postmerge_window: float | None = None,
    rl_postmerge: bool = False,
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

    # 1. Pad obs cho đủ 4 agent (respawn ramp / MarkovVectorEnv.concat_obs).
    env = PadPossibleAgentsParallelWrapper(parallel_env)
    # 2. One-hot agent ID (15,) → (19,) — không dùng ss.agent_indicator_v0 (AEC round-trip + GAMA → KeyError).
    env = AgentIndicatorParallelWrapper(env, type_only=False)
    # 3. CTDE: nối global state 60D → obs 79D cho centralized critic (MAPPO/MAA2C).
    env = GlobalStateParallelWrapper(env)

    # Xác nhận 4 agent MARL đều có mặt (bootstrap GAML sau cycle>0).
    try:
        obs, _ = env.reset(seed=cfg.simulation_seed)
        missing = [a for a in MARL_AGENTS if a not in obs]
        if missing:
            raise RuntimeError(
                f"GAMA không expose đủ agent MARL. Thiếu: {missing}. "
                f"Kiểm tra possible_agents trong Main_Traffic.gaml."
            )
    except Exception:
        env.close()
        raise

    # 4. MarkovVectorEnv: không dùng pettingzoo_env_to_vec_env_v1 (black_death=False).
    #    GAMA có bước par_env.agents != possible_agents (merge/respawn) → cần black_death=True.
    markov = MarkovVectorEnv(env, black_death=True)

    # 5. SB3 VecEnv — không dùng concat_vec_envs_v1 (pickle GAMA → lỗi coroutine).
    if num_vec_envs == 1:
        return GamaMarkovSB3VecEnv(markov, postmerge_window=postmerge_window, rl_postmerge=rl_postmerge)

    raise NotImplementedError(
        "num_vec_envs>1 cần concat_vec_envs_v1 (pickle env). GAMA bridge không pickle được; "
        "giữ num_vec_envs=1 (mặc định)."
    )
