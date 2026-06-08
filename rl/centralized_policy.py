"""Centralized-critic policy cho MARL CTDE (MAPPO / MAA2C).

Kiến trúc CTDE (Centralized Training, Decentralized Execution):
  - Actor (decentralized): chỉ đọc local observation 19D (15 sensor + 4 one-hot agent ID).
  - Critic (centralized):  đọc global state 60D (concat 15D base obs của cả 4 agents).
  - Obs đưa vào model = concat([local 19D, global 60D]) = 79D.

Vì actor chỉ đọc ``obs[:, :ACTOR_DIM]`` nên khi execution có thể điền 0 cho phần global
mà action không đổi → đúng tính chất decentralized execution.

So với critic phi tập trung (chỉ thấy local obs): critic tập trung thấy trạng thái
toàn cục → value estimate chính xác hơn, khử non-stationarity do các agent khác gây
ra → giảm nhiễu advantage (Yu et al. 2021, MAPPO).

Dùng chung cho cả PPO và A2C vì cả hai đều dựa trên ``ActorCriticPolicy`` của SB3.
Hyperparameter (hidden=64, orthogonal init) khớp MAPPO official
(marlbenchmark/on-policy/config.py).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


# DERIVE từ MARL_AGENTS (tự co giãn khi đổi số agent — tránh hardcode lệch dim).
from rl.config import MARL_AGENTS, MARL_BASE_OBS_DIM  # noqa: E402
_N_AGENTS: int = len(MARL_AGENTS)
# Local obs: 15 sensor + N one-hot agent ID (khớp MARL_OBS_DIM trong config).
ACTOR_DIM: int = MARL_BASE_OBS_DIM + _N_AGENTS          # 15 + 7 = 22
# Global state: N agents × 15D base sensor (không kèm one-hot ID — tránh dư thừa).
GLOBAL_DIM: int = MARL_BASE_OBS_DIM * _N_AGENTS         # 15 × 7 = 105
# Obs đưa vào SB3 model = actor local + global state.
CENTRALIZED_OBS_DIM: int = ACTOR_DIM + GLOBAL_DIM       # 127


class CentralizedCriticExtractor(nn.Module):
    """MlpExtractor tách obs: actor đọc phần local, critic đọc phần global.

    Thay thế ``stable_baselines3.common.torch_layers.MlpExtractor``. SB3 gọi
    ``forward`` trong rollout, ``forward_actor``/``forward_critic`` ở các nhánh khác.
    """

    def __init__(
        self,
        actor_dim: int = ACTOR_DIM,
        global_dim: int = GLOBAL_DIM,
        hidden: int = 64,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.actor_dim = actor_dim
        self.global_dim = global_dim

        # SB3 đọc 2 thuộc tính này để build action_net / value_net.
        self.latent_dim_pi = hidden
        self.latent_dim_vf = hidden

        self.policy_net = nn.Sequential(
            nn.Linear(actor_dim, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(global_dim, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
        )
        self._init_orthogonal()

    def _init_orthogonal(self) -> None:
        """Orthogonal init với gain=sqrt(2) cho Tanh hidden layers (MAPPO standard)."""
        gain = math.sqrt(2.0)
        for module in [self.policy_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.constant_(layer.bias, 0.0)

    def _split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local = features[:, : self.actor_dim]
        glob = features[:, self.actor_dim : self.actor_dim + self.global_dim]
        return local, glob

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local, glob = self._split(features)
        return self.policy_net(local), self.value_net(glob)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[:, : self.actor_dim])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        glob = features[:, self.actor_dim : self.actor_dim + self.global_dim]
        return self.value_net(glob)


class CentralizedCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy với centralized critic (CTDE). Dùng cho MAPPO + MAA2C.

    Override ``_build_mlp_extractor`` để cài ``CentralizedCriticExtractor`` thay cho
    MlpExtractor mặc định. Phần còn lại (action_net, value_net head, optimizer) do SB3
    tự build dựa trên ``latent_dim_pi`` / ``latent_dim_vf``.
    """

    def __init__(self, *args, actor_dim: int = ACTOR_DIM, global_dim: int = GLOBAL_DIM, **kwargs):
        self._actor_dim = actor_dim
        self._global_dim = global_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CentralizedCriticExtractor(
            actor_dim=self._actor_dim,
            global_dim=self._global_dim,
        )
