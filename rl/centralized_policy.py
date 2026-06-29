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

# `from __future__ import annotations`: cho phép viết type hint kiểu mới (vd `tuple[...]`)
# trên Python cũ — annotation được xử lý như chuỗi, không ép buộc lúc import.
from __future__ import annotations

import math  # dùng math.sqrt(2) cho hệ số khởi tạo trọng số (orthogonal gain).

import torch  # thư viện tensor/deep-learning nền tảng.
from torch import nn  # nn: các lớp mạng nơ-ron (Linear, Tanh, Sequential...).
# ActorCriticPolicy: lớp policy gốc của SB3 mà ta kế thừa để chèn critic tập trung.
from stable_baselines3.common.policies import ActorCriticPolicy


# ---------------------------------------------------------------------------
# Các kích thước (dim) suy ra TỰ ĐỘNG từ config — tránh hardcode để khi đổi số
# agent thì không bị lệch chiều vector gây lỗi khó tìm.
# ---------------------------------------------------------------------------
from rl.config import MARL_AGENTS, MARL_BASE_OBS_DIM  # noqa: E402
# Số lượng tác tử (đồ án dùng 4: 1 nhập làn + 3 cao tốc).
_N_AGENTS: int = len(MARL_AGENTS)
# Đầu vào của ACTOR = 15 chiều cảm biến cục bộ + N chiều one-hot định danh agent.
# One-hot ID giúp 1 mạng dùng chung biết "mình đang là agent nào" để xử lý đúng vai trò.
ACTOR_DIM: int = MARL_BASE_OBS_DIM + _N_AGENTS          # 15 + 4 = 19
# Đầu vào của CRITIC = ghép 15 chiều cảm biến gốc của TẤT CẢ agent (KHÔNG kèm one-hot
# để khỏi dư thừa) → critic "nhìn" được trạng thái toàn cục lúc huấn luyện.
GLOBAL_DIM: int = MARL_BASE_OBS_DIM * _N_AGENTS         # 15 × 4 = 60
# Vector thực sự đưa vào model SB3 = [phần cục bộ của actor | phần toàn cục của critic].
CENTRALIZED_OBS_DIM: int = ACTOR_DIM + GLOBAL_DIM       # 19 + 60 = 79


class CentralizedCriticExtractor(nn.Module):
    """MlpExtractor tách obs: actor đọc phần local, critic đọc phần global.

    Thay thế ``stable_baselines3.common.torch_layers.MlpExtractor``. SB3 gọi
    ``forward`` trong rollout, ``forward_actor``/``forward_critic`` ở các nhánh khác.
    """

    def __init__(
        self,
        actor_dim: int = ACTOR_DIM,    # số chiều phần cục bộ (đầu vector) dành cho actor.
        global_dim: int = GLOBAL_DIM,  # số chiều phần toàn cục (cuối vector) dành cho critic.
        hidden: int = 64,              # số nơ-ron mỗi lớp ẩn (cấu hình chuẩn MAPPO).
        activation: type[nn.Module] = nn.Tanh,  # hàm kích hoạt phi tuyến (Tanh chuẩn MAPPO).
    ) -> None:
        super().__init__()  # bắt buộc gọi để khởi tạo nn.Module gốc.
        # Lưu lại để hàm _split() biết cắt vector ở đâu.
        self.actor_dim = actor_dim
        self.global_dim = global_dim

        # SB3 đọc 2 thuộc tính này để tự dựng action_net (lớp ra hành động) và
        # value_net head (lớp ra giá trị) ở phía sau extractor. Phải đặt = hidden.
        self.latent_dim_pi = hidden  # kích thước đặc trưng ra của nhánh policy (actor).
        self.latent_dim_vf = hidden  # kích thước đặc trưng ra của nhánh value (critic).

        # Nhánh ACTOR: MLP 2 lớp ẩn 64 nơ-ron, mỗi lớp kèm Tanh.
        # Đầu vào actor_dim (19) → 64 → 64. Đầu ra 64 chiều đặc trưng cục bộ.
        self.policy_net = nn.Sequential(
            nn.Linear(actor_dim, hidden),  # lớp tuyến tính 1: 19 → 64.
            activation(),                  # phi tuyến Tanh.
            nn.Linear(hidden, hidden),     # lớp tuyến tính 2: 64 → 64.
            activation(),                  # phi tuyến Tanh.
        )
        # Nhánh CRITIC: cấu trúc giống hệt nhưng đầu vào là global_dim (60) → 64 → 64.
        # Critic là MẠNG RIÊNG (không chung trọng số với actor) nên θ (actor) ≠ φ (critic).
        self.value_net = nn.Sequential(
            nn.Linear(global_dim, hidden),  # lớp tuyến tính 1: 60 → 64.
            activation(),
            nn.Linear(hidden, hidden),      # lớp tuyến tính 2: 64 → 64.
            activation(),
        )
        # Khởi tạo trọng số theo kiểu trực giao (orthogonal) — giúp train ổn định hơn.
        self._init_orthogonal()

    def _init_orthogonal(self) -> None:
        """Orthogonal init với gain=sqrt(2) cho Tanh hidden layers (MAPPO standard)."""
        # gain=√2 là giá trị khuyến nghị cho lớp ẩn dùng Tanh (giữ phương sai tín hiệu ổn định).
        gain = math.sqrt(2.0)
        # Duyệt cả 2 nhánh (actor + critic).
        for module in [self.policy_net, self.value_net]:
            for layer in module:  # duyệt từng lớp trong Sequential.
                if isinstance(layer, nn.Linear):  # chỉ khởi tạo lớp tuyến tính (bỏ qua Tanh).
                    nn.init.orthogonal_(layer.weight, gain=gain)  # trọng số: ma trận trực giao.
                    nn.init.constant_(layer.bias, 0.0)            # bias: khởi tạo bằng 0.

    def _split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Cắt vector 79D thành 2 phần theo chiều cột (dim=1):
        local = features[:, : self.actor_dim]                                  # 19 chiều đầu → actor.
        glob = features[:, self.actor_dim : self.actor_dim + self.global_dim]  # 60 chiều sau → critic.
        return local, glob

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Dùng trong rollout: chạy CẢ actor lẫn critic cùng lúc.
        local, glob = self._split(features)
        # Trả về (đặc trưng actor, đặc trưng critic) cho SB3 xử lý tiếp.
        return self.policy_net(local), self.value_net(glob)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        # Chỉ chạy nhánh actor (vd lúc execution/đánh giá) → chỉ cần phần cục bộ.
        return self.policy_net(features[:, : self.actor_dim])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        # Chỉ chạy nhánh critic → cắt lấy phần toàn cục rồi đưa vào value_net.
        glob = features[:, self.actor_dim : self.actor_dim + self.global_dim]
        return self.value_net(glob)


class CentralizedCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy với centralized critic (CTDE). Dùng cho MAPPO + MAA2C.

    Override ``_build_mlp_extractor`` để cài ``CentralizedCriticExtractor`` thay cho
    MlpExtractor mặc định. Phần còn lại (action_net, value_net head, optimizer) do SB3
    tự build dựa trên ``latent_dim_pi`` / ``latent_dim_vf``.
    """

    def __init__(self, *args, actor_dim: int = ACTOR_DIM, global_dim: int = GLOBAL_DIM, **kwargs):
        # Lưu 2 dim vào thuộc tính TRƯỚC khi gọi super().__init__, vì SB3 sẽ gọi
        # _build_mlp_extractor() ngay trong __init__ của lớp cha và cần 2 giá trị này.
        self._actor_dim = actor_dim
        self._global_dim = global_dim
        # Gọi khởi tạo của ActorCriticPolicy (SB3) — nó tự dựng action_net/value_net/optimizer.
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # SB3 gọi hàm này để tạo bộ trích đặc trưng. Ta thay bằng extractor tách actor/critic.
        self.mlp_extractor = CentralizedCriticExtractor(
            actor_dim=self._actor_dim,
            global_dim=self._global_dim,
        )
