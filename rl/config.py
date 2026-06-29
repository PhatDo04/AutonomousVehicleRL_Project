"""Shared configuration for training and evaluating the highway merging RL agent.

File cấu hình TRUNG TÂM: định nghĩa đường dẫn, preset thực nghiệm, danh sách tác tử,
bảng ý nghĩa hành động và các kịch bản mật độ. Mọi module khác import từ đây để giữ
mọi con số nhất quán ở MỘT chỗ (tránh hardcode lặp lại gây lệch).
"""

from __future__ import annotations

from dataclasses import dataclass  # @dataclass: tạo lớp dữ liệu gọn (tự sinh __init__...).
from pathlib import Path           # Path: thao tác đường dẫn đa nền tảng (Windows/Linux).


# --- Đường dẫn gốc của dự án và các thư mục output ---------------------------
# __file__ = file config.py này; .resolve() → đường dẫn tuyệt đối; parents[1] = lùi 2
# cấp (rl/ → thư mục gốc dự án). Nhờ vậy chạy ở đâu cũng tìm đúng file.
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Main_Traffic.gaml"  # file mô hình GAML (môi trường).
MARL_EXPERIMENT = "TrafficMARLHeadless"             # tên experiment headless để train/eval.
OUTPUT_DIR = ROOT / "outputs"                       # thư mục chứa mọi kết quả.
MODEL_DIR = OUTPUT_DIR / "models"                   # nơi lưu model (.zip) đã train.
LOG_DIR = OUTPUT_DIR / "logs"                       # nơi lưu CSV số liệu episode.
PLOT_DIR = OUTPUT_DIR / "plots"                     # nơi lưu biểu đồ PNG.


@dataclass(frozen=True)  # frozen=True → bất biến (không sửa được sau khi tạo) → an toàn.
class GamaConnectionConfig:
    """Connection settings for the GAMA headless socket server."""

    host: str = "localhost"               # GAMA chạy cùng máy.
    port: int = 1001                      # cổng TCP của GAMA headless.
    experiment_name: str = MARL_EXPERIMENT  # experiment sẽ nạp khi kết nối.
    agent_id: str = "merging_0"           # tác tử "chủ" — khi nó kết thúc thì cả episode reset.
    max_episode_steps: int = 300          # số tick tối đa mỗi episode (mặc định, có thể override).
    # Truyền vào PettingZoo ``reset(seed=...)`` (train/eval/baseline). Giúp tái lập phía Python/bridge;
    # RNG nội bộ GAMA vẫn có thể khác — báo cáo nên nêu reproducibility **một phần**.
    simulation_seed: int | None = None    # None = ngẫu nhiên; đặt số → tái lập được.


# MARL chỉ dùng các thuật toán on-policy (PPO/A2C). DQN bị loại vì off-policy:
# replay buffer trộn transitions của nhiều agent → bất ổn trong môi trường non-stationary.
MARL_ALGORITHMS = ("ppo", "a2c")

# Presets thực nghiệm MARL (PPO/A2C, 4 agents) — pipeline chính run_experiments.py.
# Mỗi preset là 1 dict: timesteps (số bước train), episodes (số ván đánh giá),
# seeds (danh sách hạt giống), max_episode_steps (độ dài tối đa 1 ván).
#   - smoke : sanity check pipeline (~vài giây).
#   - fast  : 1 seed, 60k steps — kiểm TIỀM NĂNG nhanh khi tune reward (~6-7'/thuật toán).
#             Chưa hội tụ → chỉ xem XU HƯỚNG, không phải số cuối.
#   - short : 1 seed, 200k steps — smoke test sau khi đổi config.
#   - thesis: 3 seeds × 200k steps + 50 eval episodes — bộ số chính cho luận văn.
MARL_EXPERIMENT_PRESETS = {
    "smoke":  {"timesteps": 400,     "episodes": 2,  "seeds": [0],             "max_episode_steps": 80},
    "fast":   {"timesteps": 60_000,  "episodes": 10, "seeds": [0],             "max_episode_steps": 500},
    "short":  {"timesteps": 200_000, "episodes": 20, "seeds": [0],             "max_episode_steps": 500},
    "thesis": {"timesteps": 200_000, "episodes": 50, "seeds": [0, 1, 2], "max_episode_steps": 500},
}


# Action meanings theo vai trò agent — cùng action index nhưng ý nghĩa KHÁC NHAU.
# Shared policy phân biệt qua one-hot agent_indicator trong obs (15D → 19D).
# Bảng tra cứu chỉ-số-hành-động → tên (dùng để gán nhãn log/biểu đồ và để con người đọc).
ACTION_MEANINGS_MERGING = {
    0: "decelerate",      # giảm tốc
    1: "keep_speed",      # giữ tốc
    2: "accelerate",      # tăng tốc
    3: "merge_attempt",   # nhập làn — chỉ thực hiện khi in_accel_zone và gap an toàn
    4: "wait",            # chờ
}

ACTION_MEANINGS_HIGHWAY = {
    0: "accelerate",      # đối ngược với merging: ở đây 0=tăng, 2=giảm (cùng index, ngược nghĩa)
    1: "keep_speed",      # giữ tốc
    2: "decelerate",      # giảm tốc
    3: "lane_change_left",   # chuyển làn trái
    4: "lane_change_right",  # chuyển làn phải
}

# Alias backward-compatible (chỉ dùng để label file CSV/log — không dùng cho logic)
ACTION_MEANINGS = ACTION_MEANINGS_MERGING

# MARL: 4 agents — 1 xe nhập làn + 3 xe cao tốc (cấu hình chính của luận văn).
# Đã thử 7 agents (6 highway) để tăng penetration cho goal-2, nhưng VỀ LẠI 4 vì nhường là
# CỤC BỘ tại điểm merge → 1-2 xe gần nhất là đủ (xem ghi chú TIMING bên dưới).
MARL_AGENTS: tuple[str, ...] = (
    "merging_0", "highway_0", "highway_1", "highway_2",
)
MARL_MERGING_AGENTS: tuple[str, ...] = ("merging_0",)              # nhóm xe nhập làn.
MARL_HIGHWAY_AGENTS: tuple[str, ...] = (                            # nhóm xe cao tốc.
    "highway_0", "highway_1", "highway_2",
)
# SuperSuit agent_indicator nối one-hot agent ID vào obs: 15 sensor + 4 agents = 19D.
# 4-agent (về lại từ 7): nhường là CỤC BỘ tại điểm merge → 1-2 xe gần nhất đủ; thay vì rải nhiều
# agent, thiết kế TIMING để highway_1 gặp merger tại merge_x (xem spawn trong Main_Traffic.gaml).
MARL_OBS_DIM: int = 15 + len(MARL_AGENTS)  # 19: chiều quan sát cục bộ của actor (15 + one-hot 4).
MARL_BASE_OBS_DIM: int = 15                # 15: chiều cảm biến gốc (chưa kèm one-hot ID).


@dataclass(frozen=True)
class ScenarioConfig:
    """Traffic density scenario for labelling experiments.

    ``nb_cars_max`` / ``spawn_interval`` / ``ramp_spawn_prob`` là tham số **mục tiêu** tương ứng với ``Main_Traffic.gaml``,
    nhưng khi chỉ đổi ``--scenario`` trong ``train_marl.py`` / ``baselines.py`` thì chúng **chỉ ghi vào nhãn
    và log** — không kích hoạt ``apply_scenario_to_gaml``. Để mô phỏng khớp preset: dùng ``run_experiments.py``, hoặc gọi
    ``scenario_utils.apply_scenario_to_gaml`` rồi **restart GAMA**.
    """

    name: str = "medium"          # tên kịch bản (low/medium/high).
    nb_cars_max: int = 45         # số xe nền tối đa trong môi trường.
    spawn_interval: int = 5       # cứ bao nhiêu tick lại sinh 1 xe nền (nhỏ = đông).
    ramp_spawn_prob: float = 0.5  # xác suất sinh xe nhập làn NPC trên ramp mỗi lần.


# 2026-06: nb_cars_max scale theo road_length=240 (×1.2 từ gốc 200) để GIỮ mật độ (cars/đơn-vị) vùng merge.
# low 20→24, medium 45→54, high 70→84. (320 từng thử nhưng gridlock-bò → rút về 240.)
# Ba kịch bản mật độ dùng cho ma trận thực nghiệm giai đoạn 1.
SCENARIO_PRESETS: dict[str, ScenarioConfig] = {
    "low":    ScenarioConfig(name="low",    nb_cars_max=24, spawn_interval=10, ramp_spawn_prob=0.3),  # thưa.
    "medium": ScenarioConfig(name="medium", nb_cars_max=54, spawn_interval=5,  ramp_spawn_prob=0.5),  # vừa.
    "high":   ScenarioConfig(name="high",   nb_cars_max=84, spawn_interval=3,  ramp_spawn_prob=0.7),  # đông (bộ số chính e2e).
}


# Được in từ train_marl.py / baselines.py — tránh hiểu nhầm CLI.
# Cảnh báo: cờ --scenario CHỈ là nhãn metadata, KHÔNG tự vá mật độ vào file GAML.
SCENARIO_CLI_METADATA_ONLY_HINT = (
    "[scenario] --scenario chỉ là nhãn (CSV/metadata); không vá GAML — mật độ thật là file GAMA simulator đã load. "
    "Đồng bộ preset: rl.scenario_utils.apply_scenario_to_gaml(...) + restart headless, hoặc rl/run_experiments.py."
)
