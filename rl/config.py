"""Shared configuration for training and evaluating the highway merging RL agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Main_Traffic.gaml"
MARL_EXPERIMENT = "TrafficMARLHeadless"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"


@dataclass(frozen=True)
class GamaConnectionConfig:
    """Connection settings for the GAMA headless socket server."""

    host: str = "localhost"
    port: int = 1001
    experiment_name: str = MARL_EXPERIMENT
    agent_id: str = "merging_0"
    max_episode_steps: int = 300
    # Truyền vào PettingZoo ``reset(seed=...)`` (train/eval/baseline). Giúp tái lập phía Python/bridge;
    # RNG nội bộ GAMA vẫn có thể khác — báo cáo nên nêu reproducibility **một phần**.
    simulation_seed: int | None = None


# MARL chỉ dùng các thuật toán on-policy (PPO/A2C). DQN bị loại vì off-policy:
# replay buffer trộn transitions của nhiều agent → bất ổn trong môi trường non-stationary.
MARL_ALGORITHMS = ("ppo", "a2c")

# Presets thực nghiệm MARL (PPO/A2C, 4 agents) — pipeline chính run_experiments.py.
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
ACTION_MEANINGS_MERGING = {
    0: "decelerate",
    1: "keep_speed",
    2: "accelerate",
    3: "merge_attempt",   # chỉ thực hiện khi in_accel_zone và gap an toàn
    4: "wait",
}

ACTION_MEANINGS_HIGHWAY = {
    0: "accelerate",      # đối ngược với merging: 0=tăng, 2=giảm
    1: "keep_speed",
    2: "decelerate",
    3: "lane_change_left",
    4: "lane_change_right",
}

# Alias backward-compatible (chỉ dùng để label file CSV/log — không dùng cho logic)
ACTION_MEANINGS = ACTION_MEANINGS_MERGING

# MARL: 4 agents — 1 xe nhập làn + 3 xe cao tốc (cấu hình chính của luận văn).
# Đã thử 7 agents (6 highway) để tăng penetration cho goal-2, nhưng VỀ LẠI 4 vì nhường là
# CỤC BỘ tại điểm merge → 1-2 xe gần nhất là đủ (xem ghi chú TIMING bên dưới).
MARL_AGENTS: tuple[str, ...] = (
    "merging_0", "highway_0", "highway_1", "highway_2",
)
MARL_MERGING_AGENTS: tuple[str, ...] = ("merging_0",)
MARL_HIGHWAY_AGENTS: tuple[str, ...] = (
    "highway_0", "highway_1", "highway_2",
)
# SuperSuit agent_indicator nối one-hot agent ID vào obs: 15 sensor + 4 agents = 19D.
# 4-agent (về lại từ 7): nhường là CỤC BỘ tại điểm merge → 1-2 xe gần nhất đủ; thay vì rải nhiều
# agent, thiết kế TIMING để highway_1 gặp merger tại merge_x (xem spawn trong Main_Traffic.gaml).
MARL_OBS_DIM: int = 15 + len(MARL_AGENTS)
MARL_BASE_OBS_DIM: int = 15


@dataclass(frozen=True)
class ScenarioConfig:
    """Traffic density scenario for labelling experiments.

    ``nb_cars_max`` / ``spawn_interval`` / ``ramp_spawn_prob`` là tham số **mục tiêu** tương ứng với ``Main_Traffic.gaml``,
    nhưng khi chỉ đổi ``--scenario`` trong ``train_marl.py`` / ``baselines.py`` thì chúng **chỉ ghi vào nhãn
    và log** — không kích hoạt ``apply_scenario_to_gaml``. Để mô phỏng khớp preset: dùng ``run_experiments.py``, hoặc gọi
    ``scenario_utils.apply_scenario_to_gaml`` rồi **restart GAMA**.
    """

    name: str = "medium"
    nb_cars_max: int = 45
    spawn_interval: int = 5
    ramp_spawn_prob: float = 0.5


# 2026-06: nb_cars_max scale theo road_length=240 (×1.2 từ gốc 200) để GIỮ mật độ (cars/đơn-vị) vùng merge.
# low 20→24, medium 45→54, high 70→84. (320 từng thử nhưng gridlock-bò → rút về 240.)
SCENARIO_PRESETS: dict[str, ScenarioConfig] = {
    "low":    ScenarioConfig(name="low",    nb_cars_max=24, spawn_interval=10, ramp_spawn_prob=0.3),
    "medium": ScenarioConfig(name="medium", nb_cars_max=54, spawn_interval=5,  ramp_spawn_prob=0.5),
    "high":   ScenarioConfig(name="high",   nb_cars_max=84, spawn_interval=3,  ramp_spawn_prob=0.7),
}


# Được in từ train_marl.py / baselines.py — tránh hiểu nhầm CLI.
SCENARIO_CLI_METADATA_ONLY_HINT = (
    "[scenario] --scenario chỉ là nhãn (CSV/metadata); không vá GAML — mật độ thật là file GAMA simulator đã load. "
    "Đồng bộ preset: rl.scenario_utils.apply_scenario_to_gaml(...) + restart headless, hoặc rl/run_experiments.py."
)
