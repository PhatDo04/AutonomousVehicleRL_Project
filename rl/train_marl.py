"""Huấn luyện MARL (Multi-Agent RL) cho bài toán nhập làn cao tốc.

Sử dụng chiến lược Shared Policy (IPPO/IA2C — Independent MARL với parameter sharing):
  - 4 agent (merging_0, highway_0/1/2) cùng học 1 chính sách duy nhất
  - SuperSuit agent_indicator_v0 thêm one-hot agent ID vào obs (15D → 19D)
    để policy phân biệt vai trò mà không cần model riêng biệt

Lưu ý: DQN bị loại khỏi MARL vì:
  - DQN là off-policy → replay buffer lẫn lộn transitions của 4 agents khác nhau
  - Non-stationarity làm Q-values không hội tụ trong setting đa tác tử
  - PPO/A2C là on-policy → cập nhật ngay trên dữ liệu hiện tại, ổn định hơn trong MARL

Tham khảo: Lowe et al. (2017) MADDPG; de Witt et al. (2020) MAPPO.

Sử dụng:
    python rl/train_marl.py --algo ppo --timesteps 200000 --seed 0
    python rl/train_marl.py --algo a2c --timesteps 200000 --seed 0
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from rl.config import (
    GamaConnectionConfig,
    LOG_DIR,
    MARL_AGENTS,
    MARL_ALGORITHMS,
    MARL_OBS_DIM,
    MODEL_DIR,
    SCENARIO_CLI_METADATA_ONLY_HINT,
    SCENARIO_PRESETS,
    TrainConfig,
)
from rl.marl_env import make_marl_vec_env
from rl.metrics import classify_episode, write_episode_metrics, EpisodeMetric


# ---------------------------------------------------------------------------
# Callback: ghi CSV metrics cho merging_0 sau mỗi episode
# ---------------------------------------------------------------------------

class MARLEpisodeCSVCallback(BaseCallback):
    """Callback ghi metrics episode cho tất cả MARL agents trong training.

    SuperSuit concat 4 agents thành 1 VecEnv. Callback tách riêng:
    - CSV chính (output_path): metrics của merging_0 — để so sánh với single-agent
    - CSV phụ (highway_path): metrics tổng hợp của highway_0/1/2 — theo dõi hành vi cooperative
    """

    # SuperSuit concat agents theo thứ tự MARL_AGENTS → index 0 = merging_0, 1–3 = highway.
    # Dùng để suy ra agent_role khi GAMA trả về missing_agent (không có agent_role trong info).
    _AGENT_INDEX_TO_ROLE: dict[int, str] = {
        i: ("merging" if agent == "merging_0" else "highway")
        for i, agent in enumerate(MARL_AGENTS)
    }

    def __init__(self, algorithm: str, seed: int, output_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.algorithm = f"marl_{algorithm}"
        self.seed = seed
        self.output_path = output_path
        self.highway_path = output_path.parent / output_path.name.replace("_episodes.csv", "_highway_episodes.csv")
        self.merging_rows: list[EpisodeMetric] = []
        self.highway_rows: list[EpisodeMetric] = []
        self._ep_rewards: dict[int, float] = {}
        self._ep_lengths: dict[int, int] = {}
        self._hw_ep_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
            self._ep_rewards[i] = self._ep_rewards.get(i, 0.0) + float(reward)
            self._ep_lengths[i] = self._ep_lengths.get(i, 0) + 1

            if done:
                total_r = self._ep_rewards.pop(i, 0.0)
                length = self._ep_lengths.pop(i, 0)
                # Fallback khi GAMA trả về missing_agent (xe đã chết) — info không có agent_role.
                # Suy ra từ vị trí index trong VecEnv (0=merging, 1–3=highway).
                # i luôn trong [0, 3] vì concat_vec_envs tạo đúng 4 slots.
                # Dùng .get(i, "highway") thay vì i % len để tránh nhầm lẫn toán học.
                role = str(info.get("agent_role") or self._AGENT_INDEX_TO_ROLE.get(i, "highway"))
                outcome = str(info.get("outcome", "unknown"))
                wrapper_trunc = bool(info.get("TimeLimit.truncated", False))
                is_terminated, is_truncated = classify_episode(outcome, wrapper_trunc)

                if role == "merging":
                    self.merging_rows.append(EpisodeMetric(
                        algorithm=self.algorithm,
                        seed=self.seed,
                        episode=len(self.merging_rows) + 1,
                        reward=total_r,
                        length=length,
                        terminated=is_terminated,
                        truncated=is_truncated,
                        outcome=outcome,
                        success=outcome == "success",
                        collision=outcome == "collision",
                        failed_merge=outcome == "failed_merge",
                        timeout=outcome == "timeout",
                        mean_speed=float(info.get("mean_speed", 0.0)),
                        min_front_gap=float(info.get("min_front_gap", 999.0)),
                        min_rear_gap=float(info.get("min_rear_gap", 999.0)),
                        merge_step=int(info.get("merge_step", 0)) if outcome == "success" else 0,
                        throughput=float(info.get("throughput", 0.0)),
                        shockwave_index=float(info.get("shockwave_index", 0.0)),
                    ))
                elif role == "highway":
                    self._hw_ep_count += 1
                    self.highway_rows.append(EpisodeMetric(
                        algorithm=f"{self.algorithm}_highway",
                        seed=self.seed,
                        episode=self._hw_ep_count,
                        reward=total_r,
                        length=length,
                        terminated=is_terminated,
                        truncated=is_truncated,
                        outcome=outcome,
                        success=outcome == "exited",
                        collision=outcome == "collision",
                        failed_merge=False,
                        timeout=outcome == "timeout",
                        mean_speed=float(info.get("mean_speed", 0.0)),
                        min_front_gap=float(info.get("min_front_gap", 999.0)),
                        min_rear_gap=float(info.get("min_rear_gap", 999.0)),
                    ))
        return True

    def _on_training_end(self) -> None:
        write_episode_metrics(self.output_path, self.merging_rows)
        if self.highway_rows:
            write_episode_metrics(self.highway_path, self.highway_rows)
            print(f"  Đã lưu highway metrics: {self.highway_path}")


# ---------------------------------------------------------------------------
# Model builders — giữ nguyên hyperparameter như single-agent nhưng obs_dim=19
# ---------------------------------------------------------------------------

def build_marl_model(algo: str, env, seed: int, tensorboard_log: str):
    """Tạo SB3 on-policy model cho MARL (PPO hoặc A2C).

    DQN bị loại trừ khỏi MARL vì off-policy replay buffer không ổn định
    trong môi trường đa tác tử (non-stationarity problem).
    """
    if algo not in MARL_ALGORITHMS:
        raise ValueError(
            f"Thuật toán '{algo}' không được hỗ trợ trong MARL. "
            f"Dùng: {MARL_ALGORITHMS}. "
            f"DQN bị loại vì off-policy không phù hợp đa tác tử."
        )

    if algo == "ppo":
        # V6: ent_coef 0.08 -> 0.20 (anti policy-collapse).
        # Iter 5 cho thay PPO collapse ve Keep 72.6% du phat -0.5/tick trong zone -> argmax
        # bi stuck, exploration khong du de escape. Bump entropy de policy network giu
        # probability cao cho cac action it duoc chon -> argmax co the flip qua Merge.
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=5e-4,
            n_steps=256,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.20,
            vf_coef=0.5,
            tensorboard_log=tensorboard_log,
            seed=seed,
            verbose=1,
        )

    if algo == "a2c":
        # Iter 10: A2C iter 9 collapse hoan toan (0% success, 100% collision, 86% accel)
        # vi C1+ GAML (bo per-tick gap penalty) thay doi reward landscape qua dot ngot
        # voi A2C don gian (khong co clipping). Stabilize hyperparameter:
        #   - learning_rate 1e-3 -> 5e-4: update nhe hon, tranh collapse "spam accel"
        #   - ent_coef 0.32 -> 0.15: giam random exploration, khai thac policy an toan
        #   - n_steps 32 -> 64: advantage estimate chinh xac hon -> gradient on dinh hon
        return A2C(
            "MlpPolicy",
            env,
            learning_rate=5e-4,
            n_steps=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.15,
            tensorboard_log=tensorboard_log,
            seed=seed,
            verbose=1,
        )

    raise ValueError(f"Thuật toán không được hỗ trợ: {algo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=list(MARL_ALGORITHMS), default="ppo", help="Thuật toán RL (MARL: chỉ PPO/A2C).")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Tổng số training timesteps.")
    parser.add_argument("--seed", type=int, default=TrainConfig().seed)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1001)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Nếu >0, lưu checkpoint mỗi N timesteps để chọn best model sau train.",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_PRESETS.keys()),
        default="medium",
        help=(
            "Nhãn kịch bản (metadata). Không vá GAML — xem rl/scenario_utils + restart hoặc run_experiments.py."
        ),
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)

    run_name = args.run_name or f"marl_{args.algo}_seed{args.seed}"
    model_path = MODEL_DIR / f"{run_name}.zip"
    metric_path = LOG_DIR / f"{run_name}_episodes.csv"
    tensorboard_log = str(LOG_DIR / "tensorboard")
    checkpoint_dir = MODEL_DIR / "checkpoints" / run_name

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_interval > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scenario = SCENARIO_PRESETS.get(args.scenario)
    if scenario:
        print(f"Scenario: {scenario.name} (nb_cars_max={scenario.nb_cars_max})")
        print(SCENARIO_CLI_METADATA_ONLY_HINT)

    print(f"\n=== MARL Training: {args.algo.upper()} | seed={args.seed} | {args.timesteps:,} steps ===")
    print(f"  4 agents: merging_0 + highway_0/1/2")
    print(f"  Obs dim : {MARL_OBS_DIM}D (15 sensor + 4 one-hot agent ID)")
    print(f"  Policy  : Shared (parameter sharing across all agents)\n")

    config = GamaConnectionConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )

    env = make_marl_vec_env(config)

    try:
        model = build_marl_model(args.algo, env, args.seed, tensorboard_log)
        callbacks = [MARLEpisodeCSVCallback(args.algo, args.seed, metric_path)]
        if args.checkpoint_interval > 0:
            # SB3 callback n_calls tăng theo VecEnv step, còn num_timesteps tăng theo n_envs.
            # MARL concat 4 agents => chia interval để checkpoint đúng theo tổng timestep.
            save_freq = max(1, args.checkpoint_interval // max(1, int(getattr(env, "num_envs", 1))))
            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix=run_name,
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )
        callback = CallbackList(callbacks)
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=run_name,
        )
        model.save(model_path)
        print(f"\nĐã lưu model      : {model_path}")
        print(f"Đã lưu metrics    : {metric_path}")
        print(f"Highway metrics   : {metric_path.parent / metric_path.name.replace('_episodes.csv', '_highway_episodes.csv')}")
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
