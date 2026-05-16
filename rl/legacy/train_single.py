"""Train DQN, PPO, or A2C on the GAMA highway merging environment."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stable-Baselines3 only needs PyTorch TensorBoard writer; avoid importing local TensorFlow.
os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from rl.config import (
    ALGORITHMS,
    LOG_DIR,
    MODEL_DIR,
    SCENARIO_CLI_METADATA_ONLY_HINT,
    SCENARIO_PRESETS,
    SingleAgentGamaConfig,
    TrainConfig,
)
from rl.legacy.env_single import GamaMergingEnv
from rl.metrics import EpisodeCSVCallback


def parse_args() -> argparse.Namespace:
    """Read command-line options so experiments are reproducible from the terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGORITHMS, default="dqn", help="RL algorithm to train.")
    parser.add_argument("--timesteps", type=int, default=TrainConfig().total_timesteps, help="Training steps.")
    parser.add_argument("--seed", type=int, default=TrainConfig().seed, help="Random seed.")
    parser.add_argument("--host", default="localhost", help="GAMA headless host.")
    parser.add_argument("--port", type=int, default=1001, help="GAMA headless socket port.")
    parser.add_argument("--max-episode-steps", type=int, default=300, help="Wrapper episode timeout.")
    parser.add_argument("--run-name", default=None, help="Optional name used for model/log files.")
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
            "Nhãn kịch bản (CSV/run name). KHÔNG vá Main_Traffic.gaml — đồng bộ mật độ: scenario_utils + restart "
            "hoặc run_experiments.py."
        ),
    )
    return parser.parse_args()


def build_model(algo: str, env: Monitor, seed: int, tensorboard_log: str):
    """Create a Stable-Baselines3 model with conservative defaults for first experiments."""
    if algo == "dqn":
        return DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            tensorboard_log=tensorboard_log,
            seed=seed,
            verbose=1,
        )

    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=tensorboard_log,
            seed=seed,
            verbose=1,
        )

    if algo == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            learning_rate=7e-4,
            n_steps=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log=tensorboard_log,
            seed=seed,
            verbose=1,
        )

    raise ValueError(f"Unsupported algorithm: {algo}")


async def async_main(args: argparse.Namespace) -> None:
    """Run one training job and save both model weights and episode CSV metrics."""
    set_random_seed(args.seed)

    run_name = args.run_name or f"{args.algo}_seed{args.seed}"
    model_path = MODEL_DIR / f"{run_name}.zip"
    # Ghi scenario vào tên file CSV để phân biệt khi chạy nhiều kịch bản.
    metric_path = LOG_DIR / f"{run_name}_episodes.csv"
    monitor_dir = LOG_DIR / "monitor" / run_name
    tensorboard_log = str(LOG_DIR / "tensorboard")
    checkpoint_dir = MODEL_DIR / "checkpoints" / run_name

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_interval > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = SingleAgentGamaConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )
    scenario = SCENARIO_PRESETS.get(args.scenario)
    if scenario:
        print(f"Scenario: {scenario.name} (nb_cars_max={scenario.nb_cars_max}, spawn_interval={scenario.spawn_interval})")
        print(SCENARIO_CLI_METADATA_ONLY_HINT)
    env = Monitor(GamaMergingEnv(config), filename=str(monitor_dir / "monitor.csv"))

    try:
        model = build_model(args.algo, env, args.seed, tensorboard_log)
        callbacks = [EpisodeCSVCallback(args.algo, args.seed, metric_path)]
        if args.checkpoint_interval > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=args.checkpoint_interval,
                    save_path=str(checkpoint_dir),
                    name_prefix=run_name,
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )
        callback = CallbackList(callbacks)
        model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name=run_name)
        model.save(model_path)
        print(f"saved model: {model_path}")
        print(f"saved metrics: {metric_path}")
    finally:
        env.close()


def main() -> None:
    """Start training inside asyncio.run because gama-client requires a running event loop.

    Lưu ý kỹ thuật: asyncio.run() ở đây không phải async thực sự.
    gama-client dùng asyncio internally khi khởi tạo kết nối socket, nên cần
    event loop tồn tại. Sau khi kết nối xong, model.learn() của SB3 chạy
    synchronous blocking — không có coroutine nào chạy song song.
    """
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
