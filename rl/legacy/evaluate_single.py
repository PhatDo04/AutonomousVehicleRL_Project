"""Evaluate trained RL policies and export episode-level metrics."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stable-Baselines3 imports TensorBoard through PyTorch; do not load TensorFlow during evaluation.
os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")

from stable_baselines3 import A2C, DQN, PPO

from rl.config import ALGORITHMS, LOG_DIR, SingleAgentGamaConfig
from rl.legacy.env_single import GamaMergingEnv
from rl.metrics import EpisodeMetric, build_episode_metric, write_episode_metrics


ACTION_NAMES = {
    0: "brake",
    1: "keep",
    2: "accel",
    3: "merge",
    4: "wait",
}


def _format_action_hist(counter: Counter[int]) -> str:
    total = sum(counter.values())
    if total <= 0:
        return "(no actions)"
    parts = []
    for action_id, count in sorted(counter.items()):
        label = ACTION_NAMES.get(action_id, str(action_id))
        parts.append(f"{action_id}={label}:{count}({count / total:.1%})")
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    """Read evaluation settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGORITHMS, required=True, help="Algorithm used by the saved model.")
    parser.add_argument("--model", required=True, help="Path to the .zip model produced by train.py.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    parser.add_argument("--host", default="localhost", help="GAMA headless host.")
    parser.add_argument("--port", type=int, default=1001, help="GAMA headless socket port.")
    parser.add_argument("--max-episode-steps", type=int, default=300, help="Wrapper episode timeout.")
    parser.add_argument("--out", default=None, help="Optional CSV output path.")
    parser.add_argument("--log-actions", action="store_true", help="Print action histogram for each episode and total.")
    return parser.parse_args()


def load_model(algo: str, model_path: str, env: GamaMergingEnv):
    """Load the correct Stable-Baselines3 class for the selected algorithm."""
    if algo == "dqn":
        return DQN.load(model_path, env=env)
    if algo == "ppo":
        return PPO.load(model_path, env=env)
    if algo == "a2c":
        return A2C.load(model_path, env=env)
    raise ValueError(f"Unsupported algorithm: {algo}")


async def async_main(args: argparse.Namespace) -> None:
    """Run deterministic evaluation and save one CSV row per episode."""
    config = SingleAgentGamaConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )
    env = GamaMergingEnv(config)
    model = load_model(args.algo, args.model, env)

    rows: list[EpisodeMetric] = []
    action_counter_total: Counter[int] = Counter()
    try:
        for episode in range(1, args.episodes + 1):
            obs, _info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            length = 0
            terminated = False
            truncated = False
            info = {}
            action_counter_ep: Counter[int] = Counter()

            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                action_int = int(action)
                action_counter_ep[action_int] += 1
                action_counter_total[action_int] += 1
                obs, reward, terminated, truncated, info = env.step(action_int)
                total_reward += reward
                length += 1

            rows.append(
                build_episode_metric(
                    algorithm=args.algo,
                    seed=args.seed,
                    episode=episode,
                    reward=total_reward,
                    length=length,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
            print(
                f"episode={episode} reward={total_reward:.2f} length={length} "
                f"outcome={rows[-1].outcome}"
            )
            if args.log_actions:
                print(f"  actions ({sum(action_counter_ep.values())} steps): {_format_action_hist(action_counter_ep)}")
    finally:
        env.close()

    output_path = Path(args.out) if args.out else LOG_DIR / f"{args.algo}_eval_seed{args.seed}.csv"
    write_episode_metrics(output_path, rows)
    print(f"saved evaluation metrics: {output_path}")
    if args.log_actions:
        print(f"action histogram ({sum(action_counter_total.values())} steps): {_format_action_hist(action_counter_total)}")


def main() -> None:
    """Start evaluation inside asyncio.run because gama-client requires a running event loop."""
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
