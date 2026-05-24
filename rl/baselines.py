"""Evaluate non-learning baselines on the MARL GAMA socket (cùng Main_Traffic.gaml)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import (
    LOG_DIR,
    MARL_AGENTS,
    MARL_EXPERIMENT,
    MODEL_PATH,
    SCENARIO_CLI_METADATA_ONLY_HINT,
    SCENARIO_PRESETS,
    GamaConnectionConfig,
)
from rl.gama_compat import patch_gama_gymnasium
from rl.marl_env import AgentIndicatorParallelWrapper
from rl.gama_episode_reset import reset_marl_episode
from rl.metrics import (
    EpisodeMetric,
    build_episode_metric,
    classify_episode,
    finalize_merging_info_on_step_limit,
    write_episode_metrics,
)

patch_gama_gymnasium()

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read baseline evaluation settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=["random", "heuristic", "greedy"],
        default="heuristic",
        help=(
            "Chính sách baseline: "
            "'heuristic'/'greedy' = rule-based (tương đương, khớp thuật ngữ đề cương); "
            "'random' = ngẫu nhiên."
        ),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1001)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_PRESETS.keys()),
        default="medium",
        help=(
            "Nhãn kịch bản trong CSV/output. Không vá GAML — mật độ thực là GAMA đang load (scenario_utils + restart)."
        ),
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def heuristic_action(obs: np.ndarray) -> int:
    """Rule-based Greedy policy dùng 15D merging observation từ Main_Traffic.gaml."""
    speed = float(obs[0])
    in_accel_zone = float(obs[4]) >= 0.5
    gap_rear = float(obs[7])
    speed_rear = float(obs[8])
    gap_safe = float(obs[9]) >= 0.5
    ramp_front_gap = float(obs[10])
    urgency = float(obs[12])

    if in_accel_zone and gap_safe:
        return 3

    if gap_rear < 0.1 and speed_rear > 0.5:
        return 0

    if ramp_front_gap < 0.08:
        return 0

    if urgency > 0.9 and not gap_safe:
        return 3
    if urgency > 0.75 and not gap_safe:
        return 4

    if speed < 0.7:
        return 2

    return 1


def _choose_merging_action(policy: str, obs: np.ndarray, rng: np.random.Generator) -> int:
    if policy == "random":
        return int(rng.integers(0, 5))
    return heuristic_action(obs)


def _choose_highway_action(policy: str, rng: np.random.Generator) -> int:
    if policy == "random":
        return int(rng.integers(0, 5))
    return 1


async def async_main(args: argparse.Namespace) -> None:
    """Run baseline episodes on MARL socket and write metrics for merging_0."""
    scenario = SCENARIO_PRESETS.get(args.scenario)
    if scenario:
        print(f"Scenario: {scenario.name} (nb_cars_max={scenario.nb_cars_max})")
        print(SCENARIO_CLI_METADATA_ONLY_HINT)

    config = GamaConnectionConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )
    parallel_env = GamaParallelEnv(
        gaml_experiment_path=str(MODEL_PATH),
        gaml_experiment_name=MARL_EXPERIMENT,
        gama_ip_address=config.host,
        gama_port=config.port,
    )
    env_ss = AgentIndicatorParallelWrapper(parallel_env, type_only=False)
    rng = np.random.default_rng(args.seed)
    # Bug #4 fix: sinh seed lon, xao tron tot cho moi episode (tranh `seed <- 1.0/2.0...` không xao tron GAMA RNG).
    seed_rng = np.random.default_rng(args.seed)
    rows: list[EpisodeMetric] = []

    try:
        obs_dict, _ = reset_marl_episode(env_ss, args.seed)
        missing = [a for a in MARL_AGENTS if a not in obs_dict]
        if missing:
            raise RuntimeError(f"Thiếu agent MARL: {missing}. Kiểm tra GAML / GAMA headless.")

        for episode in range(1, args.episodes + 1):
            ep_seed = int(seed_rng.integers(1, 2**31 - 1))
            if args.policy == "random":
                rng = np.random.default_rng(ep_seed)

            obs_dict, _ = reset_marl_episode(env_ss, ep_seed)
            ep_reward = 0.0
            length = 0
            done_agents: set[str] = set()
            final_info: dict[str, dict[str, Any]] = {}
            last_infos: dict[str, dict[str, Any]] = {}
            step = 0
            hit_step_limit = False

            while step < args.max_episode_steps:
                if "merging_0" in done_agents:
                    break
                actions: dict[str, int] = {}
                for agent_id in MARL_AGENTS:
                    if agent_id in done_agents:
                        continue
                    obs = obs_dict.get(agent_id)
                    if obs is None:
                        done_agents.add(agent_id)
                        final_info[agent_id] = {"outcome": "missing_agent"}
                        continue
                    if agent_id == "merging_0":
                        arr = np.asarray(obs, dtype=np.float32)[:15]
                        actions[agent_id] = _choose_merging_action(args.policy, arr, rng)
                    else:
                        actions[agent_id] = _choose_highway_action(args.policy, rng)

                if not actions:
                    break

                next_obs, rewards, terminations, truncations, infos = env_ss.step(actions)
                step += 1
                for agent_id, info in (infos or {}).items():
                    if info:
                        last_infos[agent_id] = dict(info)
                if "merging_0" in actions:
                    ep_reward += float(rewards.get("merging_0", 0.0))
                    length += 1

                for agent_id in list(actions.keys()):
                    if terminations.get(agent_id, False) or truncations.get(agent_id, False):
                        if agent_id not in done_agents:
                            done_agents.add(agent_id)
                            final_info[agent_id] = dict(infos.get(agent_id) or {})

                obs_dict = next_obs

            hit_step_limit = step >= args.max_episode_steps
            if "merging_0" not in final_info and last_infos.get("merging_0"):
                final_info["merging_0"] = last_infos["merging_0"]
            m0_info = finalize_merging_info_on_step_limit(
                final_info.get("merging_0", {}),
                hit_step_limit=hit_step_limit,
            )
            outcome = str(m0_info.get("outcome", "timeout"))
            is_term, is_trunc = classify_episode(outcome)
            rows.append(
                build_episode_metric(
                    algorithm=args.policy,
                    seed=args.seed,
                    episode=episode,
                    reward=ep_reward,
                    length=length,
                    terminated=is_term,
                    truncated=is_trunc,
                    info=m0_info,
                )
            )
            print(
                f"episode={episode} reward={ep_reward:.2f} length={length} "
                f"outcome={rows[-1].outcome}"
            )
    finally:
        parallel_env.close()

    output_path = Path(args.out) if args.out else LOG_DIR / f"{args.policy}_baseline_seed{args.seed}.csv"
    write_episode_metrics(output_path, rows)
    print(f"saved baseline metrics: {output_path}")


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
