"""Đánh giá model MARL đã train và xuất metrics episode cho tất cả agents.

Script load model MARL (shared policy, 19D obs) và chạy qua môi trường
PettingZoo 4-agent. Xuất 2 CSV:
  - {run_label}.csv            : metrics của merging_0 (so sánh với single-agent)
  - {run_label}_highway.csv   : metrics tổng hợp của highway_0/1/2

Sử dụng:
    python rl/evaluate_marl.py --algo ppo --model outputs/models/marl_ppo_seed0_200k.zip --episodes 50
    python rl/evaluate_marl.py --algo a2c --model outputs/models/marl_a2c_seed0_200k.zip --episodes 50
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")

import numpy as np
from stable_baselines3 import A2C, PPO

from rl.config import (
    GamaConnectionConfig,
    LOG_DIR,
    MARL_AGENTS,
    MARL_ALGORITHMS,
    MARL_HIGHWAY_AGENTS,
    MODEL_PATH,
)
from rl.gama_compat import patch_gama_gymnasium
from rl.metrics import EpisodeMetric, classify_episode, write_episode_metrics
from rl.marl_env import AgentIndicatorParallelWrapper

patch_gama_gymnasium()

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv  # noqa: E402


def load_marl_model(algo: str, model_path: str):
    """Load SB3 model MARL (không cần env khi chỉ dùng predict)."""
    if algo == "ppo":
        return PPO.load(model_path)
    if algo == "a2c":
        return A2C.load(model_path)
    raise ValueError(f"Thuật toán '{algo}' không được hỗ trợ. MARL dùng: {MARL_ALGORITHMS}")


# Action map cho merging_0 theo `species car.reflex behave` trong Main_Traffic.gaml.
_MERGING_ACTION_NAMES = {
    0: "brake",
    1: "keep",
    2: "accel",
    3: "merge",
    4: "wait",
}


def _format_action_hist(counter: "Counter[int]") -> str:
    total = sum(counter.values())
    if total == 0:
        return "(không có action)"
    parts: list[str] = []
    for a in sorted(counter):
        name = _MERGING_ACTION_NAMES.get(a, f"a{a}")
        pct = 100.0 * counter[a] / total
        parts.append(f"{a}={name}:{counter[a]}({pct:.1f}%)")
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=list(MARL_ALGORITHMS), required=True)
    parser.add_argument("--model", required=True, help="Đường dẫn .zip model MARL.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1001)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--out", default=None, help="Tùy chọn đường dẫn CSV output.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Predict với deterministic=False (lấy mẫu từ phân phối policy) thay vì argmax.",
    )
    parser.add_argument(
        "--log-actions",
        action="store_true",
        help="In action histogram của từng agent cuối mỗi episode và tổng kết để debug policy.",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    """Chạy đánh giá deterministic và lưu metrics của merging_0."""
    model = load_marl_model(args.algo, args.model)

    # Tạo PettingZoo parallel env với 4 agents
    config = GamaConnectionConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )
    parallel_env = GamaParallelEnv(
        gaml_experiment_path=str(MODEL_PATH),
        gaml_experiment_name=config.experiment_name,
        gama_ip_address=config.host,
        gama_port=config.port,
    )

    # Xác nhận 4 agent có mặt
    obs_dict, _ = parallel_env.reset(seed=args.seed)
    missing = [a for a in MARL_AGENTS if a not in obs_dict]
    if missing:
        raise RuntimeError(f"Thiếu agent MARL: {missing}. Kiểm tra GAML.")

    # Cùng pipeline obs như train (không dùng SuperSuit AEC round-trip — tránh KeyError GAMA).
    env_ss = AgentIndicatorParallelWrapper(parallel_env, type_only=False)

    deterministic = not args.stochastic
    mode_label = "deterministic" if deterministic else "stochastic"
    print(f"\n=== MARL Evaluation: {args.algo.upper()} | {args.episodes} episodes | {mode_label} ===")
    print(f"  Model: {args.model}")
    print(f"  Agents: {list(MARL_AGENTS)}\n")

    rows: list[EpisodeMetric] = []
    highway_stats: list[EpisodeMetric] = []
    merging_action_counter_total: Counter[int] = Counter()
    action_counter_total: dict[str, Counter[int]] = {agent_id: Counter() for agent_id in MARL_AGENTS}

    try:
        for episode in range(1, args.episodes + 1):
            obs_dict, _ = env_ss.reset(seed=args.seed + episode)

            ep_rewards: dict[str, float] = {a: 0.0 for a in MARL_AGENTS}
            ep_lengths: dict[str, int] = {a: 0 for a in MARL_AGENTS}
            done_agents: set[str] = set()
            final_info: dict[str, Any] = {}
            step = 0
            merging_actions_ep: Counter[int] = Counter()
            actions_ep: dict[str, Counter[int]] = {agent_id: Counter() for agent_id in MARL_AGENTS}

            while len(done_agents) < len(MARL_AGENTS) and step < args.max_episode_steps:
                actions: dict[str, int] = {}
                for agent_id in MARL_AGENTS:
                    if agent_id in done_agents:
                        continue
                    obs = obs_dict.get(agent_id)
                    if obs is None:
                        # Agent không có trong obs_dict → PettingZoo đã xóa nó.
                        # Đánh dấu done để tránh vòng lặp vô tận (race condition).
                        if agent_id not in done_agents:
                            done_agents.add(agent_id)
                            final_info[agent_id] = {"outcome": "missing_agent"}
                        continue
                    obs_arr = np.asarray(obs, dtype=np.float32)
                    # Không dùng assert (bị tắt với python -O); raise rõ ràng nếu pipeline obs sai kích thước.
                    if obs_arr.shape != (19,):
                        raise RuntimeError(
                            f"{agent_id}: obs shape {obs_arr.shape} ≠ (19,). "
                            f"Kiểm tra AgentIndicatorParallelWrapper / obs 19D trong evaluate_marl."
                        )
                    action, _ = model.predict(obs_arr, deterministic=deterministic)
                    a_int = int(action)
                    actions[agent_id] = a_int
                    actions_ep[agent_id][a_int] += 1
                    action_counter_total[agent_id][a_int] += 1
                    if agent_id == "merging_0":
                        merging_actions_ep[a_int] += 1
                        merging_action_counter_total[a_int] += 1

                if not actions:
                    break  # Tất cả agent đã done hoặc missing

                active_agents = set(actions)
                next_obs, rewards, terminations, truncations, infos = env_ss.step(actions)
                step += 1

                for agent_id in active_agents:
                    r = float(rewards.get(agent_id, 0.0))
                    ep_rewards[agent_id] = ep_rewards.get(agent_id, 0.0) + r
                    ep_lengths[agent_id] = ep_lengths.get(agent_id, 0) + 1

                    if terminations.get(agent_id, False) or truncations.get(agent_id, False):
                        if agent_id not in done_agents:
                            done_agents.add(agent_id)
                            final_info[agent_id] = dict(infos.get(agent_id) or {})

                obs_dict = next_obs

            # Ghi metrics của merging_0
            m0_info = final_info.get("merging_0", {})
            outcome = str(m0_info.get("outcome", "timeout"))
            is_term, is_trunc = classify_episode(outcome)
            rows.append(EpisodeMetric(
                algorithm=f"marl_{args.algo}",
                seed=args.seed,
                episode=episode,
                reward=ep_rewards.get("merging_0", 0.0),
                length=ep_lengths.get("merging_0", 0),
                terminated=is_term,
                truncated=is_trunc,
                outcome=outcome,
                success=outcome == "success",
                collision=outcome == "collision",
                failed_merge=outcome == "failed_merge",
                timeout=outcome == "timeout",
                mean_speed=float(m0_info.get("mean_speed", 0.0)),
                min_front_gap=float(m0_info.get("min_front_gap", 999.0)),
                min_rear_gap=float(m0_info.get("min_rear_gap", 999.0)),
                merge_step=int(m0_info.get("merge_step", 0)) if outcome == "success" else 0,
                throughput=float(m0_info.get("throughput", 0.0)),
                shockwave_index=float(m0_info.get("shockwave_index", 0.0)),
            ))

            # Ghi thống kê highway agents theo từng agent
            hw_collisions = 0
            for agent_id in MARL_HIGHWAY_AGENTS:
                hw_info = final_info.get(agent_id, {})
                hw_outcome = str(hw_info.get("outcome", "unknown"))
                hw_r = ep_rewards.get(agent_id, 0.0)
                hw_len = ep_lengths.get(agent_id, 0)
                if hw_outcome == "collision":
                    hw_collisions += 1
                hw_term, hw_trunc = classify_episode(hw_outcome)
                highway_stats.append(EpisodeMetric(
                    algorithm=f"marl_{args.algo}_highway",
                    seed=args.seed,
                    episode=episode,
                    reward=hw_r,
                    length=hw_len,
                    terminated=hw_term,
                    truncated=hw_trunc,
                    outcome=hw_outcome,
                    success=hw_outcome == "exited",
                    collision=hw_outcome == "collision",
                    failed_merge=False,
                    timeout=hw_outcome == "timeout",
                    mean_speed=float(hw_info.get("mean_speed", 0.0)),
                    min_front_gap=float(hw_info.get("min_front_gap", 999.0)),
                    min_rear_gap=float(hw_info.get("min_rear_gap", 999.0)),
                ))

            print(
                f"episode={episode:3d} | merging_0: reward={ep_rewards.get('merging_0', 0.0):7.2f} "
                f"outcome={outcome:<12} | hw_collisions={hw_collisions}"
            )
            if args.log_actions:
                for agent_id in MARL_AGENTS:
                    hist = _format_action_hist(actions_ep[agent_id])
                    print(f"            {agent_id} actions ({sum(actions_ep[agent_id].values())} steps): {hist}")

    finally:
        # Gọi env_ss.close() thay vì parallel_env.close() để SuperSuit wrapper
        # được dọn dẹp đúng cách trước khi đóng kết nối GAMA socket.
        env_ss.close()

    # Lưu CSV cho merging_0
    run_label = f"marl_{args.algo}_eval_seed{args.seed}"
    output_path = Path(args.out) if args.out else LOG_DIR / f"{run_label}.csv"
    write_episode_metrics(output_path, rows)
    print(f"\nĐã lưu metrics merging_0 : {output_path}")

    # Lưu CSV cho highway agents
    if highway_stats:
        hw_path = output_path.parent / output_path.name.replace(".csv", "_highway.csv")
        write_episode_metrics(hw_path, highway_stats)
        print(f"Đã lưu metrics highway   : {hw_path}")

    # Tóm tắt
    total = len(rows)
    if total > 0:
        sr = sum(1 for r in rows if r.success) / total
        cr = sum(1 for r in rows if r.collision) / total
        fm = sum(1 for r in rows if r.failed_merge) / total
        to = sum(1 for r in rows if r.timeout) / total
        avg_r = sum(r.reward for r in rows) / total
        hw_total = len(highway_stats)
        hw_coll_rate = sum(1 for s in highway_stats if s.collision) / hw_total if hw_total else 0.0
        hw_exit_rate = sum(1 for s in highway_stats if s.success) / hw_total if hw_total else 0.0
        hw_avg_r = sum(s.reward for s in highway_stats) / hw_total if hw_total else 0.0
        print(f"\n=== Kết quả MARL ({args.algo.upper()}, {total} episodes, {mode_label}) ===")
        print(f"  merging_0  | success={sr:.1%}  collision={cr:.1%}  failed_merge={fm:.1%}  timeout={to:.1%}")
        print(f"  merging_0  | avg_reward={avg_r:.2f}")
        print(f"  highway    | exit_rate={hw_exit_rate:.1%}  collision_rate={hw_coll_rate:.1%}  avg_reward={hw_avg_r:.2f}")
        if args.log_actions and merging_action_counter_total:
            for agent_id in MARL_AGENTS:
                total_actions = sum(action_counter_total[agent_id].values())
                hist = _format_action_hist(action_counter_total[agent_id])
                print(f"  {agent_id} action histogram ({total_actions} steps): {hist}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
