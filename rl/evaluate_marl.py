"""Đánh giá model MARL đã train và xuất metrics episode cho tất cả agents.

Script load model MARL (shared policy, 19D obs) và chạy qua môi trường
PettingZoo 4-agent. Xuất 2 CSV:
  - {run_label}.csv            : metrics của merging_0 (agent nhập làn)
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
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.utils import obs_as_tensor

from rl.config import (
    GamaConnectionConfig,
    LOG_DIR,
    MARL_AGENTS,
    MARL_ALGORITHMS,
    MARL_EXPERIMENT,
    MARL_HIGHWAY_AGENTS,
    MODEL_PATH,
)
from rl.gama_compat import patch_gama_gymnasium
from rl.metrics import (
    EpisodeMetric,
    classify_episode,
    finalize_merging_info_on_step_limit,
    write_episode_metrics,
)
from rl.gama_episode_reset import reset_marl_episode
from rl.marl_env import AgentIndicatorParallelWrapper
from rl.centralized_policy import ACTOR_DIM, CENTRALIZED_OBS_DIM

patch_gama_gymnasium()

from gama_gymnasium.exceptions import GamaCommandError  # noqa: E402
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv  # noqa: E402


def _is_gama_sim_lost(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "unable to find" in msg and ("experiment" in msg or "simulation" in msg)


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


def _to_model_obs(local_obs: np.ndarray) -> np.ndarray:
    """Pad local obs 19D → 79D (zeros cho phần global) cho CTDE decentralized execution.

    Model train với obs 79D (19 local + 60 global state). Tại execution không có global
    state thực — pad 0; actor (``CentralizedCriticPolicy``) chỉ đọc ``obs[:19]`` nên
    action không đổi. Đây là chứng minh chính cho tính chất decentralized execution
    của kiến trúc CTDE.
    """
    arr = np.asarray(local_obs, dtype=np.float32)
    if arr.shape[0] == CENTRALIZED_OBS_DIM:
        return arr
    padded = np.zeros(CENTRALIZED_OBS_DIM, dtype=np.float32)
    padded[: ACTOR_DIM] = arr[:ACTOR_DIM] if arr.shape[0] >= ACTOR_DIM else arr
    return padded


def _predict_marl_action(
    model: PPO | A2C,
    obs_arr: np.ndarray,
    *,
    agent_id: str,
    deterministic: bool,
    eval_mask: bool,
) -> int:
    """Chọn action cho 1 agent; với deterministic eval của ``merging_0`` áp action mask.

    Mask boost xác suất action 3 (merge) khi đang trong accel zone + gap an toàn,
    đồng thời triệt tiêu brake/keep khi tốc độ quá thấp — giúp argmax không kẹt vào
    hành vi thụ động khi policy có xác suất gần đều giữa nhiều action.
    """
    model_obs = _to_model_obs(obs_arr)
    if not deterministic or agent_id != "merging_0" or not eval_mask:
        action, _ = model.predict(model_obs, deterministic=deterministic)
        return int(action)

    obs_tensor = obs_as_tensor(model_obs.reshape(1, -1), model.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0].astype(np.float64)

    speed = float(obs_arr[0])
    in_accel = float(obs_arr[4]) > 0.5
    gap_safe = float(obs_arr[9]) > 0.5
    masked = probs.copy()
    if in_accel:
        if speed < 0.35:
            masked[0] = 0.0
        if speed < 0.14:
            masked[1] *= 0.25
        if gap_safe:
            masked[3] *= 2.0
        masked[2] *= 1.35
    total = masked.sum()
    if total <= 1e-8:
        return int(np.argmax(probs))
    masked /= total
    return int(np.argmax(masked))


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
    parser.add_argument(
        "--no-eval-mask",
        action="store_true",
        help="Tắt mask argmax cho merging_0 (deterministic thuần SB3).",
    )
    parser.add_argument(
        "--experiment",
        default=MARL_EXPERIMENT,
        help="Tên experiment GAML. Hướng B (UI): TrafficSimulation. Headless train/eval: TrafficMARLHeadless.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Giây sleep giữa các bước (vd 0.25) để quan sát trên GAMA UI.",
    )
    parser.add_argument(
        "--watch-ui",
        action="store_true",
        help="Demo trên GAMA UI: giảm reset, in cảnh báo NOTREADY/PAUSED, mặc định pause 5s giữa các episode.",
    )
    parser.add_argument(
        "--pause-between-episodes",
        type=float,
        default=None,
        help="Giây chờ trước khi reset episode tiếp (tránh GAMA UI nháy liên tục).",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Lặp episode liên tục đến khi Ctrl+C (demo GAMA UI, không dừng sau 1 lượt).",
    )
    return parser.parse_args()


def _apply_watch_ui_defaults(args: argparse.Namespace) -> None:
    if args.pause_between_episodes is None:
        args.pause_between_episodes = 5.0 if args.watch_ui else 0.0
    if args.watch_ui:
        print(
            "\n  [watch-ui] Mỗi episode mới Python gọi reset() → GAMA báo NOTREADY/PAUSED và UI có thể "
            "nháy/reload. Để xem một lượt merge ổn định: --episodes 1. "
            "Giữa các episode: pause {:.0f}s (--pause-between-episodes).\n".format(
                args.pause_between_episodes
            )
        )


async def async_main(args: argparse.Namespace) -> None:
    """Chạy đánh giá deterministic và lưu metrics của merging_0."""
    _apply_watch_ui_defaults(args)
    model = load_marl_model(args.algo, args.model)

    # Tạo PettingZoo parallel env với 4 agents
    config = GamaConnectionConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )
    exp_name = args.experiment
    parallel_env = GamaParallelEnv(
        gaml_experiment_path=str(MODEL_PATH),
        gaml_experiment_name=exp_name,
        gama_ip_address=config.host,
        gama_port=config.port,
    )

    print(f"  GAML experiment: {exp_name}  |  socket {config.host}:{config.port}")
    if exp_name == "TrafficSimulation":
        print(
            "  [UI] Dashboard policy = Python/SB3 model. "
            "Chọn model bằng --model path/to/file.zip (không có tham số tương ứng trên GAMA)."
        )

    # Cùng pipeline obs như train: 15D GAMA + one-hot agent ID → 19D (không lấy obs thô từ parallel_env).
    env_ss = AgentIndicatorParallelWrapper(parallel_env, type_only=False)
    obs_dict, _ = env_ss.reset(seed=args.seed)
    missing = [a for a in MARL_AGENTS if a not in obs_dict]
    if missing:
        raise RuntimeError(f"Thiếu agent MARL: {missing}. Kiểm tra GAML.")
    for agent_id in MARL_AGENTS:
        if agent_id in obs_dict:
            sh = np.asarray(obs_dict[agent_id], dtype=np.float32).shape
            if sh != (19,):
                raise RuntimeError(
                    f"{agent_id}: obs shape {sh} ≠ (19,) sau AgentIndicatorParallelWrapper."
                )

    deterministic = not args.stochastic
    eval_mask = not args.no_eval_mask
    mode_label = "deterministic" if deterministic else "stochastic"
    if deterministic and eval_mask:
        mode_label = "deterministic+mask"
    ep_label = "∞ (Ctrl+C dừng)" if args.forever else str(args.episodes)
    print(f"\n=== MARL Evaluation: {args.algo.upper()} | {ep_label} episodes | {mode_label} ===")
    print(f"  Model: {args.model}")
    print(f"  Agents: {list(MARL_AGENTS)}\n")
    if args.forever:
        print("  [forever] Chạy liên tục — nhấn Ctrl+C trong terminal để dừng.\n")

    rows: list[EpisodeMetric] = []
    highway_stats: list[EpisodeMetric] = []
    merging_action_counter_total: Counter[int] = Counter()
    action_counter_total: dict[str, Counter[int]] = {agent_id: Counter() for agent_id in MARL_AGENTS}

    # Bug #4 fix: sinh seed lon, xao tron tot cho moi episode tu RNG goc co seed.
    # Truoc day truyen args.seed+episode = 1,2,3... -> GAMA `seed <- 1.0;` khong xao tron
    # du RNG noi bo -> initial conditions giong nhau -> 20 episodes co metrics y het nhau.
    seed_rng = np.random.default_rng(args.seed)
    ep_seed = args.seed

    episode = 0
    try:
        while True:
            episode += 1
            if not args.forever and episode > args.episodes:
                break
            if episode > 1:
                if args.pause_between_episodes > 0:
                    print(
                        f"\n  [episode {episode}] Chờ {args.pause_between_episodes:.1f}s "
                        "trước reset GAMA (UI có thể nháy)..."
                    )
                    await asyncio.sleep(args.pause_between_episodes)
                ep_seed = int(seed_rng.integers(1, 2**31 - 1))
                obs_dict, _ = reset_marl_episode(env_ss, ep_seed)
            # Episode 1: dùng obs sau reset khởi tạo — tránh reset GAMA lần 2 ngay đầu (UI nháy thừa).

            ep_rewards: dict[str, float] = {a: 0.0 for a in MARL_AGENTS}
            ep_lengths: dict[str, int] = {a: 0 for a in MARL_AGENTS}
            done_agents: set[str] = set()
            final_info: dict[str, Any] = {}
            last_infos: dict[str, dict[str, Any]] = {}
            step = 0
            merging_actions_ep: Counter[int] = Counter()
            actions_ep: dict[str, Counter[int]] = {agent_id: Counter() for agent_id in MARL_AGENTS}

            while step < args.max_episode_steps:
                if "merging_0" in done_agents:
                    break
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
                    a_int = _predict_marl_action(
                        model,
                        obs_arr,
                        agent_id=agent_id,
                        deterministic=deterministic,
                        eval_mask=eval_mask,
                    )
                    actions[agent_id] = a_int
                    actions_ep[agent_id][a_int] += 1
                    action_counter_total[agent_id][a_int] += 1
                    if agent_id == "merging_0":
                        merging_actions_ep[a_int] += 1
                        merging_action_counter_total[a_int] += 1

                if not actions:
                    break  # Tất cả agent đã done hoặc missing

                active_agents = set(actions)
                try:
                    next_obs, rewards, terminations, truncations, infos = env_ss.step(actions)
                except GamaCommandError as exc:
                    if not _is_gama_sim_lost(exc):
                        raise
                    print(
                        "\n  [GAMA] Simulation mat ket noi (NOTREADY/NONE). "
                        "Bam Play tren TrafficSimulation, cho 3s roi thu lai..."
                    )
                    await asyncio.sleep(3.0)
                    try:
                        obs_dict, _ = reset_marl_episode(env_ss, ep_seed)
                    except GamaCommandError:
                        print("  [GAMA] Chua ket noi lai — dung Ctrl+C, Play GAMA, chay lai lenh Python.")
                        raise
                    continue
                step += 1
                for agent_id, info in (infos or {}).items():
                    if info:
                        last_infos[agent_id] = dict(info)

                for agent_id in active_agents:
                    r = float(rewards.get(agent_id, 0.0))
                    ep_rewards[agent_id] = ep_rewards.get(agent_id, 0.0) + r
                    ep_lengths[agent_id] = ep_lengths.get(agent_id, 0) + 1

                    if terminations.get(agent_id, False) or truncations.get(agent_id, False):
                        if agent_id not in done_agents:
                            done_agents.add(agent_id)
                            final_info[agent_id] = dict(infos.get(agent_id) or {})

                obs_dict = next_obs
                if args.step_delay > 0:
                    await asyncio.sleep(args.step_delay)

            hit_step_limit = step >= args.max_episode_steps
            if "merging_0" not in final_info and last_infos.get("merging_0"):
                final_info["merging_0"] = last_infos["merging_0"]
            # Ghi metrics của merging_0
            m0_info = finalize_merging_info_on_step_limit(
                final_info.get("merging_0", {}),
                hit_step_limit=hit_step_limit,
            )
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
                mainline_mean_speed=float(m0_info.get("mainline_mean_speed", 0.0)),
            ))

            # Ghi thống kê highway agents theo từng agent.
            # Fallback last_infos (giống merging_0): episode thường kết thúc theo merging_0 khi
            # highway vẫn "running" → final_info[highway] rỗng → trước đây ghi 0.0/unknown.
            for agent_id in MARL_HIGHWAY_AGENTS:
                if agent_id not in final_info and last_infos.get(agent_id):
                    final_info[agent_id] = last_infos[agent_id]
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

    except KeyboardInterrupt:
        print(f"\n  [dừng] Đã chạy {episode} episode(s).")
    finally:
        # Gọi env_ss.close() thay vì parallel_env.close() để SuperSuit wrapper
        # được dọn dẹp đúng cách trước khi đóng kết nối GAMA socket.
        env_ss.close()

    if episode == 0:
        return

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
