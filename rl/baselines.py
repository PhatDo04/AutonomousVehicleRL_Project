"""Evaluate non-learning baselines for the highway merging task."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import LOG_DIR, SCENARIO_CLI_METADATA_ONLY_HINT, SCENARIO_PRESETS, SingleAgentGamaConfig
from rl.legacy.env_single import GamaMergingEnv
from rl.metrics import EpisodeMetric, build_episode_metric, write_episode_metrics


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
    """Rule-based Greedy policy dùng 15D merging observation từ Main_Traffic.gaml.

    Vector đủ 15 chiều — khớp README (bảng *merging_0*) và ``get_merging_state()`` trong GAML.
    Hàm này chỉ **đọc** các chiều sau (các chiều khác có thể dùng để mở rộng heuristic sau này):

    | Index | Trường GAML / README |
    |---|---|
    | 0 | norm_speed |
    | 1–3 | norm_progress, norm_dist_to_merge, norm_lateral_error — *chưa dùng trong rule hiện tại* |
    | 4 | in_accel_zone (binary) |
    | 5–6 | norm_gap_front, norm_speed_front — *chưa dùng* |
    | 7–8 | norm_gap_rear, norm_speed_rear |
    | 9 | norm_gap_safe |
    | 10–11 | norm_ramp_front_gap, norm_ramp_front_speed — chỉ dùng gap |
    | 12 | norm_urgency |
    | 13–14 | norm_last_action, norm_patience — *chưa dùng* |
    """
    speed          = float(obs[0])
    in_accel_zone  = float(obs[4]) >= 0.5
    gap_rear       = float(obs[7])
    speed_rear     = float(obs[8])
    gap_safe       = float(obs[9]) >= 0.5
    ramp_front_gap = float(obs[10])
    urgency        = float(obs[12])

    # Ngưỡng dưới đây là heuristic “mềm” trên [0,1]; có thể tinh chỉnh theo ablation (ghi trong luận văn).
    # Ưu tiên 1: merge ngay nếu điều kiện an toàn
    if in_accel_zone and gap_safe:
        return 3  # merge

    # Ưu tiên 2: xe phía sau cao tốc áp sát + đang chạy nhanh → giảm tốc nhường gap
    # (gap_rear nhỏ = nguy hiểm, speed_rear cao = xe sau đang lao tới)
    if gap_rear < 0.1 and speed_rear > 0.5:  # 0.1 ≈ khoảng cách chuẩn hóa “rất sát” trong [0,1]
        return 0  # decelerate — tránh bị đâm từ sau khi merge

    # Ưu tiên 3: có xe ramp phía trước đang chặn → giảm tốc
    if ramp_front_gap < 0.08:  # ngưỡng “quá gần” trên thang chuẩn hóa ramp_front_gap
        return 0  # decelerate behind a ramp vehicle

    # Ưu tiên 4: urgency cực cao (≈ cuối làn tăng tốc) → cố merge hoặc phanh gấp
    # Khi urgency = 1.0 mà tiếp tục wait sẽ chắc chắn failed_merge.
    # Greedy tối ưu: nếu còn thời gian (urgency 0.75–0.9) → chờ; nếu gần hết (>0.9) → cố merge.
    if urgency > 0.9 and not gap_safe:  # gần hết làn tăng tốc (chuẩn hóa urgency)
        return 3  # desperate merge attempt — đánh cược còn hơn chắc chắn failed_merge
    if urgency > 0.75 and not gap_safe:
        return 4  # wait — vẫn còn ít thời gian để chờ gap tốt hơn

    # Ưu tiên 5: tăng tốc để bắt kịp tốc độ dòng chính
    if speed < 0.7:  # dưới ~70% tốc độ chuẩn hóa
        return 2  # accelerate toward traffic flow speed

    return 1  # keep speed


def choose_action(policy: str, env: GamaMergingEnv, obs: np.ndarray) -> int:
    """Return one discrete action for the selected baseline policy.

    'greedy' là alias của 'heuristic' để khớp thuật ngữ đề cương
    ('chính sách dựa trên quy tắc tĩnh Greedy').
    """
    if policy == "random":
        return int(env.action_space.sample())
    if policy in ("heuristic", "greedy"):
        return heuristic_action(obs)
    raise ValueError(f"Unsupported baseline policy: {policy}")


async def async_main(args: argparse.Namespace) -> None:
    """Run baseline episodes and write one metrics CSV."""
    scenario = SCENARIO_PRESETS.get(args.scenario)
    if scenario:
        print(f"Scenario: {scenario.name} (nb_cars_max={scenario.nb_cars_max})")
        print(SCENARIO_CLI_METADATA_ONLY_HINT)
    config = SingleAgentGamaConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )
    env = GamaMergingEnv(config)
    rows: list[EpisodeMetric] = []

    try:
        for episode in range(1, args.episodes + 1):
            obs, _info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            length = 0
            terminated = False
            truncated = False
            info = {}

            while not (terminated or truncated):
                action = choose_action(args.policy, env, obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                length += 1

            rows.append(
                build_episode_metric(
                    algorithm=args.policy,
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
    finally:
        env.close()

    output_path = Path(args.out) if args.out else LOG_DIR / f"{args.policy}_baseline_seed{args.seed}.csv"
    write_episode_metrics(output_path, rows)
    print(f"saved baseline metrics: {output_path}")


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
