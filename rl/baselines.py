"""Evaluate non-learning baselines on the MARL GAMA socket (cùng Main_Traffic.gaml).

Chạy chính sách Greedy (KHÔNG học) trên đúng môi trường/khiên/seed như RL để làm
mốc so sánh "việc học có đáng giá không". Kết quả ghi ra CSV giống hệt eval của RL
nên có thể đặt cạnh nhau trong bảng so sánh.
"""

from __future__ import annotations

import argparse   # phân tích tham số dòng lệnh (--episodes, --seed...).
import asyncio     # chạy vòng lặp bất đồng bộ (socket GAMA dùng async).
import sys
from pathlib import Path
from typing import Any

import numpy as np  # xử lý vector quan sát.

# Bảo đảm import được package rl.* dù chạy script từ thư mục nào: thêm thư mục gốc vào sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import cấu hình + tiện ích dùng chung.
from rl.config import (
    LOG_DIR,
    MARL_AGENTS,
    MARL_EXPERIMENT,
    MODEL_PATH,
    SCENARIO_CLI_METADATA_ONLY_HINT,
    SCENARIO_PRESETS,
    GamaConnectionConfig,
)
from rl.gama_compat import patch_gama_gymnasium      # vá tương thích thư viện cầu nối.
from rl.marl_env import AgentIndicatorParallelWrapper  # nối one-hot ID vào quan sát.
from rl.gama_episode_reset import reset_marl_episode   # reset + gieo giao thông theo seed.
from rl.metrics import (                                # bộ chỉ số + ghi CSV.
    EpisodeMetric,
    build_episode_metric,
    classify_episode,
    finalize_merging_info_on_step_limit,
    write_episode_metrics,
)

patch_gama_gymnasium()  # PHẢI gọi trước khi import GamaParallelEnv (vá API gym/gymnasium).

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read baseline evaluation settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=["greedy"],   # hiện chỉ có greedy ('random' đã bỏ theo đề cương).
        default="greedy",
        help=(
            "Baseline (non-learning) cho bảng so sánh với PPO/A2C: "
            "'greedy' = THAM LAM — tăng tốc + merge sớm + vượt làn; né va chạm dọc nhờ khiên môi trường "
            "(như move-forward-greedy của NetLogo). "
            "Lưu ý: 'random' đã BỎ (đề cương không dùng). demo GUI chế độ Heuristic dùng luật GAML (get_heuristic_*), KHÔNG phải file này."
        ),
    )
    parser.add_argument("--episodes", type=int, default=20)      # số ván đánh giá.
    parser.add_argument("--seed", type=int, default=0)           # hạt giống (tái lập giao thông).
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1001)
    parser.add_argument("--max-episode-steps", type=int, default=300)  # độ dài tối đa 1 ván.
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_PRESETS.keys()),
        default="medium",
        help=(
            "Nhãn kịch bản trong CSV/output. Không vá GAML — mật độ thực là GAMA đang load (scenario_utils + restart)."
        ),
    )
    parser.add_argument("--out", default=None)  # đường dẫn CSV ra (mặc định tự đặt theo seed).
    # --- Các cờ cho chế độ end-to-end + ablation (giống RL để so sánh công bằng) ---
    parser.add_argument("--eval-continue", action="store_true",
                        help="Merger chạy tiếp sau merge để đo ĐUÔI SÓNG LÙI (đo goal-3 thật).")
    parser.add_argument("--postmerge-window", type=float, default=None,
                        help="(eval-continue) Số tick chạy tiếp sau merge. Mặc định 50 (GAML).")
    parser.add_argument("--rl-postmerge", action="store_true",
                        help="Policy (greedy) điều khiển merger SAU merge thay vì IDM — CÙNG cơ chế với RL "
                             "--rl-postmerge (cùng khiên IDM-cap) → so sánh e2e CÔNG BẰNG.")
    parser.add_argument("--no-shield", action="store_true",
                        help="ABLATION: bỏ khiên (IDM-cap + gate ngang) — greedy lái TRẦN, cùng điều kiện RL no-shield.")
    return parser.parse_args()


def greedy_merging_action(obs: np.ndarray) -> int:
    """greedy (merging): tham lam — tăng tốc + merge SỚM, không chờ/không phanh phòng bị.

    Baseline tham lam (non-learning) so với PPO/A2C. Merging obs index: [4]=in_accel_zone.
    - Trong accel zone → luôn thử merge (action 3) NGAY bước đầu tiên — không chờ khe/khớp tốc
      (gate is_merge_gap_safe đã TẮT trong Python mode cho mọi policy → merge thực thi bất kể gap).
    - Ngoài zone → luôn tăng tốc (action 2). Va chạm dọc được khiên môi trường (M3c) né phản ứng,
      giống move-forward-greedy của NetLogo (tham lam + né phản ứng).
    """
    in_accel_zone = float(obs[4]) >= 0.5  # chiều [4] là cờ đang ở vùng tăng tốc (0/1).
    if in_accel_zone:
        return 3   # trong vùng → thử nhập làn ngay (tham lam, không chờ khe).
    return 2       # ngoài vùng → tăng tốc tối đa.


def greedy_highway_action(obs: np.ndarray) -> int:
    """greedy THẬT (highway): phóng nhanh + vượt làn hung hãn, BỎ QUA gap phía sau.

    Highway obs (get_current_state): [1]=own lane, [2]=same-lane ahead dist,
    [4]=left-ahead dist, [8]=right-ahead dist (tất cả normalize /100m). Action highway:
    0=accel, 1=keep, 2=decel, 3=lane_left, 4=lane_right.

    Đổi làn đi qua gate ngang của env (chặn khi gap thiếu) như mọi policy lái slot RL — khác biệt
    của greedy nằm ở Ý ĐỊNH (luôn vượt/luôn max ga), không phải ở lưới an toàn.
    """
    own_lane = float(obs[1])      # làn hiện tại (đã chuẩn hóa).
    ahead = float(obs[2])         # khoảng cách xe trước cùng làn (/100m).
    left_ahead = float(obs[4])    # khoảng trống trước ở làn trái.
    right_ahead = float(obs[8])   # khoảng trống trước ở làn phải.
    # Bị chặn phía trước (< ~12m) → vượt sang làn THOÁNG hơn, bất chấp xe sau.
    if ahead < 0.12:
        if own_lane > 0.0 and left_ahead >= right_ahead:  # ưu tiên làn trái nếu thoáng hơn.
            return 3
        if own_lane < 1.0:                                 # còn không thì rẽ phải.
            return 4
    # Đường thoáng → tăng tốc tối đa (không bao giờ keep/brake).
    return 0


def _choose_merging_action(policy: str, obs: np.ndarray, rng: np.random.Generator) -> int:
    # Dispatch theo policy (giữ chỗ cho baseline tương lai, vd 'idm'); hiện chỉ greedy.
    return greedy_merging_action(obs)


def _choose_highway_action(policy: str, obs: np.ndarray, rng: np.random.Generator) -> int:
    return greedy_highway_action(obs)


async def async_main(args: argparse.Namespace) -> None:
    """Run baseline episodes on MARL socket and write metrics for merging_0."""
    # In nhãn kịch bản + cảnh báo "scenario chỉ là metadata".
    scenario = SCENARIO_PRESETS.get(args.scenario)
    if scenario:
        print(f"Scenario: {scenario.name} (nb_cars_max={scenario.nb_cars_max})")
        print(SCENARIO_CLI_METADATA_ONLY_HINT)

    # Cấu hình kết nối GAMA.
    config = GamaConnectionConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)  # RNG cho quyết định (greedy không dùng, để dành).
    # Bug #4 fix: sinh seed lon, xao tron tot cho moi episode (tranh `seed <- 1.0/2.0...` không xao tron GAMA RNG).
    seed_rng = np.random.default_rng(args.seed)  # RNG riêng để sinh seed-giao-thông mỗi ván.
    rows: list[EpisodeMetric] = []  # gom kết quả từng ván để ghi CSV cuối.
    parallel_env = None

    try:
        # Mở môi trường GAMA qua PettingZoo + nối one-hot ID vào quan sát.
        parallel_env = GamaParallelEnv(
            gaml_experiment_path=str(MODEL_PATH),
            gaml_experiment_name=MARL_EXPERIMENT,
            gama_ip_address=config.host,
            gama_port=config.port,
        )
        env_ss = AgentIndicatorParallelWrapper(parallel_env, type_only=False)
        # Reset khởi động (bootstrap) + kiểm tra đủ 4 tác tử.
        obs_dict, _ = reset_marl_episode(env_ss, args.seed, eval_continue=args.eval_continue, postmerge_window=args.postmerge_window, rl_postmerge=args.rl_postmerge, no_shield=args.no_shield)
        missing = [a for a in MARL_AGENTS if a not in obs_dict]
        if missing:
            raise RuntimeError(f"Thiếu agent MARL: {missing}. Kiểm tra GAML / GAMA headless.")

        # === Vòng lặp qua từng ván đánh giá ===
        for episode in range(1, args.episodes + 1):
            ep_seed = int(seed_rng.integers(1, 2**31 - 1))  # mỗi ván 1 seed giao thông khác (paired).
            obs_dict, _ = reset_marl_episode(env_ss, ep_seed, eval_continue=args.eval_continue, postmerge_window=args.postmerge_window, rl_postmerge=args.rl_postmerge, no_shield=args.no_shield)
            ep_reward = 0.0           # tổng reward của merging_0 trong ván.
            length = 0                # số bước merging_0 thực sự đi.
            done_agents: set[str] = set()           # tác tử đã kết thúc.
            final_info: dict[str, dict[str, Any]] = {}  # info lúc kết thúc của mỗi tác tử.
            last_infos: dict[str, dict[str, Any]] = {}  # info gần nhất (phòng khi thiếu final).
            step = 0
            hit_step_limit = False

            # === Vòng lặp từng tick trong ván ===
            while step < args.max_episode_steps:
                if "merging_0" in done_agents:
                    break  # merging_0 kết thúc → cả ván kết thúc.
                actions: dict[str, int] = {}
                # Chọn hành động greedy cho từng tác tử còn sống.
                for agent_id in MARL_AGENTS:
                    if agent_id in done_agents:
                        continue
                    obs = obs_dict.get(agent_id)
                    if obs is None:  # tác tử biến mất → đánh dấu kết thúc.
                        done_agents.add(agent_id)
                        final_info[agent_id] = {"outcome": "missing_agent"}
                        continue
                    arr = np.asarray(obs, dtype=np.float32)[:15]  # lấy 15 chiều cảm biến (bỏ one-hot).
                    if agent_id == "merging_0":
                        actions[agent_id] = _choose_merging_action(args.policy, arr, rng)
                    else:
                        actions[agent_id] = _choose_highway_action(args.policy, arr, rng)

                if not actions:
                    break

                # Gửi hành động xuống GAMA, nhận lại quan sát/phần thưởng/cờ kết thúc.
                next_obs, rewards, terminations, truncations, infos = env_ss.step(actions)
                step += 1
                # Lưu info mới nhất của mỗi tác tử.
                for agent_id, info in (infos or {}).items():
                    if info:
                        last_infos[agent_id] = dict(info)
                # Cộng dồn reward + đếm độ dài cho merging_0.
                if "merging_0" in actions:
                    ep_reward += float(rewards.get("merging_0", 0.0))
                    length += 1

                # Đánh dấu tác tử kết thúc (terminated hoặc truncated).
                for agent_id in list(actions.keys()):
                    if terminations.get(agent_id, False) or truncations.get(agent_id, False):
                        if agent_id not in done_agents:
                            done_agents.add(agent_id)
                            final_info[agent_id] = dict(infos.get(agent_id) or {})

                obs_dict = next_obs  # chuyển sang trạng thái kế tiếp.

            # === Chốt sổ ván ===
            hit_step_limit = step >= args.max_episode_steps  # có chạm trần bước không (→ timeout).
            if "merging_0" not in final_info and last_infos.get("merging_0"):
                final_info["merging_0"] = last_infos["merging_0"]  # fallback nếu thiếu info cuối.
            # Suy ra kết cục (success/collision/failed_merge/timeout) cho merging_0.
            m0_info = finalize_merging_info_on_step_limit(
                final_info.get("merging_0", {}),
                hit_step_limit=hit_step_limit,
            )
            outcome = str(m0_info.get("outcome", "timeout"))
            is_term, is_trunc = classify_episode(outcome)  # phân loại terminated vs truncated.
            # Gom thành 1 dòng metric chuẩn (giống hệt eval RL).
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
            # Metric "ỷ khiên" (so công bằng với RL): khiên can thiệp bao nhiêu lần/ván.
            sh_m = int(m0_info.get("shield_interventions", 0) or 0)  # khiên cho merger.
            sh_h = 0
            for aid, ai in {**last_infos, **final_info}.items():     # cộng khiên của 3 xe highway.
                if aid != "merging_0" and isinstance(ai, dict):
                    sh_h += int(ai.get("shield_interventions", 0) or 0)
            print(
                f"episode={episode} reward={ep_reward:.2f} length={length} "
                f"outcome={rows[-1].outcome} shield_mrg={sh_m} shield_hw={sh_h}"
            )
    finally:
        # Luôn đóng kết nối GAMA dù có lỗi (tránh treo socket).
        if parallel_env is not None:
            parallel_env.close()

    # Ghi toàn bộ kết quả ra CSV.
    output_path = Path(args.out) if args.out else LOG_DIR / f"{args.policy}_baseline_seed{args.seed}.csv"
    write_episode_metrics(output_path, rows)
    print(f"saved baseline metrics: {output_path}")


def main() -> None:
    # asyncio.run: chạy hàm async ở vòng lặp sự kiện mới rồi đóng.
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
