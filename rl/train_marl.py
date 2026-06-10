"""Huấn luyện MARL (Multi-Agent RL) cho bài toán nhập làn cao tốc — MAPPO / MAA2C với CTDE.

Kiến trúc CTDE (Centralized Training, Decentralized Execution):
  - 4 agent (merging_0, highway_0/1/2) cùng học 1 chính sách (parameter sharing).
  - Actor decentralized: chỉ đọc local 19D (15 sensor + 4 one-hot agent ID).
  - Critic centralized: thấy global state 60D (concat 15D base obs của cả 4 agents).
  - Obs đưa vào SB3 model = 19 + 60 = 79D (xem ``rl/centralized_policy.py``).

DQN bị loại vì off-policy replay buffer trộn transitions của nhiều agent → bất ổn trong
môi trường non-stationary multi-agent.

Tham khảo:
  - Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021).
  - Lowe et al. (2017) MADDPG; de Witt et al. (2020) MAPPO.

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
)
from rl.marl_env import make_marl_vec_env
from rl.centralized_policy import CentralizedCriticPolicy
from rl.metrics import classify_episode, write_episode_metrics, EpisodeMetric


# ---------------------------------------------------------------------------
# Callback: ghi CSV metrics cho merging_0 + highway agents sau mỗi episode
# ---------------------------------------------------------------------------

class MARLEpisodeCSVCallback(BaseCallback):
    """Ghi metrics episode cho tất cả MARL agents trong training.

    SuperSuit concat 4 agents thành 1 VecEnv với 4 slots cố định. Callback tách riêng:
      - CSV chính (output_path): metrics của merging_0 (agent nhập làn).
      - CSV phụ (highway_path):  metrics tổng hợp của highway_0/1/2 (theo dõi hành vi cooperative).
    """

    # SuperSuit concat agents theo thứ tự MARL_AGENTS → index 0 = merging_0, 1–3 = highway.
    # Dùng để suy ra role khi info không có ``agent_role`` (vd: GAMA trả về missing_agent).
    _AGENT_INDEX_TO_ROLE: dict[int, str] = {
        i: ("merging" if agent == "merging_0" else "highway")
        for i, agent in enumerate(MARL_AGENTS)
    }

    def __init__(self, algorithm: str, seed: int, output_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.algorithm = f"marl_{algorithm}"
        self.seed = seed
        self.output_path = output_path
        self.highway_path = output_path.parent / output_path.name.replace(
            "_episodes.csv", "_highway_episodes.csv"
        )
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
                # Fallback role khi info thiếu ``agent_role`` (dùng index slot, 0=merging, 1–3=highway).
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
                        mainline_mean_speed=float(info.get("mainline_mean_speed", 0.0)),
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
# Model builders
# ---------------------------------------------------------------------------

class EntCoefAnnealCallback(BaseCallback):
    """Giảm dần entropy coefficient theo tiến trình train (cách B — anneal trong 1 run từ đầu).

    ent_coef = start + (end - start) * (num_timesteps / total). Early entropy CAO (explore, học
    action-3); late entropy THẤP (nhọn → action-3 thành argmax → deterministic chạy sạch).
    Loss của A2C/PPO đọc self.ent_coef mỗi update nên cập nhật ở đây có hiệu lực ngay."""

    def __init__(self, ent_start: float, ent_end: float, total_timesteps: int, hold_frac: float = 0.0):
        super().__init__()
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.total = max(1, total_timesteps)
        self.hold_frac = max(0.0, min(0.99, hold_frac))  # giữ entropy cao tới mốc này rồi mới giảm

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total)
        if frac <= self.hold_frac:
            self.model.ent_coef = self.ent_start          # GIỮ cao (explore, học action-3)
        else:
            t = (frac - self.hold_frac) / (1.0 - self.hold_frac)  # giảm trong phần còn lại
            self.model.ent_coef = self.ent_start + (self.ent_end - self.ent_start) * t
        return True


def build_marl_model(algo: str, env, seed: int, tensorboard_log: str, ent_coef: float | None = None):
    """Tạo SB3 on-policy model MARL với centralized critic (MAPPO / MAA2C).

    ent_coef=None → dùng mặc định (PPO 0.01 / A2C 0.05). Truyền giá trị để override
    (vd fine-tune entropy thấp 0.003 → policy nhọn lại → deterministic/argmax chọn action-3)."""
    if algo not in MARL_ALGORITHMS:
        raise ValueError(
            f"Thuật toán '{algo}' không được hỗ trợ trong MARL. "
            f"Dùng: {MARL_ALGORITHMS}. DQN bị loại vì off-policy không phù hợp đa tác tử."
        )

    if algo == "ppo":
        return PPO(
            CentralizedCriticPolicy,
            env,
            learning_rate=1e-4,   # hạ 3e-4→1e-4: train DÀI ổn định hơn (giảm drift rời nhường ở giai đoạn exploit / catastrophic forgetting). Goal-2 cần train lâu mà không degrade.
            n_steps=256,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=(0.01 if ent_coef is None else ent_coef),
            vf_coef=0.5,
            tensorboard_log=tensorboard_log,
            seed=seed,
            verbose=1,
        )

    if algo == "a2c":
        return A2C(
            CentralizedCriticPolicy,
            env,
            learning_rate=3e-4,
            n_steps=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=(0.05 if ent_coef is None else ent_coef),
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
    parser.add_argument("--algo", choices=list(MARL_ALGORITHMS), default="ppo", help="Thuật toán RL (MARL: PPO/A2C).")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Tổng số training timesteps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1001)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--postmerge-window",
        type=float,
        default=None,
        help=(
            "THỬ NGHIỆM: train post-merge-bounded — sau merge chạy tiếp N tick (set pz_eval_continue=1) "
            "rồi end, để học lái post-merge. None = terminate-at-merge (mặc định, ổn định). Cảnh báo: "
            "post-merge-continue unbounded từng làm merger né-merge (yt22); window có-chặn là thử nghiệm mới."
        ),
    )
    parser.add_argument(
        "--rl-postmerge",
        action="store_true",
        help=(
            "RL điều khiển merger SAU merge (end-to-end, thesis): merger lái tiếp tới cuối đường bằng RL "
            "(không IDM) → đo trọn hành trình + merge-không-tai-nạn-sau. Dùng kèm --postmerge-window lớn "
            "(vd 99999 = tới cuối) + curriculum --resume-from (warmstart nền biết-merge) để chống collapse."
        ),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Đường dẫn .zip model để train tiếp. None = train mới từ đầu.",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("continue", "warmstart"),
        default="continue",
        help=(
            "continue = PPO/A2C.load (khôi phục cả optimizer + counter, train tiếp liền mạch). "
            "warmstart = chỉ nạp weights (set_parameters) lên model mới — dùng cho curriculum/đổi cấu hình."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Nếu >0, lưu checkpoint mỗi N timesteps để chọn best model sau train.",
    )
    parser.add_argument(
        "--norm-reward",
        action="store_true",
        help="Bọc VecNormalize(norm_reward=True) chống value-failure (ev≈0) ở e2e train do reward-scale "
             "mismatch (+200 vs ±1). Chỉ chuẩn reward (norm_obs=False) → eval/policy không đổi.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.05,
        help="Entropy coefficient ĐẦU (cao = explore). Mặc định 0.05 — đầu của lịch anneal (recipe a3_anneal2).",
    )
    parser.add_argument(
        "--ent-coef-end",
        type=float,
        default=0.002,
        help="Entropy coefficient CUỐI (thấp = policy nhọn → deterministic sạch). Mặc định 0.002. "
             "ANNEAL từ --ent-coef về giá trị này theo tiến trình train (1 run từ đầu, không cần BC).",
    )
    parser.add_argument(
        "--ent-anneal-hold",
        type=float,
        default=0.65,
        help="Phần train GIỮ entropy ở --ent-coef trước khi giảm (0-1). Mặc định 0.65 = giữ cao tới 65%% "
             "(action-3 học vững) rồi 35%% cuối mới giảm về --ent-coef-end (làm nhọn). Tránh sụp sớm.",
    )
    parser.add_argument(
        "--no-ent-anneal",
        action="store_true",
        help="TẮT entropy annealing (giữ --ent-coef cố định suốt train). Dùng khi muốn so sánh "
             "thuần không-anneal; mặc định anneal BẬT (recipe a3_anneal2 cho deterministic sạch).",
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

    # mkdir parent của từng output: hỗ trợ --run-name dạng "<tag>/<base>" (chia output theo lần chạy).
    # Khi run_name không có subfolder, parent = MODEL_DIR/LOG_DIR như cũ (backward compatible).
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_interval > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scenario = SCENARIO_PRESETS.get(args.scenario)
    if scenario:
        print(f"Scenario: {scenario.name} (nb_cars_max={scenario.nb_cars_max})")
        print(SCENARIO_CLI_METADATA_ONLY_HINT)

    ma_name = {"ppo": "MAPPO", "a2c": "MAA2C"}.get(args.algo, args.algo.upper())
    print(f"\n=== MARL Training: {ma_name} (CTDE) | seed={args.seed} | {args.timesteps:,} steps ===")
    print(f"  4 agents: merging_0 + highway_0/1/2")
    print(f"  Actor   : {MARL_OBS_DIM}D local obs (15 sensor + 4 one-hot ID) — decentralized")
    print(f"  Critic  : 60D global state (15D base × 4 agents) — centralized")
    print(f"  Policy  : Shared params + centralized critic (CTDE)\n")

    config = GamaConnectionConfig(
        host=args.host,
        port=args.port,
        max_episode_steps=args.max_episode_steps,
        simulation_seed=args.seed,
    )

    env = make_marl_vec_env(config, postmerge_window=args.postmerge_window, rl_postmerge=args.rl_postmerge)
    if args.norm_reward:
        # C-FIX value-failure: e2e train có explained_variance≈0 + value_loss≈1e-6 (critic KHÔNG học được
        # do reward-scale mismatch: merge +200 sparse vs per-tick ±1 trên horizon dài → returns dải động
        # quá rộng → critic không fit → advantage rác → collapse policy warmstart). VecNormalize(norm_reward)
        # chuẩn hoá return-scale → critic fit được → ev tăng → hết collapse. norm_obs=False (obs đã [0,1] sẵn
        # trong GAML → policy/actor thấy obs THÔ → eval (PPO.load + predict obs thô) KHÔNG cần load stats).
        from stable_baselines3.common.vec_env import VecNormalize
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=10.0)
        print(f"  [norm-reward] VecNormalize(norm_reward=True) — chuẩn hoá return-scale chống value-failure (ev≈0).")
    if args.rl_postmerge:
        print(f"  [RL-post-merge] merger lái tiếp SAU merge bằng RL (end-to-end). Cần curriculum warmstart chống collapse.")
    if args.postmerge_window is not None:
        print(f"  [post-merge-bounded] TRAIN chạy tiếp {args.postmerge_window:g} tick sau merge (pz_eval_continue=1). "
              f"THỬ NGHIỆM — khác yt22 unbounded; theo dõi merge success kẻo collapse.")

    try:
        reset_counter = True
        if args.resume_from:
            resume_path = Path(args.resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(f"--resume-from không tồn tại: {resume_path}")
            if args.resume_mode == "continue":
                algo_cls = PPO if args.algo == "ppo" else A2C
                model = algo_cls.load(str(resume_path), env=env, tensorboard_log=tensorboard_log)
                reset_counter = False  # train tiếp liền mạch (giữ step counter + optimizer)
                print(f"  [resume] CONTINUE từ {resume_path.name} — khôi phục optimizer + counter.")
            else:  # warmstart: model mới, CHỈ nạp POLICY weights (optimizer mới).
                # set_parameters() nạp CẢ optimizer state → crash khi cross-algo (PPO Adam → A2C RMSprop:
                # KeyError 'square_avg'). Warmstart đúng nghĩa = weights-only + optimizer mới. Load riêng
                # params["policy"] (network state_dict) → hỗ trợ PPO↔A2C, và đúng hơn cho cả PPO→PPO.
                model = build_marl_model(args.algo, env, args.seed, tensorboard_log, ent_coef=args.ent_coef)
                from stable_baselines3.common.save_util import load_from_zip_file
                _, _wparams, _ = load_from_zip_file(str(resume_path), device="cpu")
                model.policy.load_state_dict(_wparams["policy"])
                _opt = "RMSprop" if args.algo == "a2c" else "Adam"
                print(f"  [resume] WARM-START từ {resume_path.name} — CHỈ nạp policy weights (optimizer {_opt} mới). Hỗ trợ cross-algo PPO↔A2C.")
        else:
            model = build_marl_model(args.algo, env, args.seed, tensorboard_log, ent_coef=args.ent_coef)
        callbacks = [MARLEpisodeCSVCallback(args.algo, args.seed, metric_path)]
        if args.ent_coef_end is not None and not args.no_ent_anneal:
            ent_start = args.ent_coef if args.ent_coef is not None else (0.01 if args.algo == "ppo" else 0.05)
            callbacks.append(EntCoefAnnealCallback(ent_start, args.ent_coef_end, args.timesteps, args.ent_anneal_hold))
            print(f"  [ent-anneal] entropy {ent_start} → {args.ent_coef_end} (giữ cao tới {args.ent_anneal_hold:.0%}) qua {args.timesteps:,} steps")
        elif args.no_ent_anneal:
            print(f"  [ent-anneal] TẮT — entropy cố định {args.ent_coef} suốt train")
        if args.checkpoint_interval > 0:
            # SB3 callback ``n_calls`` tăng theo VecEnv step (=1 step VecEnv tương ứng
            # ``n_envs`` transitions). Chia interval cho num_envs để checkpoint đúng theo
            # tổng timestep yêu cầu.
            save_freq = max(1, args.checkpoint_interval // max(1, int(getattr(env, "num_envs", 1))))
            # name_prefix phải là BASENAME (không chứa "<tag>/"): nếu để cả run_name có dấu "/",
            # SB3 sinh file vào checkpoints/<tag>/<base>/<tag>/... (lồng kép) → select_best_checkpoint
            # glob ở checkpoints/<tag>/<base>/*.zip không thấy. save_path đã là checkpoint_dir (kèm tag).
            ckpt_prefix = Path(run_name).name
            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix=ckpt_prefix,
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )
        callback = CallbackList(callbacks)
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=run_name,
            reset_num_timesteps=reset_counter,
        )
        model.save(model_path)
        print(f"\nĐã lưu model      : {model_path}")
        print(f"Đã lưu metrics    : {metric_path}")
        print(
            f"Highway metrics   : "
            f"{metric_path.parent / metric_path.name.replace('_episodes.csv', '_highway_episodes.csv')}"
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
