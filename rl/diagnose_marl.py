"""Chẩn đoán nhanh: va chạm sớm, baseline MARL socket, eval + action histogram.

Yêu cầu GAMA headless: gama-headless.bat -socket 1001

    python rl/diagnose_marl.py
    python rl/diagnose_marl.py --port 1001 --model outputs/models/marl_ppo_seed0_50k.zip
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.baselines import greedy_merging_action
from rl.config import LOG_DIR, MARL_AGENTS, MARL_EXPERIMENT, MODEL_PATH
from rl.gama_compat import patch_gama_gymnasium
from rl.marl_env import AgentIndicatorParallelWrapper

patch_gama_gymnasium()

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv  # noqa: E402

_MERGING_ACTIONS = {
    0: "brake",
    1: "keep",
    2: "accel",
    3: "merge",
    4: "wait",
}


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


def _fmt_obs(obs: np.ndarray) -> str:
    labels = [
        "speed", "progress", "dist_merge", "lateral", "in_accel",
        "gap_front", "spd_front", "gap_rear", "spd_rear", "gap_safe",
        "ramp_gap", "ramp_spd", "urgency", "last_act", "patience",
    ]
    # Sua bug #12: truoc day chi in 13 chieu, drop `last_act` + `patience` (2 chieu cuoi cua 15D obs).
    parts = [f"{labels[i]}={obs[i]:.3f}" for i in range(min(len(labels), len(obs)))]
    return " ".join(parts)


async def trace_episode(
    env_ss: AgentIndicatorParallelWrapper,
    parallel_env: GamaParallelEnv,
    *,
    label: str,
    seed: int,
    policy: str,
    model=None,
    max_steps: int = 20,
) -> dict:
    """Chạy 1 episode, in từng bước merging_0."""
    from rl.gama_episode_reset import reset_marl_episode

    obs_dict, _ = reset_marl_episode(env_ss, seed)
    ep_reward = 0.0
    actions: list[int] = []
    final_outcome = "unknown"

    print(f"\n--- Trace: {label} | seed={seed} | policy={policy} ---")

    for step in range(1, max_steps + 1):
        actions_map: dict[str, int] = {}
        for agent_id in MARL_AGENTS:
            obs = obs_dict.get(agent_id)
            if obs is None:
                continue
            arr = np.asarray(obs, dtype=np.float32)[:15]
            if agent_id == "merging_0":
                if policy == "greedy":
                    a = greedy_merging_action(arr)
                elif policy == "random":
                    a = int(np.random.randint(0, 5))
                elif policy == "model" and model is not None:
                    obs19 = np.asarray(obs, dtype=np.float32)
                    a = int(model.predict(obs19, deterministic=True)[0])
                else:
                    a = 1
                actions_map[agent_id] = a
                actions.append(a)
            else:
                if policy == "random":
                    actions_map[agent_id] = int(np.random.randint(0, 5))
                else:
                    actions_map[agent_id] = 1  # keep — highway không phải focus chẩn đoán

        if "merging_0" not in actions_map:
            print(f"  step {step}: merging_0 missing trong obs_dict keys={list(obs_dict.keys())}")
            break

        obs_dict, rewards, terms, truncs, infos = env_ss.step(actions_map)
        r0 = float(rewards.get("merging_0", 0.0))
        ep_reward += r0
        m0_info = dict(infos.get("merging_0") or {})
        done = bool(terms.get("merging_0")) or bool(truncs.get("merging_0"))
        a = actions_map["merging_0"]
        m0_obs = np.asarray(obs_dict.get("merging_0", np.zeros(15)), dtype=np.float32)[:15]

        print(
            f"  step {step:2d} | act={a}({_MERGING_ACTIONS.get(a, '?')}) "
            f"reward={r0:+.2f} cum={ep_reward:+.2f} | {_fmt_obs(m0_obs)}"
        )
        if done:
            final_outcome = str(m0_info.get("outcome", "unknown"))
            print(
                f"  >>> DONE outcome={final_outcome} "
                f"collision={m0_info.get('collision')} success={m0_info.get('success')}"
            )
            break
    else:
        final_outcome = "max_steps"

    hist = Counter(actions)
    hist_str = ", ".join(f"{_MERGING_ACTIONS.get(k, k)}:{v}" for k, v in sorted(hist.items()))
    return {
        "label": label,
        "policy": policy,
        "seed": seed,
        "steps": len(actions),
        "reward": ep_reward,
        "outcome": final_outcome,
        "actions": actions,
        "action_hist": hist_str,
    }


async def async_main(args: argparse.Namespace) -> int:
    host, port = args.host, args.port
    print(f"Kiểm tra socket {host}:{port} ...")
    if not _port_open(host, port):
        print(
            "  [LỖI] Không kết nối được GAMA. Mở terminal khác và chạy:\n"
            "        gama-headless.bat -socket 1001\n"
            "  (thư mục cài GAMA headless, thường AppData\\Local\\Programs\\Gama\\headless)"
        )
        return 1
    print("  [OK] Port mở — kết nối GAMA ...\n")

    parallel = GamaParallelEnv(
        gaml_experiment_path=str(MODEL_PATH),
        gaml_experiment_name=MARL_EXPERIMENT,
        gama_ip_address=host,
        gama_port=port,
    )
    env_ss = AgentIndicatorParallelWrapper(parallel, type_only=False)

    model = None
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        print(f"Đã load model: {args.model}")

    results: list[dict] = []
    try:
        for i, policy in enumerate(("greedy", "random", "model"), start=1):
            if policy == "model" and model is None:
                continue
            label = f"{policy}_ep{i}"
            seed = args.seed + i
            if policy == "random":
                np.random.seed(seed)
            results.append(
                await trace_episode(
                    env_ss,
                    parallel,
                    label=label,
                    seed=seed,
                    policy=policy,
                    model=model,
                    max_steps=args.max_steps,
                )
            )
    finally:
        parallel.close()

    print("\n" + "=" * 60)
    print("  TÓM TẮT CHẨN ĐOÁN")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['label']:12s} | {r['policy']:6s} | steps={r['steps']:3d} | "
            f"reward={r['reward']:+8.2f} | outcome={r['outcome']:<12s} | {r['action_hist']}"
        )

    out = LOG_DIR / "diagnose_summary.txt"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["label\tpolicy\tsteps\treward\toutcome\taction_hist"]
    for r in results:
        lines.append(
            f"{r['label']}\t{r['policy']}\t{r['steps']}\t{r['reward']:.4f}\t{r['outcome']}\t{r['action_hist']}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nĐã lưu: {out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=20, help="Số bước tối đa mỗi episode trace.")
    p.add_argument(
        "--model",
        default=str(ROOT / "outputs" / "models" / "marl_ppo_seed0_50k.zip"),
        help="Model PPO để trace (bỏ trống '' để chỉ baseline).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "":
        args.model = None
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
