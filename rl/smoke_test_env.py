"""
Smoke test: GAMA headless (socket) + gama-pettingzoo — MARL (mặc định).

Cách chạy:
  1) gama-headless.bat -socket 1001
  2) python rl/smoke_test_env.py
     python rl/smoke_test_env.py --legacy-single   # archive TrafficSingleHeadless

Cần: pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import MARL_AGENTS, MARL_EXPERIMENT, MODEL_PATH, SINGLE_AGENT_EXPERIMENT, SINGLE_AGENT_GAML_PATH
from rl.gama_compat import patch_gama_gymnasium

patch_gama_gymnasium()

from gama_pettingzoo.gama_parallel_env import GamaParallelEnv  # noqa: E402

STEPS = 150
_REQUIRED_MERGING = ("outcome", "agent_role", "throughput", "shockwave_index", "merge_step")


def _print_sep(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


async def smoke_marl() -> bool:
    """Kiểm tra kết nối và obs/action cho 4 MARL agents."""
    _print_sep("SMOKE TEST — MARL (merging_0 + highway_0/1/2)")
    env = GamaParallelEnv(
        gaml_experiment_path=str(MODEL_PATH),
        gaml_experiment_name=MARL_EXPERIMENT,
        gama_ip_address="localhost",
        gama_port=1001,
    )
    try:
        obs, _ = env.reset(seed=0)
        present = list(obs.keys())
        print(f"  Agents available : {present}")
        missing = [a for a in MARL_AGENTS if a not in obs]
        if missing:
            print(f"  [FAIL] Thiếu agents: {missing}")
            return False

        for agent_id in MARL_AGENTS:
            o = obs[agent_id]
            assert getattr(o, "shape", None) == (15,), \
                f"{agent_id}: mong đợi obs 15D, nhận {getattr(o,'shape',None)}"

        merging_info_seen: dict | None = None
        for t in range(STEPS):
            actions = {a: int(env.action_space(a).sample()) for a in env.agents}
            obs, rewards, term, trunc, infos = env.step(actions)
            if t % 10 == 0:
                r_str = {a: f"{v:.2f}" for a, v in rewards.items()}
                print(f"  t={t:2d} | rewards={r_str}")
            if term.get("merging_0", False) or trunc.get("merging_0", False):
                merging_info_seen = infos.get("merging_0") if isinstance(infos.get("merging_0"), dict) else None
            if env.agents and all(term.get(a, False) or trunc.get(a, False) for a in env.agents):
                obs, _ = env.reset(seed=None)
    finally:
        env.close()

    if merging_info_seen is not None:
        missing_keys = [k for k in _REQUIRED_MERGING if k not in merging_info_seen]
        if missing_keys:
            print(f"  [WARN] merging_0 info thiếu: {missing_keys}")
        else:
            print(f"  info check merging_0: throughput={merging_info_seen.get('throughput', 0):.3f}  ✓")
    else:
        print(f"  [WARN] merging_0 chưa kết thúc episode trong {STEPS} bước.")

    print("  [PASS] MARL smoke test OK")
    return True


async def smoke_legacy_single() -> bool:
    """Kiểm tra archive single-agent (merging_0 only)."""
    _print_sep("SMOKE TEST — Legacy single (TrafficSingleHeadless)")
    env = GamaParallelEnv(
        gaml_experiment_path=str(SINGLE_AGENT_GAML_PATH),
        gaml_experiment_name=SINGLE_AGENT_EXPERIMENT,
        gama_ip_address="localhost",
        gama_port=1001,
    )
    try:
        obs, _ = env.reset(seed=0)
        print(f"  Agents available : {env.agents}")
        if "merging_0" not in obs:
            print(f"  [FAIL] merging_0 không có. Có: {list(obs.keys())}")
            return False
        if len(obs) != 1:
            print(f"  [WARN] Mong đợi 1 agent, nhận {len(obs)}: {list(obs.keys())}")
        assert getattr(obs["merging_0"], "shape", None) == (15,)
        for t in range(min(STEPS, 50)):
            actions = {a: int(env.action_space(a).sample()) for a in env.agents}
            obs, rewards, term, trunc, _ = env.step(actions)
            if env.agents and all(term.get(a, False) or trunc.get(a, False) for a in env.agents):
                obs, _ = env.reset(seed=None)
    finally:
        env.close()
    print("  [PASS] Legacy single-agent smoke OK")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--legacy-single",
        action="store_true",
        help="Thêm smoke archive GAML (models/single_agent/Main_Traffic_SingleAgent.gaml).",
    )
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    results = [await smoke_marl()]
    if args.legacy_single:
        results.append(await smoke_legacy_single())
    print(f"\n{'='*55}")
    print("  Tất cả smoke test PASSED" if all(results) else "  Có smoke test FAILED")
    print(f"{'='*55}\n")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
