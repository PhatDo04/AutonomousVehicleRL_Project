"""Demo MARL tren GAMA UI."""
from __future__ import annotations
import argparse, asyncio, os, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")
import numpy as np
from rl.config import MARL_AGENTS, MODEL_PATH
from rl.evaluate_marl import _predict_marl_action, load_marl_model
from rl.gama_compat import patch_gama_gymnasium
from rl.gama_episode_reset import reset_marl_episode
from rl.marl_env import AgentIndicatorParallelWrapper
patch_gama_gymnasium()
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

async def run(args):
    model = load_marl_model(args.algo, args.model)
    pe = GamaParallelEnv(str(MODEL_PATH), args.experiment, args.host, args.port)
    env_ss = AgentIndicatorParallelWrapper(pe, type_only=False)
    obs, _ = env_ss.reset(seed=args.seed)
    infos = {}
    det = not args.stochastic
    try:
        for ep in range(1, args.episodes + 1):
            if ep > 1:
                await asyncio.sleep(args.pause_between_episodes)
                obs, _ = reset_marl_episode(env_ss, args.seed + ep)
            done, step, hist = set(), 0, Counter()
            while step < args.max_steps and "merging_0" not in done:
                actions = {}
                for aid in MARL_AGENTS:
                    if aid in done: continue
                    o = obs.get(aid)
                    if o is None: done.add(aid); continue
                    arr = np.asarray(o, dtype=np.float32)
                    if arr.shape != (19,):
                        raise RuntimeError(f"{aid}: obs {arr.shape}, can 19")
                    actions[aid] = _predict_marl_action(model, arr, agent_id=aid, deterministic=det, eval_mask=not args.no_eval_mask)
                    if aid == "merging_0": hist[actions[aid]] += 1
                if not actions: break
                obs, _, term, trunc, infos = env_ss.step(actions)
                step += 1
                for aid in actions:
                    if term.get(aid) or trunc.get(aid): done.add(aid)
                if args.step_delay > 0: await asyncio.sleep(args.step_delay)
            print(f"ep{ep} steps={step} outcome={(infos or {}).get('merging_0', {}).get('outcome')} hist={dict(hist)}")
    finally:
        env_ss.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo")
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--port", type=int, default=1000)
    p.add_argument("--experiment", default="TrafficSimulation")
    p.add_argument("--step-delay", type=float, default=0.35)
    p.add_argument("--stochastic", action="store_true")
    asyncio.run(run(p.parse_args()))

if __name__ == "__main__":
    main()
