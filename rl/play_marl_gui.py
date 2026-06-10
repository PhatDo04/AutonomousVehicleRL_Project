"""Demo MARL tren GAMA UI."""
from __future__ import annotations
import argparse, asyncio, os, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")
import numpy as np
from rl.config import MARL_AGENTS, MARL_OBS_DIM, MODEL_PATH
from rl.evaluate_marl import _predict_marl_action, load_marl_model
from rl.gama_compat import patch_gama_gymnasium
from rl.gama_episode_reset import reset_marl_episode, _exec_global
from rl.marl_env import AgentIndicatorParallelWrapper
patch_gama_gymnasium()
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv

async def run(args):
    # --scenario va nb_cars_max vao file GAML truoc khi load. GAMA reload doc lai file tu dia
    # moi episode -> mat do giu nguyen. Khoi phuc o finally.
    if args.scenario:
        from rl.scenario_utils import apply_scenario_to_gaml
        apply_scenario_to_gaml(args.scenario)
    try:
        model = load_marl_model(args.algo, args.model)
        pe = GamaParallelEnv(
            gaml_experiment_path=str(MODEL_PATH),
            gaml_experiment_name=args.experiment,
            gama_ip_address=args.host,
            gama_port=args.port,
        )
        print(f"[debug] post-load experiment_id={pe.experiment_id!r} type={type(pe.experiment_id).__name__}", flush=True)
        env_ss = AgentIndicatorParallelWrapper(pe, type_only=False)
        obs, _ = env_ss.reset(seed=args.seed)
        if args.eval_continue:
            _exec_global(env_ss, "pz_eval_continue <- 1.0;")
            _exec_global(env_ss, f"pz_postmerge_window <- {float(args.postmerge_window)};")
        if args.rl_postmerge:
            _exec_global(env_ss, "pz_rl_postmerge <- 1.0;")
        if args.no_shield:
            _exec_global(env_ss, "pz_no_shield <- 1.0;")
        print(f"[debug] post-reset experiment_id={pe.experiment_id!r} agents={pe.agents}", flush=True)
        infos = {}
        det = not args.stochastic
        try:
            for ep in range(1, args.episodes + 1):
                if ep > 1:
                    await asyncio.sleep(args.pause_between_episodes)
                    obs, _ = reset_marl_episode(env_ss, args.seed + ep, eval_continue=args.eval_continue, postmerge_window=args.postmerge_window, rl_postmerge=args.rl_postmerge, no_shield=args.no_shield)
                done, step, hist = set(), 0, Counter()
                while step < args.max_steps and "merging_0" not in done:
                    actions = {}
                    for aid in MARL_AGENTS:
                        if aid in done: continue
                        o = obs.get(aid)
                        if o is None: done.add(aid); continue
                        arr = np.asarray(o, dtype=np.float32)
                        if arr.shape != (MARL_OBS_DIM,):
                            raise RuntimeError(f"{aid}: obs {arr.shape}, can {MARL_OBS_DIM}")
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
    finally:
        if args.scenario:
            from rl.scenario_utils import restore_gaml_backup
            restore_gaml_backup()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo")
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1000)
    p.add_argument("--experiment", default="TrafficSimulation")
    p.add_argument("--scenario", choices=["low", "medium", "high"], default=None,
                   help="Va nb_cars_max vao GAML (low=24/medium=54/high=84) de khop mat do model da train. Khoi phuc sau khi chay.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--step-delay", type=float, default=0.35)
    p.add_argument("--pause-between-episodes", type=float, default=1.0)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-eval-mask", action="store_true")
    p.add_argument("--eval-continue", action="store_true",
                   help="Merger chay tiep SAU merge (khong end o merge) de xem di toi cuoi + song lui.")
    p.add_argument("--postmerge-window", type=float, default=200.0,
                   help="So tick chay tiep sau merge (mac dinh 200 ~ toi gan cuoi duong).")
    p.add_argument("--rl-postmerge", action="store_true",
                   help="RL dieu khien merger SAU merge (e2e) thay vi IDM. Dung voi model train --rl-postmerge (vd yt42).")
    p.add_argument("--no-shield", action="store_true",
                   help="ABLATION: bo khien RL (IDM-cap + gate ngang) — xem policy lai TRAN tren GUI.")
    asyncio.run(run(p.parse_args()))

if __name__ == "__main__":
    main()
