"""PROBE chẩn đoán value-failure: log dòng reward THÔ mà SB3 thấy từ env train.

Nghi vấn: value_loss≈1e-6 + ev≈0 ⇒ returns gần hằng số ⇒ spike terminal (+200/+50/-100/-50)
có thể KHÔNG đến được buffer SB3 (mất ở bridge GAML→Python). Probe build env y hệt train
(make_marl_vec_env + postmerge_window + rl_postmerge), chạy policy yt43, in:
  - mọi reward |r|>5 (spike) kèm step + slot
  - stats per-slot (min/max/mean/std)
  - đếm episode boundary (dones) + outcome
"""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")
import numpy as np

import asyncio

from rl.config import GamaConnectionConfig
from rl.marl_env import make_marl_vec_env
from stable_baselines3 import PPO

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 1001
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
MODEL = sys.argv[3] if len(sys.argv) > 3 else "outputs/models/checkpoints/yt43_hwshield/yt43_hwshield_250000_steps.zip"


async def amain() -> None:
    config = GamaConnectionConfig(port=PORT, max_episode_steps=600, simulation_seed=0)
    env = make_marl_vec_env(config, postmerge_window=400.0, rl_postmerge=True)
    model = PPO.load(MODEL)
    print(f"[probe] env num_envs={env.num_envs} | model loaded | steps={STEPS}", flush=True)

    obs = env.reset()
    slots = ["merging_0", "highway_0", "highway_1", "highway_2"]  # 4 slot (đúng thứ tự VecEnv).
    all_r = [[] for _ in range(env.num_envs)]  # gom reward từng slot để thống kê cuối.
    spikes = 0   # đếm số reward "đỉnh" (|r|>5 → terminal +50/-100...).
    ep = 0       # đếm số episode kết thúc.
    for t in range(STEPS):
        actions, _ = model.predict(obs, deterministic=True)  # policy chọn hành động (argmax).
        obs, rewards, dones, infos = env.step(actions)       # tiến 1 bước, lấy reward THÔ.
        for i in range(env.num_envs):
            r = float(rewards[i])
            all_r[i].append(r)
            if abs(r) > 5.0:   # reward đỉnh → in ra (kiểm spike terminal có tới SB3 không).
                spikes += 1
                print(f"[SPIKE] t={t} slot={slots[i]} r={r:+.1f} done={bool(dones[i])}", flush=True)
        if bool(np.asarray(dones).all()):  # cả 4 slot done → 1 episode kết thúc.
            ep += 1
            oc = infos[0].get("outcome") if infos and isinstance(infos[0], dict) else None
            print(f"[EP-END] t={t} ep={ep} outcome={oc} (dones all True)", flush=True)

    print("\n===== REWARD STREAM STATS (cái SB3 thật sự thấy) =====", flush=True)
    for i, name in enumerate(slots):
        a = np.asarray(all_r[i], dtype=np.float64)
        print(f"{name:10s} | n={len(a)} min={a.min():+.2f} max={a.max():+.2f} mean={a.mean():+.3f} std={a.std():.3f} | |r|>5: {(np.abs(a)>5).sum()}", flush=True)
    print(f"episodes={ep} spikes_total={spikes}", flush=True)
    env.close()


asyncio.run(amain())
