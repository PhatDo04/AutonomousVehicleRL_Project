"""Imitation / Behavior Cloning warmstart (#3) — mồi policy biết bấm action-3.

Vấn đề: RL không tự khám phá được "bấm action-3 tại gap an toàn" (sparse exploration).
Cách giải: dùng TEACHER kịch bản (đọc obs: in_accel_zone=obs[4], gap_safe=obs[9]) để sinh
demonstration → behavior-clone vào policy CTDE → policy khởi đầu đã biết bấm action-3 →
sau đó warmstart RL (train_marl --resume-from ... --resume-mode warmstart) để tinh chỉnh.

Teacher (theo VAI, nhận diện qua one-hot obs[15]):
  - merging_0 : in_zone & gap_safe -> 3 (merge); speed thấp -> 2 (accel); else -> 1 (keep).
  - highway   : 1 (keep). LƯU Ý highway đảo ngữ nghĩa: 0=accel, 2=decel — KHÔNG dùng 2.

Chạy: python rl/imitation_bc.py --port 1001 --demo-steps 6000 --bc-epochs 12 --out outputs/models/bc_seed.zip
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")

import nest_asyncio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.config import GamaConnectionConfig  # noqa: E402
from rl.marl_env import make_marl_vec_env  # noqa: E402
from rl.train_marl import build_marl_model  # noqa: E402

nest_asyncio.apply()


def scripted_action(o: np.ndarray) -> int:
    """Teacher thuần đọc obs 19D (one-hot ở [15:19]; merging_0 = bit 15)."""
    if o[15] > 0.5:  # merging_0
        in_zone = o[4] > 0.5
        if in_zone:
            return 3            # COMMIT-TO-MERGE: vào zone là cam kết action-3 (env thực thi khi gap an toàn)
        if o[0] < 0.6:
            return 2            # chưa vào zone + chậm → accel để tới zone đúng tốc (merging: 2=accel)
        return 1                # keep
    return 1                    # highway: keep (0=accel,2=decel — tránh 2)


def collect_demos(env, n_steps: int):
    """Chạy env bằng teacher kịch bản, thu (obs_79, action) cho cả 4 agent."""
    obs = env.reset()
    obs_buf, act_buf = [], []
    a3 = 0
    for _ in range(n_steps):
        acts = np.array([scripted_action(obs[i]) for i in range(obs.shape[0])], dtype=np.int64)
        obs_buf.append(obs.copy())
        act_buf.append(acts.copy())
        a3 += int((acts == 3).sum())
        obs, _, _, _ = env.step(acts)
    X = np.concatenate(obs_buf, axis=0).astype(np.float32)
    y = np.concatenate(act_buf, axis=0).astype(np.int64)
    return X, y, a3


def behavior_clone(model, X, y, epochs: int, batch: int = 256, lr: float = 1e-3):
    """Huấn luyện có giám sát actor của CTDE policy bắt chước teacher (CE = -log_prob)."""
    device = model.policy.device
    Xt = torch.as_tensor(X, device=device)   # obs demo → tensor.
    yt = torch.as_tensor(y, device=device)   # action teacher → tensor (nhãn cần bắt chước).
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)  # tối ưu Adam cho actor.
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)  # xáo trộn thứ tự mẫu mỗi epoch.
        tot, correct, nb = 0.0, 0, 0
        for i in range(0, n, batch):             # duyệt từng minibatch.
            idx = perm[i:i + batch]
            ob, ac = Xt[idx], yt[idx]
            # Cross-entropy = -log P(action_teacher) → ép policy gán xác suất cao cho hành động teacher.
            _, log_prob, _ = model.policy.evaluate_actions(ob, ac)
            loss = -log_prob.mean()
            opt.zero_grad(); loss.backward(); opt.step()  # lan truyền ngược + cập nhật trọng số.
            tot += float(loss) * len(idx); nb += len(idx)
            with torch.no_grad():
                # Đo độ khớp với teacher (argmax của policy == action teacher).
                pred = model.policy.get_distribution(ob).distribution.probs.argmax(-1)
                correct += int((pred == ac).sum())
        print(f"  BC epoch {ep+1}/{epochs}  loss={tot/nb:.3f}  teacher-acc={correct/nb:.1%}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1001)
    p.add_argument("--host", default="localhost")
    p.add_argument("--demo-steps", type=int, default=6000)
    p.add_argument("--bc-epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--out", default=str(ROOT / "outputs" / "models" / "bc_seed.zip"))
    args = p.parse_args()

    config = GamaConnectionConfig(host=args.host, port=args.port,
                                  max_episode_steps=args.max_episode_steps,
                                  simulation_seed=args.seed)
    env = make_marl_vec_env(config)
    try:
        print(f"[demo] thu {args.demo_steps} step bằng teacher kịch bản…")
        X, y, a3 = collect_demos(env, args.demo_steps)
        print(f"[demo] {len(y)} cặp (obs,action) | action-3 = {a3} ({a3/len(y):.1%}) | "
              f"phân bố action: {np.bincount(y, minlength=5).tolist()}")
        if a3 < 50:
            print("[WARN] quá ít demo action-3 — teacher hiếm khi gặp gap an toàn; cân nhắc nới gap/giảm mật độ.")
        model = build_marl_model("a2c", env, args.seed, None)
        print(f"[bc] behavior cloning {args.bc_epochs} epoch…")
        behavior_clone(model, X, y, args.bc_epochs)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
        print(f"[done] đã lưu BC seed: {args.out}")
    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())
