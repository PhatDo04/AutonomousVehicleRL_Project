"""Wrapper reset MARL — reload GAMA + bootstrap đủ cycle trước khi train/eval.

Khi reset, GAMA cần vài chu kỳ "khởi động" để sinh đủ 4 tác tử. File này lo việc:
(1) reset, (2) đẩy seed giao thông xuống GAML (tái lập), (3) bật các cờ chế độ
(eval-continue / no-shield...), (4) chạy không (no-op) tới khi đủ 4 agent.
"""

from __future__ import annotations

from typing import Any

from rl.config import MARL_AGENTS

_BOOTSTRAP_STEPS = 40  # số bước "chạy không" tối đa để chờ GAMA sinh đủ 4 tác tử.


def _exec_global(env_ss: Any, expr: str) -> None:
    """Chạy 1 expression GAML ở sim-scope (set global) qua gama_client. Phải gọi NGAY SAU reset,
    TRƯỚC khi step sang cycle>0 (để bootstrap đọc giá trị mới)."""
    # env_ss bị bọc nhiều tầng wrapper → đi sâu (tối đa 8 tầng) tìm đối tượng có gama_client.
    node = env_ss
    for _ in range(8):
        gc = getattr(node, "gama_client", None)       # client socket tới GAMA.
        exp = getattr(node, "experiment_id", None)     # id experiment đang chạy.
        if gc is not None and exp is not None:
            try:
                gc._execute_expression(exp, expr)      # gửi câu lệnh GAML xuống chạy.
            except Exception:
                pass                                   # lỗi mạng nhất thời → bỏ qua (không làm sập).
            return
        node = getattr(node, "env", None)              # chưa thấy → xuống tầng wrapper trong.
        if node is None:
            return


def _set_sim_seed(env_ss: Any, seed: int) -> None:
    """Đẩy seed xuống GAML ``pz_sim_seed`` để traffic biến thiên theo seed (reproducible + paired)."""
    # Reflex sinh xe nền trong GAML đọc pz_sim_seed để gieo lại RNG → cùng seed = cùng tình huống.
    _exec_global(env_ss, f"pz_sim_seed <- {float(int(seed))};")


def reset_marl_episode(
    env_ss: Any,
    seed: int | None,
    eval_continue: bool = False,
    postmerge_window: float | None = None,
    rl_postmerge: bool = False,
    no_shield: bool = False,
) -> tuple[dict, dict]:
    """Reset và chờ đủ agent (bootstrap_initial_cars chạy sau cycle>0).

    ``eval_continue=True`` (EVAL): set ``pz_eval_continue=1`` → merger chạy tiếp sau merge thêm
    ``pz_postmerge_window`` tick rồi end → đo SÓNG LÙI sau merge (KHÔNG dùng khi train).
    ``postmerge_window`` (nếu set) ghi đè số tick chạy tiếp (rất lớn ≈ chạy tới cuối làn)."""
    obs, info = env_ss.reset(seed=seed)        # reset môi trường (GAMA reload mô phỏng).
    if seed is not None:
        _set_sim_seed(env_ss, seed)            # gieo lại giao thông theo seed.
    # --- Bật các cờ chế độ xuống GAML (reload đã đưa chúng về mặc định nên phải bơm lại) ---
    if eval_continue:
        _exec_global(env_ss, "pz_eval_continue <- 1.0;")   # cho merger chạy tiếp sau merge.
        if postmerge_window is not None:
            _exec_global(env_ss, f"pz_postmerge_window <- {float(postmerge_window)};")  # số tick chạy tiếp.
    if rl_postmerge:
        _exec_global(env_ss, "pz_rl_postmerge <- 1.0;")    # RL lái merger sau merge (e2e).
    if no_shield:
        # ABLATION: bỏ khiên RL (IDM-cap + gate ngang) — đo nội-tâm-hóa an toàn. EVAL-only.
        _exec_global(env_ss, "pz_no_shield <- 1.0;")
    # --- Bootstrap: chạy "giữ tốc" (action 1) tới khi GAMA sinh đủ 4 tác tử ---
    noop = {a: 1 for a in MARL_AGENTS}         # hành động "giữ tốc" cho mọi agent.
    for _ in range(_BOOTSTRAP_STEPS):
        missing = [a for a in MARL_AGENTS if a not in obs]
        if not missing:
            break                              # đã đủ 4 agent → xong.
        obs, _, terminations, _, info = env_ss.step(noop)
        if all(terminations.get(a, False) for a in MARL_AGENTS if a in obs):
            break                              # tất cả đã kết thúc (bất thường) → dừng.
    return obs, info
