"""Lõi điều khiển dự án AutonomousVehicleRL — KHÔNG phụ thuộc UI.

Gom mọi thao tác rời rạc (cấu hình path, bật/tắt GAMA headless, train mới/train-tiếp,
eval, chạy preset, soi log/NPE, tóm tắt kết quả) thành các hàm tái dùng. Dùng chung cho
console hoặc Streamlit (app_streamlit.py). Chỉ thư viện chuẩn — không thêm dependency.

Lưu ý kiến trúc đã rút ra:
  - GAMA headless biên dịch lại GAML mỗi lần client connect → đổi GAML rồi train là đủ
    (không cần restart), nhưng restart cho chắc khi đổi scenario/shield.
  - train_marl.py: --scenario chỉ là NHÃN (không vá density GAML). Muốn density thật =>
    patch_gaml() + restart GAMA, hoặc dùng run_experiments (tự patch+restore).
  - 1 GAMA port = 1 training tại một thời điểm.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "console_config.json"
LOGS_DIR = ROOT / "logs"
OUT_LOGS = ROOT / "outputs" / "logs"
OUT_MODELS = ROOT / "outputs" / "models"

GAMA_STDOUT = LOGS_DIR / "gama_stdout.log"
GAMA_STDERR = LOGS_DIR / "gama_stderr.log"

ALGOS = ("ppo", "a2c")
SCENARIOS = ("low", "medium", "high")
PRESETS = ("smoke", "fast", "short", "thesis")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _default_venv_python() -> str:
    p = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(p) if p.exists() else sys.executable


def default_config() -> dict:
    return {
        "gama_bat_path": "",
        "gama_headless_dir": "",
        "project_dir": str(ROOT),
        "venv_python": _default_venv_python(),
        "gama_port": 1001,
    }


def load_config() -> dict:
    cfg = default_config()
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    # auto-suy gama_headless_dir từ bat nếu trống
    if cfg.get("gama_bat_path") and not cfg.get("gama_headless_dir"):
        cfg["gama_headless_dir"] = str(Path(cfg["gama_bat_path"]).parent)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_config(cfg: dict) -> dict[str, bool]:
    """Trả về {key: ok} để UI hiện ✓/✗."""
    bat = cfg.get("gama_bat_path", "")
    venv = cfg.get("venv_python", "")
    proj = cfg.get("project_dir", "")
    return {
        "gama_bat_path": bool(bat) and Path(bat).is_file(),
        "gama_headless_dir": bool(cfg.get("gama_headless_dir")) and Path(cfg["gama_headless_dir"]).is_dir(),
        "venv_python": bool(venv) and Path(venv).is_file(),
        "project_dir": bool(proj) and Path(proj).is_dir(),
        "gama_port": isinstance(cfg.get("gama_port"), int) and cfg["gama_port"] > 0,
    }


def config_ok(cfg: dict) -> bool:
    return all(validate_config(cfg).values())


# ---------------------------------------------------------------------------
# GAMA headless
# ---------------------------------------------------------------------------

def port_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def gama_start(cfg: dict) -> dict:
    """Bật GAMA headless (detached), redirect stdout/stderr ra logs/. Idempotent: nếu đã
    listening thì bỏ qua."""
    port = int(cfg["gama_port"])
    if port_listening(port):
        return {"ok": True, "msg": f"GAMA đã chạy sẵn (port {port}).", "started": False}
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    bat = cfg["gama_bat_path"]
    wd = cfg.get("gama_headless_dir") or str(Path(bat).parent)
    out = open(GAMA_STDOUT, "w", encoding="utf-8", errors="replace")
    err = open(GAMA_STDERR, "w", encoding="utf-8", errors="replace")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        [bat, "-socket", str(port)],
        cwd=wd, stdout=out, stderr=err, creationflags=flags,
    )
    return {"ok": True, "msg": f"Đang bật GAMA (port {port})…", "started": True}


def gama_wait_listening(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_listening(port):
            return True
        time.sleep(2.0)
    return False


def gama_stop(port: int) -> dict:
    """Tắt process đang giữ port (Windows: netstat + taskkill)."""
    pid = _pid_on_port(port)
    if pid is None:
        return {"ok": True, "msg": "Không có process nào trên port.", "killed": False}
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        return {"ok": True, "msg": f"Đã tắt GAMA (PID {pid}).", "killed": True}
    except OSError as e:
        return {"ok": False, "msg": f"Lỗi taskkill: {e}", "killed": False}


def _pid_on_port(port: int) -> int | None:
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True
        ).stdout
    except OSError:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
            try:
                return int(parts[4])
            except ValueError:
                continue
    return None


def gama_status(cfg: dict) -> dict:
    port = int(cfg["gama_port"])
    listening = port_listening(port)
    return {"listening": listening, "pid": _pid_on_port(port) if listening else None, "port": port}


def gama_check_errors() -> dict:
    """Soi log GAMA: đếm NPE/Exception (bắt nhiễu). Phân loại race (getTempVar) vs
    teardown (GlobalVariableExpression)."""
    res = {"stdout_bytes": 0, "stderr_bytes": 0, "npe": 0, "exception": 0, "samples": []}
    for p in (GAMA_STDOUT, GAMA_STDERR):
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if p == GAMA_STDOUT:
            res["stdout_bytes"] = len(txt)
        else:
            res["stderr_bytes"] = len(txt)
        for line in txt.splitlines():
            if "NullPointerException" in line:
                res["npe"] += 1
                if len(res["samples"]) < 3:
                    kind = "race(getTempVar)" if "getTempVar" in txt else (
                        "teardown(GlobalVar)" if "GlobalVariableExpression" in txt else "other")
                    res["samples"].append(f"{kind}: {line.strip()[:120]}")
            elif "Exception" in line and "log.level=ERROR" not in line:
                res["exception"] += 1
    return res


# ---------------------------------------------------------------------------
# GAML patch (scenario density + shield) — uỷ thác scenario_utils có sẵn
# ---------------------------------------------------------------------------

def patch_gaml(scenario: str | None = None, shield: str | None = None) -> str:
    msgs = []
    if scenario:
        from rl.scenario_utils import apply_scenario_all_gaml
        apply_scenario_all_gaml(scenario)
        msgs.append(f"density={scenario}")
    if shield in ("on", "off"):
        from rl.scenario_utils import set_safety_shield
        set_safety_shield(shield == "on")
        msgs.append(f"shield={shield}")
    return "patch GAML: " + (", ".join(msgs) if msgs else "không đổi")


def restore_gaml() -> str:
    try:
        from rl.scenario_utils import restore_gaml_backup
        restore_gaml_backup()
        return "đã restore GAML."
    except Exception as e:  # noqa: BLE001
        return f"restore lỗi: {e}"


# ---------------------------------------------------------------------------
# Launch process (train / eval / experiment) — Popen nền, ghi log file
# ---------------------------------------------------------------------------

@dataclass
class Job:
    pid: int
    log_path: str
    cmd: str
    kind: str
    target_timesteps: int | None = None  # mốc timesteps/seed để Monitor vẽ progress đúng tỉ lệ


def _launch(cfg: dict, args: list[str], log_name: str, kind: str) -> Job:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / log_name
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    py = cfg["venv_python"]
    cmd = [py] + args
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd, cwd=cfg["project_dir"], stdout=logf, stderr=subprocess.STDOUT,
        env=env, creationflags=flags,
    )
    return Job(pid=proc.pid, log_path=str(log_path), cmd=" ".join(str(c) for c in cmd), kind=kind)


def launch_train(
    cfg: dict, *, algo: str, timesteps: int, seed: int, run_name: str,
    scenario: str = "medium", shield: str | None = None,
    patch_gaml_real: bool = False, checkpoint_interval: int = 0,
    resume_from: str | None = None, resume_mode: str = "continue",
    ent_anneal: bool = True,
) -> tuple[Job, str]:
    """Trả (Job, msg). Nếu patch_gaml_real=True: vá GAML (density theo scenario + shield)
    TRƯỚC khi train — GAMA biên dịch lại GAML khi train_marl connect (không cần restart).
    Dùng cho train-tiếp-zip để khớp đúng mật độ/shield mà zip đã train.
    LƯU Ý: không tự restore GAML sau train (train chạy nền) — dùng nút Restore GAML ở Monitor."""
    patch_msg = ""
    if patch_gaml_real:
        patch_msg = patch_gaml(scenario=scenario, shield=shield) + " (GAMA sẽ recompile khi train connect)"
    args = [
        "rl/train_marl.py", "--algo", algo, "--timesteps", str(timesteps),
        "--seed", str(seed), "--port", str(cfg["gama_port"]),
        "--run-name", run_name, "--scenario", scenario,
    ]
    if checkpoint_interval and checkpoint_interval > 0:
        args += ["--checkpoint-interval", str(int(checkpoint_interval))]
    if not ent_anneal:
        args += ["--no-ent-anneal"]
    if resume_from:
        args += ["--resume-from", resume_from, "--resume-mode", resume_mode]
    job = _launch(cfg, args, f"train_{run_name.replace('/', '_')}.log", "train")
    job.target_timesteps = int(timesteps)
    return job, patch_msg


def launch_train_multiseed(
    cfg: dict, *, algo: str, timesteps: int, seeds: list[int], run_name: str,
    scenario: str = "medium", shield: str | None = None, patch_gaml_real: bool = False,
    checkpoint_interval: int = 0, resume_from: str | None = None,
    resume_mode: str = "warmstart", ent_anneal: bool = True,
) -> tuple[Job, str]:
    """Train NHIỀU seed TUẦN TỰ trong 1 job nền (chain powershell). Mỗi seed → run-name/seedN.
    resume: nên dùng warm-start (mỗi seed init từ cùng 1 zip; continue không hợp đa-seed vì
    optimizer state dùng chung vô nghĩa). Patch GAML 1 lần trước cả chuỗi. Trả (Job, msg)."""
    patch_msg = ""
    if patch_gaml_real:
        patch_msg = patch_gaml(scenario=scenario, shield=shield)
    py = cfg["venv_python"]
    port = int(cfg["gama_port"])
    parts = []
    for s in seeds:
        a = (f'& "{py}" rl/train_marl.py --algo {algo} --timesteps {int(timesteps)} '
             f'--seed {int(s)} --port {port} --run-name "{run_name}/seed{int(s)}" --scenario {scenario}')
        if checkpoint_interval and checkpoint_interval > 0:
            a += f' --checkpoint-interval {int(checkpoint_interval)}'
        if not ent_anneal:
            a += ' --no-ent-anneal'
        if resume_from:
            a += f' --resume-from "{resume_from}" --resume-mode {resume_mode}'
        parts.append(a)
    ps_cmd = "; ".join(parts)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"train_multi_{run_name.replace('/', '_')}.log"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        cwd=cfg["project_dir"], stdout=logf, stderr=subprocess.STDOUT, env=env, creationflags=flags,
    )
    job = Job(pid=proc.pid, log_path=str(log_path), cmd=f"multiseed seeds={seeds} {algo}", kind="train",
              target_timesteps=int(timesteps))
    return job, patch_msg


def launch_eval(cfg: dict, *, algo: str, model_zip: str, episodes: int,
                stochastic: bool, log_actions: bool, out_csv: str | None = None) -> Job:
    args = [
        "rl/evaluate_marl.py", "--algo", algo, "--model", model_zip,
        "--episodes", str(episodes), "--port", str(cfg["gama_port"]),
    ]
    if stochastic:
        args.append("--stochastic")
    if log_actions:
        args.append("--log-actions")
    if out_csv:
        args += ["--out", out_csv]
    return _launch(cfg, args, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.log", "eval")


def launch_experiment(cfg: dict, *, preset: str, scenario: str, shield: str,
                      algos: list[str], skip_baselines: bool, run_tag: str | None = None,
                      eval_stochastic: bool = False, mode: str = "both",
                      dry_run: bool = False, ent_anneal: bool = True) -> Job:
    args = [
        "rl/run_experiments.py", "--preset", preset, "--scenario", scenario,
        "--shield", shield, "--port", str(cfg["gama_port"]), "--algos", *algos,
    ]
    if not ent_anneal:
        args.append("--no-ent-anneal")
    if skip_baselines:
        args.append("--skip-baselines")
    if run_tag:
        args += ["--run-tag", run_tag]
    if eval_stochastic:
        args.append("--eval-stochastic")
    if mode and mode != "both":
        args += ["--mode", mode]
    if dry_run:
        args.append("--dry-run")
    tag = run_tag or preset
    return _launch(cfg, args, f"experiment_{tag}_{shield}.log", "experiment")


def kill_job(job: "Job | None") -> dict:
    """Kill job đang chạy + TOÀN BỘ cây con (run_experiments spawn train_marl con → cần /T)."""
    if job is None:
        return {"ok": False, "msg": "Không có job nào để kill."}
    try:
        r = subprocess.run(["taskkill", "/F", "/T", "/PID", str(job.pid)],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        return {"ok": ok, "msg": (f"Đã kill {job.kind} (PID {job.pid} + cây con)."
                                  if ok else f"taskkill: {r.stderr.strip() or 'PID không còn'}")}
    except OSError as e:
        return {"ok": False, "msg": f"Lỗi: {e}"}


def launch_play(cfg: dict, *, algo: str, model_zip: str, episodes: int,
                experiment: str = "TrafficSimulation", port: int = 1000,
                step_delay: float = 0.35, stochastic: bool = False,
                scenario: str | None = None) -> tuple[Job, str]:
    """Chạy play_marl_gui.py: Python load zip → điều khiển GAMA Desktop (port serve có hiển thị)
    chạy experiment GUI để XEM model. Nếu scenario != None: PATCH density GAML trước — play gọi
    load_experiment(gaml) nên server Desktop nạp lại file đã patch → mật độ xe áp dụng.
    Trả (Job, msg)."""
    patch_msg = ""
    if scenario:
        patch_msg = patch_gaml(scenario=scenario) + " (Desktop nạp lại khi play load_experiment)"
    args = [
        "rl/play_marl_gui.py", "--algo", algo, "--model", model_zip,
        "--episodes", str(episodes), "--experiment", experiment,
        "--port", str(port), "--step-delay", str(step_delay),
    ]
    # Lưu ý: play_marl_gui không có --scenario; mật độ đặt qua patch_gaml ở trên (server nạp lại file).
    if stochastic:
        args.append("--stochastic")
    job = _launch(cfg, args, f"play_{time.strftime('%Y%m%d_%H%M%S')}.log", "play")
    return job, patch_msg


def proc_alive(pid: int) -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    except OSError:
        return False


def tail_log(log_path: str, n: int = 40) -> str:
    p = Path(log_path)
    if not p.exists():
        return "(chưa có log)"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(không đọc được log)"
    return "\n".join(lines[-n:])


def train_status(log_path: str) -> dict:
    """Parse log train/experiment thành trạng thái rõ ràng để Monitor hiện (pha + algo + seed +
    số lần train + timesteps seed hiện tại + episode gần nhất)."""
    p = Path(log_path)
    if not p.exists():
        return {"phase": "(chưa có log)", "trainings": 0}
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"phase": "(không đọc được log)", "trainings": 0}
    lines = txt.splitlines()
    trainings = re.findall(r"MARL Training: (MAPPO|MAA2C) \(CTDE\) \| seed=(\d+)", txt)
    cur_algo = cur_seed = None
    if trainings:
        a, s = trainings[-1]
        cur_algo = "PPO" if a == "MAPPO" else "A2C"
        cur_seed = int(s)
    ts = None
    for ln in lines:
        if "total_timesteps" in ln:
            digs = "".join(c for c in ln if c.isdigit())
            if digs:
                ts = int(digs)
    low = txt.lower()
    if "hoàn thành toàn" in low or "biểu đồ và bảng tổng" in low:
        phase = "✅ HOÀN THÀNH (plots + bảng)"
    elif "đã lưu model" in low and "marl training" in low and not trainings:
        phase = "✅ train xong"
    elif "marl evaluation" in low and trainings:
        phase = "đang train/eval"
    elif trainings:
        phase = "đang train"
    elif "baseline" in low:
        phase = "đang chạy baseline"
    else:
        phase = "khởi động…"
    last_ep = ""
    for ln in lines:
        if "outcome=" in ln and "episode=" in ln:
            last_ep = ln.strip()
    return {"phase": phase, "trainings": len(trainings), "algo": cur_algo,
            "seed": cur_seed, "timesteps": ts, "last_ep": last_ep[-90:] if last_ep else ""}


def latest_timestep(log_path: str) -> int | None:
    """Đọc total_timesteps gần nhất từ log SB3 (theo dõi tiến độ train)."""
    p = Path(log_path)
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "total_timesteps" in line:
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                last = int(digits)
    return last


# ---------------------------------------------------------------------------
# Liệt kê / tóm tắt kết quả
# ---------------------------------------------------------------------------

def list_model_zips() -> list[str]:
    if not OUT_MODELS.exists():
        return []
    return sorted(str(p) for p in OUT_MODELS.rglob("*.zip"))


def list_run_tags() -> list[str]:
    """Các run-tag đã train (thư mục con outputs/models/<tag>/ có chứa marl_*.zip) — để chọn khi
    Experiments mode=eval (re-eval cả batch không cần train lại)."""
    if not OUT_MODELS.exists():
        return []
    tags = []
    for d in OUT_MODELS.iterdir():
        if d.is_dir() and d.name != "checkpoints" and any(d.glob("marl_*.zip")):
            tags.append(d.name)
    return sorted(tags)


def list_run_dirs() -> list[str]:
    if not OUT_LOGS.exists():
        return []
    dirs = [p for p in OUT_LOGS.iterdir() if p.is_dir() and p.name != "tensorboard"]
    return [str(p) for p in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


def list_eval_csvs() -> list[str]:
    """Mọi CSV eval merging (bỏ highway) trên các run — để chọn vẽ biểu đồ/bảng."""
    out: list[str] = []
    for d in list_run_dirs():
        for pat in ("*_eval_stochastic.csv", "*_eval.csv", "*_episodes.csv"):
            out += [f for f in sorted(glob.glob(os.path.join(d, pat)))
                    if "_highway" not in os.path.basename(f) and "_steps_eval" not in os.path.basename(f)]
    # bỏ trùng, giữ thứ tự
    seen = set()
    uniq = []
    for f in out:
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def generate_report(cfg: dict, csv_files: list[str], out_dir: str) -> dict:
    """Chạy plots.py + analysis.py (đồng bộ) trên các CSV đã chọn → PNG + comparison_table + LaTeX.
    Đây CHÍNH là bước cuối mà run_experiments làm — mở ra dùng độc lập cho model lẻ/resume."""
    py = cfg["venv_python"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    msgs = []
    for script in ("rl/plots.py", "rl/analysis.py"):
        try:
            r = subprocess.run(
                [py, script, *csv_files, "--out-dir", out_dir],
                cwd=cfg["project_dir"], env=env, capture_output=True, text=True, timeout=300,
            )
            ok = r.returncode == 0
            msgs.append(f"{Path(script).stem}: {'OK' if ok else 'FAIL'}")
            if not ok:
                msgs.append((r.stderr or r.stdout).strip()[-300:])
        except (OSError, subprocess.TimeoutExpired) as e:
            msgs.append(f"{Path(script).stem}: lỗi {e}")
    pngs = sorted(str(p) for p in Path(out_dir).glob("*.png"))
    table = os.path.join(out_dir, "comparison_table.csv")
    return {"out_dir": out_dir, "msgs": msgs, "pngs": pngs,
            "table": table if os.path.exists(table) else None}


def summarize_run(run_dir: str) -> dict:
    """Đọc CSV eval merging_0 trong run_dir, trả mean±std success/collision theo seed."""
    fs = sorted(glob.glob(os.path.join(run_dir, "*_eval_stochastic.csv")))
    if not fs:
        fs = sorted(glob.glob(os.path.join(run_dir, "*_eval.csv")))
    # CHỈ giữ eval của model FINAL — loại eval checkpoint (..._<steps>_steps_eval...) vì checkpoint
    # 40k/80k chưa train xong sẽ kéo mean xuống + phồng variance (bug đếm "30 seed").
    fs = [f for f in fs if "_steps_eval" not in os.path.basename(f)]
    out = {"run": os.path.basename(run_dir), "per_algo": {}}
    by_algo: dict[str, list[tuple[float, float]]] = {}
    for f in fs:
        base = os.path.basename(f)
        algo = "ppo" if "_ppo_" in base else ("a2c" if "_a2c_" in base else "?")
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if not rows:
            continue
        n = len(rows)
        succ = 100 * sum(1 for r in rows if r.get("success") == "True") / n
        coll = 100 * sum(1 for r in rows if r.get("collision") == "True") / n
        by_algo.setdefault(algo, []).append((succ, coll))
    for algo, vals in by_algo.items():
        ss = [v[0] for v in vals]
        cs = [v[1] for v in vals]
        out["per_algo"][algo] = {
            "seeds": len(vals),
            "success_mean": round(statistics.mean(ss), 1),
            "success_std": round(statistics.pstdev(ss), 1) if len(ss) > 1 else 0.0,
            "collision_mean": round(statistics.mean(cs), 1),
            "collision_std": round(statistics.pstdev(cs), 1) if len(cs) > 1 else 0.0,
        }
    return out
