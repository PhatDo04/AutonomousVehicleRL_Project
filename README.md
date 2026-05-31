# Phát triển kịch bản nhập làn trên đường cao tốc và đánh giá các thuật toán học tăng cường đa tác tử cho điều khiển phương tiện tự hành

**Sinh viên**: Đỗ Tiến Phát — 2251172447 — ĐH Thuỷ Lợi, Khoa CNTT, Lớp 64KTPM1
**GVHD**: TS. Lê Nguyễn Tuấn Thành
**Tham chiếu**: Le Nguyen Tuan Thanh (2023). *Multi-agent reinforcement learning for traffic congestion on one-way multi-lane highways*. Journal of Information and Telecommunication, 7:3, 255–269.

---

## Tổng quan

Đồ án mô phỏng kịch bản nhập làn trên đường cao tốc bằng **GAMA Platform** và đánh giá hai thuật toán học tăng cường đa tác tử (**MAPPO** và **MAA2C**) cho bài toán điều khiển phương tiện tự hành. Kiến trúc dùng **CTDE** (Centralized Training, Decentralized Execution): mỗi agent có **actor** chỉ đọc local observation (decentralized) trong khi **critic** thấy trạng thái toàn cục của 4 agents (centralized) để khử non-stationarity do agent khác gây ra.

GAMA được chọn thay NetLogo do hỗ trợ MARL native qua PettingZoo bridge, không cần extension bên thứ 3, và headless mode ổn định cho training tự động.

---

## Kết quả chính

Thesis preset: 5 seeds × 200k timesteps × 50 episode eval stochastic, scenario `low` (mật độ ~20 xe).

| Algorithm | Success rate | CI95 | Per-seed |
|---|---|---|---|
| **MAA2C** (CTDE + reset fix) | **84.0%** ⭐ | ±4.6% | 80 / 82 / 84 / 86 / 88 |
| **MAPPO** (CTDE + reset fix) | 71.2% | ±5.7% | 58 / 66 / 74 / 76 / 82 |

**Phát hiện kỹ thuật then chốt** — *reset-on-merging-death wrapper* trong [`rl/marl_env.py:GamaMarkovSB3VecEnv.step_wait`](rl/marl_env.py): pipeline train ban đầu **không** reset env khi `merging_0` chết (do `MarkovVectorEnv(black_death=True)` che `done` signal), khiến rollout buffer tràn dead-agent transitions. Fix này khớp pipeline train với pipeline eval (đã có `reset_marl_episode` sau mỗi termination) và giúp cả MAPPO/MAA2C đạt kết quả cao + ổn định.

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│  GAMA Platform  —  models/Main_Traffic.gaml                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Môi trường vật lý                                  │    │
│  │  • 3 làn cao tốc (lane_width = 3.5m, 200m)          │    │
│  │  • Làn ramp (6 waypoints, 18m → 180m)               │    │
│  │  • Vùng tăng tốc accel_start=48m → accel_end=162m   │    │
│  │  • Điểm merge: x = 180m                             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌──────────────┐   ┌──────────────────────────────────┐    │
│  │ Xe nền (NPC) │   │ RL Agents (PettingZoo Parallel)  │    │
│  │ Rule-based   │   │  • merging_0  (magenta) — ramp   │    │
│  │ Greedy NPC   │   │  • highway_0  (cyan)   ┐         │    │
│  │              │   │  • highway_1  (cyan)   ├ MARL    │    │
│  │              │   │  • highway_2  (cyan)   ┘         │    │
│  └──────────────┘   └──────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │ gama-pettingzoo  (socket :1001)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Python RL Pipeline (CTDE: MAPPO + MAA2C)                    │
│                                                              │
│  Env wrappers (rl/marl_env.py)                               │
│    GamaParallelEnv                                           │
│      → PadPossibleAgentsParallelWrapper   (giữ 4 slots)      │
│      → AgentIndicatorParallelWrapper      (15D → 19D)        │
│      → GlobalStateParallelWrapper         (19D → 79D)        │
│      → MarkovVectorEnv(black_death=True)                     │
│      → GamaMarkovSB3VecEnv                                   │
│        • step_wait: reset-on-merging-death                   │
│                                                              │
│  Policy (rl/centralized_policy.py)                           │
│    CentralizedCriticPolicy                                   │
│      • Actor MLP: input 19D local obs → action logits        │
│      • Critic MLP: input 60D global state → value            │
│      • Orthogonal init (gain=√2), Tanh activation            │
│                                                              │
│  Training (rl/train_marl.py)                                 │
│    PPO + CentralizedCriticPolicy   → MAPPO                   │
│    A2C + CentralizedCriticPolicy   → MAA2C                   │
│                                                              │
│  Pipeline tools                                              │
│    run_experiments.py — orchestrator (train + eval + plots)  │
│    analysis.py, plots.py — bảng so sánh + 12 biểu đồ         │
│    baselines.py — greedy / random baseline (rule-based)      │
└──────────────────────────────────────────────────────────────┘
```

---

## Bài toán MARL

### Agents

| Agent | Vai trò | Màu | Mục tiêu |
|---|---|---|---|
| `merging_0` | Xe nhập làn (ramp) | Magenta | Nhập làn an toàn, không va chạm |
| `highway_0/1/2` | 3 xe cao tốc làn dưới | Cyan | Duy trì tốc độ + nhường gap cho xe ramp |

**Chiến lược học**: Shared policy + agent indicator (parameter sharing). Mỗi agent dùng cùng một mạng nhưng có one-hot ID trong obs để phân biệt vai trò.

### Không gian quan sát

- **Local actor obs**: 19D = 15D base sensor + 4D one-hot agent ID.
- **Global critic state**: 60D = concat 15D base obs của cả 4 agents (cố định thứ tự).
- **Obs đưa vào SB3 model**: 79D = `[local 19D | global 60D]`.

Phần local 19D cho `merging_0` và `highway_0/1/2` **cùng kích thước** nhưng **khác semantics**:

`merging_0`:
| # | Chiều | Ý nghĩa |
|---|---|---|
| 0 | norm_speed | Tốc độ xe nhập làn |
| 1 | norm_progress | Tiến độ trên ramp |
| 2 | norm_dist_to_merge | Khoảng cách đến điểm merge |
| 3 | norm_lateral_error | Lệch ngang so với làn mục tiêu |
| 4 | in_accel_zone | Có trong accel zone không (binary) |
| 5 | norm_gap_front | Gap phía trước trên làn mục tiêu |
| 6 | norm_speed_front | Tốc độ xe phía trước |
| 7 | norm_gap_rear | Gap phía sau |
| 8 | norm_speed_rear | Tốc độ xe phía sau |
| 9 | norm_gap_safe | Gap đủ an toàn để merge (binary) |
| 10 | norm_ramp_front_gap | Gap tới xe trước trên ramp |
| 11 | norm_ramp_front_speed | Tốc độ xe trước trên ramp |
| 12 | norm_urgency | Khẩn cấp (tăng về cuối accel lane) |
| 13 | norm_last_action | Action gần nhất |
| 14 | norm_patience | Chỉ số patience tích lũy |
| 15–18 | one-hot agent ID | Bit 1 ở vị trí tương ứng `merging_0` |

`highway_0/1/2` (cùng 19D nhưng 15D đầu khác semantics): các chiều mô tả xe quanh (front/left/right + tốc độ tương ứng) trên cao tốc — chi tiết trong `get_current_state()` của `Main_Traffic.gaml`.

### Không gian hành động (`Discrete(5)`, ngữ nghĩa khác nhau theo agent)

| Action | `merging_0` | `highway_0/1/2` |
|---|---|---|
| 0 | Giảm tốc | Tăng tốc |
| 1 | Giữ tốc | Giữ tốc |
| 2 | Tăng tốc | Giảm tốc |
| 3 | Nhập làn | Rẽ trái (lên làn trên) |
| 4 | Chờ | Rẽ phải (xuống làn dưới) |

Bảng chuẩn ngữ nghĩa: `ACTION_MEANINGS_MERGING` / `ACTION_MEANINGS_HIGHWAY` trong [`rl/config.py`](rl/config.py).

### Hàm phần thưởng (nguồn sự thật: `models/Main_Traffic.gaml`)

Toàn bộ reward theo bước và terminal được tính trong GAML (`calculate_merging_reward`, `calculate_reward`, `check_*_terminal_state`). Python chỉ nhận `reward_val` qua PettingZoo và ghi CSV — **không có** reward shaping riêng phía Python.

**`merging_0`** (tóm tắt — chi tiết trong `calculate_merging_reward`):

| Thành phần | Giá trị | Mục đích |
|---|---|---|
| Terminal merge success | +200 (khi đi gần hết mainline) | Signal mạnh dẫn dắt convergence |
| Terminal collision | -100 | Phạt va chạm |
| Terminal failed_merge | -50 | Phạt hết đường chưa merge |
| Base mỗi tick | -0.01 + speed × 0.10 | Time pressure + khuyến khích di chuyển |
| Vào accel zone (one-shot) | +15.0 | Dense shaping signal |
| Trong accel zone | +0.05/tick + thưởng gap safe (+0.05) | Incentive duy trì vị trí merge |
| Action 3 + zone + gap safe | +25 → +50 (scale theo urgency) | Sharp gradient hướng merge khi an toàn |
| Action 3 + zone + gap unsafe | -2.0 | Phạt nhẹ để học chọn đúng gap |
| Action 3 ngoài zone | -0.5 | "Merge chỉ trong zone" |
| Action 0/1/4 trong zone | -2.0 / -0.5 / -3.0 | Phạt non-progress (wait nặng nhất) |
| Reward cap | [-60, 60] | Ràng buộc magnitude cho value stability |

**`highway_0/1/2`** (`calculate_reward`):

| Thành phần | Giá trị | Mục đích |
|---|---|---|
| Terminal exit | +50 | Thưởng đi hết đường |
| Terminal collision | -100 | Phạt va chạm |
| Base mỗi tick | speed × 0.04 + action_penalty | Khuyến khích duy trì tốc độ |
| Low-speed penalty | -0.05 khi speed < 0.2 × speed_max | Tránh policy "đứng yên" |
| TTC penalty | -((20 - ahead_gap) / 20) × speed × 1.2 khi gap < 20m | Phanh khi gần xe trước |
| Action 0 (HW accel) + gap < 12m | -0.5 | Phạt trực tiếp accel ẩu |
| Cooperative bonus | +0.3 khi gần điểm merge thành công | Khuyến khích nhường gap |

---

## Thuật toán

Code dùng `CentralizedCriticPolicy` chung cho cả PPO và A2C, mỗi thuật toán dùng `n_steps` và `ent_coef` riêng phù hợp với cơ chế update của nó.

| Hyperparameter | MAPPO | MAA2C |
|---|---|---|
| Policy class | `CentralizedCriticPolicy` (CTDE) | `CentralizedCriticPolicy` (CTDE) |
| Learning rate | 3e-4 | 3e-4 |
| `n_steps` | 256 | 64 |
| `batch_size` | 128 | (full rollout) |
| `gamma` | 0.99 | 0.99 |
| `gae_lambda` | 0.95 | 0.95 |
| `clip_range` | 0.2 | n/a |
| `ent_coef` | 0.01 | 0.05 |
| `vf_coef` | 0.5 | (default) |
| Network width | 64 | 64 |
| Activation | Tanh | Tanh |
| Init | Orthogonal (gain=√2) | Orthogonal (gain=√2) |

DQN bị loại khỏi MARL vì replay buffer off-policy trộn transitions của 4 agents → Q-values không hội tụ do non-stationarity. PPO/A2C on-policy update ngay trên dữ liệu hiện tại, ổn định hơn.

---

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Yêu cầu:**
- Python 3.10+
- GAMA Platform 1.9.x+ (có `gama-headless`)
- RAM khuyến nghị ≥ 8GB

**Windows + tiếng Việt**: nếu gặp `UnicodeEncodeError` khi in log, set trước:
```powershell
$env:PYTHONIOENCODING='utf-8'
```

---

## Khởi động GAMA Headless

Trước khi chạy bất kỳ script Python nào:

```powershell
cd <ĐƯỜNG_DẪN>\gama-headless
.\gama-headless.bat -socket 1001
```

Để quan sát trực quan: mở `models/Main_Traffic.gaml` trong GAMA GUI → chạy experiment `TrafficSimulation`.

---

## Smoke test

Kiểm tra kết nối GAMA + env trước khi train dài:

```powershell
python rl/smoke_test_env.py
```

Output kỳ vọng: thông báo `[PASS] MARL smoke test OK — 4 agents hoạt động đúng`.

---

## Chạy thực nghiệm

### Tự động (khuyến nghị)

```powershell
# Thesis (chính) — 5 seeds × 200k × 50 ep eval — ~5h
python rl/run_experiments.py --preset thesis --port 1001

# Smoke test pipeline (nhanh, dry-run)
python rl/run_experiments.py --preset smoke --dry-run

# Short (1 seed × 200k × 20 ep) — kiểm tra nhanh sau khi đổi config
python rl/run_experiments.py --preset short --port 1001
```

Tùy chọn:
- `--skip-baselines`: bỏ Greedy/Random baseline.
- `--algos ppo` hoặc `--algos a2c`: chỉ chạy một thuật toán.
- `--eval-stochastic`: ép eval dùng sampling (mặc định bật cho `thesis`).
- `--run-tag <tên>`: đặt tên thư mục cho lần chạy (vd `baocao_v1`). Truyền `none` để ghi phẳng (không tách).

**Tách output theo từng lần chạy (mặc định):** mỗi lần chạy `run_experiments.py` tự sinh một thư mục riêng theo timestamp `<preset>_YYYYmmdd_HHMMSS`, áp cho **cả model, log và plots**:

```
outputs/models/thesis_20260530_143022/marl_ppo_seed0_200k_low.zip
outputs/logs/thesis_20260530_143022/...csv
outputs/plots/thesis/thesis_20260530_143022/...png
```

→ Mỗi lần chạy không bao giờ đè kết quả lần trước. Muốn ghi phẳng (như bộ thesis cũ ở `outputs/models/*.zip`): thêm `--run-tag none`.

Presets MARL (`rl/config.py`):

| Preset | Timesteps | Episodes eval | Seeds | Dùng cho |
|---|---|---|---|---|
| `smoke` | 400 | 2 | [0] | Pipeline sanity |
| `short` | 200,000 | 20 | [0] | Smoke 1 seed sau khi đổi config |
| `thesis` | 200,000 | 50 | [0, 1, 2, 3, 4] | Bộ số chính cho luận văn |

### Thủ công (debug từng bước)

```powershell
# Train một seed
python rl/train_marl.py --algo ppo --timesteps 200000 --seed 0 --host localhost --port 1001 --max-episode-steps 500

# Eval
python rl/evaluate_marl.py --algo ppo --model outputs/models/marl_ppo_seed0.zip --episodes 50 --seed 0 --stochastic --log-actions

# Baseline greedy
python rl/baselines.py --policy greedy --episodes 50 --seed 0 --host localhost --port 1001
```

---

## Phân tích kết quả

Sau khi pipeline xong, output nằm trong thư mục theo lần chạy: CSV ở `outputs/logs/<tag>/`, model ở `outputs/models/<tag>/`, plots + bảng ở `outputs/plots/<preset>/<tag>/` (với `<tag>` = `<preset>_<timestamp>` mặc định). Nếu chạy `--run-tag none` thì output ghi phẳng ở `outputs/logs/`, `outputs/models/`, `outputs/plots/<preset>/`.

```powershell
# Sinh 12 biểu đồ
python rl/plots.py outputs/logs/marl_*.csv --out-dir outputs/plots/thesis

# Bảng so sánh + LaTeX
python rl/analysis.py outputs/logs/marl_*.csv --out-dir outputs/plots/thesis
```

`run_experiments.py` tự động gọi cả `plots.py` và `analysis.py` cuối pipeline.

### Biểu đồ sinh ra (12 file)

| File | Nội dung |
|---|---|
| `reward_curve.png`, `smoothed_reward_curve.png` | Đường cong reward thô + rolling avg + CI |
| `boxplot_final_reward.png` | Box plot 50 ep cuối — độ ổn định policy |
| `radar_comparison.png` | Spider chart 5 chỉ số chính |
| `success_rate.png` | Tỷ lệ merge thành công (chỉ số chính) |
| `collision_rate.png`, `failed_merge_rate.png`, `timeout_rate.png` | Các loại failure |
| `merge_step.png` | Số bước đến merge thành công |
| `episode_length.png` | Độ dài episode trung bình |
| `throughput.png` | Thông lượng (merge success / total attempts) |
| `shockwave_index.png` | Chỉ số sóng lùi (CV tốc độ mainline) |

### File phân tích

| File | Nội dung |
|---|---|
| `comparison_table.csv` | Wide table: mean ± std cho mọi metric × thuật toán |
| `latex_table.tex` | LaTeX bảng copy-paste vào báo cáo |
| `learning_efficiency.csv` | Episode đầu tiên đạt success rate > threshold |
| `summary_metrics.csv` | Tóm tắt nhanh |

---

## Metrics báo cáo

| Metric | Ý nghĩa | Tốt hơn khi |
|---|---|---|
| `success_rate` | Tỷ lệ merge thành công | Cao |
| `collision_rate` | Tỷ lệ va chạm | Thấp |
| `failed_merge_rate` | Hết đường chưa merge | Thấp |
| `timeout_rate` | Vượt max_episode_steps | Thấp |
| `avg_reward` | Tổng reward trung bình mỗi episode | Cao |
| `merge_step` | Số bước đến merge thành công | Thấp |
| `mean_speed` | Tốc độ trung bình xe nhập làn | Cao |
| `throughput` | merge_success / total_ramp_attempts | Cao |
| `shockwave_index` | std/mean tốc độ mainline (CV) | Thấp |

**Throughput / shockwave** đo trên toàn bộ traffic ramp trong cửa sổ episode gắn với agent RL (kể cả NPC ramp) — semantics hợp lệ, không phải lỗi.

---

## Cấu trúc thư mục

```
.
├── models/
│   └── Main_Traffic.gaml              # MARL env (4 agents)
├── rl/
│   ├── centralized_policy.py          # CentralizedCriticPolicy (CTDE)
│   ├── marl_env.py                    # Env wrappers + reset-on-merging-death
│   ├── train_marl.py                  # MAPPO + MAA2C training
│   ├── evaluate_marl.py               # Eval + action histogram
│   ├── run_experiments.py             # Pipeline orchestrator
│   ├── baselines.py                   # Greedy / Random baseline (rule-based)
│   ├── analysis.py                    # Bảng so sánh + LaTeX
│   ├── plots.py                       # 12 biểu đồ
│   ├── metrics.py                     # EpisodeMetric + CSV writer
│   ├── config.py                      # Preset, paths, ACTION_MEANINGS
│   ├── scenario_utils.py              # Vá GAML theo scenario
│   ├── smoke_test_env.py              # Smoke test kết nối
│   ├── diagnose_marl.py               # Diagnostic tool
│   ├── play_marl_gui.py               # GUI demo
│   ├── model_registry.py              # Liệt kê model → JSON
│   ├── gama_compat.py                 # GAMA compatibility shim
│   └── gama_episode_reset.py          # Reset helper (eval)
├── outputs/                           # Sinh khi train/eval
│   ├── models/                        # *.zip + checkpoints
│   ├── logs/                          # episodes.csv + tensorboard/
│   └── plots/thesis/                  # 12 PNG + comparison_table.csv + latex_table.tex
├── tests/                             # pytest (chủ yếu offline)
└── requirements.txt
```

---

## Gỡ lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: supersuit` | `pip install -r requirements.txt` trong venv |
| `UnicodeEncodeError` (Windows) | `$env:PYTHONIOENCODING='utf-8'` hoặc `chcp 65001` |
| MARL: `KeyError` trên agent khi dùng `ss.agent_indicator_v0` trực tiếp | Repo đã có `AgentIndicatorParallelWrapper` (cùng one-hot, giữ Parallel API) |
| MARL: `TypeError: cannot pickle 'coroutine'` | Không dùng `concat_vec_envs_v1` khi `num_vec_envs=1` (repo đã handle) |
| MARL: SB3 báo env không phải `VecEnv` | Repo dùng `GamaMarkovSB3VecEnv` wrap |
| GAMA: `unable to find experiment/simulation` | Restart GAMA headless (`gama-headless.bat -socket 1001`) |

---

## Phạm vi và hạn chế (cho phần Discussion luận văn)

- **Mô phỏng giản lược**: action rời rạc + waypoint — không phải động học xe thật. Kết luận giới hạn trong phạm vi MDP đã định nghĩa.
- **Heterogeneous agents + shared policy**: `merging_0` và `highway_0/1/2` cùng kích thước obs nhưng khác semantics, cùng action space nhưng khác meaning. Shared policy + agent_indicator là lựa chọn thiết kế hợp lệ, ưu/nhược điểm cần nêu rõ.
- **Một instance GAMA**: `num_vec_envs=1` — vector hóa đến từ concat 4 agents trong cùng thế giới. Wall-clock chậm hơn pipeline pure CPU/GPU; bù lại đảm bảo dynamics nhất quán.
- **Scenario ↔ GAML**: `--scenario` chỉ là metadata; mật độ thật theo file GAMA đã load. Để đồng bộ: dùng `run_experiments.py` (tự `apply_scenario_to_gaml`) hoặc gọi `apply_scenario_to_gaml` + restart headless.
- **Reproducibility một phần**: `simulation_seed` cố định phía Python; RNG nội bộ GAMA có thể khác giữa máy.
- **Reset-on-merging-death**: thay đổi semantic của pipeline train so với mặc định SuperSuit `black_death=True` (không hide done). Cần nêu rõ trong báo cáo khi so sánh với MARL baselines khác.

---

## Ghi chú kỹ thuật

- **GAMA thay NetLogo**: GAMA hỗ trợ MARL native (PettingZoo bridge), headless ổn định hơn cho training tự động.
- **SB3 thay TF/Keras**: Stable-Baselines3 cung cấp PPO/A2C đã kiểm thử, tích hợp TensorBoard, hỗ trợ VecEnv.
- **DQN loại khỏi MARL**: off-policy replay buffer + non-stationarity → Q-values không hội tụ.
- **CTDE qua custom MlpExtractor**: `CentralizedCriticExtractor` tách obs 79D thành (local 19D, global 60D) và route đến `policy_net` / `value_net` riêng. Tham khảo: MAPPO (Yu et al. 2021).
- **Reset-on-merging-death**: phát hiện `merging_0` chết qua nhiều tín hiệu (terms array + `par_env.agents` + info outcome) vì `MarkovVectorEnv(black_death=True)` che `terms` ngay sau step chết đầu tiên.
- **pytest + GAMA**: tests chủ yếu offline (analysis, metrics, scenario_utils). Tests cần GAMA bật cờ `RUN_GAMA_INTEGRATION=1` hoặc dùng `smoke_test_env.py`.
- **Throughput / shockwave reset theo cửa sổ episode**: khi `merging_0` respawn, các bộ đếm reset — giá trị trong `info` phản ánh giai đoạn giữa 2 lần respawn.
