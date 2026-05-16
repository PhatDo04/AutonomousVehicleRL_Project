# Phát triển kịch bản nhập làn trên đường cao tốc và đánh giá các thuật toán học tăng cường cho bài toán điều khiển phương tiện tự hành

**Sinh viên**: Đỗ Tiến Phát — 2251172447 — ĐH Thuỷ Lợi, Khoa CNTT, Lớp 64KTPM1  
**GVHD**: TS. Lê Nguyễn Tuấn Thành  
**Tham chiếu**: Le Nguyen Tuan Thanh (2023). *Multi-agent reinforcement learning for traffic congestion on one-way multi-lane highways*. Journal of Information and Telecommunication, 7:3, 255–269.

---

## Tổng quan

Đồ án mô phỏng kịch bản nhập làn trên đường cao tốc bằng **GAMA Platform** (thay thế NetLogo so với đề cương — GAMA hỗ trợ MARL native, không cần extension bên thứ 3) và đánh giá hiệu năng của các thuật toán học tăng cường đa tác tử (MARL) trong việc điều khiển phương tiện tự hành.

### Thực nghiệm và thư mục `outputs/`

Toàn bộ train/eval/baseline **yêu cầu GAMA headless đang chạy** và socket đúng cổng. Trước khi chạy, `outputs/logs/`, `outputs/models/`, `outputs/plots/` có thể **chỉ chứa `.gitkeep`** — không có CSV/model/plot **không** có nghĩa mã Python sai; nghĩa là **chưa thu thập dữ liệu thực nghiệm**. Để bảo vệ chương *Thực nghiệm và đánh giá*, cần chạy xong pipeline (xem `outputs/README.md` và mục *Quy trình đầy đủ cho báo cáo* bên dưới).

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│  GAMA Platform  —  models/Main_Traffic.gaml                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Môi trường vật lý                                  │   │
│  │  • 3 làn cao tốc (lane_width = 3.5m, 200m)         │   │
│  │  • Làn ramp (6 waypoints, 18m → 180m)              │   │
│  │  • Vùng tăng tốc accel_start=48m → accel_end=162m  │   │
│  │  • Điểm merge: x = 180m                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐   ┌──────────────────────────────────┐   │
│  │ Xe nền (NPC) │   │ RL Agents (PettingZoo Parallel)  │   │
│  │ Rule-based   │   │  • merging_0  (magenta) — ramp   │   │
│  │ Greedy NPC,  │   │  • highway_0  (cyan)   ─┐        │   │
│  │ xe hỏng 0.001│   │  • highway_1  (cyan)    ├ MARL   │   │
│  │  probability │   │  • highway_2  (cyan)   ─┘        │   │
│  └──────────────┘   └──────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ gama-pettingzoo  (socket :1001)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Python RL Pipeline                                         │
│                                                             │
│  MARL (chính)                Archive single-agent          │
│  ├── marl_env.py             models/single_agent/          │
│  ├── train_marl.py           Main_Traffic_SingleAgent.gaml │
│  │   PPO / A2C (IPPO/IA2C)   rl/legacy/ (train_single…)   │
│  └── evaluate_marl.py                                       │
│                                                             │
│  BASELINE (archive GAML)   TOOLS                            │
│  baselines.py              ├── run_experiments.py (MARL)  │
│  random / greedy   ├── analysis.py         (LaTeX table)    │
│                    ├── plots.py            (12 biểu đồ)     │
│                    └── smoke_test_env.py   (kết nối test)   │
└─────────────────────────────────────────────────────────────┘
```

---

## Bài toán MARL: Thiết kế MDP

### Agents

| Agent | Vai trò | Màu | Mục tiêu |
|---|---|---|---|
| `merging_0` | Xe nhập làn (ramp) | Magenta | Nhập làn an toàn, không va chạm |
| `highway_0` | Xe cao tốc làn dưới | Cyan | Duy trì tốc độ cao, hỗ trợ nhường gap |
| `highway_1` | Xe cao tốc làn dưới | Cyan | Duy trì tốc độ cao, hỗ trợ nhường gap |
| `highway_2` | Xe cao tốc làn dưới | Cyan | Duy trì tốc độ cao, hỗ trợ nhường gap |

**Chiến lược học**: Shared Policy (IPPO/IA2C — parameter sharing). Vector quan sát 19D = 15D sensor + **one-hot agent ID (4 chiều)**; phần mở rộng dùng **cùng công thức** `supersuit.utils.agent_indicator` nhưng áp trực tiếp trên **Parallel API** qua `AgentIndicatorParallelWrapper` trong `rl/marl_env.py` (tránh lỗi PettingZoo AEC — xem *Ghi chú kỹ thuật → MARL*).

### Không gian quan sát — merging_0 (15D, Box [0, 1])

| # | Chiều | Ý nghĩa |
|---|---|---|
| 0 | norm_speed | Tốc độ xe nhập làn |
| 1 | norm_progress | Tiến độ trên ramp (0=đầu, 1=cuối) |
| 2 | norm_dist_to_merge | Khoảng cách còn lại đến điểm merge |
| 3 | norm_lateral_error | Lệch ngang so với làn mục tiêu |
| 4 | in_accel_zone | Có trong acceleration zone không (binary) |
| 5 | norm_gap_front | Gap phía trước trên làn mục tiêu |
| 6 | norm_speed_front | Tốc độ xe phía trước trên làn mục tiêu |
| 7 | norm_gap_rear | Gap phía sau trên làn mục tiêu |
| 8 | norm_speed_rear | Tốc độ xe phía sau trên làn mục tiêu |
| 9 | norm_gap_safe | Gap đủ an toàn để merge (binary) |
| 10 | norm_ramp_front_gap | Gap tới xe trước trên ramp |
| 11 | norm_ramp_front_speed | Tốc độ xe trước trên ramp |
| 12 | norm_urgency | Khẩn cấp (urgency tăng về cuối accel lane) |
| 13 | norm_last_action | Action gần nhất |
| 14 | norm_patience | Chỉ số patience tích lũy |

Thứ tự trên **khớp** `get_merging_state()` trong `models/Main_Traffic.gaml` và vector `obs` mà `rl/baselines.heuristic_action()` đọc (baseline chỉ dùng một tập con chỉ số; bảng đủ 15 chiều dùng cho tài liệu và kiểm tra đồng bộ).

### Không gian quan sát — highway_0/1/2 (15D, `get_current_state()` — làn cao tốc)

Cùng kích thước `Box(15,)` nhưng **khác semantics** với merging. Thứ tự vector (GAML `get_current_state` khi không phải `merging_0`):

| # | Chiều | Ý nghĩa |
|---|---|---|
| 0 | norm_speed | Tốc độ bản thân / `speed_max` |
| 1 | norm_current_lane | Làn hiện tại / `(n_lanes - 1)` |
| 2 | norm_dist_ahead | Khoảng cách xe trước cùng làn (chuẩn hóa) |
| 3 | norm_speed_ahead | Tốc độ xe trước cùng làn |
| 4 | norm_dist_ahead_left | Xe trước làn trái (0 nếu không có làn trái) |
| 5 | norm_speed_ahead_left | Tốc độ xe trước làn trái |
| 6 | norm_dist_behind_left | Xe sau làn trái |
| 7 | norm_speed_behind_left | Tốc độ xe sau làn trái |
| 8 | norm_dist_ahead_right | Xe trước làn phải |
| 9 | norm_speed_ahead_right | Tốc độ xe trước làn phải |
| 10 | norm_dist_behind_right | Xe sau làn phải |
| 11 | norm_speed_behind_right | Tốc độ xe sau làn phải |
| 12 | norm_dist_merger | Khoảng cách tới xe ramp/merge gần nhất trong tầm quan sát |
| 13 | norm_speed_merger | Tốc độ xe merger gần nhất |
| 14 | norm_patience | `patience / 100` |

### Không gian hành động (Discrete, 5 actions — dùng chung)

| Action | merging_0 | highway_0/1/2 |
|---|---|---|
| 0 | Giảm tốc | Tăng tốc |
| 1 | Giữ tốc | Giữ tốc |
| 2 | Tăng tốc | Giảm tốc |
| 3 | Nhập làn | Rẽ trái (lên làn trên) |
| 4 | Chờ / giảm nhẹ | Rẽ phải (xuống làn dưới) |

### Hàm phần thưởng

**Nguồn sự thật (SoT)**: toàn bộ reward theo bước và terminal được tính trong **`models/Main_Traffic.gaml`** (`calculate_merging_reward`, `calculate_reward`, `rl_merging_behavior`, `check_*_terminal_state`). Python chỉ nhận **một số thực** mỗi bước qua PettingZoo (`reward_val`) và ghi `cumulative_reward` / outcome trong CSV — **không** có lớp reward shaping riêng trong Python.

#### merging_0 — phân rã đa mục tiêu (tóm tắt từ GAML)

| Thành phần | Giá trị / quy tắc | Ghi chú |
|---|---|---|
| Terminal merge OK | +100 (đặt trong nhánh action 3) | Đồng thời `terminal_reason = success` |
| Terminal va chạm | −100 | `collision_distance` |
| Terminal hết đường chưa merge | −50 | `failed_merge` |
| Nền mỗi bước | −0.02 | `calculate_merging_reward` |
| `action_penalty` | cộng dồn (vd. merge sai gap −25, chờ −0.02) | Từ `rl_merging_behavior` |
| Theo tốc độ | +(speed/speed_max)×0.05 | Khuyến khích không đứng yên |
| Gap trước < `gap_front_min` | −(gap_front_min − gap_front)×0.5 | Phạt dense |
| Gap sau < `gap_rear_min` | −(gap_rear_min − gap_rear)×0.5 | Phạt dense |
| `is_merge_gap_safe()` | +0.05 | Thưởng dense |
| Trong accel zone | −urgency×0.05 | Urgency từ vị trí trên làn tăng tốc |
| Action 3 + in accel + gap safe | +1.0 | Tín hiệu “đúng lúc merge” |

**highway_0/1/2** (tóm tắt): +speed×0.1, phạt TTC khi bám đuôi, phạt chậm, phạt/thưởng chuyển làn (`action_penalty`), +0.3 cooperative khi merge gần, terminal va chạm −100, thoát đường +50 — chi tiết trong `calculate_reward` / `check_terminal_state` (GAML).

#### Bảng nhanh (tham chiếu slide / tóm tắt)

**merging_0:**

| Sự kiện | Reward |
|---|---|
| Merge thành công, không va chạm | +100 |
| Va chạm | −100 |
| Hết accel lane chưa merge | −50 |
| Mỗi timestep (phạt thời gian) | −0.02 |
| Gap an toàn (thưởng nhỏ) | +0.05 |
| Chọn action merge đúng lúc gap an toàn | +1.0 |
| Urgency penalty (gần cuối accel lane) | −urgency × 0.05 |
| Action merge khi gap không an toàn | −25 |
| Action chờ (action 4) | −0.02 |
| Duy trì tốc độ hợp lý | +speed × 0.05 |

**highway_0/1/2:**

| Sự kiện | Reward |
|---|---|
| Va chạm | −100 |
| Tốc độ cao (throughput) | +speed × 0.1 |
| Bám đuôi nguy hiểm (TTC penalty) | −TTC × 0.5 |
| Tốc độ rùa bò | −0.05 |
| Chuyển làn (action 3/4) | −0.05 |
| Cooperative: xe ramp merge gần mình | +0.3 |
| Về đích (exit road) | +50 |

### Điều kiện kết thúc episode

| Agent | Kết quả | Điều kiện |
|---|---|---|
| merging_0 | `success` | Merge thành công |
| merging_0 | `collision` | Va chạm |
| merging_0 | `failed_merge` | Hết đường chưa merge |
| merging_0 | `timeout` | Vượt max_episode_steps |
| highway_i | `exited` | Đi hết đường (x ≥ 195m) |
| highway_i | `collision` | Va chạm |

---

## Thuật toán

### Single-agent: DQN / PPO / A2C

| | DQN | PPO | A2C |
|---|---|---|---|
| Loại | Off-policy | On-policy | On-policy |
| Hyperparameter | lr=1e-4, buffer=50k, batch=64 | lr=3e-4, n_steps=512, clip=0.2 | lr=7e-4, n_steps=64 |
| Timesteps | 100k | 100k | 100k |

### MARL: PPO / A2C (DQN bị loại)

DQN không phù hợp MARL vì replay buffer off-policy lẫn lộn transitions của 4 agents → Q-values không hội tụ do non-stationarity. PPO/A2C on-policy cập nhật ngay trên dữ liệu hiện tại, ổn định hơn trong MARL.

| | MARL-PPO (IPPO) | MARL-A2C (IA2C) |
|---|---|---|
| Obs dim | 19D (15 + 4 one-hot) | 19D |
| Agents | 4 (shared weights) | 4 (shared weights) |
| Timesteps | 200k | 200k |

---

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows / `pip install -r`:** Trên một số máy, `pip` đọc `requirements.txt` theo mã trang hệ thống (vd. cp1252). File trong repo dùng **chỉ comment ASCII** để tránh `UnicodeDecodeError`. Nếu bạn tự sửa file và thêm tiếng Việt vào comment mà lỗi lại xuất hiện: hoặc đổi comment sang tiếng Anh, hoặc trước khi cài đặt chạy `$env:PYTHONUTF8='1'` (PowerShell) rồi `pip install -r requirements.txt`.

**Yêu cầu hệ thống:**
- Python 3.10+
- GAMA Platform 1.9.x trở lên (với gama-headless)
- RAM khuyến nghị: 8GB+

---

## Khởi động GAMA Headless

Trước khi chạy bất kỳ script Python nào:

```powershell
path C:\Users\...\AppData\Local\Programs\Gama\headless
# Windows
.\gama-headless.bat -socket 1001

# Linux/macOS
./gama-headless.sh -socket 1001
```

Để quan sát trực quan, mở `models/Main_Traffic.gaml` trong GAMA Platform GUI → chạy experiment `TrafficSimulation`.

---

## Gỡ lỗi — máy mới clone / Windows / MARL

Áp dụng khi **mới cài môi trường** hoặc **train MARL** (`python rl/train_marl.py`) để tránh lặp lại các lỗi đã xử lý trong codebase.

| Hiện tượng | Nguyên nhân gợi ý | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'supersuit'` | Chưa cài dependency MARL | Trong venv: `pip install -r requirements.txt` |
| `UnicodeEncodeError` / `charmap` khi in log tiếng Việt (Windows) | Console mặc định không UTF-8 | Trước khi chạy Python: `$env:PYTHONIOENCODING='utf-8'` (PowerShell) hoặc `chcp 65001`, hoặc dùng **Windows Terminal** với UTF-8 |
| MARL: `KeyError` trên agent (`highway_0`, …) khi dùng SuperSuit `agent_indicator_v0` trực tiếp | SuperSuit bọc ParallelEnv qua **Parallel→AEC→Parallel**; GAMA đổi tập agent giữa các bước → state PettingZoo lệch | **Không cần sửa tay**: repo đã dùng **`AgentIndicatorParallelWrapper`** trong `rl/marl_env.py` (cùng logic one-hot với SuperSuit nhưng giữ nguyên API Parallel). Chỉ lưu ý nếu bạn tự refactor và gọi lại `ss.agent_indicator_v0` |
| MARL: `TypeError: cannot pickle 'coroutine'` với `concat_vec_envs_v1` | Pickle env để nhân bản VecEnv; bridge GAMA không pickle được | Code đã **không** gọi `concat_vec_envs_v1` khi chỉ một instance GAMA (`num_vec_envs=1`) |
| MARL: `AssertionError` MarkovVectorEnv / agent biến động | Một số bước `agents ≠ possible_agents` | Code dùng **`MarkovVectorEnv(..., black_death=True)`** |
| MARL: SB3 báo env không phải Gymnasium `VecEnv` | `MarkovVectorEnv` là Gymnasium vector, không phải SB3 `VecEnv` | Code bọc bằng **`GamaMarkovSB3VecEnv`** (`rl/marl_env.py`) |

**Tóm tắt MARL (để tái lập đúng sau này):** huấn luyện MARL đi qua `make_marl_vec_env()` → `AgentIndicatorParallelWrapper` → `MarkovVectorEnv(..., black_death=True)` → `GamaMarkovSB3VecEnv`. Không phụ thuộc `supersuit.agent_indicator_v0` hay `concat_vec_envs_v1` như các ví dụ SuperSuit cổ điển.

---

## Bước 1 — Kiểm tra kết nối (Smoke test)

```powershell
# MARL 4 agents (mặc định)
python rl/smoke_test_env.py

# Thêm archive single-agent (TrafficSingleHeadless)
python rl/smoke_test_env.py --legacy-single
```

Output kỳ vọng:
```
  merging_0    | obs shape=(15,)  obs_space=Box(0.0, 1.0, (15,))
  highway_0    | obs shape=(15,)  ...
  [PASS] MARL smoke test OK — 4 agents hoạt động đúng
```

---

## Bước 2 — Chạy thực nghiệm

### Tự động (khuyến nghị)

```powershell
# Baseline Greedy/Random + MARL PPO/A2C (preset thesis: 50k steps, 3 seeds)
python rl/run_experiments.py --preset thesis --port 1001

# Kiểm thử nhanh không chạy thật
python rl/run_experiments.py --preset smoke --dry-run
```

Preset MARL (`MARL_EXPERIMENT_PRESETS`):

| Preset | Timesteps | Episodes eval | Seeds |
|---|---|---|---|
| `smoke` | 400 | 2 | [0] |
| `short` | 50,000 | 20 | [0] |
| `thesis` | 50,000 | 50 | [0, 1, 2] |

### Thủ công từng bước

**MARL — Huấn luyện:**
```powershell
python rl/train_marl.py --algo ppo --timesteps 200000 --seed 0
python rl/train_marl.py --algo a2c --timesteps 200000 --seed 0
```

**Baseline:**
```powershell
python rl/baselines.py --policy random  --episodes 50 --seed 0
python rl/baselines.py --policy greedy  --episodes 50 --seed 0
```

---

## Bước 3 — Đánh giá model

**Single-agent:**
```powershell
python rl/evaluate_marl.py --algo ppo --model outputs/models/marl_ppo_seed0_50k.zip --episodes 50
python rl/evaluate_marl.py --algo a2c --model outputs/models/marl_a2c_seed0_50k.zip --episodes 50
```

**MARL:**
```powershell
python rl/evaluate_marl.py --algo ppo --model outputs/models/marl_ppo_seed0_200k.zip --episodes 50
python rl/evaluate_marl.py --algo a2c --model outputs/models/marl_a2c_seed0_200k.zip --episodes 50
```

---

## Bước 4 — Sinh biểu đồ và bảng phân tích

```powershell
# Sinh tất cả biểu đồ
python rl/plots.py outputs/logs/*.csv --out-dir outputs/plots/thesis

# Sinh bảng so sánh + file LaTeX
python rl/analysis.py outputs/logs/*.csv --out-dir outputs/plots/thesis
```

### Biểu đồ được sinh ra (12 file)

| File | Nội dung | Dùng cho |
|---|---|---|
| `reward_curve.png` | Đường cong phần thưởng thô | Chương kết quả |
| `smoothed_reward_curve.png` | Đường cong học (rolling avg + CI) | So sánh tốc độ hội tụ |
| `boxplot_final_reward.png` | Box plot 50 episode cuối | Độ ổn định policy |
| `radar_comparison.png` | Spider chart 5 chỉ số | Tổng quan so sánh |
| `success_rate.png` | Tỷ lệ merge thành công | Chỉ số chính |
| `collision_rate.png` | Tỷ lệ va chạm | An toàn |
| `failed_merge_rate.png` | Tỷ lệ hết đường chưa merge | Hiệu quả |
| `timeout_rate.png` | Tỷ lệ hết giờ | Hiệu quả |
| `merge_step.png` | Số bước đến merge thành công | Tốc độ quyết định |
| `episode_length.png` | Độ dài episode trung bình | Hiệu quả |
| `throughput.png` | Thông lượng (merge success / total attempts) | Lưu lượng giao thông |
| `shockwave_index.png` | Chỉ số sóng lùi (std/mean tốc độ mainline) | Ùn tắc sóng lùi |

### File phân tích

| File | Nội dung |
|---|---|
| `comparison_table.csv` | Bảng wide: mean ± std tất cả 13 chỉ số × thuật toán |
| `latex_table.tex` | Bảng LaTeX copy-paste trực tiếp vào báo cáo |
| `learning_efficiency.csv` | Episode đầu tiên đạt success rate > 50% |
| `summary_metrics.csv` | Tóm tắt nhanh tất cả metric |

---

## Hướng dẫn đầy đủ: huấn luyện chi tiết → đánh giá

Phần *Bước 1–4* phía trên là các khối lệnh rời; mục này **nối thành một quy trình** và liệt kê **toàn bộ script có trong repo** để không bỏ sót bước sau khi clone máy mới.

### Checklist nhanh (máy mới hoặc làm lại từ đầu)

| Thứ tự | Việc cần làm |
|:---:|:---|
| 1 | **Cài đặt**: `python -m venv .venv` → kích hoạt venv → `pip install -r requirements.txt` |
| 2 | **GAMA headless** đúng cổng (vd. `-socket 1001`), trùng với `--port` Python |
| 3 | **Smoke test**: `python rl/smoke_test_env.py` |
| 4 | **Huấn luyện**: `run_experiments.py` (MARL) **hoặc** `train_marl.py` thủ công |
| 5 | **Đánh giá** `evaluate_marl.py` nếu train thủ công; `run_experiments --preset thesis` đã gọi eval kèm |
| 6 | **Biểu đồ + bảng**: `plots.py` và `analysis.py` trên các CSV trong `outputs/logs/` |
| 7 | *(Tùy chọn)* **Đăng ký model**: `model_registry.py`; **TensorBoard** xem học; **`pytest`** kiểm tra không GAMA |

### Chuỗi lệnh đầy đủ (copy-paste)

**Điều kiện**: hai cửa sổ terminal — một giữ **GAMA headless** mở suốt lúc train/eval/baseline; một chạy Python. Luôn **đứng tại thư mục gốc repo** (thư mục chứa `rl/` và `models/`). Trên Windows nên **kích hoạt venv** trước mọi lệnh `python`.

**Terminal A — GAMA (đường dẫn tùy máy bạn):**

```powershell
cd <ĐƯỜNG_DẪN_TỚI_THƯ_MỤC_CHỨA_gama-headless.bat>
.\gama-headless.bat -socket 1001
```

**Terminal B — Python (từ thư mục gốc dự án):**

```powershell
cd D:\path\to\AutonomousVehicleRL_Project   # chỉnh lại đường dẫn máy bạn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*(Tuỳ chọn — tránh lỗi in tiếng Việt trên console Windows)*

```powershell
$env:PYTHONIOENCODING='utf-8'
```

**Kiểm tra kết nối (bắt buộc trước khi train dài):**

```powershell
python rl/smoke_test_env.py
```

---

**Cách 1 — Một lệnh chạy gần hết pipeline** (`baseline` → train → eval → `plots` → `analysis`; GAMA vẫn phải bật):

```powershell
# Nhanh (preset ngắn)
python rl/run_experiments.py --preset short --port 1001

# Đầy đủ cho báo cáo (rất lâu)
python rl/run_experiments.py --preset thesis --port 1001
```

Bỏ baseline: `--skip-baselines`. Xem lệnh không chạy thật: `--dry-run`. Legacy single-agent: `rl/legacy/README.md`.

---

**Cách 2 — Làm tay từng bước** (sau khi smoke test OK):

```powershell
# --- Baseline (tùy chọn so sánh) ---
python rl/baselines.py --policy random  --episodes 50 --seed 0 --host localhost --port 1001
python rl/baselines.py --policy greedy  --episodes 50 --seed 0 --host localhost --port 1001

# --- MARL: ví dụ PPO 50k (thesis) ---
python rl/train_marl.py --algo ppo --timesteps 50000 --seed 0 --host localhost --port 1001 --max-episode-steps 300

python rl/evaluate_marl.py --algo ppo --model outputs/models/marl_ppo_seed0.zip --episodes 50 --seed 0 --host localhost --port 1001 --max-episode-steps 300
```

*(Đổi `--model` nếu bạn dùng `--run-name marl_ppo_seed0_200k` khi train → file `outputs/models/marl_ppo_seed0_200k.zip`.)*

---

**Sinh biểu đồ và bảng** (sau khi đã có CSV trong `outputs/logs/`):

```powershell
python rl/plots.py outputs/logs/*.csv --out-dir outputs/plots/my_run
python rl/analysis.py outputs/logs/*.csv --out-dir outputs/plots/my_run
```

---

**Theo dõi học trong lúc train:**

```powershell
tensorboard --logdir outputs/logs/tensorboard
```

**Đăng ký model (tuỳ chọn):**

```powershell
python rl/model_registry.py --model-dir outputs/models --out outputs/models/registry.json
```

**Kiểm thử code không cần GAMA:**

```powershell
python -m pytest tests/ -q
```

---

### Huấn luyện thủ công — tham số CLI (GAMA phải đang chạy)

**MARL** (`rl/train_marl.py`) — shared policy, 4 agents:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--algo` | `ppo` | Chỉ **`ppo`** \| **`a2c`** (DQN không dùng MARL) |
| `--timesteps` | `50000` (`TrainConfig`) | Preset thesis MARL: **50_000** |
| Các flag còn lại | *(như trên)* | `run-name` mặc định `marl_<algo>_seed<seed>` |

**Preset** (`MARL_EXPERIMENT_PRESETS` trong `rl/config.py`). Baseline Greedy dùng archive `models/single_agent/Main_Traffic_SingleAgent.gaml`. Legacy train DQN/PPO/A2C: `rl/legacy/`.

### Sau khi train xong — artifact và việc nên làm tiếp

| Xuất ra | Đường dẫn / ghi chú |
|---|---|
| **Model** | `outputs/models/<run-name>.zip` — dùng cho `evaluate*.py` và báo cáo |
| **CSV học (theo episode)** | `outputs/logs/<run-name>_episodes.csv` — có exploration; MARL thêm `*_highway_episodes.csv` |
| **TensorBoard** | `outputs/logs/tensorboard/` — mỗi run một thư mục con theo `tb_log_name` |

**Nếu train bằng `train_marl.py` (không qua `run_experiments`):**

1. Chạy **`evaluate_marl.py`** trỏ đúng `--model` và `--algo` để có CSV **đánh giá deterministic**.  
2. Thu thêm baseline (`baselines.py`) nếu cần đường cong so sánh Random/Greedy.  
3. Gom CSV vào một thư mục hoặc wildcard rồi chạy **`plots.py`** và **`analysis.py`**.

**Nếu dùng `run_experiments.py --preset thesis`:** pipeline đã gọi baseline → train → eval → plots → analysis; chỉ cần kiểm tra `outputs/plots/<preset>/`.

### Tự động so với thủ công (`run_experiments.py`)

| Khía cạnh | `run_experiments.py` | Thủ công từng lệnh |
|---|---|---|
| Baseline | Có (trừ khi `--skip-baselines`) | Chạy `baselines.py` hai lần (random + greedy) |
| Train + eval | Theo preset (nhiều algo × seed) | Lặp `train` → `evaluate` từng model |
| Scenario vá GAML | Có (`--scenario` ≠ medium) + khôi phục file sau khi xong | Phải tự `apply_scenario_to_gaml` + restart GAMA |
| Plots / Analysis | Gọi cuối cùng | Tự chạy `plots.py`, `analysis.py` |

### Theo dõi học (TensorBoard)

```powershell
tensorboard --logdir outputs/logs/tensorboard
```

Mở URL hiển thị trong terminal (thường `http://localhost:6006`). Hữu ích để xem reward/value loss trong lúc train.

### Bản đồ chức năng — script trong repo

| Script | Vai trò |
|---|---|
| `rl/train_marl.py` | Huấn luyện MARL PPO/A2C (shared policy) |
| `rl/evaluate_marl.py` | Đánh giá model MARL (merging + highway metrics) |
| `rl/baselines.py` | Random / greedy baseline (archive GAML single-agent) |
| `rl/legacy/train_single.py` | *(Archive)* Huấn luyện DQN/PPO/A2C 1 agent |
| `rl/run_experiments.py` | Tự động hóa toàn pipeline + plots + analysis |
| `rl/plots.py` | 12 biểu đồ từ CSV logs |
| `rl/analysis.py` | Bảng so sánh, LaTeX, learning efficiency |
| `rl/model_registry.py` | Liệt kê model trong `outputs/models/` → JSON |
| `rl/smoke_test_env.py` | Kiểm tra kết nối GAMA + env |
| `rl/scenario_utils.py` | Vá `Main_Traffic.gaml` theo scenario (dùng gián tiếp qua `run_experiments` hoặc gọi tay) |
| `tests/` | `pytest` — phần lớn không cần GAMA; tích hợp GAMA tùy chọn |

**Kiểm thử không cần simulator (sau khi cài dependency):**

```powershell
python -m pytest tests/ -q
```

---

## Metrics đo lường

### Chỉ số chính (cần báo cáo)

| Metric | Ý nghĩa | Tốt hơn khi |
|---|---|---|
| **success_rate** | Tỷ lệ merge thành công | Cao hơn |
| **collision_rate** | Tỷ lệ va chạm | Thấp hơn |
| **failed_merge_rate** | Tỷ lệ hết đường chưa merge | Thấp hơn |
| **avg_reward** | Phần thưởng trung bình | Cao hơn |
| **merge_step** | Số bước đến merge thành công | Thấp hơn |
| **throughput** | merge_success / total_ramp_attempts (xem mục dưới) | Cao hơn |
| **shockwave_index** | std/mean tốc độ mainline vùng merge (CV) | Thấp hơn |
| **mean_speed** | Tốc độ trung bình xe nhập làn | Cao hơn |
| **min_front_gap** | Khoảng cách tối thiểu phía trước | Cao hơn |

### Lưu ý semantics & thống kê (cho báo cáo)

- **Throughput**: Bộ đếm `total_ramp_attempts` / `total_merge_success` tăng cho **mọi** xe ramp (kể cả NPC) khi vào vùng tăng tốc / merge thành công; cửa sổ số liệu được reset khi `merging_0` respawn. Trong báo cáo nên diễn đạt rõ: *throughput đo trên toàn bộ traffic ramp trong cửa sổ episode gắn với tác tử RL*, không phải chỉ riêng số lần thử merge của `merging_0`. Nếu NPC ramp cũng vào accel zone trong cùng cửa sổ, mẫu số tăng → throughput có thể **thấp hơn nhẹ** so với trường hợp chỉ có RL; đó là đặc tả hợp lệ của metric, không phải lỗi.

- **Nhiễu môi trường**: `balance_traffic` (mặc định medium: mỗi 5 cycle) vẫn sinh xe NPC với nhánh `flip(0.05)` cho **obstacle**; kết hợp `prob_damaged_car = 0.001`/cycle, môi trường có **yếu tố ngẫu nhiên** bổ sung ngoài policy. Để kết luận thống kê đáng tin, nên giữ **preset thesis**: nhiều seed (ví dụ 3) và **đủ episode đánh giá** (ví dụ 50 episode eval mỗi run) như pipeline đã thiết kế.

- **Cân bằng reward (sparse / dense)**: Thang phần thưởng theo bước (∼0.01–0.1) và terminal (±50…100, merge sai −25, …) khác một bậc cỡ — codebase **không đổi** thiết kế đó; trong báo cáo thêm phần *Hyperparameter và sự cân bằng reward*: giải thích trade-off học terminal vs tinh chỉnh hành vi theo bước, và (nếu có) đề xuất curriculum / chỉnh scale sau thực nghiệm.

### Giải thích Shockwave Index

```
shockwave_index = std(tốc độ mainline) / mean(tốc độ mainline)
```

- Thu thập mỗi 5 cycle, vùng `accel_start − 20m` đến `merge_x + 30m`
- Dùng Welford's online algorithm (không cần lưu toàn bộ mẫu)
- **0.0 – 0.15**: dòng chảy êm, không sóng lùi
- **0.15 – 0.35**: dao động vừa phải
- **> 0.35**: sóng lùi đáng kể

Kỳ vọng: Greedy > Random > DQN > A2C > PPO > MARL-PPO (thấp nhất)

---

## Kịch bản mật độ giao thông

| Kịch bản | nb_cars_max | spawn_interval | ramp_prob | Mô tả |
|---|---|---|---|---|
| `low` | 20 | 10 cycles | 30% | Giao thông thưa |
| `medium` | 45 | 5 cycles | 50% | Giao thông trung bình (mặc định) |
| `high` | 70 | 3 cycles | 70% | Giao thông dày đặc |

```powershell
python rl/run_experiments.py --preset thesis --scenario high --port 1001
```

> **`--scenario` thật sự vá file GAML** (qua `rl/scenario_utils.py`), rồi `run_experiments` khôi phục `.gaml` khi xong. **Phải khởi động (hoặc restart) `gama-headless` sau khi file được vá và trước khi huấn luyện** — nếu GAMA đã mở từ trước, nó giữ giá trị cũ trong bộ nhớ cho đến khi reload model.

---

## Quản lý model

```powershell
python rl/model_registry.py --model-dir outputs/models --out outputs/models/registry.json
```

Tên file chuẩn: `{algo}_seed{N}_{timesteps}k.zip` hoặc `marl_{algo}_seed{N}_{timesteps}k.zip`

---

## Quy trình đầy đủ cho báo cáo

```
1. gama-headless.bat -socket 1001
2. python rl/smoke_test_env.py                              ← xác nhận kết nối
3. python rl/run_experiments.py --preset thesis ← 5–8 giờ (MARL)
4. python rl/plots.py outputs/logs/*.csv --out-dir outputs/plots/thesis
5. python rl/analysis.py outputs/logs/*.csv --out-dir outputs/plots/thesis
6. Copy outputs/plots/thesis/latex_table.tex vào báo cáo
7. Nhúng 12 file .png vào các chương kết quả
```

---

## Cấu trúc output

```
outputs/
├── models/
│   ├── dqn_seed0_100k.zip
│   ├── dqn_seed1_100k.zip
│   ├── dqn_seed2_100k.zip
│   ├── ppo_seed0_100k.zip  ...
│   ├── a2c_seed0_100k.zip  ...
│   ├── marl_ppo_seed0_200k.zip
│   ├── marl_ppo_seed1_200k.zip
│   ├── marl_a2c_seed0_200k.zip  ...
│   └── registry.json
├── logs/
│   ├── dqn_seed0_100k_episodes.csv
│   ├── ppo_seed0_100k_episodes.csv
│   ├── a2c_seed0_100k_episodes.csv
│   ├── marl_ppo_seed0_200k_episodes.csv
│   ├── marl_a2c_seed0_200k_episodes.csv
│   ├── random_baseline_seed0.csv
│   ├── greedy_baseline_seed0.csv
│   └── tensorboard/
└── plots/
    └── thesis/
        ├── reward_curve.png
        ├── smoothed_reward_curve.png
        ├── boxplot_final_reward.png
        ├── radar_comparison.png
        ├── success_rate.png
        ├── collision_rate.png
        ├── failed_merge_rate.png
        ├── timeout_rate.png
        ├── merge_step.png
        ├── episode_length.png
        ├── throughput.png
        ├── shockwave_index.png
        ├── summary_metrics.csv
        ├── comparison_table.csv
        ├── latex_table.tex
        └── learning_efficiency.csv
```

---

## Phạm vi kết luận và hạn chế (đề xuất đưa vào luận văn)

- **Độ giản lược môi trường**: Điều khiển qua waypoint, tốc độ và hành động **rời rạc**; **không** phải động học / cảm biến đầy đủ của xe thật. Kết luận nên **giới hạn** trong phạm vi: *so sánh thuật toán học tăng cường trên MDP định nghĩa trong mô phỏng GAMA* — **không** mở rộng thành khẳng định trực tiếp cho triển khai xe thực đường trường (có thể nêu hướng mở rộng tương lai như realism, transfer).

- **Single-agent vs MARL**: Single-agent chỉ bọc **`merging_0`** — RL cho **xe ramp**; xe cao tốc và phần lớn tương tác là **NPC/rule-based**. MARL học **đồng thời** ramp + các tác tử highway (nhường / đổi làn trong không gian hành động đã định nghĩa). So sánh hai nhánh là **không đối xứng về vai trò RL**; chỉ có thể so sánh công bằng nếu nêu rõ giả định và khác biệt thiết lập như một **hạn chế có chủ đích**.

- **Ngữ nghĩa chỉ số**: Trích đúng định nghĩa **throughput** và **shockwave** trong báo cáo (README: mục *Lưu ý semantics & thống kê*) — đặc biệt throughput **không** đồng nghĩa “thông lượng chỉ của agent RL”.

- **Độ lệch vai trò, cùng 15 chiều quan sát**: Merging và highway **cùng kích thước vector** nhưng **khác semantics** các chiều; shared policy + `agent_indicator` **không tương đương** hai kiến trúc policy chuyên biệt. Nên nêu thẳng trong luận văn (IPPO/IA2C + parameter sharing) — **ưu điểm tái lập và so sánh thuật toán**, **nhược điểm**: một policy phải học hai “chế độ” quan sát.

- **Cùng không gian hành động Discrete(5), khác ý nghĩa**: Dù chung **5 hành động rời rạc**, **nghĩa từng index** cho `merging_0` và cho highway **khác nhau** (ví dụ 0 = giảm tốc vs tăng tốc). Bảng chuẩn nằm trong `rl/config.py` — `ACTION_MEANINGS_MERGING` và `ACTION_MEANINGS_HIGHWAY`. Đây là điểm **mạnh về tinh giản** một policy dùng chung; khi **đối chiếu văn bản** các dạng MARL / multi-agent khác (OBS/ACT tách theo role, CTDE…), cần **giải thích** lựa chọn thiết kế và **hạn chế** khi tổng quát hóa kết luận.

- **Hiệu năng mẫu và thời gian chạy**: `make_marl_vec_env(..., num_vec_envs=1)` chỉ dùng **một instance GAMA** (một socket mô phỏng). Vector hóa đến từ việc **concat bốn agent** trong **cùng** thế giới (SuperSuit → SB3), **không** phải nhiều bản GAMA song song — số **environment step** theo đồng hồ wall-clock thường **chậm** hơn so với pipeline chỉ CPU/GPU không socket. Trong luận văn nên **làm rõ giới hạn tính toán** và **ý nghĩa thống kê** của bộ (timesteps, seeds, số episode eval) đã chọn: so sánh **thuật toán trong cùng budget** hợp lý hơn so với khẳng định “đã đủ lớn absolute” nếu không có thêm thực nghiệm scale-up.

- **Chuẩn “scenario ↔ GAML” và `--scenario` chỉ là nhãn / metadata**: `ScenarioConfig` + CLI (`train.py` / `train_marl.py` / `baselines.py`) đã cảnh báo: **đổi `--scenario` không tự vá** `Main_Traffic.gaml`; mật độ thật = file GAMA simulator đang load, trừ khi dùng `run_experiments.py` hoặc `apply_scenario_to_gaml` + **restart** headless. **Rủi ro cho báo cáo**: CSV/run name ghi `high` trong khi mô phỏng vẫn **medium** (hoặc ngược lại) nếu tác giả không đối chiếu nhãn với bước vá + phiên bản GAML đã load. Khi tranh luận khoa học, nên cố định **một** luồng (vd. toàn bộ đường cong chạy theo `run_experiments` + preset + commit hash GAML).

- **`GamaMergingEnv.__init__` và phụ thuộc GAMA lúc khởi tạo**: Gọi `parallel_env.reset()` **ngay** để lấy `observation_space` / `action_space` — **đúng** cho SB3 nhưng **bắt buộc** socket GAMA sống; phần lớn luồng “nóng” **không** nằm trong pytest mặc định (offline/CI không socket). Docstring lớp đã mô tả — nên **ghi trong luận văn** (limitations / kiến trúc), không quy là lỗi.

- **Regex (`scenario_utils`)**: Áp patch theo đúng reflex/biểu thức hiện tại; refactor GAML mà không chạy `tests/test_scenario_utils.py` có thể dẫn tới không vá ([WARN]); cần **kỷ luật** khi chỉnh tên reflex hoặc format.

- **`classify_episode` (“outcome unknown”)**: Fallback truncated + **cảnh báo có kiểm soát**: mỗi chuỗi outcome lạ chỉ đưa một `warnings.warn` trong process để log không tràn nhưng vẫn phát hiện khi GAMA đổi protocol.

- **`EpisodeCSVCallback`**: Giả định ``rewards[0]`` / ``dones[0]`` — đúng với một `Monitor(GamaMergingEnv)`; bọc VecEnv replica sau này phải sửa (đã docstring trong `metrics.py`).

- **Tái lập thực nghiệm**: Phụ thuộc **phiên bản GAMA headless**, `gama-pettingzoo`, Python/SB3, và **socket** mô phỏng — nên ghi **phiên bản phần mềm** (environment) và quy trình (ví dụ **restart GAMA sau khi vá scenario**).

- **Độ phủ kiểm thử tích hợp thấp**: `tests/` tập trung **analysis**, **metrics**, **scenario_utils**, **baselines** (pure Python); **train/eval thật** qua socket **gần như không** có trong pytest trừ khi bật `RUN_GAMA_INTEGRATION` hoặc **`smoke_test_env.py`** thủ công. **Rủi ro regress** khi đổi `Main_Traffic.gaml`, phiên bản **gama-pettingzoo**, hoặc bridge — CI **không** thay thế hồi quy end-to-end trên mô phỏng; luận văn nên nêu rõ khi bàn **độ tin cậy kết quả**.

### Rủi ro & đặc tính kiến trúc (nên nêu trong luận văn — không coi là lỗi phần mềm)

- **Tái lập & tiến trình ngoài**: Xem các mục *GamaMergingEnv* (reset ngay khi khởi tạo), *Độ phủ kiểm thử tích hợp thấp*, và *Tái lập thực nghiệm* phía trên — **quan hệ simulator–driver**, không phải defect, nhưng ảnh hưởng **CI** và **người mới clone** (GAMA, socket, smoke).

- **`Main_Traffic.gaml` lớn & đồng bộ với Python**: File mô hình **rất dài**; bảo trì/review đòi **kỷ luật** đồng bộ với Python (vd. chỉ số `obs[…]` trong `baselines.heuristic_action` khớp README và GAML). Sửa một chiều quan sát ở GAML mà không cập nhật heuristic/tests → **lệch hành vi baseline** — rủi ro quy trình, không phải lỗi cú pháp tự hiện.

- **`TrainConfig.total_timesteps` mặc định 20_000**: Là mặc định **smoke/debug** (đã ghi trong `TrainConfig` docstring); ai chạy `train.py` **không** truyền `--timesteps` và không đọc README có thể huấn luyện **quá ngắn** so với preset **thesis** (100k / MARL 200k). **Rủi ro thực nghiệm / báo cáo**, không phải bug logic.

- **Phiên bản stack bên ngoài**: `requirements.txt` ghim **torch / SB3 / scipy / …**; tái lập trên máy khác vẫn phụ thuộc **phiên bản GAMA headless**, **socket**, và có thể khác wheel theo OS — luận văn nên **ghi rõ môi trường** (phiên bản + cách cài) để hội đồng đánh giá **khả năng tái lập**, không quy là thiếu sót mã nguồn thuần túy.

- **Regex `scenario_utils` vs. consistency**: Bổ sung cho mục *Chuẩn scenario ↔ GAML* và bullet *Regex* phía trên — khi đổi GAML, kiểm tra [WARN] và chạy `tests/test_scenario_utils.py`.

- **MARL: IPPO-style, không centralized critic**: Thiết kế **Independent PPO/A2C + shared policy + `agent_indicator`** là **thực hành hợp lệ**. Code **không** có centralized critic hay lớp phối hợp tường minh — đặt **phạm vi đóng góp** đúng: đồ án là **đánh giá/so sánh thuật toán học trên một lựa chọn MARL cố định**, **không** nhất thiết là đóng góp **phương pháp MARL** mới — trừ phần luận văn chủ động **đối chiếu** với các dạng MARL trong tài liệu tham khảo (CTDE, QMIX, …, tùy độ sâu đồ án).

- **`asyncio` trong các script train**: Dùng `async_main` + `asyncio.run(main)` không sai — chỉ là **pattern dày hơn** so với một script CLI SB3 chỉ sync; khi bảo trì, entry `main()` sync + docstring ngắn đã giúp đọc dõi — có thể **ghi nhận nhẹ** trong luận văn (readability/kỹ năng đọc codebase) nhưng **không** bắt buộc refactor.

---

## Ghi chú kỹ thuật

- **GAMA thay NetLogo**: GAMA hỗ trợ MARL native (PettingZoo bridge), không cần extension bên thứ 3, headless mode ổn định hơn cho training tự động.
- **SB3 thay TF/Keras**: Stable-Baselines3 cung cấp DQN/PPO/A2C đã kiểm thử, tích hợp TensorBoard, hỗ trợ VecEnv cho MARL.
- **DQN loại khỏi MARL**: Off-policy, replay buffer lẫn transitions 4 agents → non-stationarity → Q-values không hội tụ.
- **GAMA không load `.zip` trực tiếp**: File model do Python quản lý. GAMA GUI (`TrafficSimulation`) dùng để demo Heuristic/Random/Manual.
- **asyncio**: `gama-client` yêu cầu event loop; các script dùng `asyncio.run()`.
- **Tái lập một phần (seed)**: `GamaConnectionConfig.simulation_seed` được truyền vào `PettingZoo.reset(seed=...)` khi khởi tạo env / MARL vec env (train, eval, baseline). Giúp thống nhất phía Python/bridge; **RNG nội bộ GAMA** vẫn có thể khác — luận văn nên nêu **reproducibility không hoàn toàn** nếu không kiểm soát thêm phía simulator.
- **Timeout wrapper**: `max_episode_steps=300` bảo vệ training khỏi episode treo khi GAMA gặp lỗi nội bộ.
- **Throughput / shockwave theo cửa sổ episode**: khi tái sinh `merging_0`, các bộ đếm throughput và Welford shockwave được reset — giá trị trong `info` phản ánh giai đoạn giữa hai lần respawn chứ không lũy kế suốt simulation. Chi tiết semantics throughput (bao gồm NPC ramp) xem mục *Lưu ý semantics & thống kê* trong phần Metrics.
- **Greedy trên xe cao tốc NPC**: `get_heuristic_highway_action()` (car-following cơ bản + nhường khi ramp gần merge), không chỉ ”luôn giữ tốc”.
- **`prob_damaged_car`**: mặc định `0.001`/cycle trong GAML (điều kiện ngẫu nhiên — tăng phương sai, cần nhiều seed). Đặt `0.0` nếu muốn mô phỏng “sạch” để chỉ đánh giá thuật toán.
- **Shared policy MARL**: 15D quan sát của merging và highway khác semantics; báo cáo nên nêu đây là IPPO/IA2C shared obs + indicator, có thể cần nhiều timesteps / reward shaping đã mô tả.
- **pytest + GAMA**: không có kiểm thử tự động mô phỏng trong CI; có thể chạy `python rl/smoke_test_env.py` khi máy có socket GAMA hoặc (tùy chọn) đặt `RUN_GAMA_INTEGRATION=1` cho các test được đánh dấu trong `tests/test_gama_optional.py`.

### MARL: pipeline Python, máy mới, và xử lý sự cố

**Mục đích**: clone sang máy khác vẫn chạy được `train_marl.py` / `evaluate_marl.py` mà không gặp các lỗi bridge đã xử lý trong codebase.

| Yêu cầu / lệnh | Ghi chú |
|---|---|
| `pip install -r requirements.txt` | Gồm `supersuit`, `gama-pettingzoo`, `stable-baselines3`, `torch`, … — **bắt buộc** trước khi train MARL. |
| GAMA headless đúng cổng | Train/eval cần `gama-headless.bat -socket 1001` (hoặc cổng khớp `--port`). |
| Windows + log tiếng Việt | Nếu gặp `UnicodeEncodeError` khi in docstring cảnh báo scenario, chạy PowerShell: `$env:PYTHONIOENCODING='utf-8'` trước `python rl/train_marl.py ...`, hoặc dùng terminal UTF-8 (chẳng hạn `chcp 65001`). |

**Thiết kế MARL trong repo (không đổi semantics MDP trong GAML)**:

- **`AgentIndicatorParallelWrapper`**: thêm one-hot agent ID (15D → 19D) bằng **`supersuit.utils.agent_indicator`** (`get_indicator_map`, `change_observation`, `change_obs_space`). **Không** dùng `supersuit.agent_indicator_v0` trực tiếp vì wrapper đó chuyển ParallelEnv sang AEC rồi về Parallel (`parallel_to_aec` → `aec_to_parallel`), dễ **`KeyError`** khi GAMA thay đổi tập agent đang hoạt động giữa các bước.
- **`MarkovVectorEnv(..., black_death=True)`**: cho phép `agents` khác `possible_agents` trong cùng episode (merge / respawn), khớp mô phỏng GAMA.
- **`GamaMarkovSB3VecEnv`**: bọc `MarkovVectorEnv` thành `stable_baselines3.common.vec_env.VecEnv` (SB3 2.x không nhận raw Gymnasium `VectorEnv`; không dùng `concat_vec_envs_v1` với `num_vec_envs=1` vì bước đó **pickle** env và GAMA bridge **không pickle** được sau `reset`).

**Có làm mất “logic mã cũ” không?** — **Không** đối với phần **định nghĩa bài toán và dữ liệu học**:

- Phần thưởng, điều kiện kết thúc, quan sát thô từ GAMA vẫn do **`models/Main_Traffic.gaml`** và `GamaParallelEnv` — không đổi.
- Vector 19D cho policy vẫn là **15 sensor + one-hot 4 agent** với **cùng quy tắc gán bit** như tài liệu SuperSuit `agent_indicator` (chỉ đổi **cách bọc** để tránh lỗi PettingZoo, không đổi công thức số).
- Shared policy + PPO/A2C (SB3) — không đổi kiến trúc thuật toán; chỉ **lớp adapter** giữa PettingZoo/Gymnasium và SB3 ổn định hơn trên stack hiện tại.

**Lưu ý nhỏ**: PPO có thể kết thúc với `total_timesteps` thực tế **lớn hơn** một chút so với `--timesteps` do làm tròn rollout (`n_steps` × số sub-env) — hành vi chuẩn của SB3, không phải lỗi MARL.
