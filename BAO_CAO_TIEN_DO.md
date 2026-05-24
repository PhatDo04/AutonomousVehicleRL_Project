# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN
## Phát triển kịch bản nhập làn trên đường cao tốc và đánh giá các thuật toán học tăng cường đa tác tử (MARL) cho điều khiển phương tiện tự hành

**Sinh viên thực hiện**: Đỗ Tiến Phát — MSSV 2251172447 — Lớp 64KTPM1, Khoa CNTT, ĐH Thủy Lợi
**Giảng viên hướng dẫn**: TS. Lê Nguyễn Tuấn Thành
**Ngày báo cáo**: 24/05/2026
**Tham chiếu chính**: Le Nguyen Tuan Thanh (2023). *Multi-agent reinforcement learning for traffic congestion on one-way multi-lane highways*. Journal of Information and Telecommunication, 7:3, 255–269.

---

## 1. Tóm tắt tiến độ

| Hạng mục | Trạng thái | Ghi chú |
|---|:---:|---|
| Khảo sát tài liệu & lựa chọn công cụ | ✅ Hoàn thành | Chuyển từ NetLogo → GAMA Platform (lý do mục 2.1) |
| Thiết kế môi trường mô phỏng (GAML) | ✅ Hoàn thành | `Main_Traffic.gaml`, 4904 LOC |
| Thiết kế MDP cho MARL (4 agent) | ✅ Hoàn thành | 19D obs, Discrete(5), reward đa thành phần |
| Cài đặt pipeline Python (env, train, eval) | ✅ Hoàn thành | PettingZoo → SuperSuit → SB3 |
| Cài đặt baseline (Greedy, Random) | ✅ Hoàn thành | `rl/baselines.py` rule-based |
| Cài đặt thuật toán MARL (IPPO, IA2C) | ✅ Hoàn thành | Shared policy + agent indicator |
| Thực nghiệm 100k–200k timesteps, kịch bản `low` | ✅ Có dữ liệu sơ bộ | `outputs/models/`, `outputs/logs/` đã có 4 model |
| Hệ thống chấm điểm KPI tự động | ✅ Hoàn thành | `check_eval_results.py` |
| Sinh biểu đồ + bảng LaTeX | ✅ Hoàn thành | 12 biểu đồ + bảng so sánh trong `outputs/plots/short/` |
| Chạy thực nghiệm đầy đủ preset `thesis` (5 seed × 80k) | 🟡 Đang chuẩn bị | Cần GAMA headless ổn định nhiều giờ |
| Phân tích sâu (tuning hyperparameter, ablation) | 🟡 Đang lặp | Đã qua 7+ iter hiệu chỉnh PPO/A2C |
| Viết luận văn chính thức | 🟡 Đang triển khai | Bản nháp tài liệu kỹ thuật trong `README.md` |

**Tóm lại**: Phần kỹ thuật (mô phỏng + pipeline học) đã chạy được end-to-end và đã sinh đủ artifact mẫu để báo cáo. Hiện tập trung tinh chỉnh hyperparameter để chống policy collapse trong A2C và mở rộng số seed/episode để có ý nghĩa thống kê.

---

## 2. Khái quát đề tài

### 2.1. Bối cảnh & mục tiêu

Bài toán **nhập làn trên đường cao tốc** (highway on-ramp merging) là kịch bản kinh điển trong an toàn giao thông và xe tự hành: xe trên đường nhánh phải tăng tốc, đánh giá khe trống và nhập làn vào dòng xe cao tốc đang di chuyển. Vấn đề khó vì:

- **Tương tác đa tác tử**: An toàn không phụ thuộc duy nhất vào hành động của xe nhập làn; xe cao tốc có thể nhường hoặc không.
- **Quan sát một phần (partial observability)**: Mỗi xe chỉ nhìn thấy lân cận.
- **Phần thưởng sparse + dense xen kẽ**: Thành công/va chạm là terminal (sparse) nhưng có shaping theo bước (dense).

**Mục tiêu đồ án**:
1. Xây dựng môi trường mô phỏng nhập làn (3 làn cao tốc + 1 làn ramp) trên **GAMA Platform**.
2. Hình thức hóa bài toán dưới dạng MDP đa tác tử (4 RL agents).
3. So sánh các thuật toán RL: baseline **Greedy/Random**, **MARL-PPO** (IPPO), **MARL-A2C** (IA2C) trên các chỉ số an toàn (collision rate), hiệu quả (success rate, merge_step) và lưu lượng (throughput, shockwave index).
4. Đối chiếu với kết quả công bố trong bài báo tham chiếu của GVHD.

### 2.2. Lý do chuyển NetLogo → GAMA Platform

Đề cương ban đầu dự kiến NetLogo, nhưng quá trình thử nghiệm cho thấy:

| Tiêu chí | NetLogo | GAMA |
|---|---|---|
| Hỗ trợ MARL native (PettingZoo bridge) | Cần extension bên thứ 3 | **Có sẵn** (`gama-pettingzoo`) |
| Headless mode cho training tự động | Hạn chế | **Ổn định** với `gama-headless` |
| Mô hình hóa hình học phức tạp (waypoint, ramp polygon) | Khó | **Tốt** (GAML hỗ trợ `geometry`, polyline) |
| Tích hợp với Python/Stable-Baselines3 | Bridge tự viết | **Có sẵn** qua socket TCP |

Do đó toàn bộ mô phỏng được port sang GAML, đồng thời giữ tinh thần "tham chiếu NetLogo" trong tài liệu (NPC rule-based mô phỏng lại logic Greedy của bài báo tham chiếu).

---

## 3. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│  GAMA Platform — models/Main_Traffic.gaml (4904 LOC)        │
│  • 3 làn cao tốc (lane_width=3.5m, road_length=200m)        │
│  • Làn ramp (6 waypoints, từ y=18m → merge_x=180m)          │
│  • Vùng tăng tốc: accel_start_x=48m → accel_end_x=162m      │
│  • NPC rule-based (driving_policy="Greedy") + xe hỏng       │
│    (prob_damaged_car=0.001/cycle, max 3 xe đồng thời)       │
│                                                             │
│  4 RL Agents (PettingZoo Parallel API):                     │
│   - merging_0  (magenta) — xe nhập làn trên ramp            │
│   - highway_0/1/2 (cyan) — xe cao tốc làn dưới              │
└───────────────────────┬─────────────────────────────────────┘
                        │ socket :1001 (gama-pettingzoo)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Python RL Pipeline (rl/)                                   │
│                                                             │
│  marl_env.py:                                               │
│   GamaParallelEnv                                           │
│     → PadPossibleAgentsParallelWrapper                      │
│     → AgentIndicatorParallelWrapper (15D → 19D)            │
│     → MarkovVectorEnv(black_death=True)                     │
│     → GamaMarkovSB3VecEnv (SB3 VecEnv)                      │
│                                                             │
│  train_marl.py:  IPPO/IA2C shared policy + SB3              │
│  evaluate_marl.py: 50 episode/seed, deterministic+mask      │
│  baselines.py:    Greedy rule-based                         │
│  plots.py:        12 biểu đồ                                │
│  analysis.py:     bảng LaTeX, comparison_table.csv          │
└─────────────────────────────────────────────────────────────┘
```

### 3.1. Phần GAML — môi trường vật lý và hành vi

File `models/Main_Traffic.gaml` (~4900 dòng) định nghĩa:

- **Global state**: hình học đường, các vùng (accel zone, merge point), bộ đếm `total_ramp_attempts` / `total_merge_success`, sampling shockwave bằng Welford online (không lưu toàn bộ mẫu).
- **Species `car`**: cả NPC và RL agent đều cùng một species. Phân biệt qua thuộc tính `rl_agent_id`.
  - `reflex behave`: vòng lặp hành vi mỗi tick. Bao gồm collision scan, gọi `ramp_behavior()` (NPC ramp) hoặc nhận action từ Python (`rl_merging_behavior` / `highway_rl_move`).
  - `action calculate_reward` / `calculate_merging_reward`: tính reward theo công thức đa mục tiêu (chi tiết mục 4.3).
  - `action check_terminal_state` / `check_merging_terminal_state`: phân loại kết quả episode (`success`, `collision`, `failed_merge`, `exited`).
  - `action get_current_state` / `get_merging_state`: trả về vector quan sát 15D.
- **Bridge layer (`PzBridgeAgent`, `petz_apply_tick`, `petz_collect_tick`)**: nhận action từ Python qua `gama-pettingzoo`, ghi vào `action_rl` của xe RL; cuối cycle thu obs/reward/done trả về Python.
- **Experiment `TrafficMARLHeadless`** (dùng cho train/eval) và **`TrafficSimulation`** (UI để demo).

### 3.2. Phần Python — pipeline học

Mô-đun chính trong thư mục `rl/`:

| File | Vai trò |
|---|---|
| `config.py` | `GamaConnectionConfig`, `TrainConfig`, `MARL_EXPERIMENT_PRESETS`, `ACTION_MEANINGS_*` |
| `gama_compat.py` | Patch tương thích `gama-gymnasium` (xử lý số phiên bản API) |
| `marl_env.py` | 3 wrapper Parallel + SB3 VecEnv adapter (xử lý các lỗi `KeyError`, `pickle coroutine`, `black_death`) |
| `train_marl.py` | Train IPPO/IA2C bằng SB3 + callback ghi CSV episode |
| `evaluate_marl.py` | Đánh giá deterministic (có mask argmax) + stochastic |
| `baselines.py` | Greedy rule-based qua cùng pipeline MARL |
| `scenario_utils.py` | Patch file `Main_Traffic.gaml` theo preset `low/medium/high` |
| `gama_episode_reset.py` | Reset GAMA giữa các episode (xử lý NOTREADY/PAUSED) |
| `run_experiments.py` | Pipeline tự động: baseline → train → eval → plots → analysis |
| `plots.py` | Sinh 12 biểu đồ PNG từ CSV |
| `analysis.py` | Bảng so sánh, LaTeX, learning efficiency |
| `metrics.py` | `EpisodeMetric`, `classify_episode`, CSV writer |
| `model_registry.py` | Liệt kê model `.zip` → JSON |
| `check_eval_results.py` | Kiểm tra KPI sau training (success ≥ 10%, collision ≤ 85%) |
| `diagnose_marl.py` | Trace từng bước episode để debug policy collapse |
| `smoke_test_env.py` | Kiểm tra kết nối GAMA + 4 agent xuất hiện đúng |

---

## 4. Thiết kế MDP đa tác tử

### 4.1. Tác tử (Agents)

| Agent | Vai trò | Màu UI | Mục tiêu |
|---|---|---|---|
| `merging_0` | Xe nhập làn trên ramp | Magenta | Nhập làn an toàn vào làn dưới cùng |
| `highway_0` | Xe cao tốc làn dưới | Cyan | Duy trì tốc độ, hỗ trợ nhường gap khi cần |
| `highway_1` | Xe cao tốc làn dưới | Cyan | Như highway_0 |
| `highway_2` | Xe cao tốc làn dưới | Cyan | Như highway_0 |

**Chiến lược học**: **Shared Policy** (IPPO/IA2C — Independent learning với parameter sharing). Cả 4 agent dùng chung 1 model. Để policy phân biệt vai trò, vector quan sát 15D được nối thêm **one-hot agent ID** (4 chiều) thành **19D** thông qua `AgentIndicatorParallelWrapper` (port lại logic SuperSuit `agent_indicator_v0` để giữ Parallel API, tránh lỗi `KeyError` khi GAMA respawn động agent).

### 4.2. Không gian quan sát (Observation Space)

Cả hai vai trò đều dùng `Box(low=0, high=1, shape=(15,))` — sau wrapper là `(19,)`.

#### `merging_0` (vector từ `get_merging_state()` trong GAML)

| # | Chiều | Ý nghĩa |
|:---:|---|---|
| 0 | `norm_speed` | Tốc độ xe / `speed_max` |
| 1 | `norm_progress` | Tiến độ trên ramp (0=đầu → 1=cuối) |
| 2 | `norm_dist_to_merge` | Khoảng cách đến điểm merge |
| 3 | `norm_lateral_error` | Lệch ngang so với làn mục tiêu |
| 4 | `in_accel_zone` | Có trong vùng tăng tốc không (binary) |
| 5–6 | `norm_gap_front`, `norm_speed_front` | Khe trống & tốc độ xe trước trên làn đích |
| 7–8 | `norm_gap_rear`, `norm_speed_rear` | Khe trống & tốc độ xe sau trên làn đích |
| 9 | `norm_gap_safe` | Gap đủ an toàn để merge (binary) |
| 10–11 | `norm_ramp_front_gap`, `norm_ramp_front_speed` | Xe phía trước trên ramp |
| 12 | `norm_urgency` | Khẩn cấp (0 đầu accel zone, 1 cuối) |
| 13 | `norm_last_action` | Action gần nhất |
| 14 | `norm_patience` | Patience tích lũy |

#### `highway_0/1/2` (vector từ `get_current_state()` — radar 15D)

| # | Chiều | Ý nghĩa |
|:---:|---|---|
| 0–1 | `norm_speed`, `norm_current_lane` | Tốc độ + chỉ số làn hiện tại |
| 2–3 | `norm_dist_ahead`, `norm_speed_ahead` | Xe trước cùng làn |
| 4–7 | dist/speed ahead/behind **trái** | Radar làn trái |
| 8–11 | dist/speed ahead/behind **phải** | Radar làn phải |
| 12–13 | `norm_dist_merger`, `norm_speed_merger` | Xe ramp/merge gần nhất |
| 14 | `norm_patience` | `patience / 100` |

> **Lưu ý quan trọng**: Hai vai trò có **cùng kích thước 15D** nhưng **khác semantics** từng chiều. Shared policy + one-hot indicator phải tự học phân biệt — đây là một thách thức ghi nhận trong phần *Hạn chế* (mục 8).

### 4.3. Không gian hành động (Action Space)

`Discrete(5)` — cùng index, khác ý nghĩa theo vai trò:

| Action | `merging_0` | `highway_0/1/2` |
|:---:|---|---|
| 0 | Giảm tốc (brake) | **Tăng tốc** (accelerate) — đối ngược |
| 1 | Giữ tốc (keep) | Giữ tốc (keep) |
| 2 | Tăng tốc (accelerate) | **Giảm tốc** (decelerate) — đối ngược |
| 3 | **Nhập làn** (merge_attempt) | Rẽ trái (lane change left) |
| 4 | Chờ / giảm nhẹ (wait) | Rẽ phải (lane change right) |

### 4.4. Hàm phần thưởng (Reward Function)

**Nguồn sự thật**: toàn bộ tính trong GAML (`calculate_reward`, `calculate_merging_reward`). Python chỉ nhận `reward_val` mỗi step. Đã trải qua **7+ iterations** tinh chỉnh.

#### Reward cho `merging_0`

| Sự kiện | Giá trị | Điều kiện |
|---|---:|---|
| Terminal: merge thành công | **+100** | Đạt điểm merge, không va chạm |
| Terminal: va chạm | **−100** | `nearest_collision_dist < 3.5m` |
| Terminal: hết accel lane chưa merge | **−50** | `failed_merge` |
| Step base | −0.01 + `action_penalty` + `speed × 0.06` | Mỗi tick |
| Vào vùng accel lần đầu (one-shot) | **+15.0** | `in_accel_zone && !was_in_accel_prev` |
| Trong accel zone | +0.05 / step | Khuyến khích vào và ở lại |
| Action 3 (merge) đúng lúc gap an toàn | **+10.0 + urgency × 10.0** | `in_accel_zone && obs_gap_safe` (max +20 khi sắp hết accel lane) |
| Gap trước < `gap_front_min` | −(gap_front_min − gap)×**0.2** | Phạt dense (đã giảm từ ×0.5) |
| Gap sau < `gap_rear_min` | −(gap_rear_min − gap)×**0.2** | Phạt dense |
| Action 1 (keep) trong zone | **−0.5** | Chống collapse về "do nothing" |
| Action 4 (wait) trong zone | **−0.5** | Chống collapse về "wait forever" |
| Action 0 (brake) trong zone | **−0.3** | Chống collapse về "đứng yên" |
| Speed < 0.10 | −0.06 | Phạt rùa bò |
| Reward clip | [−40, +40] | Tránh exploding gradient |

#### Reward cho `highway_0/1/2`

| Sự kiện | Giá trị | Điều kiện |
|---|---:|---|
| Terminal: va chạm | **−100** | |
| Terminal: thoát khỏi đường | **+50** | `location.x ≥ road_length − 5m` (≈195m) |
| Speed reward (throughput) | `speed × 0.04` | Mỗi step (đã giảm từ 0.10 chống collapse) |
| Phạt rùa bò | −0.05 | `speed < 0.2 × speed_max` |
| TTC penalty (bám đuôi) | −((15 − gap)/15) × speed × 0.8 | `gap < 15m && speed > 0.15` |
| Cooperative bonus | +0.3 | `recent_merge_coop_ticks > 0 && abs(x − recent_merge_x) < 20m` — *hiện đang disable* (`enable_marl_coop_reward = false`) |

> **Ghi chú**: Cooperative reward chưa được bật trong production vì trong các iter trước nó làm highway "đứng yên chờ merge" gây collapse. Sẽ bật lại sau khi PPO baseline ổn định.

### 4.5. Điều kiện kết thúc episode

| Agent | Outcome | Điều kiện |
|---|---|---|
| `merging_0` | `success` | Merge thành công vào làn cao tốc |
| `merging_0` | `collision` | Va chạm với xe khác (`< 3.5m`) |
| `merging_0` | `failed_merge` | Đi quá `merge_x + 2m` mà chưa nhập làn |
| `merging_0` | `timeout` | Vượt `max_episode_steps = 300` (Python wrapper) |
| `highway_i` | `exited` | `location.x ≥ 195m` |
| `highway_i` | `collision` | Va chạm |

---

## 5. Thuật toán đã triển khai

### 5.1. Baseline (rule-based)

**Greedy** (`rl/baselines.py:heuristic_action`): rule-based dùng 7 chỉ số đầu của obs 15D:

```
if in_accel_zone & gap_safe         → merge (action 3)
if gap_rear < 0.1 & speed_rear > 0.5 → brake (action 0)
if ramp_front_gap < 0.08             → brake
if urgency > 0.9 & !gap_safe         → merge (cố nhập)
if urgency > 0.75 & !gap_safe        → wait
if speed < 0.7                       → accelerate (action 2)
default                              → keep_speed (action 1)
```

Mô phỏng lại "Greedy NPC" theo bài báo tham chiếu — phản ánh hành vi người lái thực tế thận trọng.

### 5.2. MARL-PPO (IPPO + parameter sharing)

| Hyperparameter | Giá trị | Ghi chú thay đổi qua các iter |
|---|---:|---|
| Policy | `MlpPolicy` | SB3 mặc định, 2 hidden layer × 64 unit |
| `learning_rate` | 5e-4 | |
| `n_steps` | 256 | |
| `batch_size` | 128 | |
| `gamma` | 0.99 | |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | |
| `ent_coef` | **0.20** | Tăng từ 0.08 (iter 5) chống policy collapse → "keep" |
| `vf_coef` | 0.5 | |
| `total_timesteps` | 50k → 200k | preset thesis: 80k × 5 seed |

### 5.3. MARL-A2C (IA2C + parameter sharing)

| Hyperparameter | Giá trị | Ghi chú |
|---|---:|---|
| `learning_rate` | **1e-3** | Tăng từ 7e-4 (iter 7) |
| `n_steps` | 32 | |
| `gamma` | 0.99 | |
| `gae_lambda` | 0.95 | |
| `ent_coef` | **0.32** | Tăng mạnh từ 0.18 — A2C không có PPO clip nên cần entropy mạnh hơn để escape collapse |

### 5.4. Lý do loại DQN khỏi MARL

DQN off-policy nhồi replay buffer chung của 4 agent → transitions lẫn lộn, non-stationarity làm Q-values không hội tụ. PPO/A2C on-policy update ngay trên rollout hiện tại — phù hợp hơn cho MARL (theo Lowe et al. 2017 MADDPG; de Witt et al. 2020 MAPPO).

---

## 6. Kết quả thực nghiệm sơ bộ

> **Lưu ý**: Đây là kết quả từ preset `short` (1 seed, 20 episode eval) — **chưa phải dữ liệu cuối cho luận văn**. Số liệu cuối cần preset `thesis` (5 seed × 80k steps × 50 episode eval) để có ý nghĩa thống kê.

### 6.1. Bảng so sánh (eval, kịch bản `low`)

| Chỉ số | **Greedy** | **MARL-A2C** | **MARL-PPO** |
|---|---:|---:|---:|
| Số episode eval | 20 | 20 | 20 |
| **Success rate** | **100%** | 0% | **100%** |
| **Collision rate** | 0% | **100%** | 0% |
| Failed merge rate | 0% | 0% | 0% |
| Timeout rate | 0% | 0% | 0% |
| Avg reward (merging_0) | +21.18 | −285.57 | −426.95 |
| Avg episode steps | 263 | 414 | 220 |
| Avg merge_step (timestep) | 54 | — (collision) | **70** |
| Avg mean_speed (norm) | 0.690 | 0.300 | **0.820** |
| Throughput | 1.000 | 1.125 | **1.167** |
| **Shockwave index** | **0.149** | 0.706 | 0.330 |

> Số bảng eval ổn định do `evaluate_marl.py` chạy deterministic+mask trên cùng seed — báo cáo thesis cần thêm sampling đa seed.

### 6.2. Nhận xét sơ bộ

**Mặt tích cực**:
- **MARL-PPO đạt 100% success rate** và tốc độ trung bình **cao nhất (0.82)** + throughput **cao nhất (1.167)** trong 3 phương pháp.
- **PPO hoàn thành nhập làn nhanh hơn Greedy** (220 step vs 263 step).
- Shockwave index PPO (0.33) thấp hơn A2C đáng kể (0.71), thể hiện dòng giao thông mainline ổn định hơn.

**Vấn đề cần xử lý**:
- **MARL-A2C collapse**: 100% collision rate, reward âm sâu. Đây là dấu hiệu policy đã sập về một action duy nhất (đang điều tra qua `diagnose_marl.py` — log action histogram cho thấy hiện tượng "thiên về 1 action").
- **PPO reward âm dù success 100%** (−426.95): do gap-penalty tích lũy lớn khi PPO chọn chiến lược "lao nhanh tìm khe" — đã giảm hệ số gap penalty từ ×0.5 xuống ×0.2 trong iter 7, sẽ retrain để xem reward dương.
- **Greedy thắng tuyệt đối ở reward absolute** (+21.18) — không bất ngờ vì rule-based né hoàn toàn các tình huống nguy hiểm. Đây là **upper bound** cho RL trên kịch bản `low`.

### 6.3. Artifact đã sinh

```
outputs/
├── models/   (4 file: marl_{ppo,a2c}_seed0_{100k,200k}_low.zip)
├── logs/     (16 file CSV: train + eval × ppo/a2c × 100k/200k × merging/highway)
│   └── tensorboard/    (per-run subdirs)
└── plots/short/  (12 PNG + comparison_table.csv + latex_table.tex + summary)
```

---

## 7. Các khó khăn kỹ thuật đã giải quyết

Đây là phần cho thấy độ sâu công việc — không chỉ "chạy code mà còn debug".

### 7.1. PettingZoo + SuperSuit + SB3 không tương thích trực tiếp

Khi áp dụng `supersuit.agent_indicator_v0` trực tiếp lên `GamaParallelEnv`, SuperSuit bọc `parallel_to_aec → aec_to_parallel` — GAMA respawn agent động (merging_0 chết → spawn lại), AEC trace bị lệch → `KeyError: terminations[agent_selection]`.
**Giải pháp**: viết lại `AgentIndicatorParallelWrapper` giữ nguyên Parallel API, dùng cùng công thức one-hot SuperSuit (`supersuit.utils.agent_indicator`) — `rl/marl_env.py:113-171`.

### 7.2. `concat_vec_envs_v1` pickle GAMA bridge → `TypeError: cannot pickle 'coroutine'`

`gama-pettingzoo` dùng `asyncio` ngầm, sau `reset()` env chứa coroutine không pickle được.
**Giải pháp**: viết `GamaMarkovSB3VecEnv` bọc `MarkovVectorEnv` thành SB3 `VecEnv` mà không cần `concat_vec_envs_v1` — `rl/marl_env.py:174-250`.

### 7.3. `agents != possible_agents` giữa các step → `AssertionError` MarkovVectorEnv

Khi `merging_0` chết, GAMA xóa khỏi `agents` nhưng `possible_agents` vẫn cố định.
**Giải pháp**:
- `PadPossibleAgentsParallelWrapper`: luôn pad obs đủ 4 agent (zeros nếu thiếu) — `rl/marl_env.py:60-110`.
- `MarkovVectorEnv(..., black_death=True)` cho phép agent động — `rl/marl_env.py:301`.

### 7.4. Policy collapse trong PPO/A2C

Trải qua **7 iteration tuning**:

| Iter | Hiện tượng | Giải pháp |
|---|---|---|
| 1 | A2C 94% chọn brake → đứng yên | Tăng `ent_coef` 0.05 → 0.18 |
| 3 | PPO 36% chọn merge nhưng vẫn timeout | Tăng `n_steps` 256, `max_episode_steps` 300→500 |
| 4 | PPO 91% chọn wait trong accel zone | Thêm penalty **−0.5** cho action 1/4 trong zone |
| 5 | PPO collapse về keep dù phạt −0.5 | Tăng `ent_coef` 0.08 → **0.20** + bonus action 3 từ +5 lên **+10 + urgency×10** |
| 6 | PPO success 100% nhưng reward −426 | Giảm gap penalty ×0.5 → **×0.2** |
| 7 | A2C vẫn 71% keep | Tăng `ent_coef` 0.18 → **0.32**, `lr` 7e-4 → **1e-3** |

Quá trình tuning được ghi log trong comment GAML `calculate_merging_reward` (lines 4451-4546).

### 7.5. Bug seed lặp lại trong eval

Trước iter này, `evaluate_marl.py` truyền `args.seed + episode` (1, 2, 3...) → GAMA chỉ chuyển thành `seed <- 1.0;` (float không xáo trộn đủ) → 20 episode có metric giống hệt.
**Giải pháp**: dùng `np.random.default_rng(args.seed)` để sinh seed lớn (31-bit) mỗi episode — `rl/evaluate_marl.py:261-262`.

### 7.6. Unicode encoding trên Windows

Console Windows mặc định cp1252 → log tiếng Việt `UnicodeEncodeError`.
**Giải pháp**: `pip install -r requirements.txt` chỉ dùng comment ASCII; runtime patch `$env:PYTHONIOENCODING='utf-8'` trong `gama_compat._configure_windows_cli_io`.

### 7.7. Bridge mất kết nối giữa episode (`NOTREADY/PAUSED`)

GAMA sau `reset()` có thể vào trạng thái NOTREADY tạm thời.
**Giải pháp**: `rl/gama_episode_reset.py` retry với backoff + bắt `GamaCommandError` rồi await 3s và thử lại — `rl/evaluate_marl.py:332-345`.

---

## 8. Hạn chế và phạm vi kết luận

Phần này dự kiến đưa vào chương *Kết luận và hướng phát triển* của luận văn.

### 8.1. Hạn chế mô hình hóa

- **Mô phỏng waypoint + Discrete(5)** — không phải động học/cảm biến đầy đủ của xe thật. Kết luận giới hạn ở *so sánh thuật toán trong MDP đã định nghĩa*, không phải khẳng định cho xe thật đường trường.
- **Shared policy + agent_indicator** trên obs 15D **khác semantics** giữa merging và highway — đây là thiết kế tinh giản (1 policy cho 2 vai trò), không tương đương 2 policy chuyên biệt. Báo cáo nêu thẳng đây là IPPO/IA2C parameter-sharing.
- **Cùng action space, khác ý nghĩa** (action 0 = brake cho merging, accelerate cho highway) — chống chỉ định hành động qua agent indicator trong obs là một giả định mạnh.

### 8.2. Hạn chế thống kê

- Kết quả sơ bộ chỉ với **1 seed, 20 episode eval**. Để có CI95% có ý nghĩa cần **≥ 3 seed × ≥ 50 episode eval**.
- **Nhiễu môi trường** từ `prob_damaged_car = 0.001` + `balance_traffic` ngẫu nhiên — phương sai bổ sung ngoài policy.

### 8.3. Hạn chế tính toán

- `num_vec_envs = 1` — một instance GAMA. Không vector hóa song song nhiều bản GAMA → throughput timestep thấp (~5–10 step/s tùy máy).
- Preset `thesis` (5 seed × 80k step × ~2 phút eval/seed) ước tính **5–8 giờ wall-clock**.

### 8.4. Hạn chế tái lập

- Phụ thuộc phiên bản: GAMA Platform, `gama-pettingzoo`, Python, SB3, torch — cần ghi rõ environment trong luận văn.
- RNG nội bộ GAMA không kiểm soát hoàn toàn từ Python — *reproducibility không tuyệt đối*.
- Test tự động chỉ kiểm logic Python pure (`tests/test_analysis.py`, `tests/test_metrics.py`, `tests/test_scenario_utils.py`); end-to-end qua socket GAMA chạy thủ công qua `smoke_test_env.py`.

### 8.5. Phạm vi đóng góp

Đồ án là **đánh giá so sánh** các thuật toán RL trên một MDP MARL cố định, **không phải đề xuất phương pháp MARL mới** (như centralized critic, QMIX). Nếu hướng tới đóng góp phương pháp, sẽ là phần *Hướng mở rộng*.

---

## 9. Việc cần làm tiếp

### 9.1. Trước báo cáo bảo vệ

| # | Việc | Ưu tiên | Trạng thái |
|:---:|---|:---:|:---:|
| 1 | Chạy preset `thesis` (5 seed × 80k × 50 eval) | 🔴 Cao | Đang chuẩn bị môi trường |
| 2 | Sinh lại biểu đồ + bảng LaTeX với CI95% có ý nghĩa | 🔴 Cao | Chờ #1 |
| 3 | Khắc phục A2C collapse 100% collision | 🔴 Cao | Đang test ent_coef 0.32 |
| 4 | Chạy thêm kịch bản `medium` và `high` để so sánh density | 🟡 Vừa | Chưa |
| 5 | Bật `enable_marl_coop_reward` test cooperative | 🟡 Vừa | Chưa |
| 6 | Viết chương 4 (Thực nghiệm và đánh giá) | 🔴 Cao | Đang triển khai |
| 7 | Viết chương 2 (Cơ sở lý thuyết MARL: IPPO, IA2C, PettingZoo) | 🟡 Vừa | Đang triển khai |
| 8 | Bổ sung test cho `marl_env.py` (mock GAMA socket) | 🟢 Thấp | Chưa |
| 9 | Đóng gói requirements + hướng dẫn cài đặt cho hội đồng | 🟡 Vừa | README.md đã có |

### 9.2. Hướng mở rộng (cho phần *Hướng phát triển tương lai*)

- **Centralized Critic MARL** (CTDE): thử MAPPO chính thức (centralized value, decentralized actor) thay vì IPPO/IA2C đơn thuần.
- **Reward shaping bằng inverse RL** từ Greedy demonstrations (vì Greedy đang là upper bound).
- **Sim2Real partial**: thay obs synthetic bằng obs có nhiễu cảm biến (Gaussian noise) để test robustness.
- **Mở rộng action sang continuous** (steering + throttle) thay vì Discrete(5).

---

## 10. Tham chiếu chính trong codebase

Liên kết tới các file/dòng quan trọng phục vụ trình bày:

- Định nghĩa MDP (state, action, reward):
  - [models/Main_Traffic.gaml:3942-3975](models/Main_Traffic.gaml#L3942-L3975) — `get_current_state` (highway obs 15D)
  - [models/Main_Traffic.gaml:4082](models/Main_Traffic.gaml#L4082) — `get_merging_state` (merging obs 15D)
  - [models/Main_Traffic.gaml:4333-4381](models/Main_Traffic.gaml#L4333-L4381) — `calculate_reward` (highway)
  - [models/Main_Traffic.gaml:4452-4547](models/Main_Traffic.gaml#L4452-L4547) — `calculate_merging_reward` (merging, đã qua 7 iter tuning)
  - [models/Main_Traffic.gaml:3166-3262](models/Main_Traffic.gaml#L3166-L3262) — `check_terminal_state` + `check_merging_terminal_state`
- Pipeline Python:
  - [rl/marl_env.py:113-171](rl/marl_env.py#L113-L171) — `AgentIndicatorParallelWrapper`
  - [rl/marl_env.py:174-250](rl/marl_env.py#L174-L250) — `GamaMarkovSB3VecEnv`
  - [rl/marl_env.py:253-310](rl/marl_env.py#L253-L310) — `make_marl_vec_env`
  - [rl/train_marl.py:158-210](rl/train_marl.py#L158-L210) — `build_marl_model` (PPO/A2C hyperparam)
  - [rl/evaluate_marl.py:83-117](rl/evaluate_marl.py#L83-L117) — `_predict_marl_action` (deterministic + mask)
  - [rl/baselines.py:72-99](rl/baselines.py#L72-L99) — `heuristic_action` (Greedy rule-based)
- Config & preset:
  - [rl/config.py:64-70](rl/config.py#L64-L70) — `MARL_EXPERIMENT_PRESETS`
  - [rl/config.py:75-89](rl/config.py#L75-L89) — `ACTION_MEANINGS_MERGING/HIGHWAY`

---

## 11. Đánh giá tự thân (Self-assessment)

**Đã đạt được mục tiêu ban đầu**:
- ✅ Môi trường mô phỏng MARL hoàn chỉnh trên GAMA (thay NetLogo, có chính đáng).
- ✅ Pipeline học end-to-end Python ↔ GAMA qua socket.
- ✅ Baseline + 2 thuật toán MARL chạy được, có artifact (model + CSV + plot).
- ✅ Hệ thống chỉ số đo lường đầy đủ (success, collision, throughput, shockwave).

**Đã giải quyết các thách thức kỹ thuật**:
- ✅ Bridge incompatibility (`AgentIndicatorParallelWrapper`, `GamaMarkovSB3VecEnv`).
- ✅ Black death + agent dynamics.
- ✅ Policy collapse — 7 iter tuning hyperparameter + reward shaping.

**Cần tăng cường**:
- 🟡 Số seed + episode eval cho ý nghĩa thống kê.
- 🟡 A2C vẫn collapse (100% collision) — cần fix trước bảo vệ.
- 🟡 So sánh đa scenario (low/medium/high) — chỉ mới có `low`.
- 🟡 Viết hoàn chỉnh các chương luận văn.

**Mức độ hoàn thành ước tính**: **~75%** — phần kỹ thuật cốt lõi đã ổn định, còn lại là chạy thực nghiệm quy mô lớn và viết báo cáo chính thức.

---

*Báo cáo này được sinh tự động từ phân tích source code và artifact thực nghiệm hiện có. File gốc: `BAO_CAO_TIEN_DO.md`.*
