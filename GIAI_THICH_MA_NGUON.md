# Giải thích toàn bộ mã nguồn — Đồ án RL nhập làn cao tốc

> **Dành cho ai “vibe code” xong mà chưa nắm logic.**  
> Đọc file này theo thứ tự: mục 1 → 2 → 3 → 4. Khi cần chi tiết kỹ thuật sâu, xem thêm `README.md` và `BAO_CAO_TIEN_DO.md`.

---

## Mục lục

1. [Đồ án này làm gì?](#1-đồ-án-này-làm-gì)
2. [Bức tranh tổng thể — 2 nửa của hệ thống](#2-bức-tranh-tổng-thể--2-nửa-của-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Luồng dữ liệu một bước huấn luyện](#4-luồng-dữ-liệu-một-bước-huấn-luyện)
5. [Phần GAMA — `models/Main_Traffic.gaml`](#5-phần-gama--modelsmain_trafficgaml)
6. [Phần Python — thư mục `rl/`](#6-phần-python--thư-mục-rl)
7. [Bài toán RL (MDP) — quan sát, hành động, thưởng](#7-bài-toán-rl-mdp--quan-sát-hành-động-thưởng)
8. [Thuật toán và cách train](#8-thuật-toán-và-cách-train)
9. [Kết quả thực nghiệm — CSV, biểu đồ, bảng](#9-kết-quả-thực-nghiệm--csv-biểu-đồ-bảng)
10. [Chạy dự án từ đầu](#10-chạy-dự-án-từ-đầu)
11. [Phần archive (single-agent)](#11-phần-archive-single-agent)
12. [Bẫy thường gặp khi đọc code](#12-bẫy-thường-gặp-khi-đọc-code)
13. [Bản đồ “muốn sửa X thì mở file nào”](#13-bản-đồ-muốn-sửa-x-thì-mở-file-nào)

---

## 1. Đồ án này làm gì?

**Bài toán:** Xe trên **làn nhập (ramp)** phải tăng tốc và **nhập làn** vào **3 làn cao tốc** đang chạy — giống cảnh trên đường cao tốc thật.

**Mục tiêu kỹ thuật:**

- Mô phỏng bằng **GAMA Platform** (ngôn ngữ **GAML**).
- Huấn luyện **4 tác tử RL** cùng lúc (MARL):
  - `merging_0` — xe nhập làn (màu magenta).
  - `highway_0`, `highway_1`, `highway_2` — 3 xe trên làn cao tốc dưới cùng (màu cyan).
- So sánh **PPO / A2C** (học) với **Greedy** (rule-based, không học).
- Đo **success rate**, **collision rate**, **throughput**, **shockwave index**, v.v.

**Điểm quan trọng:** Phần thưởng và physics **không** tính trong Python. Python chỉ gửi **số hành động 0–4** và nhận **vector quan sát 15 số** + **reward một số**. Mọi logic “xe chạy thế nào, va chạm thì −100” nằm trong **GAML**.

---

## 2. Bức tranh tổng thể — 2 nửa của hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│  GAMA (simulator) — Main_Traffic.gaml ~4900 dòng                 │
│  • Vẽ đường, ramp, spawn xe NPC                                   │
│  • Di chuyển xe, va chạm, merge                                  │
│  • Tính reward, obs 15D, outcome (success/collision/...)         │
│  • Experiment: TrafficMARLHeadless (train) / TrafficSimulation (UI)│
└────────────────────────────┬────────────────────────────────────┘
                             │ TCP socket cổng 1001
                             │ thư viện gama-pettingzoo
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Python (học máy) — thư mục rl/                                 │
│  • marl_env.py: bọc PettingZoo → SB3 VecEnv (obs 19D)            │
│  • train_marl.py: PPO / A2C (Stable-Baselines3)                 │
│  • evaluate_marl.py, baselines.py: đánh giá                     │
│  • run_experiments.py: chạy cả pipeline                         │
│  • plots.py, analysis.py: biểu đồ + bảng LaTeX                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tại sao cần 2 chương trình?**

- GAMA giỏi mô phỏng giao thông (geometry, nhiều xe, reflex mỗi “tick”).
- Python giỏi neural network (PyTorch + SB3).
- Cầu nối: **PettingZoo Parallel API** — chuẩn đa tác tử trong RL.

---

## 3. Cấu trúc thư mục

| Đường dẫn | Vai trò (nói ngắn) |
|-----------|-------------------|
| `models/Main_Traffic.gaml` | **Trái tim** — mô phỏng + reward + 4 RL agents |
| `models/single_agent/Main_Traffic_SingleAgent.gaml` | Bản cũ chỉ 1 agent `merging_0` (archive) |
| `models/Old Model.gaml` | Model NetLogo/GAMA cũ, tham khảo |
| `rl/config.py` | Hằng số: đường dẫn, preset, ý nghĩa action |
| `rl/marl_env.py` | Nối GAMA → SB3 (wrapper 19D obs) |
| `rl/gama_compat.py` | Vá lỗi tương thích gama-pettingzoo / GAMA 2025.6 |
| `rl/gama_episode_reset.py` | Reset episode + chờ đủ 4 agent spawn |
| `rl/train_marl.py` | Huấn luyện MARL |
| `rl/evaluate_marl.py` | Test model đã train |
| `rl/baselines.py` | Greedy / Random không học |
| `rl/run_experiments.py` | Tự động: baseline → train → eval → plots |
| `rl/scenario_utils.py` | Đổi mật độ xe (low/medium/high) bằng regex vá GAML |
| `rl/metrics.py` | Định nghĩa CSV từng episode |
| `rl/plots.py` | 12 file PNG từ CSV |
| `rl/analysis.py` | Bảng so sánh + `latex_table.tex` |
| `rl/smoke_test_env.py` | Test nhanh “GAMA có sống không?” |
| `rl/diagnose_marl.py` | Debug từng bước obs/action |
| `rl/check_eval_results.py` | Kiểm KPI có đạt ngưỡng không |
| `rl/play_marl_gui.py` | Chạy policy trên GAMA GUI |
| `rl/legacy/` | Train/eval single-agent cũ (DQN/PPO/A2C 1 xe) |
| `outputs/models/` | File `.zip` (mạng neural) |
| `outputs/logs/` | CSV episode + TensorBoard |
| `outputs/plots/` | Biểu đồ PNG |
| `tests/` | pytest không cần GAMA (phần lớn) |
| `requirements.txt` | Thư viện Python |
| `README.md` | Hướng dẫn chạy (tiếng Việt, chi tiết) |
| `BAO_CAO_TIEN_DO.md` | Báo cáo tiến độ đồ án |

---

## 4. Luồng dữ liệu một bước huấn luyện

Hiểu **một vòng** là hiểu cả project:

```
1. SB3 (PPO/A2C) nhìn obs 19D × 4 agent → chọn action 0–4 cho mỗi agent
2. marl_env gộp 4 action → gửi qua socket tới GAMA
3. GAMA (mỗi cycle):
   a. apply_pz_python_each_cycle: gán action_rl cho từng xe RL
   b. reflex behave: xe NPC + RL di chuyển
   c. rl_merging_behavior / highway behavior: tốc độ, merge, reward
   d. petz_collect_tick: đọc obs/reward/done → pz_data
4. Python nhận reward, obs mới, terminated
5. SB3 cập nhật policy (on-policy: PPO/A2C)
```

**1 Python `env.step()`** ≈ **1 bước mô phỏng GAMA** (1 cycle), không phải 1 episode.

**1 episode** kết thúc khi ví dụ `merging_0` merge thành công, va chạm, hết đường chưa merge, hoặc Python cắt ở `max_episode_steps` (mặc định 300–500).

---

## 5. Phần GAMA — `models/Main_Traffic.gaml`

File ~4900 dòng. Không cần đọc hết — chia **khối logic**:

### 5.1. `global { ... }` — Tham số thế giới

| Tham số | Ý nghĩa |
|---------|---------|
| `number_of_lanes = 3` | 3 làn cao tốc |
| `lane_width = 3.5` | Rộng mỗi làn (m) |
| `road_length = 200` | Chiều dài đoạn mô phỏng (m) |
| `offset_y = 18` | Cao độ Y gốc của đường |
| `accel_start_x = 48`, `accel_end_x = 162` | Vùng **tăng tốc** trên ramp |
| `merge_x = 180` | Điểm **nhập làn** vào cao tốc |
| `ramp_waypoints` | Danh sách điểm polyline — xe ramp bám theo |
| `nb_cars_max` | Số xe NPC tối đa (mật độ) |
| `gap_front_min`, `gap_rear_min` | Khoảng cách an toàn để merge |
| `collision_distance` | Ngưỡng va chạm |
| `driving_policy = "Greedy"` | **Chỉ cho xe NPC**, không cho RL |
| `prob_damaged_car` | Xe hỏng ngẫu nhiên trên cao tốc |
| `total_ramp_attempts`, `total_merge_success` | Đếm throughput |
| `sw_mean`, `sw_M2` | Welford — tính **shockwave index** |
| `pz_*` maps | Biến global mirror cho socket Python |

**Experiments (cuối file):**

- `TrafficMARLHeadless` — Python train/eval (không cần UI).
- `TrafficSimulation` — mở trong GAMA GUI, có dashboard chọn Manual/Heuristic/Python.

### 5.2. Khởi tạo xe — `bootstrap_initial_cars`

Chạy **một lần** sau khi simulation bắt đầu (`cycle > 0`):

1. **NPC trên cao tốc:** `nb_cars_max / 3` xe mỗi làn, rải đều theo trục X.
2. **`merging_0`:** 1 xe RL màu magenta, spawn **đầu ramp** (`ramp_waypoints[0]`), `merge_mode = 1`.
3. **3 xe highway RL:** `highway_0/1/2` màu cyan trên **làn dưới cùng**, rải 25%/50%/75% chiều dài đường.

Sau đó `initial_cars_created <- true`.

### 5.3. Species `car` — Mỗi chiếc xe

**Trường quan trọng:**

| Trường | Ý nghĩa |
|--------|---------|
| `speed`, `location`, `heading` | Vật lý (không dùng MovingSkill mặc định) |
| `current_lane_index`, `physical_lane` | Làn logic vs làn vật lý (tránh “ma xe” khi đổi làn) |
| `merge_mode = 1` | Đang trên ramp |
| `is_rl_agent`, `rl_agent_id` | `"merging_0"` hoặc `"highway_0"`... |
| `action_rl` | Hành động từ Python (0–4) |
| `reward_val`, `cumulative_reward` | Thưởng bước hiện tại / tích lũy |
| `terminal_reason` | `"running"`, `"success"`, `"collision"`, `"failed_merge"` |
| `is_done` | Episode của agent này đã xong? |
| `episode_step`, `merge_step` | Số bước / bước merge thành công |

**Reflex chính (thứ tự ý niệm mỗi cycle):**

- `behave` — điều phối: NPC greedy, highway RL, merging RL.
- `rl_merging_behavior` — **chỉ `merging_0`**: đọc action Python, tăng/giảm tốc, thử merge (action 3).
- `die_if_out_of_road` — ra khỏi map → chết, set outcome.
- `compute_nearest_collision` — khoảng cách va chạm gần nhất.

### 5.4. Hành vi nhập làn — `rl_merging_behavior`

Đây là nơi **action 0–4 có nghĩa với xe ramp**:

| Action | Tên | Hành vi trong GAML |
|--------|-----|-------------------|
| 0 | brake | Giảm tốc (`speed - deceleration`) |
| 1 | keep | Giữ tốc; nếu đứng yên thì tăng nhẹ |
| 2 | accel | Tăng tốc |
| 3 | merge | Nếu trong accel zone + `is_merge_gap_safe()` → `execute_merge()`, thưởng lớn |
| 4 | wait | Giảm tốc nhẹ, phạt nhỏ `action_penalty` |

Sau khi chỉnh tốc độ, xe **bám polyline ramp** (`step_along_ramp_polyline`), rồi gọi `calculate_reward()` và `check_merging_terminal_state()`.

**Lưu ý thiết kế:** Có `rl_ramp_creep_min = 0.30` — tốc độ sàn để xe không kẹt ở đầu ramp quá 300 bước (không vào được vùng accel).

### 5.5. Quan sát 15D — `get_merging_state()` (merging_0)

Vector **15 số trong [0, 1]** (thứ tự cố định, khớp `rl/baselines.py`):

| Index | Tên | Ý nghĩa |
|-------|-----|---------|
| 0 | norm_speed | Tốc độ / speed_max |
| 1 | norm_progress | Tiến độ trên vùng accel |
| 2 | norm_dist_to_merge | Còn bao xa điểm merge |
| 3 | norm_lateral | Lệch ngang so với làn đích |
| 4 | norm_in_accel | 1 nếu trong accel zone |
| 5–6 | gap/speed front | Xe trước trên làn đích |
| 7–8 | gap/speed rear | Xe sau trên làn đích |
| 9 | norm_gap_safe | 1 nếu đủ gap để merge |
| 10–11 | ramp front gap/speed | Xe trước trên ramp |
| 12 | norm_urgency | Càng gần cuối accel càng cao |
| 13 | norm_last_action | Action vừa chọn |
| 14 | norm_patience | patience / 100 |

**Highway agents** dùng `get_current_state()` — **cùng 15 chiều nhưng khác nghĩa** (radar 6 hướng + xe merger gần nhất). Shared policy phân biệt nhờ **one-hot agent ID** (phần Python).

### 5.6. Phần thưởng — `calculate_merging_reward` / `calculate_reward`

**Nguồn sự thật duy nhất** — Python không reshape reward.

**merging_0 (tóm tắt):**

| Sự kiện | Reward xấp xỉ |
|---------|----------------|
| Merge thành công (terminal) | +100 (trong nhánh terminal) |
| Va chạm | −100 |
| Hết đường chưa merge | −50 |
| Mỗi bước | −0.02 (phạt thời gian) |
| Gap an toàn | +0.05 |
| Action merge đúng lúc | +1.0 (shaping) |
| Urgency trong accel zone | −urgency × 0.05 |

**highway_0/1/2:** thưởng tốc độ, phạt TTC (bám đuôi), phạt chậm, ± chuyển làn, +0.3 cooperative khi có merge gần, terminal va chạm −100, ra khỏi đường +50.

### 5.7. Cầu nối Python — `petz_collect_tick` + `PzBridgeAgent`

Mỗi cycle, `petz_collect_tick.tick_sync_from_world`:

1. Tìm xe có `rl_agent_id == "merging_0"` ...
2. Gán `pz_observations`, `pz_rewards`, `pz_terminations`, `pz_infos`.
3. Gói vào `pz_data` để socket trả về Python.

Species `PzBridgeAgent` chủ yếu giữ map trống lúc init; logic thật nằm ở **global** `apply_pz_python_each_cycle`.

### 5.8. NPC và mật độ

- Xe xám trên cao tốc: rule-based **Greedy** (car-following, đổi làn).
- `spawn_ramp_cars`: sinh thêm xe ramp NPC mỗi ~25 cycle (có cap).
- `balance_traffic`: duy trì số xe trên đường.
- Xe hỏng: 2 phase (dừng làn → kéo lề → có thể rejoin).

---

## 6. Phần Python — thư mục `rl/`

### 6.1. `config.py` — “Sổ tay hằng số”

- Đường dẫn: `MODEL_PATH`, `OUTPUT_DIR`, `LOG_DIR`, ...
- `MARL_AGENTS = ("merging_0", "highway_0", "highway_1", "highway_2")`
- `MARL_OBS_DIM = 19` = 15 sensor + 4 bit one-hot
- `ACTION_MEANINGS_MERGING` vs `ACTION_MEANINGS_HIGHWAY` — **cùng index 0–4, khác nghĩa**
- `MARL_EXPERIMENT_PRESETS`: `smoke`, `short`, `thesis` (timesteps, seeds, episodes)
- `SCENARIO_PRESETS`: `low` / `medium` / `high` (chỉ metadata trừ khi dùng `scenario_utils`)

### 6.2. `gama_compat.py` — Vá bridge GAMA ↔ Python

**Vì sao tồn tại:** Phiên bản `gama-pettingzoo` + GAMA 2025.6 có bug/race (obs None sau reset, species name `PetzAgent[0]`, v.v.).

**Làm gì:**

- Patch `GamaParallelEnv.reset/step` — bootstrap obs sau reset.
- Patch `GamaClientWrapperPtZ` — gọi `pz_agents`, `pz_data` thay vì species cũ.
- Cache `observation_space` / `action_space` — tránh gọi GAMA lặp gây lỗi.
- UTF-8 console Windows.

**Quy tắc:** Mọi script train/eval gọi `patch_gama_gymnasium()` **trước** import `GamaParallelEnv`.

### 6.3. `marl_env.py` — Pipeline wrapper (quan trọng nhất)

Hàm `make_marl_vec_env()` xây chuỗi:

```
GamaParallelEnv
  → PadPossibleAgentsParallelWrapper   # luôn đủ 4 obs (zero nếu thiếu)
  → AgentIndicatorParallelWrapper      # 15D → 19D (one-hot agent)
  → MarkovVectorEnv(black_death=True)  # gộp 4 agent thành 1 vector env
  → GamaMarkovSB3VecEnv                # SB3 nhận được VecEnv
```

**Vì sao không dùng SuperSuit `agent_indicator_v0` trực tiếp?**  
SuperSuit chuyển Parallel → AEC → Parallel; GAMA đổi tập agent giữa các bước → `KeyError`.

**Vì sao `num_vec_envs=1`?**  
Chỉ có **một** GAMA headless trên socket; không pickle được env để nhân bản.

### 6.4. `train_marl.py` — Huấn luyện

- Load `make_marl_vec_env()`.
- `build_marl_model()`: **PPO** (lr 5e-4, ent_coef 0.20) hoặc **A2C** (lr 1e-3, ent_coef 0.32).
- **Shared policy:** 1 mạng cho cả 4 agent; phân biệt role bằng 4 bit cuối obs.
- `MARLEpisodeCSVCallback`: khi episode `merging_0` kết thúc → ghi 1 dòng CSV; highway ghi file `_highway_episodes.csv`.
- Lưu `outputs/models/marl_{algo}_seed{N}.zip` + TensorBoard.

**DQN không dùng cho MARL** — replay buffer lẫn transition nhiều agent → không hội tụ.

### 6.5. `evaluate_marl.py` — Đánh giá model .zip

- Không dùng full `make_marl_vec_env`; chỉ `AgentIndicatorParallelWrapper` + vòng lặp PettingZoo tay.
- `deterministic=True` + **eval mask** cho `merging_0`: chỉnh xác suất action (ưu tiên merge khi gap safe) — tránh policy “chỉ brake”.
- Seed **ngẫu nhiên lớn** mỗi episode (tránh GAMA seed 1,2,3... giống hệt).
- Xuất CSV eval cho merging + highway.

### 6.6. `baselines.py` — Không học

- `heuristic_action(obs[:15])` — rule Greedy khớp đề cương (merge khi gap safe, urgency cao thì chờ, v.v.).
- Highway baseline: thường action `1` (keep).
- Ghi `greedy_baseline_seed0.csv`.

### 6.7. `run_experiments.py` — Một lệnh chạy hết

Thứ tự:

1. `apply_scenario_all_gaml(scenario)` — vá mật độ vào GAML (+ backup `.gaml.bak`).
2. Baseline greedy (tùy chọn).
3. Với mỗi algo × seed: `train_marl.py` → (thesis) chọn checkpoint tốt → `evaluate_marl.py`.
4. `plots.py` + `analysis.py`.
5. `restore_gaml_backup()` — trả GAML về medium.

**Preset thesis:** 80k steps, 5 seeds, eval stochastic (vì argmax PPO hay collapse).

### 6.8. `scenario_utils.py` — Đổi mật độ giao thông

Regex thay trong GAML:

- `nb_cars_max <- 20/45/70`
- `spawn_ramp_tick >= N`

**Cảnh báo:** Chỉ đổi `--scenario` trong `train_marl.py` **không** đổi GAML — chỉ ghi nhãn CSV. Phải `run_experiments` hoặc vá tay + **restart GAMA**.

### 6.9. `metrics.py` — Định dạng CSV

`EpisodeMetric` mỗi dòng: algorithm, seed, episode, reward, length, outcome, success, collision, merge_step, throughput, shockwave_index, ...

`classify_episode(outcome)` — thống nhất terminated vs truncated cho báo cáo.

### 6.10. `plots.py` + `analysis.py`

- **plots:** 12 PNG (reward curve, success rate, collision, radar chart, ...).
- **analysis:** `comparison_table.csv`, `latex_table.tex`, learning efficiency.

Phân loại CSV theo `phase`: `eval` (KPI thật) vs `training` (rollout khi train, có thể nhiễu).

### 6.11. Script phụ trợ

| Script | Khi nào dùng |
|--------|----------------|
| `smoke_test_env.py` | Sau khi bật `gama-headless -socket 1001` |
| `diagnose_marl.py` | In từng bước obs + action để debug |
| `check_eval_results.py` | Tự động kiểm success rate có quá thấp không |
| `play_marl_gui.py` | Xem xe chạy trên UI GAMA |
| `model_registry.py` | Liệt kê file .zip trong outputs |

### 6.12. `gama_episode_reset.py`

Sau `reset()`, GAMA cần vài chục bước noop để `bootstrap_initial_cars` tạo đủ 4 RL agents. Hàm `reset_marl_episode()` lặp `step({agent: 1})` tối đa 40 lần.

---

## 7. Bài toán RL (MDP) — quan sát, hành động, thưởng

### 7.1. Agents

| ID | Vai trò | Màu |
|----|---------|-----|
| merging_0 | Nhập làn từ ramp | Magenta |
| highway_0/1/2 | Xe cao tốc làn dưới, có thể nhường/đổi làn | Cyan |

### 7.2. Observation 19D (đầu vào mạng neural)

```
[ 15 số từ GAMA (khác nghĩa merging vs highway) | 4 bit one-hot agent ]
```

One-hot ví dụ: `merging_0` → `[1,0,0,0]`, `highway_1` → `[0,1,0,0]`.

### 7.3. Action chung Discrete(5) — **nghĩa khác nhau theo role**

**merging_0:**

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Giảm tốc | Giữ | Tăng tốc | Merge | Chờ |

**highway:**

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Tăng tốc | Giữ | Giảm tốc | Sang trái | Sang phải |

→ **Một policy shared** học 2 “ngôn ngữ” hành động; đây là điểm cần nêu trong luận văn (IPPO + parameter sharing).

### 7.4. Episode kết thúc khi nào?

| Agent | Outcome | Điều kiện |
|-------|---------|-----------|
| merging_0 | success | Merge OK |
| merging_0 | collision | Va chạm |
| merging_0 | failed_merge | Hết đường chưa merge |
| merging_0 | timeout | Quá `max_episode_steps` |
| highway_i | exited | Chạy hết đường (~x ≥ 195m) |
| highway_i | collision | Va chạm |

---

## 8. Thuật toán và cách train

| Thuật toán | Loại | MARL? | Ghi chú |
|------------|------|-------|---------|
| Greedy | Rule | Baseline | `baselines.py` |
| Random | Ngẫu nhiên | Baseline | Ít dùng trong báo cáo |
| PPO | On-policy | ✅ IPPO | Shared policy, ổn định hơn A2C |
| A2C | On-policy | ✅ IA2C | Cần entropy cao để tránh collapse |
| DQN | Off-policy | ❌ | Chỉ trong `rl/legacy/` single-agent |

**IPPO/IA2C ở đây** = Independent MARL + **cùng một bộ trọng số** (không phải MADDPG, không centralized critic).

---

## 9. Kết quả thực nghiệm — CSV, biểu đồ, bảng

```
outputs/
├── models/     marl_ppo_seed0_80k.zip  ...
├── logs/       *_episodes.csv, *_eval.csv, tensorboard/
└── plots/      thesis/*.png, comparison_table.csv, latex_table.tex
```

**Metrics chính cần nhớ khi báo cáo:**

| Metric | Ý nghĩa đơn giản |
|--------|------------------|
| success_rate | % episode merge thành công |
| collision_rate | % va chạm |
| failed_merge_rate | % hết đường chưa kịp merge |
| merge_step | Bước nào merge thành công (càng thấp càng nhanh) |
| throughput | merge thành công / số lần thử ramp (cả NPC) |
| shockwave_index | Độ dao động tốc độ vùng merge (càng thấp càng êm) |

---

## 10. Chạy dự án từ đầu

### Bước 0 — Cài đặt

```powershell
cd D:\DoAn_RL\GAMA\temp\AutonomousVehicleRL_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Cần **GAMA Platform** (có `gama-headless.bat`).

### Bước 1 — Terminal A: GAMA

```powershell
.\gama-headless.bat -socket 1001
```

Giữ cửa sổ này mở suốt khi train.

### Bước 2 — Terminal B: Kiểm tra

```powershell
python rl/smoke_test_env.py
```

Kỳ vọng: `[PASS] MARL smoke test OK`.

### Bước 3 — Train / thực nghiệm

```powershell
# Nhanh
python rl/run_experiments.py --preset short --port 1001

# Đầy đủ (lâu)
python rl/run_experiments.py --preset thesis --port 1001
```

Hoặc tay:

```powershell
python rl/train_marl.py --algo ppo --timesteps 200000 --seed 0
python rl/evaluate_marl.py --algo ppo --model outputs/models/marl_ppo_seed0.zip --episodes 50
```

### Bước 4 — Biểu đồ

```powershell
python rl/plots.py outputs/logs/*.csv --out-dir outputs/plots/my_run
python rl/analysis.py outputs/logs/*.csv --out-dir outputs/plots/my_run
```

### Xem học realtime

```powershell
tensorboard --logdir outputs/logs/tensorboard
```

---

## 11. Phần archive (single-agent)

| Thành phần | Khác MARL chính |
|------------|-----------------|
| `models/single_agent/Main_Traffic_SingleAgent.gaml` | Chỉ `merging_0` học; highway là NPC |
| `rl/legacy/train_single.py` | DQN / PPO / A2C 1 agent |
| Experiment | `TrafficSingleHeadless` |

Pipeline đồ án **chính** dùng MARL 4 agents — legacy để so sánh / tham khảo đề cương cũ.

---

## 12. Bẫy thường gặp khi đọc code

1. **Đổi `--scenario medium` nhưng quên restart GAMA** → mật độ thật vẫn là file cũ trong RAM.
2. **Nhầm obs 15D merging với highway** — cùng shape, khác semantics từng chiều.
3. **Nhầm action 0** — merging = brake, highway = accelerate.
4. **Reward trong Python** — không có; sửa reward phải vào GAML `calculate_*_reward`.
5. **CSV training vs eval** — đừng gộp `*_episodes.csv` (train) với `*_eval.csv` khi báo cáo KPI cuối.
6. **Policy collapse** — PPO argmax toàn action 1 (keep); thesis eval dùng **stochastic** + entropy cao.
7. **Throughput** — đếm cả NPC ramp, không chỉ `merging_0`.
8. **File GAML 4900 dòng** — đừng sợ; tìm theo tên hàm (`get_merging_state`, `rl_merging_behavior`).

---

## 13. Bản đồ “muốn sửa X thì mở file nào”

| Muốn thay đổi | File |
|---------------|------|
| Tốc độ xe, khoảng cách merge, điểm merge | `Main_Traffic.gaml` → `global` |
| Cách action 3 merge hoạt động | `rl_merging_behavior`, `execute_merge` |
| Hệ số reward (+100, −0.02, ...) | `calculate_merging_reward`, `calculate_reward` |
| Vector quan sát 15 chiều | `get_merging_state`, `get_current_state` |
| Số xe / mật độ | `nb_cars_max`, `spawn_ramp_*` hoặc `scenario_utils.py` |
| Hyperparameter PPO/A2C | `train_marl.py` → `build_marl_model` |
| Thêm wrapper env | `marl_env.py` |
| Lỗi socket / obs None | `gama_compat.py` |
| Baseline Greedy | `baselines.py` + `heuristic_action` |
| Biểu đồ báo cáo | `plots.py`, `analysis.py` |
| Preset thực nghiệm | `config.py` → `MARL_EXPERIMENT_PRESETS` |
| Tự động hóa cả pipeline | `run_experiments.py` |

---

## Tóm tắt một câu

**GAML mô phỏng đường + xe và trả obs/reward; Python bọc thành môi trường 4 agent (19D), train shared PPO/A2C, rồi xuất CSV/plot để so sánh với Greedy — toàn bộ đồ án xoay quanh việc `merging_0` học nhập làn an toàn trong khi `highway_*` học phối hợp nhường đường.**

---

*Tài liệu sinh tự động từ mã nguồn repo — cập nhật khi cấu trúc project thay đổi.*
