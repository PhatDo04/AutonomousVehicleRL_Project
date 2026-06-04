# BÁO CÁO CHI TIẾT MÃ NGUỒN

**Đề tài:** Phát triển kịch bản nhập làn trên đường cao tốc và đánh giá các thuật toán học tăng cường đa tác tử cho điều khiển phương tiện tự hành.
**Sinh viên:** Đỗ Tiến Phát — 2251172447 — ĐH Thủy Lợi, 64KTPM1
**GVHD:** TS. Lê Nguyễn Tuấn Thành

> Tài liệu này mô tả **chính xác những gì mã nguồn đang có**: kiến trúc, từng file, MDP, luồng dữ liệu, và một bảng thuật ngữ chuyên ngành. Người đọc cần nền tảng kỹ thuật cơ bản. Bản dành cho người mới hoàn toàn: xem [GIAI_THICH_CHO_NGUOI_MOI.md](GIAI_THICH_CHO_NGUOI_MOI.md).

---

## 1. Bảng thuật ngữ chuyên ngành (Glossary)

| Thuật ngữ | Viết tắt | Giải thích |
|---|---|---|
| **Reinforcement Learning** | RL | Học tăng cường: tác tử (agent) học bằng cách thử–sai, nhận **phần thưởng** (reward) cho hành động tốt, dần tối ưu chính sách. |
| **Multi-Agent RL** | MARL | RL với **nhiều tác tử** cùng học trong một môi trường, ảnh hưởng lẫn nhau. |
| **Markov Decision Process** | MDP | Khung toán mô hình hóa bài toán RL: gồm **State** (trạng thái), **Action** (hành động), **Reward** (thưởng), và hàm chuyển trạng thái. |
| **State / Observation** | — | Những gì agent "nhìn thấy" tại một thời điểm (ở đây là vector 15 số mô tả tốc độ, khoảng cách xe xung quanh…). |
| **Action** | — | Hành động agent chọn (ở đây là 1 trong 5: tăng/giảm/giữ tốc, nhập làn/chờ…). |
| **Reward** | — | Điểm số phản hồi sau mỗi bước (vd nhập làn an toàn +200, va chạm −100). |
| **Policy** | π | "Chính sách" — hàm ánh xạ State → Action mà agent học được (chính là mạng nơ-ron). |
| **Episode** | — | Một "ván" từ lúc bắt đầu đến khi kết thúc (xe nhập làn thành công / va chạm / hết đường / hết giờ). |
| **On-ramp merging** | — | Kịch bản **nhập làn**: xe từ nhánh phụ (ramp) hòa vào dòng xe cao tốc — bài toán cốt lõi của đề tài. |
| **PPO** | Proximal Policy Optimization | Thuật toán RL **on-policy** hiện đại, ổn định, dùng "clip" để giới hạn bước cập nhật. Phiên bản đa tác tử gọi là **MAPPO**. |
| **A2C** | Advantage Actor-Critic | Thuật toán RL on-policy đơn giản hơn PPO. Phiên bản đa tác tử gọi là **MAA2C**. |
| **DQN** | Deep Q-Network | Thuật toán **off-policy** (dùng replay buffer). **Bị loại** khỏi đề tài vì trong môi trường đa tác tử non-stationary, buffer trộn dữ liệu nhiều agent làm Q-value không hội tụ. |
| **On-policy / Off-policy** | — | On-policy: học trên dữ liệu vừa sinh bởi chính chính sách hiện tại (PPO/A2C). Off-policy: học lại từ dữ liệu cũ trong bộ nhớ (DQN). |
| **Actor / Critic** | — | **Actor** = mạng chọn hành động (policy). **Critic** = mạng ước lượng "giá trị" trạng thái để hướng dẫn Actor học. |
| **CTDE** | Centralized Training, Decentralized Execution | Huấn luyện **tập trung** (critic thấy trạng thái toàn cục của mọi agent) nhưng thực thi **phi tập trung** (actor mỗi agent chỉ đọc quan sát cục bộ của nó). Khử **non-stationarity**. |
| **Non-stationarity** | — | Trong MARL, khi các agent khác cũng đang học/đổi hành vi, "môi trường" dưới góc nhìn của 1 agent liên tục thay đổi → khó hội tụ. CTDE giúp giảm vấn đề này. |
| **Parameter sharing** | — | Mọi agent dùng **chung một mạng nơ-ron**; thêm "agent indicator" (one-hot ID) vào quan sát để mạng phân biệt vai trò. |
| **Agent indicator** | — | Vector one-hot gắn vào quan sát để cho biết đây là agent nào (merging_0 hay highway_0/1/2). |
| **Shockwave (sóng lùi)** | — | Hiện tượng "sóng phanh" lan ngược dòng xe gây ùn tắc. Đo bằng **shockwave_index** = độ lệch chuẩn / trung bình tốc độ xe trên cao tốc (hệ số biến thiên CV). Thấp = dòng chảy mượt. |
| **Throughput (thông lượng)** | — | Tỷ lệ xe ramp nhập làn thành công / tổng số lần thử nhập. |
| **Success rate** | — | Tỷ lệ episode mà xe nhập làn (merging_0) nhập làn an toàn và chạy hết đường. **Chỉ số chính** của báo cáo. |
| **GAMA Platform** | — | Phần mềm mô phỏng **dựa trên tác tử** (agent-based). Ngôn ngữ mô hình hóa là **GAML**. Đóng vai trò "môi trường" cho RL. |
| **PettingZoo** | — | Thư viện chuẩn API cho môi trường **MARL** (giống Gymnasium nhưng nhiều agent). Cầu nối GAMA ↔ Python. |
| **Stable-Baselines3** | SB3 | Thư viện Python (trên PyTorch) cung cấp sẵn PPO/A2C đã kiểm thử + TensorBoard + VecEnv. |
| **VecEnv** | — | "Vectorized environment" — giao diện SB3 cho phép xử lý nhiều môi trường/agent song song trong một lần `step`. |
| **Headless** | — | Chạy mô phỏng **không giao diện đồ họa** (qua socket) để huấn luyện tự động nhanh. |
| **Checkpoint** | — | Ảnh chụp mô hình **trong lúc** train (mốc 40k/80k/…/200k step), để chọn ra bản tốt nhất thay vì dùng bản cuối. |
| **GAE** | Generalized Advantage Estimation | Kỹ thuật ước lượng "advantage" (lợi thế của hành động) mượt hơn, dùng tham số `gae_lambda`. |
| **Entropy coefficient** | ent_coef | Hệ số khuyến khích agent "khám phá" (chọn đa dạng hành động), tránh kẹt vào thói quen sớm. |
| **Orthogonal init** | — | Cách khởi tạo trọng số mạng (gain=√2) chuẩn của MAPPO, giúp train ổn định. |

---

## 2. Tổng quan & kiến trúc

Hệ thống gồm **2 nửa** nối với nhau qua socket:

```
┌──────────────────────────────────────────────────────────┐
│  GAMA Platform — models/Main_Traffic.gaml  (MÔI TRƯỜNG)   │
│  • Đường cao tốc 3 làn (200m) + nhánh nhập làn (ramp)      │
│  • 4 RL agent: merging_0 (xe nhập làn) + highway_0/1/2     │
│  • Xe nền NPC rule-based (greedy) + hệ thống xe hỏng       │
│  • Tính observation(15D) / reward / terminal cho mỗi agent │
└───────────────────────────┬──────────────────────────────┘
                            │ PettingZoo bridge (socket :1001)
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Python RL Pipeline  (BỘ NÃO HỌC)                          │
│  marl_env.py:  GAMA → wrappers → SB3 VecEnv                │
│  centralized_policy.py:  Actor(local) + Critic(global) CTDE│
│  train_marl.py:  PPO/A2C  +  checkpoint                    │
│  evaluate_marl.py:  đánh giá + histogram action            │
│  baselines.py:  Greedy/Random để so sánh                  │
│  run_experiments.py:  điều phối toàn bộ pipeline            │
│  analysis.py + plots.py:  bảng + 12 biểu đồ                │
└──────────────────────────────────────────────────────────┘
```

**Vì sao chọn các công cụ này:**
- **GAMA thay NetLogo**: hỗ trợ MARL native qua PettingZoo, headless ổn định cho train tự động.
- **SB3 thay TensorFlow/Keras**: PPO/A2C đã kiểm thử, tích hợp TensorBoard/VecEnv.
- **MAPPO/MAA2C thay DQN**: on-policy phù hợp đa tác tử; DQN off-policy không hội tụ do non-stationarity.

---

## 3. Bài toán MARL — định nghĩa MDP

### 3.1. Các tác tử (agents)

| Agent | Vai trò | Mục tiêu |
|---|---|---|
| `merging_0` | Xe nhập làn (trên ramp) | Nhập vào dòng cao tốc **an toàn**, không va chạm, không hết đường |
| `highway_0/1/2` | 3 xe cao tốc làn dưới | Giữ tốc độ + **nhường khoảng trống** cho xe ramp |

**Chiến lược học:** Shared policy + agent indicator (parameter sharing) — 1 mạng dùng chung, phân biệt vai trò bằng one-hot ID 4 chiều.

### 3.2. Không gian quan sát (Observation)

- **Local actor obs**: `19D` = `15D` sensor gốc + `4D` one-hot agent ID.
- **Global critic state**: `60D` = nối `15D` base của cả 4 agent (thứ tự cố định).
- **Obs đưa vào model SB3**: `79D` = `[local 19D | global 60D]`.

**15D của `merging_0`** (nguồn: `get_merging_state` trong GAML):

| # | Tên | Ý nghĩa |
|---|---|---|
| 0 | norm_speed | Tốc độ xe nhập làn |
| 1 | norm_progress | Tiến độ trên ramp |
| 2 | norm_dist_to_merge | Khoảng cách đến điểm merge |
| 3 | norm_lateral | Lệch ngang so với làn mục tiêu |
| 4 | in_accel_zone | Có trong vùng tăng tốc không (0/1) |
| 5–6 | gap/speed_front | Khoảng cách & tốc độ xe **trước** trên làn mục tiêu |
| 7–8 | gap/speed_rear | Khoảng cách & tốc độ xe **sau** |
| 9 | gap_safe | Khoảng trống đủ an toàn để merge (0/1) |
| 10–11 | ramp_front_gap/speed | Xe trước **trên ramp** |
| 12 | urgency | Độ khẩn (tăng dần về cuối zone) |
| 13 | last_action | Hành động trước đó |
| 14 | patience | Chỉ số kiên nhẫn tích lũy |

**15D của `highway_0/1/2`** (nguồn: `get_current_state`): tốc độ, làn hiện tại, và radar 6 hướng (trước/trái/phải × khoảng cách+tốc độ) + thông tin xe ramp đang merge.

### 3.3. Không gian hành động — `Discrete(5)` (ý nghĩa KHÁC theo vai trò)

| Action | `merging_0` | `highway_0/1/2` |
|---|---|---|
| 0 | Giảm tốc | **Tăng tốc** |
| 1 | Giữ tốc | Giữ tốc |
| 2 | Tăng tốc | **Giảm tốc** |
| 3 | Nhập làn (chỉ khi trong zone + gap an toàn) | Rẽ trái (lên làn trên) |
| 4 | Chờ | Rẽ phải (xuống làn dưới) |

> ⚠️ Cùng chỉ số action nhưng **ngữ nghĩa ngược nhau** giữa 2 loại agent (action 0/2). Mạng phân biệt nhờ one-hot ID. Đây là điểm thiết kế cần nêu trong phần Discussion.

### 3.4. Hàm phần thưởng (nguồn sự thật: GAML)

**`merging_0`** (`calculate_merging_reward` + terminal):

| Thành phần | Giá trị |
|---|---|
| Terminal nhập làn thành công (chạy gần hết mainline) | **+200** |
| Terminal va chạm | **−100** |
| Terminal hết đường chưa merge | **−50** |
| Base mỗi tick | −0.01 + speed×0.10 |
| Vào accel zone (một lần) | +15.0 |
| Trong zone (mỗi tick) | +0.05 (+0.05 nếu gap an toàn) |
| Action 3 + zone + gap an toàn | +25 → +50 (theo urgency) |
| Action 3 + zone + gap không an toàn | −2.0 |
| Action 3 ngoài zone | −0.5 |
| Action 0/1/4 trong zone | −2.0 / −0.5 / −3.0 |
| Giới hạn (cap) | [−60, 60] |

**`highway_0/1/2`** (`calculate_reward`):

| Thành phần | Giá trị |
|---|---|
| Terminal thoát đường an toàn | +50 |
| Terminal va chạm | −100 |
| Base mỗi tick | speed×0.04 + action_penalty |
| Tốc độ quá thấp (<0.2×max) | −0.05 |
| TTC penalty (gap<20m) | −((20−gap)/20)×speed×1.2 |
| Action 0 (tăng tốc) khi gap<12m | −0.5 |
| Cooperative bonus (gần điểm merge thành công) | +0.3 |

### 3.5. Điều kiện kết thúc (Terminal)

- `merging_0`: **success** (merge xong + chạy gần hết đường), **collision** (va chạm), **failed_merge** (hết ramp/quá điểm merge chưa nhập), **timeout** (vượt max step).
- `highway_0/1/2`: **exited** (ra khỏi đường an toàn), **collision**, **timeout**.

---

## 4. Thành phần MÔI TRƯỜNG — `models/Main_Traffic.gaml`

File GAML (~4.5k dòng) định nghĩa thế giới mô phỏng. Các phần chính:

### 4.1. Tham số & hình học
- Đường: `number_of_lanes=3`, `lane_width=3.5m`, `road_length=200m`.
- Ramp: `ramp_waypoints` (8 điểm đường cong), vùng tăng tốc `accel_start_x=48` → `accel_end_x=162`, điểm merge `merge_x=180`.
- Động học: `speed_max=1.0`, `acceleration=0.05`, `deceleration=0.1`, `collision_distance=3.5`.
- Mật độ: `nb_cars_max` (20 ở scenario `low`, 45 medium, 70 high).

### 4.2. Các species (loại agent GAML)
- **`car`** (phần lõi): mỗi chiếc xe. Thuộc tính quan trọng: `rl_agent_id`, `is_merging`, `merge_mode` (1=ramp, 0=mainline), `in_accel_zone`, `action_rl`, `reward_val`, `cumulative_reward`, `terminal_reason`, cùng cache observation (gap/speed các hướng). Có hệ thống **xe hỏng 2 pha** (dừng trong làn → dạt vào lề) tạo nút thắt để quan sát.
- **`PzBridgeAgent`**: giữ các map `observations/rewards/terminations/infos/action_spaces/observation_spaces` cho socket; init bootstrap 4 agent + spec (Box 15D, Discrete 5).
- **`petz_collect_tick`**: mỗi cycle gọi `tick_sync_from_world()` — quét 4 agent, lấy observation (`get_merging_state`/`get_current_state`) + reward + terminal → đẩy lên socket.
- **`petz_apply_tick`** / **`road_segment`**: phụ trợ (nạp action đã chuyển sang global reflex; vẽ làn).

### 4.3. Vòng đồng bộ mỗi cycle (GAMA ↔ Python)
1. `apply_pz_python_each_cycle` (global reflex): đọc `pz_actions` từ Python → gán `action_rl` cho từng xe RL.
2. `reflex behave` (mỗi xe): thực thi action → đổi tốc độ/làn/vị trí, quét va chạm, tính reward.
3. `petz_collect_tick.push_outputs`: thu observation/reward/terminal → đẩy lên socket cho Python.
4. Python đọc obs → policy chọn action → ghi `pz_actions` → quay lại bước 1.

### 4.4. Reflex quan trọng
`bootstrap_initial_cars` (spawn 4 RL agent + NPC), `spawn_ramp_cars` (NPC ramp định kỳ), `balance_traffic` (giữ mật độ mainline), `respawn_rl_merging_agent` / `respawn_highway_rl_agents` (hồi sinh agent khi chết), `sample_shockwave` (đo CV tốc độ bằng thuật toán Welford), `detect_gama_sim_reset` (phát hiện reset episode).

### 4.5. Experiment
- **`TrafficMARLHeadless`**: dùng cho train/eval headless qua socket (tắt parallel để ổn định) — pipeline chính.
- **`TrafficSimulation`**: GUI demo có dashboard trực quan (chọn policy Python/Heuristic/Manual).

---

## 5. Thành phần BỘ NÃO HỌC — thư mục `rl/`

| File | Vai trò |
|---|---|
| **`config.py`** | Hằng số dùng chung: đường dẫn, `MARL_AGENTS`, preset (`smoke`/`short`/`thesis`), `SCENARIO_PRESETS` (low/medium/high), bảng `ACTION_MEANINGS_*`, `GamaConnectionConfig`. |
| **`marl_env.py`** | Chuỗi wrapper biến GAMA PettingZoo → SB3 VecEnv: `PadPossibleAgents` (giữ đủ 4 slot) → `AgentIndicator` (15D→19D) → `GlobalState` (19D→79D, ghép critic) → `MarkovVectorEnv` → `GamaMarkovSB3VecEnv`. Chứa **reset-on-merging-death** (khi merging_0 chết thì reset cả env, khớp pipeline eval). |
| **`centralized_policy.py`** | `CentralizedCriticPolicy` (CTDE): `CentralizedCriticExtractor` tách obs 79D → actor đọc 19D local, critic đọc 60D global. MLP 64, Tanh, orthogonal init. Dùng chung cho cả PPO & A2C. |
| **`train_marl.py`** | Huấn luyện: tạo model PPO/A2C **mới** (không load lại), callback ghi CSV metrics theo episode, lưu checkpoint định kỳ. **Mỗi lần train là độc lập từ đầu**. |
| **`evaluate_marl.py`** | Đánh giá model đã train: chạy N episode (deterministic hoặc stochastic), histogram action, ghi CSV eval cho merging_0 + highway. Hỗ trợ `--watch-ui` để demo trên GAMA GUI. |
| **`baselines.py`** | Chính sách **không học** để so sánh: `greedy` (tham lam — tăng tốc + merge sớm + vượt làn) và `random` (sàn). *(Bản `base_rule`/`heuristic` thận trọng đã bỏ khỏi Python — demo GUI Heuristic dùng luật phía GAML.)* |
| **`metrics.py`** | `EpisodeMetric` (dataclass 1 dòng CSV), `classify_episode`, `build_episode_metric`, `write_episode_metrics`. 4 cờ outcome (success/collision/timeout/failed_merge) suy TỪ `outcome` (nhất quán eval/train). Module **thuần stdlib**. |
| **`scenario_utils.py`** | Vá GAML theo scenario (regex thay `nb_cars_max`, `spawn_ramp_tick`) + `set_safety_shield(bool)` bật/tắt cờ `enable_safety_shield` cho A/B khiên; khôi phục từ `.bak` sau khi xong. |
| **`run_experiments.py`** | **Điều phối toàn pipeline**: baseline (greedy+random) → train (PPO+A2C × nhiều seed) → select-best-checkpoint → eval → plots → analysis. Cờ `--scenario low/medium/high`, `--shield on/off`, `--run-tag` (tách output). |
| **`analysis.py`** | Sinh bảng so sánh (mean±std, CI95) + bảng LaTeX + learning efficiency. |
| **`plots.py`** | Sinh 12 biểu đồ (reward curve, success/collision/failed_merge rate, radar, boxplot, shockwave, throughput…). |
| **`gama_compat.py`** | Lớp tương thích (monkey-patch) cho gama-pettingzoo/gama-gymnasium: vá race condition socket, cache space, bootstrap sau reset, ép UTF-8 trên Windows. |
| **`gama_episode_reset.py`** | `reset_marl_episode`: reset env + gửi noop step tới khi cả 4 agent sẵn sàng (bootstrap GAMA). Kèm `_set_sim_seed` đẩy `pz_sim_seed` xuống GAML → traffic biến thiên theo hạt giống, tái lập + paired (xem §7). Dùng trong eval/baseline. |

**Công cụ standalone (chạy tay, không thuộc pipeline chính):**
- `smoke_test_env.py`: kiểm tra nhanh kết nối GAMA + env trước khi train dài.
- `diagnose_marl.py`: chạy 1 episode với 3 policy (greedy/random/model), in trace từng bước để debug.
- `model_registry.py`: quét thư mục model `.zip` → sinh `registry.json` (metadata suy từ tên file).

---

## 6. Thuật toán & siêu tham số

| Hyperparameter | MAPPO (PPO) | MAA2C (A2C) |
|---|---|---|
| Policy | `CentralizedCriticPolicy` (CTDE) | `CentralizedCriticPolicy` (CTDE) |
| Learning rate | 3e-4 | 3e-4 |
| `n_steps` | 256 | 64 |
| `batch_size` | 128 | (full rollout) |
| `gamma` | 0.99 | 0.99 |
| `gae_lambda` | 0.95 | 0.95 |
| `clip_range` | 0.2 | n/a |
| `ent_coef` | 0.01 | 0.05 |
| Network / activation / init | 64 / Tanh / Orthogonal(√2) | nt |

---

## 7. Quy trình thực nghiệm

```
run_experiments.py --preset thesis --scenario <low|medium|high> --shield <on|off>
   │
   ├─ apply_scenario_to_gaml(scenario)   # vá mật độ vào GAML
   ├─ [nếu --shield off] set_safety_shield(false)   # tắt khiên cho cả run (A/B)
   ├─ baseline greedy + random (50 ep)   # mốc so sánh
   ├─ với mỗi (algo ∈ {ppo,a2c}, seed ∈ {0..4}):
   │     ├─ train 200k step (+ checkpoint 40k/80k/120k/160k/200k)
   │     ├─ select_best_checkpoint       # eval từng mốc, chọn điểm cao nhất
   │     └─ evaluate 50 ep (stochastic)  # số liệu báo cáo
   ├─ plots.py    → biểu đồ
   ├─ analysis.py → comparison_table.csv + latex_table.tex
   └─ restore GAML từ .bak (khôi phục shield + mật độ mặc định)
```

**Preset:** `smoke` (400 step, sanity), `short` (1 seed × 200k), `thesis` (5 seed × 200k × 50 ep eval — bộ số chính).

**Tách output theo lần chạy:** mặc định mỗi lần chạy tạo thư mục `<preset>_<timestamp>/` cho `models/`, `logs/`, `plots/<preset>/` → không đè lần trước. Dùng `--run-tag none` để ghi phẳng.

**Phát hiện kỹ thuật then chốt — reset-on-merging-death:** pipeline train ban đầu KHÔNG reset env khi `merging_0` chết (do `MarkovVectorEnv(black_death=True)` che tín hiệu done) → rollout buffer tràn transition của agent chết → gradient nhiễu. Fix trong `marl_env.py` ép reset toàn bộ khi merging_0 chết, khớp pipeline eval → cả MAPPO/MAA2C đạt kết quả cao + ổn định.

**select_best_checkpoint:** model bước cuối (200k) thường KHÔNG phải tốt nhất (có thể "sập" muộn). Pipeline eval cả 5 checkpoint, chọn điểm cao nhất (`success − collision − timeout − 0.5×hw_collision`) copy thành model chính.

**Khiên an toàn (gate `is_merge_gap_safe` + shield car-following M3c) + cờ A/B `enable_safety_shield`:** hai cơ chế ghi đè hành động chống va chạm, áp đồng đều mọi xe (RL + baseline). `--shield off` vá `enable_safety_shield ← false` cho cả run (chỉ headless; GUI Heuristic luôn giữ khiên nhờ điều kiện `dashboard_policy="Python/SB3 model"`). Đây là công cụ tách bạch "năng lực nội tại" với "trợ giúp môi trường" (xem kết quả §8).

**Cải tiến seed traffic (`pz_sim_seed`):** wrapper gama_gymnasium đặt `seed` ở experiment-scope SAU reload → KHÔNG gieo lại RNG simulation đang chạy → mọi episode cùng một traffic. Fix: Python (`_set_sim_seed`) đẩy hạt giống xuống biến GAML `pz_sim_seed` qua `_execute_expression` ngay sau reset; reflex `bootstrap_initial_cars` gieo lại `seed ← pz_sim_seed` trước khi spawn → traffic biến thiên theo episode, **tái lập + paired** (cùng seed → cùng tình huống cho mọi policy). Nhờ đó eval baseline (tất định) có ý nghĩa thống kê.

---

## 8. Cấu trúc output & kết quả

```
outputs/
├── models/<tag>/            # 10 model best (.zip): ppo+a2c × seed0-4
│   └── checkpoints/<tag>/<run>/   # các mốc trung gian 40k..200k
├── logs/<tag>/              # CSV: baseline, train episodes, eval (+highway)
└── plots/thesis/<tag>/      # 12 PNG + comparison_table.csv + latex_table.tex
```

**Kết quả — ma trận 3 mật độ × 2 khiên (thesis, 5 seeds × 50 ep stochastic).** Định dạng ô `khiên ON | khiên OFF`; RL ghi mean±std.

SUCCESS %:

| Mật độ | MAA2C | MAPPO | Greedy | Random |
|---|---|---|---|---|
| low | 86.4±3.9 \| 87.2±4.1 | 83.6±11.8 \| 75.2±22.8 | 96 \| 82 | 0 |
| medium | 89.2±1.6 \| 84.8±3.0 | 83.6±8.6 \| 76.8±17.9 | 96 \| 2 | 0 |
| high | 82.8±4.3 \| 85.2±5.2 | 68.4±7.1 \| 69.6±9.4 | 74 \| 0 | 0 |

COLLISION %:

| Mật độ | MAA2C | MAPPO | Greedy | Random |
|---|---|---|---|---|
| low | 13.6 \| 12.8 | 16.0 \| 21.6 | 4 \| 18 | 100 |
| medium | 10.8 \| 15.2 | 16.0 \| 22.4 | 0 \| 98 | 100 |
| high | 16.4 \| 14.8 | 31.2 \| 30.4 | 22 \| 100 | 100 |

> **Đọc kết quả:** (1) **MAA2C** cao + ổn định nhất mọi cấu hình (std 1.6–5.2), vượt MAPPO (std nổ tới ±23 khi không khiên). (2) **A/B khiên** là phát hiện chính: Greedy có khiên 96% nhưng gỡ khiên sụp theo mật độ (82→2→0%) → "năng lực" của nó là vay mượn từ lưới an toàn; RL nội hóa an toàn nên bền (MAA2C 86→87, 89→85, 83→85%). (3) Cùng điều kiện không-khiên, RL thắng Greedy áp đảo (medium 85% vs 2%). Eval **stochastic** (sampling) cho số thực tế có phương sai; deterministic (argmax) cho số cao hơn nhưng dễ ảo.

---

## 9. Phạm vi & hạn chế (cho Discussion)

- **Mô phỏng giản lược**: action rời rạc + waypoint, không phải động học xe thật.
- **Heterogeneous agents + shared policy**: cùng obs/action space nhưng khác ngữ nghĩa — lựa chọn thiết kế hợp lệ, cần nêu ưu/nhược.
- **Một instance GAMA** (`num_vec_envs=1`): wall-clock chậm hơn pipeline thuần CPU/GPU; bù lại dynamics nhất quán.
- **reset-on-merging-death** đổi semantic train so với mặc định SuperSuit — cần nêu khi so với MARL baseline khác.
- **A/B khiên chỉ tắt 2 cơ chế chính** (gate nhập làn + shield car-following). Các cơ chế chống-kẹt/hồi-sinh khác (anti-deadlock, respawn, kick-start) luôn bật cho mọi policy → đảm bảo công bằng nhưng không phải môi trường "trần trụi" hoàn toàn.
- **Seed traffic chỉ áp cho eval/baseline** (qua `reset_marl_episode`); vòng train RL (SB3 VecEnv) vẫn dùng traffic cố định mỗi reset → train/eval có chênh nhẹ về phân bố traffic.
- **Reproducibility một phần**: hạt giống điều khiển traffic (eval) + policy (PyTorch); RNG nội bộ GAMA có thể khác giữa máy.

---

*Tài liệu này phản ánh trạng thái mã nguồn tại thời điểm viết. Bản giải thích cho người mới: [GIAI_THICH_CHO_NGUOI_MOI.md](GIAI_THICH_CHO_NGUOI_MOI.md). Tổng quan vận hành: [README.md](README.md).*
