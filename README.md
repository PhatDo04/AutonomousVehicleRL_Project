# Học Tăng Cường Đa Tác Tử cho Bài Toán Nhập Làn Đường Cao Tốc

> Mô phỏng kịch bản **nhập làn (on-ramp merging)** trên đường cao tốc bằng **GAMA Platform** và huấn luyện các tác tử xe tự hành **tự học cách nhập làn an toàn & phối hợp nhường đường** bằng học tăng cường đa tác tử (**MAPPO** và **MAA2C**).

**Sinh viên:** Đỗ Tiến Phát — 2251172447 — ĐH Thủy Lợi, Khoa CNTT, Lớp 64KTPM1
**GVHD:** TS. Lê Nguyễn Tuấn Thành
**Tham chiếu:** Le Nguyen Tuan Thanh (2023). *Multi-agent reinforcement learning for traffic congestion on one-way multi-lane highways*. Journal of Information and Telecommunication, 7:3, 255–269.

---

## Tổng quan

Đồ án xây dựng một môi trường mô phỏng giao thông dựa trên tác tử (đường cao tốc 3 làn + một nhánh nhập làn) và dùng **học tăng cường đa tác tử (MARL)** để dạy 4 chiếc xe phối hợp giải quyết bài toán nhập làn — một trong những tình huống khó và nguy hiểm nhất trong điều khiển phương tiện tự hành.

Kiến trúc học áp dụng **CTDE** (Centralized Training, Decentralized Execution): mỗi tác tử có một **actor** chỉ đọc quan sát cục bộ (thực thi phi tập trung), trong khi **critic** thấy trạng thái toàn cục của cả 4 tác tử trong lúc huấn luyện (tập trung) để ổn định quá trình học.

## Tính năng nổi bật

- **Môi trường nhập làn đầy đủ** trên GAMA: 3 làn cao tốc + nhánh ramp + vùng tăng tốc + xe nền NPC + hệ thống xe hỏng tạo ùn tắc.
- **4 tác tử học đồng thời** (1 xe nhập làn + 3 xe cao tốc) với chính sách dùng chung (parameter sharing + agent indicator).
- **Hai thuật toán học** (MAPPO, MAA2C) theo kiến trúc CTDE, so với **baseline không học**: `greedy` (tham lam) và `random` (sàn).
- **A/B khiên an toàn** (`--shield on/off`): bật/tắt lưới an toàn của môi trường (gate nhập làn + car-following guard) cho **toàn bộ** policy → đo ảnh hưởng của khiên lên từng thuật toán một cách công bằng.
- **Pipeline thực nghiệm tự động**: một lệnh chạy trọn bộ huấn luyện → đánh giá đa hạt giống → biểu đồ + bảng so sánh có CI95%.
- **Traffic tái lập theo seed**: mỗi episode (baseline + eval) sinh giao thông khác nhau nhưng tái lập được và **paired** (cùng seed → cùng tình huống cho mọi policy).
- **Bộ chỉ số đầy đủ**: tỷ lệ thành công, va chạm, thông lượng, tốc độ dòng chính, chỉ số sóng lùi (shockwave).
- **Bảng điều khiển Streamlit**: giao diện web điều khiển toàn bộ pipeline (train/eval/experiment/theo dõi/xem kết quả/demo) — thay cho gõ lệnh tay.

## Kết quả chính

Ma trận đầy đủ: **3 mật độ** (low ~20 xe / medium ~45 / high ~70) × **2 chế độ khiên** (ON/OFF) × **4 chính sách**. Mỗi chính sách học: 5 hạt giống × 200k bước × 50 episode đánh giá (stochastic), traffic biến thiên theo seed.

**SUCCESS % — định dạng `khiên ON | khiên OFF`** (RL: mean ± std qua 5 seed):

| Mật độ | MAA2C | MAPPO | Greedy | Random |
|---|---|---|---|---|
| low | 86.4±3.9 \| 87.2±4.1 | 83.6±11.8 \| 75.2±22.8 | 96 \| 82 | 0 \| 0 |
| medium | **89.2±1.6** \| 84.8±3.0 | 83.6±8.6 \| 76.8±17.9 | 96 \| 2 | 0 \| 0 |
| high | 82.8±4.3 \| 85.2±5.2 | 68.4±7.1 \| 69.6±9.4 | 74 \| **0** | 0 \| 0 |

**COLLISION % (`ON | OFF`):**

| Mật độ | MAA2C | MAPPO | Greedy | Random |
|---|---|---|---|---|
| low | 13.6 \| 12.8 | 16.0 \| 21.6 | 4 \| 18 | 100 |
| medium | 10.8 \| 15.2 | 16.0 \| 22.4 | 0 \| **98** | 100 |
| high | 16.4 \| 14.8 | 31.2 \| 30.4 | 22 \| **100** | 100 |

### Ba kết luận chính

1. **MAA2C mạnh + ổn định nhất** — 82–89% thành công ở MỌI cấu hình, std chỉ 1.6–5.2%. Vượt MAPPO cả độ chính xác lẫn độ ổn định (MAPPO std nổ tới ±23% khi không khiên).
2. **Năng lực của Greedy là "vay mượn" từ khiên môi trường.** Có khiên: 96/96/74%. **Gỡ khiên: sụp theo mật độ → 82 / 2 / 0%** (va chạm 18/98/100%). RL thì *nội hóa* an toàn → bền (MAA2C 86→87, 89→85, 83→85%). Cùng điều kiện không-khiên, **RL thắng Greedy áp đảo** (medium 85% vs 2%, high 85% vs 0%) → đây là minh chứng giá trị của học.
3. **Tầm quan trọng của khiên tăng theo mật độ** — low còn tha thứ (Greedy-off 82%), high thì luật tĩnh vô dụng (0%). Random = sàn 0%/100% mọi nơi.

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│  GAMA Platform  —  models/Main_Traffic.gaml  (MÔI TRƯỜNG)   │
│                                                             │
│  • 3 làn cao tốc (lane_width = 3.5m, dài 200m)              │
│  • Nhánh ramp (8 waypoints) + vùng tăng tốc (48m → 162m)    │
│  • Điểm merge x = 180m                                      │
│  • Khiên an toàn: gate is_merge_gap_safe + shield M3c        │
│                                                             │
│  Xe nền NPC           │   RL Agents (PettingZoo Parallel)   │
│                       │    • merging_0  (magenta) — ramp     │
│                       │    • highway_0/1/2 (cyan) — cao tốc  │
└──────────────────────────┬──────────────────────────────────┘
                           │ gama-pettingzoo  (socket :1001)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Python RL Pipeline  (BỘ NÃO HỌC)                            │
│                                                              │
│  marl_env.py        GAMA → wrappers → SB3 VecEnv             │
│  centralized_policy Actor(local 19D) + Critic(global 60D)    │
│  train_marl.py      PPO/A2C + CentralizedCriticPolicy        │
│  evaluate_marl.py   đánh giá + histogram action              │
│  baselines.py       greedy / random (non-learning)          │
│  run_experiments.py điều phối: train → eval → plots → bảng   │
└──────────────────────────────────────────────────────────────┘
```

> **Greedy/random vs khiên:** đây là baseline **không học** (chỉ là mốc so sánh, không train). Chế độ Heuristic trên GUI GAMA dùng luật phía GAML (`get_heuristic_*`) — độc lập với `baselines.py`, để xe demo chạy hợp lý khi quan sát trực quan.

---

## Thiết kế bài toán (MDP đa tác tử)

### Tác tử

| Agent | Vai trò | Mục tiêu |
|---|---|---|
| `merging_0` | Xe nhập làn (trên ramp) | Nhập làn an toàn, không va chạm |
| `highway_0/1/2` | 3 xe cao tốc làn dưới | Giữ tốc độ + nhường khoảng trống cho xe ramp |

**Chiến lược học:** chính sách dùng chung + *agent indicator* — cả 4 tác tử dùng một mạng, phân biệt vai trò bằng one-hot ID 4 chiều gắn vào quan sát.

### Không gian quan sát

- **Actor (cục bộ):** 19D = 15D sensor + 4D one-hot agent ID.
- **Critic (toàn cục):** 60D = nối 15D base của cả 4 tác tử.
- **Đưa vào mô hình:** 79D = `[local 19D | global 60D]`.

15D của `merging_0`: tốc độ, tiến độ ramp, khoảng cách tới điểm merge, lệch ngang, cờ vùng tăng tốc, khoảng trống/tốc độ xe trước–sau trên làn đích, cờ gap an toàn, xe trước trên ramp, độ khẩn, hành động trước, kiên nhẫn.
15D của `highway`: tốc độ, làn hiện tại, radar đa hướng (trước/trái/phải) + thông tin xe ramp đang merge.

### Không gian hành động — `Discrete(5)`

| Action | `merging_0` | `highway_0/1/2` |
|---|---|---|
| 0 | Giảm tốc | Tăng tốc |
| 1 | Giữ tốc | Giữ tốc |
| 2 | Tăng tốc | Giảm tốc |
| 3 | Nhập làn | Rẽ trái |
| 4 | Chờ | Rẽ phải |

> Cùng chỉ số nhưng ngữ nghĩa khác theo vai trò — mạng phân biệt nhờ agent indicator. Bảng chuẩn: `ACTION_MEANINGS_*` trong [`rl/config.py`](rl/config.py).

### Hàm phần thưởng

Toàn bộ reward được tính trong GAML (`calculate_merging_reward`, `calculate_reward`); Python chỉ nhận giá trị qua PettingZoo.

| `merging_0` | | `highway_0/1/2` |
|---|---|---|
| Nhập làn thành công (lớn) | | Thoát đường an toàn (+) |
| Va chạm (−100) | | Va chạm (−100) |
| Hết đường chưa merge (−) | | Bám đuôi quá gần (TTC): phạt theo gap |
| Vào vùng tăng tốc / merge đúng lúc gap an toàn (+) | | Nhường gap khi xe ramp merge (+) |

---

## Thuật toán & siêu tham số

| Hyperparameter | MAPPO | MAA2C |
|---|---|---|
| Policy | `CentralizedCriticPolicy` (CTDE) | `CentralizedCriticPolicy` (CTDE) |
| Learning rate | 3e-4 | 3e-4 |
| `n_steps` | 256 | 64 |
| `batch_size` | 128 | (full rollout) |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 0.99 / 0.95 |
| `clip_range` | 0.2 | — |
| `ent_coef` | 0.01 | 0.05 |
| Mạng / kích hoạt / init | 64 / Tanh / Orthogonal(√2) | nt |

> Dự án dùng các thuật toán **on-policy** (PPO/A2C) phù hợp với môi trường đa tác tử non-stationary; DQN (off-policy) không được dùng vì replay buffer trộn dữ liệu nhiều tác tử khiến Q-value khó hội tụ.

---

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Yêu cầu:** Python 3.10+ · GAMA Platform 1.9.x+ (có `gama-headless`) · RAM ≥ 8GB.

**Windows + tiếng Việt:** nếu gặp `UnicodeEncodeError` khi in log: `$env:PYTHONIOENCODING='utf-8'`.

## Khởi động GAMA Headless

Trước khi chạy script Python, bật server mô phỏng:

```powershell
cd <ĐƯỜNG_DẪN>\gama-headless
.\gama-headless.bat -socket 1001
```

Quan sát trực quan: mở `models/Main_Traffic.gaml` trong GAMA GUI → chạy experiment `TrafficSimulation` (chế độ Heuristic dùng luật rule-based phía GAML).

## Chạy thực nghiệm

Kiểm tra nhanh kết nối trước khi train dài:

```powershell
python rl/smoke_test_env.py        # kỳ vọng: [PASS] MARL smoke test OK
```

Chạy pipeline tự động (khuyến nghị):

```powershell
python rl/run_experiments.py --preset thesis --port 1001 --scenario medium                 # bộ số chính (~5h)
python rl/run_experiments.py --preset thesis --port 1001 --scenario medium --shield off     # A/B: gỡ khiên an toàn
python rl/run_experiments.py --preset short  --port 1001                                    # nhanh: 1 seed
python rl/run_experiments.py --preset smoke  --dry-run                                      # kiểm tra cấu hình
```

Tùy chọn: `--scenario low|medium|high` (mật độ) · `--shield on|off` (khiên an toàn cho cả run) · `--algos ppo|a2c` · `--skip-baselines` · `--run-tag <tên>` (đặt tên thư mục lần chạy; `none` = ghi phẳng).

| Preset | Timesteps | Eval episodes | Seeds | Dùng cho |
|---|---|---|---|---|
| `smoke` | 400 | 2 | [0] | Sanity pipeline |
| `short` | 200,000 | 20 | [0] | Kiểm tra nhanh |
| `thesis` | 200,000 | 50 | [0–4] | Bộ số chính |

| Scenario | nb_cars_max | spawn_interval | Mô tả |
|---|---|---|---|
| `low` | 20 | 10 | Mật độ thấp (đường thưa) |
| `medium` | 45 | 5 | Mật độ vừa (bộ số chính) |
| `high` | 70 | 3 | Mật độ cao (ùn tắc) |

**Tách output theo từng lần chạy (mặc định):** mỗi lần chạy tự sinh thư mục riêng theo timestamp `<preset>_YYYYmmdd_HHMMSS` (hoặc tên `--run-tag`) cho cả model, log và plots — không bao giờ đè kết quả lần trước.

## Bảng điều khiển (UI Streamlit)

Thay cho gõ lệnh tay, có thể điều khiển toàn bộ pipeline qua giao diện web: cấu hình đường dẫn, bật/tắt GAMA, train/eval, chạy experiment, theo dõi tiến trình, xem biểu đồ kết quả, và demo model trên GAMA Desktop.

Do `streamlit` xung đột `websockets` với `gama-client`, UI chạy bằng **venv riêng** (`app_streamlit.py` chỉ gọi `.venv` chính qua subprocess nên không vướng xung đột):

```powershell
python -m venv .venv-ui
.venv-ui\Scripts\pip install streamlit
.venv-ui\Scripts\streamlit run app_streamlit.py
```

Tab **Results** hiển thị biểu đồ của đúng lần chạy đang chọn (ảnh thu nhỏ, bấm để phóng to).

## Phân tích kết quả

`run_experiments.py` tự sinh biểu đồ + bảng cuối pipeline. Output (trong thư mục theo lần chạy):

- **Biểu đồ PNG**: reward curve, success/collision/failed_merge/timeout rate, merge_step, episode_length, throughput, shockwave_index, radar, boxplot.
- **Bảng**: `comparison_table.csv` (mean±std mọi metric), `latex_table.tex` (copy vào báo cáo), `summary_metrics.csv`, `learning_efficiency.csv`.

Có thể chạy thủ công: `python rl/plots.py <csv...> --out-dir <dir>` và `python rl/analysis.py <csv...> --out-dir <dir>`.

> **Lưu ý:** các chỉ số/biểu đồ KPI được tính trên episode **đánh giá (eval) + baseline** (không trộn rollout huấn luyện) nên luôn khớp `comparison_table.csv`.

### Các chỉ số

| Metric | Ý nghĩa | Tốt khi |
|---|---|---|
| `success_rate` | Tỷ lệ episode nhập làn xong + hoàn thành lộ trình | Cao |
| `collision_rate` | Tỷ lệ va chạm | Thấp |
| `failed_merge_rate` / `timeout_rate` | Các loại thất bại | Thấp |
| `avg_reward` | Tổng reward trung bình/episode | Cao |
| `mean_speed` / `throughput` | Tốc độ / thông lượng nhập làn | Cao |
| `mainline_mean_speed` | Tốc độ trung bình dòng chính (đo ùn tắc robust) | Cao |
| `shockwave_index` | Độ dao động tốc độ mainline (sóng lùi) | Thấp |

---

## Cấu trúc thư mục

```
.
├── models/
│   └── Main_Traffic.gaml      # Môi trường mô phỏng MARL (4 agents)
├── rl/
│   ├── centralized_policy.py  # CentralizedCriticPolicy (CTDE)
│   ├── marl_env.py            # Chuỗi wrapper GAMA → SB3 VecEnv
│   ├── train_marl.py          # Huấn luyện MAPPO + MAA2C
│   ├── evaluate_marl.py       # Đánh giá + histogram action
│   ├── baselines.py           # greedy / random (non-learning)
│   ├── run_experiments.py     # Điều phối pipeline (+ --shield, --scenario)
│   ├── analysis.py            # Bảng so sánh + LaTeX
│   ├── plots.py               # Biểu đồ
│   ├── metrics.py             # EpisodeMetric + ghi CSV
│   ├── config.py              # Preset, paths, ACTION_MEANINGS, scenario
│   ├── scenario_utils.py      # Vá GAML: scenario mật độ + khiên an toàn
│   ├── gama_compat.py         # Lớp tương thích gama-pettingzoo
│   ├── gama_episode_reset.py  # Reset helper + reseed traffic theo seed
│   ├── smoke_test_env.py      # Kiểm tra kết nối (chạy tay)
│   ├── diagnose_marl.py       # Công cụ debug (chạy tay)
│   └── model_registry.py      # Liệt kê model → JSON (chạy tay)
├── tests/                     # pytest (offline)
├── outputs/                   # Sinh khi train/eval (models, logs, plots)
└── requirements.txt
```

---

## Gỡ lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: supersuit` | `pip install -r requirements.txt` trong venv |
| `UnicodeEncodeError` (Windows) | `$env:PYTHONIOENCODING='utf-8'` hoặc `chcp 65001` |
| GAMA: `unable to find experiment/simulation` | Restart GAMA headless (`gama-headless.bat -socket 1001`) |
| GAMA chết giữa run dài | Khởi động lại headless rồi chạy lại lần đó |

## Phạm vi & hạn chế

- Mô phỏng tinh giản: hành động rời rạc + waypoint, không phải động học/cảm biến xe thật.
- Chính sách dùng chung cho hai vai trò có ngữ nghĩa khác nhau — lựa chọn thiết kế đơn giản, không tương đương hai chính sách chuyên biệt.
- Một instance GAMA (`num_vec_envs=1`) → thời gian huấn luyện dài.
- Traffic của **baseline + eval** đã tái lập theo seed (reproducible + paired); riêng **huấn luyện** RL chạy trên một layout cố định (chưa randomize traffic theo seed trong vòng train).
