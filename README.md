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
- **Hai thuật toán học** (MAPPO, MAA2C) theo kiến trúc CTDE, so với **baseline không học** `greedy` (tham lam; `random` đã bỏ theo đề cương).
- **Pipeline thực nghiệm tự động**: một lệnh chạy trọn bộ huấn luyện → đánh giá đa hạt giống → biểu đồ + bảng so sánh có CI95%.
- **Traffic tái lập theo seed**: mỗi episode (baseline + eval) sinh giao thông khác nhau nhưng tái lập được và **paired** (cùng seed → cùng tình huống cho mọi policy).
- **Bộ chỉ số đầy đủ**: tỷ lệ thành công, va chạm, thông lượng, tốc độ dòng chính, chỉ số sóng lùi (CV), **%nhường-hướng-merger** (đo hợp tác thật, không nhiễu histogram), **số lần khiên can thiệp/episode** (đo mức lệ thuộc lưới an toàn).
- **Đánh giá end-to-end**: merger lái tiếp sau nhập làn tới cuối đường bằng RL (`--eval-continue --rl-postmerge`) — bắt được tai nạn hậu-merge + đo sóng lùi trọn hành trình.
- **Ablation khiên** (`--no-shield`): tắt toàn bộ lưới an toàn lúc đánh giá → định lượng mức nội tâm hóa kỹ năng lái của policy.
- **Reward shaping hiện đại**: dense progress (potential-based, Ng 1999) + phạt-khiên-tỷ-lệ theo mức cắt ga (thay binary) — hai chìa khóa giúp critic học được (explained_variance 0 → 0.85) và policy bớt ỷ khiên.
- **Curriculum 2 giai đoạn tái lập được**: `bash train_curriculum.sh` — train from-scratch (học nhập làn → học e2e + hợp tác) không cần sửa code, xuất full learning curve.
- **Bảng điều khiển Streamlit**: giao diện web điều khiển toàn bộ pipeline (train/eval/experiment/theo dõi/kết quả/demo/biểu đồ thesis) — thay cho gõ lệnh tay.

## Kết quả chính

Chế độ đánh giá chuẩn: **end-to-end (e2e)** — xe RL nhập làn xong **tự lái tiếp tới cuối đường** (ván kết tại cuối đường; đo trọn sóng lùi + tai nạn sau nhập làn). Mật độ cao (84 xe, road 240), 10 episode/seed.

| Model | Nguồn gốc huấn luyện | Thành công | Va chạm | Nhường (G2) | Không-khiên |
|---|---|---|---|---|---|
| **Curriculum 2-stage** | from scratch 600k (`train_curriculum.sh`) | 90% | 10% | 41% | 30% |
| **yt50-150k (hợp tác)** | warmstart nhiều giai đoạn | 90% | 10% | **94%** | 10% |
| **yt52-150k (propshield)** | yt50 + phạt khiên tỷ lệ k=5 | **100%** | **0%** | thấp | **50%** |
| Greedy (baseline) | luật tĩnh | 40% | 60% | 0% | 0% |

### Ba kết luận chính

1. **RL thắng baseline áp đảo ở mật độ cao**: cùng toàn bộ lưới an toàn (so sánh công bằng từng lớp), RL đạt 90–100% trong khi Greedy đâm 60% (lao sớm, không khớp tốc độ). Khác biệt thuần là **chất lượng quyết định học được**: chọn khe + căn thời điểm + khớp tốc — kiểu "zipper merge".
2. **Phổ hành vi theo trục huấn luyện**: checkpoint giữa (150k) đạt hợp tác đỉnh — highway **chủ động nhường 94%** số lần giảm tốc khi có xe nhập làn gần (Goal 2); train tiếp, hệ tiến hóa sang "tự chủ" — merger tự tìm khe tự nhiên, không cần ai nhường mà vẫn 100% (Goal 1+3 cực đại). Trade-off hợp tác ↔ lưu lượng là thật, đo được, và chọn được bằng checkpoint.
3. **Khiên an toàn là bộ phận kiến trúc, mức lệ thuộc đo được**: ablation `--no-shield` cho thấy policy chỉ còn 10–30% khi lái trần (kỹ năng điều-tốc-tinh được ủy thác cho IDM-cap); chuyển phạt can-thiệp từ **binary −1 sang tỷ lệ mức-cắt (k=5)** nâng không-khiên lên 50% mà vẫn giữ 100% có khiên — chuỗi 10% → 30% → 50% chứng minh kỹ năng này *nội tâm hóa* dần được qua thiết kế reward.

> Số liệu gốc trong `outputs/logs/` (mỗi con số truy được về file nguồn). Biểu đồ báo cáo: `python rl/make_thesis_plots.py` → `outputs/plots/fig1..fig7` (learning curve 2-stage, G1/G2/G3, ablation khiên, mức ỷ-khiên).

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│  GAMA Platform  —  models/Main_Traffic.gaml  (MÔI TRƯỜNG)   │
│                                                             │
│  • 3 làn cao tốc (lane_width = 3.5m, dài 240m)              │
│  • Nhánh ramp (8 waypoints) + vùng tăng tốc (48m → 162m)    │
│  • Điểm merge x = 180m                                      │
│  • Khiên RL: IDM-cap dọc + gate ngang + phạt tỷ lệ k=5       │
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
│  baselines.py       greedy (non-learning baseline)          │
│  run_experiments.py điều phối: train → eval → plots → bảng   │
└──────────────────────────────────────────────────────────────┘
```

> **Greedy vs khiên:** đây là baseline **không học** (chỉ là mốc so sánh, không train). Chế độ Heuristic trên GUI GAMA dùng luật phía GAML (`get_heuristic_*`) — độc lập với `baselines.py`, để xe demo chạy hợp lý khi quan sát trực quan.

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

| `merging_0` | `highway_0/1/2` |
|---|---|
| **Dense progress** (potential-based, Ng 1999): +k×Δ(x/đích) mỗi tick — chìa khóa giúp critic học được (ev 0→0.85) | Giữ tốc/flow (+ theo speed, thưởng dòng chính sw_mean) |
| Nhập làn thành công (+50, one-shot) | Thoát cuối đường: terminal `exited` |
| Va chạm (−100) | Va chạm (−100) |
| Hết đường chưa merge (−50) | Bám đuôi quá gần (TTC): phạt theo gap |
| **Phạt khiên tỷ lệ**: −k×(ga RL − ga khiên cho phép) mỗi tick (k=5) — ép nội tâm hóa điều tốc, không ỷ khiên | Như merger + thưởng nhường khi merger gần (+0.4) / phạt phanh-vô-cớ |

---

## Thuật toán & siêu tham số

| Hyperparameter | MAPPO | MAA2C |
|---|---|---|
| Policy | `CentralizedCriticPolicy` (CTDE) | `CentralizedCriticPolicy` (CTDE) |
| Learning rate | 1e-4 (ổn định train dài) | 3e-4 |
| `n_steps` | 256 | 64 |
| `batch_size` | 128 | (full rollout) |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 0.99 / 0.95 |
| `clip_range` | 0.2 | — |
| `ent_coef` | anneal 0.05 → 0.002 (giữ 65% đầu) | anneal 0.05 → 0.002 |
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
| `low` | 24 | 10 | Mật độ thấp (đường thưa) |
| `medium` | 54 | 5 | Mật độ vừa |
| `high` | 84 | 3 | Mật độ cao (ùn tắc — bộ số chính e2e, road 240) |

### Huấn luyện curriculum 2 giai đoạn (tái lập model thesis từ đầu)

```bash
bash train_curriculum.sh            # ~3h: Stage 1 học NHẬP LÀN (300k, from scratch)
                                    #      Stage 2 học E2E + HỢP TÁC (300k, warmstart Stage 1)
```

Sau đó sweep checkpoint Stage 2 (50k…300k) bằng tab Evaluate (bật E2E) và chọn checkpoint **cân bằng 3 mục tiêu** — checkpoint cuối thường tối ưu merger nhưng quên nhường (catastrophic forgetting), nên chọn theo số liệu, không lấy mù. Learning curve ghi tại `outputs/logs/curr_stage*_episodes.csv` (tự dùng cho `make_thesis_plots.py`).

### Đánh giá e2e + ablation khiên (chuẩn thesis)

```powershell
# E2E có khiên (chuẩn):
python rl/evaluate_marl.py --algo ppo --model outputs/models/thesis_e2e_yt50_150k.zip `
  --episodes 10 --seed 0 --port 1002 --max-episode-steps 800 `
  --eval-continue --postmerge-window 400 --rl-postmerge

# Ablation bỏ khiên (đo nội tâm hóa): thêm --no-shield
```

**Tách output theo từng lần chạy (mặc định):** mỗi lần chạy tự sinh thư mục riêng theo timestamp `<preset>_YYYYmmdd_HHMMSS` (hoặc tên `--run-tag`) cho cả model, log và plots — không bao giờ đè kết quả lần trước.

## Bảng điều khiển (UI Streamlit)

Thay cho gõ lệnh tay, có thể điều khiển toàn bộ pipeline qua giao diện web: cấu hình đường dẫn, bật/tắt GAMA, train/eval, chạy experiment, theo dõi tiến trình, xem biểu đồ kết quả, và demo model trên GAMA Desktop.

Do `streamlit` xung đột `websockets` với `gama-client`, UI chạy bằng **venv riêng** (`app_streamlit.py` chỉ gọi `.venv` chính qua subprocess nên không vướng xung đột):

```powershell
python -m venv .venv-ui
.venv-ui\Scripts\pip install streamlit
.venv-ui\Scripts\streamlit run app_streamlit.py
```

Tab **Results** hiển thị biểu đồ của đúng lần chạy đang chọn (ảnh thu nhỏ, bấm để phóng to). Tab **Evaluate** và **Play GUI** có công tắc **E2E** (chạy tới cuối đường) và **BỎ KHIÊN** (ablation). Tab **Thesis** sinh + xem bộ biểu đồ báo cáo (`make_thesis_plots.py`) và bảng model thesis.

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
│   ├── baselines.py           # greedy (non-learning; random đã bỏ)
│   ├── run_experiments.py     # Điều phối pipeline (+ --shield, --scenario)
│   ├── analysis.py            # Bảng so sánh + LaTeX
│   ├── plots.py               # Biểu đồ
│   ├── metrics.py             # EpisodeMetric + ghi CSV
│   ├── config.py              # Preset, paths, ACTION_MEANINGS, scenario
│   ├── scenario_utils.py      # Vá GAML: scenario mật độ + khiên an toàn
│   ├── gama_compat.py         # Lớp tương thích gama-pettingzoo
│   ├── gama_episode_reset.py  # Reset helper + reseed traffic theo seed
│   ├── make_thesis_plots.py   # 7 biểu đồ PNG cho báo cáo
│   ├── probe_reward_stream.py # Soi dòng reward/done THÔ SB3 thấy (chẩn ev≈0)
│   ├── smoke_test_env.py      # Kiểm tra kết nối (chạy tay)
│   ├── diagnose_marl.py       # Công cụ debug (chạy tay)
│   └── model_registry.py      # Liệt kê model → JSON (chạy tay)
├── train_curriculum.sh        # Curriculum 2-stage from-scratch (tái lập model)
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
| Train không học (`explained_variance≈0`, `value_loss≈1e-6`) | Episode boundary có thể bị nuốt — chạy `python rl/probe_reward_stream.py` xem done=True có tới SB3 không; xe RL trong GAML phải grace-die (KHÔNG `do die` cùng cycle) |
| Density sai sau khi Play GUI | Play patch GAML theo scenario; nếu bị kill giữa chừng → kiểm `nb_cars_max` trước khi train (`grep nb_cars_max models/Main_Traffic.gaml`) |

## Phạm vi & hạn chế

- Mô phỏng tinh giản: hành động rời rạc + waypoint, không phải động học/cảm biến xe thật.
- **Hành động rời rạc 5 mức không biểu diễn được điều-tốc-liên-tục** → policy ủy thác phần này cho khiên IDM-cap (ablation: bỏ khiên còn 10–50% tùy thiết kế phạt). Hướng mở: action liên tục hoặc shield-annealing.
- Chính sách dùng chung cho hai vai trò có ngữ nghĩa khác nhau — lựa chọn thiết kế đơn giản, không tương đương hai chính sách chuyên biệt.
- Một instance GAMA (`num_vec_envs=1`) → thời gian huấn luyện dài.
- Traffic của **baseline + eval** đã tái lập theo seed (reproducible + paired); riêng **huấn luyện** RL chạy trên một layout cố định (chưa randomize traffic theo seed trong vòng train).
