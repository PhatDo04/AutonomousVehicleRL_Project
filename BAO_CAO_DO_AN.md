# BÁO CÁO ĐỒ ÁN TỐT NGHIỆP

## Phát triển kịch bản nhập làn trên đường cao tốc và đánh giá các thuật toán học tăng cường đa tác tử cho điều khiển phương tiện tự hành

---

**Sinh viên thực hiện:** Đỗ Tiến Phát
**Mã sinh viên:** 2251172447 — Lớp 64KTPM1
**Khoa:** Công nghệ Thông tin — Trường Đại học Thủy Lợi
**Giảng viên hướng dẫn:** TS. Lê Nguyễn Tuấn Thành
**Tham chiếu nền tảng:** Le Nguyen Tuan Thanh (2023). *Multi-agent reinforcement learning for traffic congestion on one-way multi-lane highways*. Journal of Information and Telecommunication, 7:3, 255–269.

> Tài liệu kèm theo: [BAO_CAO_MA_NGUON_CHI_TIET.md](BAO_CAO_MA_NGUON_CHI_TIET.md) (chi tiết kỹ thuật + glossary), [GIAI_THICH_CHO_NGUOI_MOI.md](GIAI_THICH_CHO_NGUOI_MOI.md) (giải thích cho người mới), [README.md](README.md) (hướng dẫn vận hành).

---

## TÓM TẮT

Nhập làn trên đường cao tốc (highway on-ramp merging) là một trong những kịch bản khó và nguy hiểm nhất trong điều khiển phương tiện tự hành: xe trên nhánh phụ phải tăng tốc, đánh giá khoảng trống và hòa vào dòng xe đang di chuyển nhanh, đòi hỏi sự **phối hợp** giữa xe nhập làn và các xe trên cao tốc. Đồ án này (1) xây dựng một **môi trường mô phỏng nhập làn** dựa trên tác tử bằng nền tảng **GAMA**; (2) hình thức hóa bài toán dưới dạng **Quá trình Quyết định Markov đa tác tử** với 4 tác tử học đồng thời; (3) cài đặt và đánh giá hai thuật toán học tăng cường đa tác tử **MAPPO** và **MAA2C** theo kiến trúc **CTDE** (Centralized Training, Decentralized Execution), so sánh với hai baseline không học là **Greedy** (luật tham lam) và **Random** (ngẫu nhiên).

Khác với phiên bản khảo sát ban đầu (chỉ một kịch bản mật độ), đồ án tiến hành một **ma trận thực nghiệm có kiểm soát**: ba mức mật độ giao thông (`low`/`medium`/`high`) × hai chế độ lưới an toàn của môi trường (khiên **bật**/**tắt**) × bốn chính sách (MAA2C, MAPPO, Greedy, Random), mỗi cấu hình học chạy 5 hạt giống × 200.000 bước × 50 episode đánh giá. Kết quả cho thấy: **MAA2C là thuật toán mạnh và ổn định nhất** (82–89% thành công ở mọi cấu hình, độ lệch chuẩn chỉ 2–5%), vượt MAPPO cả về độ chính xác lẫn độ ổn định. Quan trọng hơn, thí nghiệm A/B trên lưới an toàn phát hiện một kết luận có giá trị: **năng lực của Greedy là "vay mượn" từ lưới an toàn của môi trường** — khi gỡ lưới ở mật độ cao, Greedy sụp từ 96% xuống 0% trong khi chính sách học giữ được 70–85%. Điều này cho thấy chính sách học đã **nội hóa hành vi lái an toàn**, còn luật tham lam tĩnh thì phụ thuộc hoàn toàn vào cơ chế bảo vệ bên ngoài.

Giai đoạn 2 nâng bài toán lên **end-to-end**: xe nhập làn phải **tự lái tiếp tới cuối đường** sau khi nhập (đường kéo dài lên 240m, mật độ cao 84 xe) — bắt được cả tai nạn hậu-nhập-làn và sóng lùi trọn hành trình. Đóng góp giai đoạn này: (i) chẩn đoán và sửa **lỗi nuốt tín hiệu kết-thúc-episode** ở tầng cầu nối GAMA–Python (nguyên nhân khiến critic không học được, explained variance xấp xỉ 0); (ii) **dense progress reward** (potential-based) giúp critic học thật (ev tới 0,85) và chính sách đạt **90–100% thành công ở mật độ cao** trong khi Greedy chỉ 40%; (iii) chứng minh **phổ hành vi hợp tác - tự chủ** theo trục huấn luyện (checkpoint giữa: highway chủ động nhường 94%; checkpoint muộn: merger tự tìm khe kiểu zipper, 100% không cần nhường); (iv) **ablation lưới an toàn** định lượng mức nội tâm hóa và đề xuất **phạt-can-thiệp-tỷ-lệ** nâng năng lực lái-không-khiên từ 10% lên 50%; (v) **curriculum 2 giai đoạn tái lập được** train từ đầu ra model tương đương (90%) kèm trọn đường cong học.

**Từ khóa:** học tăng cường đa tác tử, nhập làn cao tốc, CTDE, MAPPO, MAA2C, baseline tham lam, lưới an toàn, GAMA, PettingZoo.

---

## CHƯƠNG 1. MỞ ĐẦU

### 1.1. Đặt vấn đề

Khác với dòng chảy tự do trên đường thẳng, điểm nhập làn (on-ramp) tạo ra **nút thắt cổ chai** phức tạp. Bài toán khó vì ba đặc trưng:

- **Tương tác đa tác tử:** An toàn không chỉ phụ thuộc xe nhập làn — xe trên cao tốc có thể nhường hoặc không. Đây bản chất là bài toán phối hợp nhiều tác tử.
- **Quan sát một phần (partial observability):** Mỗi xe chỉ "thấy" lân cận của nó, không có thông tin toàn cục.
- **Phần thưởng thưa xen đặc (sparse + dense):** Thành công/va chạm là tín hiệu terminal hiếm, trong khi hành vi từng bước cần tín hiệu dày để hướng dẫn học.

Mô hình hóa và mô phỏng dựa trên tác tử (Agent-Based Modeling and Simulation — ABMS) là cách tiếp cận tự nhiên cho các hệ giao thông phức hợp. Kết hợp ABMS với **học tăng cường sâu**, **học tăng cường đa tác tử (MARL)** trở thành kỹ thuật phù hợp để giải bài toán này.

### 1.2. Mục tiêu

1. Xây dựng môi trường mô phỏng kịch bản nhập làn (3 làn cao tốc + 1 nhánh ramp) trên nền tảng **GAMA**.
2. Hình thức hóa bài toán dưới dạng **MDP đa tác tử** (4 tác tử: 1 xe nhập làn + 3 xe cao tốc).
3. Thiết kế không gian trạng thái, hành động và **hàm phần thưởng đa mục tiêu** phù hợp hành vi nhập làn.
4. Cài đặt và huấn luyện các thuật toán MARL (**MAPPO**, **MAA2C**) theo kiến trúc CTDE bằng thư viện **Stable-Baselines3**.
5. **Đánh giá và so sánh** hiệu năng với các baseline không học (Greedy, Random) trên **ma trận mật độ × lưới an toàn**, nhằm tách bạch đóng góp thực sự của việc học so với luật tay.

### 1.3. Phạm vi

- Mô phỏng tinh giản: hành động **rời rạc** (Discrete) + di chuyển theo **waypoint**, không mô phỏng động học/cảm biến đầy đủ của xe thật.
- Tập trung **đánh giá so sánh** các thuật toán RL trên một MDP cố định, **không đề xuất phương pháp MARL mới**.
- Ba kịch bản mật độ (`low` ~20 xe, `medium` ~45 xe, `high` ~70 xe) và hai chế độ lưới an toàn.

### 1.4. Đóng góp

- Một **môi trường mô phỏng nhập làn** đầy đủ trên GAMA, kết nối hai chiều với Python qua PettingZoo.
- Thiết kế **MDP đa tác tử dị thể** (heterogeneous: xe nhập làn và xe cao tốc có cùng kích thước không gian nhưng khác ngữ nghĩa) với chính sách dùng chung + agent indicator.
- **Khung thực nghiệm A/B lưới an toàn**: một cờ duy nhất bật/tắt đồng đều cơ chế bảo vệ của môi trường cho mọi chính sách, cho phép đo lường tách bạch "năng lực nội tại của chính sách" với "trợ giúp từ môi trường".
- **So sánh định lượng** MAA2C / MAPPO / Greedy / Random trên ma trận 3 mật độ × 2 lưới an toàn, kèm biểu đồ và bảng thống kê có khoảng tin cậy.
- Hai cải tiến kỹ thuật cho độ tin cậy thực nghiệm: cơ chế **reset-on-merging-death** (ổn định huấn luyện MARL với tác tử động) và cơ chế **gieo lại RNG giao thông theo hạt giống** (mỗi episode có tình huống khác nhau nhưng tái lập được và đồng bộ giữa các chính sách).

---

## CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

### 2.1. Học tăng cường và MDP

**Học tăng cường (RL)** mô hình hóa quá trình một tác tử học qua tương tác với môi trường dưới dạng **Quá trình Quyết định Markov (MDP)** gồm bộ `(S, A, P, R, γ)`: tập trạng thái `S`, tập hành động `A`, hàm chuyển trạng thái `P`, hàm phần thưởng `R`, và hệ số chiết khấu `γ ∈ [0,1)`. Mục tiêu là tìm **chính sách** `π: S → A` tối đa hóa kỳ vọng tổng phần thưởng chiết khấu `E[Σ γ^t r_t]`. Hai đại lượng trung tâm: **hàm giá trị trạng thái** `V(s)` (kỳ vọng lợi ích từ trạng thái `s`) và **hàm lợi thế** `A(s,a) = Q(s,a) − V(s)` (hành động `a` tốt hơn mức trung bình bao nhiêu).

### 2.2. Học tăng cường đa tác tử (MARL)

Khi nhiều tác tử cùng học trong một môi trường, hành vi của tác tử này trở thành một phần "môi trường" của tác tử khác. Vì các tác tử liên tục thay đổi chính sách, môi trường dưới góc nhìn từng tác tử trở nên **không dừng (non-stationary)** — nguyên nhân chính khiến MARL khó hội tụ. Bài toán nhập làn vừa **hợp tác** (xe cao tốc nhường để xe ramp vào an toàn) vừa có yếu tố **cạnh tranh** (mỗi xe muốn giữ tốc độ riêng), nên thuộc lớp *mixed cooperative-competitive*.

### 2.3. Thuật toán on-policy: PPO và A2C

- **A2C (Advantage Actor-Critic):** kiến trúc Actor (xuất phân phối hành động) + Critic (ước lượng `V(s)`), cập nhật theo hướng `∇ log π(a|s) · A(s,a)`, trực tiếp trên rollout hiện tại. Đơn giản, ít siêu tham số, nhưng bước cập nhật không bị giới hạn nên dễ dao động.
- **PPO (Proximal Policy Optimization):** cải tiến của Actor-Critic, dùng *clipped surrogate objective* giới hạn tỉ lệ thay đổi chính sách `π_new/π_old` trong khoảng `[1−ε, 1+ε]` → mỗi bước không "nhảy" quá xa → ổn định hơn về lý thuyết, đổi lại thêm siêu tham số (`clip_range`, số epoch).

Cả hai là **on-policy** (học trên dữ liệu vừa sinh bởi chính chính sách hiện tại), phù hợp với môi trường non-stationary của MARL. Phiên bản đa tác tử kết hợp CTDE được gọi là **MAPPO** và **MAA2C**.

**Phân biệt biến thể đa tác tử (làm rõ thuật ngữ):**
- *Independent* (IPPO/IA2C): mỗi tác tử là một bản PPO/A2C độc lập, Critic chỉ thấy quan sát cục bộ; coi các tác tử khác như một phần môi trường.
- *Multi-agent CTDE* (MAPPO/MAA2C): Actor vẫn phi tập trung (chỉ đọc quan sát cục bộ khi thực thi), nhưng **Critic tập trung** thấy trạng thái toàn cục khi huấn luyện → giảm non-stationarity. Đồ án dùng biến thể CTDE này.

**Vì sao loại DQN:** DQN là thuật toán *off-policy* dùng *replay buffer*. Trong môi trường đa tác tử, buffer trộn lẫn transitions của nhiều tác tử sinh ở các thời điểm chính sách khác nhau → ước lượng Q không hội tụ. Do đó đồ án chỉ dùng các thuật toán on-policy (theo Lowe et al. 2017 — MADDPG; Yu et al. 2021 — MAPPO).

### 2.4. CTDE — Centralized Training, Decentralized Execution

Để giảm non-stationarity, đồ án áp dụng kiến trúc **CTDE**:
- **Actor (phi tập trung):** mỗi tác tử chỉ đọc **quan sát cục bộ** của nó → có thể thực thi độc lập trong triển khai thực tế.
- **Critic (tập trung):** trong huấn luyện, Critic thấy **trạng thái toàn cục** (gộp quan sát của cả 4 tác tử) → ước lượng giá trị chính xác hơn, khử nhiễu do tác tử khác gây ra.

Đây là ý tưởng cốt lõi của MAPPO (Yu et al. 2021). Đồ án hiện thực bằng một bộ trích đặc trưng tùy biến tách quan sát thành phần cục bộ (cho Actor) và phần toàn cục (cho Critic).

### 2.5. Nền tảng công cụ

- **GAMA Platform** (ngôn ngữ GAML): nền tảng mô phỏng dựa trên tác tử, hỗ trợ hình học không gian phức tạp (polyline ramp, vùng gia tốc) và chế độ *headless* qua socket cho huấn luyện tự động.
- **PettingZoo:** chuẩn API cho môi trường MARL; cầu nối `gama-pettingzoo` cho phép GAMA giao tiếp hai chiều với Python.
- **Stable-Baselines3 (SB3):** thư viện RL trên PyTorch, cung cấp PPO/A2C đã kiểm thử, tích hợp TensorBoard và giao diện VecEnv.

> **Ghi chú điều chỉnh công nghệ so với đề cương:** đề cương ban đầu dự kiến NetLogo + TensorFlow/Keras + DQN/A2C. Trong quá trình thực hiện, nền tảng được chuyển sang **GAMA** (hỗ trợ MARL native, headless ổn định), thư viện học sâu sang **Stable-Baselines3/PyTorch**, và thuật toán tập trung vào **MAPPO/MAA2C** (phù hợp đa tác tử hơn DQN). Các điều chỉnh này đều có cơ sở kỹ thuật và được nêu rõ trong báo cáo.

---

## CHƯƠNG 3. THIẾT KẾ HỆ THỐNG

### 3.1. Kiến trúc tổng thể

Hệ thống gồm hai nửa nối qua socket TCP (cổng 1001):

```
GAMA (môi trường, models/Main_Traffic.gaml)
   • Đường 3 làn 200m + nhánh ramp + vùng tăng tốc + điểm merge
   • 4 tác tử RL (merging_0, highway_0/1/2) + xe nền NPC
   • Tính observation(15D) / reward / terminal mỗi tác tử
   • Lưới an toàn: gate is_merge_gap_safe + shield car-following M3c
        │  PettingZoo bridge (socket :1001)
        ▼
Python (bộ não học, thư mục rl/)
   • marl_env.py: GAMA → wrappers → SB3 VecEnv
   • centralized_policy.py: Actor(local 19D) + Critic(global 60D) — CTDE
   • train_marl.py / evaluate_marl.py / baselines.py
   • run_experiments.py điều phối; analysis.py + plots.py xuất kết quả
```

### 3.2. Môi trường mô phỏng (GAML)

- **Hình học:** 3 làn (rộng 3,5 m), dài 200 m; nhánh ramp định nghĩa bằng 8 waypoint; vùng tăng tốc từ x=48 m đến x=162 m; điểm merge tại x=180 m.
- **Tác tử `car`:** dùng chung cho cả NPC và RL, phân biệt qua thuộc tính `rl_agent_id`. Có hệ thống "xe hỏng" hai pha tạo nút thắt để quan sát hiện tượng ùn tắc.
- **Lớp cầu nối PettingZoo** (`PzBridgeAgent`, `petz_collect_tick`): mỗi chu kỳ thu quan sát/phần thưởng/tín hiệu kết thúc từ GAMA và đẩy sang Python; nhận hành động từ Python gán vào tác tử.
- **Hai experiment:** `TrafficMARLHeadless` (huấn luyện/đánh giá headless) và `TrafficSimulation` (demo GUI có dashboard, chế độ Heuristic dùng luật rule-based phía GAML).

### 3.3. Thiết kế MDP đa tác tử

**Tác tử:** `merging_0` (xe nhập làn) và `highway_0/1/2` (xe cao tốc). Chiến lược: **chính sách dùng chung + agent indicator** (parameter sharing) — một mạng cho cả 4 tác tử, phân biệt vai trò bằng one-hot ID 4 chiều.

**Không gian quan sát:**
- Cục bộ (Actor): 19D = 15D sensor + 4D one-hot ID.
- Toàn cục (Critic): 60D = nối 15D base của cả 4 tác tử.
- Đưa vào mô hình: 79D = [local 19D | global 60D].

15 chiều của `merging_0` gồm: tốc độ, tiến độ ramp, khoảng cách tới điểm merge, lệch ngang, cờ vùng tăng tốc, khoảng trống/tốc độ xe trước–sau trên làn đích, cờ gap an toàn, xe trước trên ramp, độ khẩn, hành động trước, kiên nhẫn. 15 chiều của `highway` là radar đa hướng (trước/trái/phải) + thông tin xe ramp đang merge.

**Không gian hành động:** `Discrete(5)` — cùng chỉ số nhưng khác ngữ nghĩa giữa hai vai trò (xe nhập làn: giảm/giữ/tăng/nhập làn/chờ; xe cao tốc: tăng/giữ/giảm/rẽ trái/rẽ phải).

**Hàm phần thưởng đa mục tiêu** (chi tiết Bảng — xem [BAO_CAO_MA_NGUON_CHI_TIET.md](BAO_CAO_MA_NGUON_CHI_TIET.md) §3.4): với `merging_0`, nhập làn thành công nhận thưởng lớn, va chạm bị phạt nặng, hết đường chưa nhập bị phạt, kèm các tín hiệu dày (vào vùng tăng tốc, nhập làn đúng lúc gap an toàn, phạt do dự). Với `highway`: thoát đường an toàn được thưởng, va chạm bị phạt, phạt bám đuôi quá gần (TTC), thưởng nhường gap khi xe ramp merge.

### 3.4. Baseline không học và lưới an toàn (điểm thiết kế cho thực nghiệm A/B)

Để đánh giá việc học có thực sự đáng giá hay không, đồ án dùng hai baseline **không học**:
- **Greedy** (tham lam): tăng tốc tối đa + cố nhập làn sớm + vượt làn khi bị chặn; tránh va chạm dọc nhờ lưới an toàn của môi trường (tương tự `move-forward-greedy` trong mô hình NetLogo tham chiếu).
- **Random:** chọn hành động ngẫu nhiên đều — đóng vai trò *sàn tuyệt đối*, xác nhận bài toán không tự dễ.

> Ghi chú: chế độ "Heuristic" trên GUI dùng một bộ luật thận trọng phía GAML (`get_heuristic_*`) phục vụ demo trực quan; baseline so sánh chạy headless do Python điều khiển.

**Lưới an toàn của môi trường** (gọi tắt "khiên") gồm hai cơ chế ghi đè hành động để chống va chạm, áp **đồng đều cho mọi xe** (cả RL lẫn baseline):
1. **Gate nhập làn (`is_merge_gap_safe`):** hành động "nhập làn" chỉ thực thi khi khoảng trống trước/sau trên làn đích đủ an toàn.
2. **Shield car-following (M3c):** xe trên làn chính bị ép giảm tốc/khựng lại khi khoảng cách tới xe trước quá nhỏ, bất kể chính sách chọn gì.

Đồ án bổ sung một cờ `enable_safety_shield` cho phép **bật/tắt đồng đều cả hai cơ chế cho toàn bộ một lần chạy**. Đây là công cụ cốt lõi của thực nghiệm A/B ở Chương 5: so sánh cùng một chính sách trong môi trường *có* và *không có* lưới an toàn để tách bạch "năng lực nội tại" với "trợ giúp môi trường". (Các cơ chế chống kẹt/hồi sinh khác — anti-deadlock, respawn — là thuộc tính nền của simulator, luôn bật cho mọi chính sách, nên không ảnh hưởng tính công bằng của so sánh.)

---

## CHƯƠNG 4. CÀI ĐẶT

### 4.1. Pipeline Python

Chuỗi wrapper biến môi trường GAMA PettingZoo thành VecEnv tương thích SB3:
`GamaParallelEnv → PadPossibleAgents (giữ đủ 4 slot) → AgentIndicator (15D→19D) → GlobalState (19D→79D) → MarkovVectorEnv → GamaMarkovSB3VecEnv`.

Chính sách CTDE (`CentralizedCriticPolicy`) tách quan sát 79D: Actor đọc 19D cục bộ, Critic đọc 60D toàn cục; MLP rộng 64, hàm kích hoạt Tanh, khởi tạo trực giao (orthogonal, gain=√2) theo chuẩn MAPPO.

### 4.2. Siêu tham số

| Tham số | MAPPO | MAA2C |
|---|---|---|
| Learning rate | 3e-4 | 3e-4 |
| `n_steps` | 256 | 64 |
| `batch_size` | 128 | (full rollout) |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 0.99 / 0.95 |
| `clip_range` | 0.2 | — |
| `ent_coef` | 0.01 | 0.05 |
| Mạng / kích hoạt / init | 64 / Tanh / Orthogonal(√2) | nt |

### 4.3. Quy trình thực nghiệm tự động

`run_experiments.py` điều phối một lần chạy: vá mật độ (và tùy chọn `--shield on/off`) vào GAML → chạy baseline Greedy + Random → với mỗi (thuật toán × hạt giống): huấn luyện 200k bước (kèm lưu checkpoint định kỳ) → **chọn checkpoint tốt nhất** → đánh giá 50 episode (stochastic) → sinh biểu đồ + bảng so sánh → khôi phục GAML về mặc định.

**Chọn checkpoint tốt nhất (select-best-checkpoint):** mô hình ở bước cuối (200k) không nhất thiết tốt nhất do hiện tượng "sập" muộn. Pipeline đánh giá các checkpoint định kỳ (40k/80k/120k/160k/200k) và chọn điểm có score cao nhất.

**Cờ A/B lưới an toàn:** `--shield off` vá `enable_safety_shield ← false` cho cả run (mọi chính sách không khiên), `--shield on` (mặc định) giữ khiên; cuối run tự khôi phục về mặc định.

### 4.4. Hai cải tiến kỹ thuật cho độ tin cậy

**(a) reset-on-merging-death:** pipeline ban đầu không reset môi trường khi `merging_0` chết (do `MarkovVectorEnv(black_death=True)` che tín hiệu kết thúc), khiến rollout buffer tràn transition của tác tử đã chết và làm nhiễu gradient. Cơ chế khắc phục ép reset toàn bộ khi merging_0 chết, đồng bộ với pipeline đánh giá.

**(b) Gieo lại RNG giao thông theo hạt giống:** wrapper mặc định đặt seed ở phạm vi *experiment* sau khi reload, **không** gieo lại bộ sinh số ngẫu nhiên của *simulation* đang chạy → mọi episode sinh ra cùng một cấu hình giao thông. Đồ án bổ sung biến `pz_sim_seed` được Python đẩy xuống ngay sau reset; reflex khởi tạo xe nền gieo lại RNG từ biến này trước khi spawn → **mỗi episode có tình huống giao thông khác nhau, tái lập được, và đồng bộ (paired)** giữa các chính sách (cùng hạt giống → cùng tình huống). Cải tiến này khiến đánh giá baseline (chính sách tất định) có ý nghĩa thống kê thay vì lặp lại một tình huống duy nhất.

### 4.5. Kiểm thử & tái lập

Bộ kiểm thử tự động (pytest) kiểm tra logic Python thuần (phân tích, metrics, scenario, baseline). Kiểm tra end-to-end qua socket GAMA thực hiện bằng `smoke_test_env.py`. Mỗi lần chạy tự tạo thư mục riêng theo nhãn, không đè kết quả lần trước.

---

## CHƯƠNG 5. THỰC NGHIỆM VÀ ĐÁNH GIÁ

### 5.1. Thiết lập — Giai đoạn 1 (terminate-at-merge, đường 200m)

> Giai đoạn 1 khảo sát trên môi trường kết-thúc-tại-điểm-nhập (đường 200m, mật độ 20/45/70). Giai đoạn 2 (mục 5.8) nâng lên end-to-end trên đường 240m, mật độ 24/54/84.

Ma trận thực nghiệm: **3 mật độ** (`low` ~20 xe, `medium` ~45 xe, `high` ~70 xe) × **2 chế độ lưới an toàn** (khiên ON/OFF) × **4 chính sách** (MAA2C, MAPPO, Greedy, Random). Mỗi chính sách học chạy preset `thesis`: **5 hạt giống** × **200.000 bước** × **50 episode** đánh giá (stochastic — lấy mẫu theo phân phối chính sách). Giao thông được gieo lại theo hạt giống (mục 4.4b) nên mỗi episode là một tình huống khác nhau, đồng bộ giữa các chính sách.

Chỉ số chính: **success rate** (nhập làn xong + hoàn thành lộ trình), **collision rate**, kèm các chỉ số phụ (timeout, throughput, mean speed, shockwave).

Mọi biểu đồ và bảng tổng hợp được tính **chỉ trên các episode đánh giá (eval) và baseline** — mỗi dòng tương ứng một episode hoàn chỉnh — tách biệt khỏi rollout trong quá trình huấn luyện (vốn có thể bị bộ bao vector hóa môi trường đánh dấu sai thành "episode" và gây thống kê ảo); đường cong học dùng riêng dữ liệu huấn luyện. Riêng chỉ số "tốc độ nhập làn" trên biểu đồ radar và bước nhập làn trung bình chỉ tính trên các episode **nhập làn thành công**, tránh sai lệch khi một chính sách gần như không nhập được (bước nhập mặc định bằng 0 dễ bị diễn giải nhầm thành "nhập tức thì"). Nhờ vậy biểu đồ luôn nhất quán với bảng so sánh định lượng.

### 5.2. Kết quả tổng hợp — Tỷ lệ thành công (%)

Định dạng ô: `khiên ON | khiên OFF`; chính sách học ghi `trung bình ± độ lệch chuẩn` trên 5 hạt giống.

| Mật độ | MAA2C | MAPPO | Greedy | Random |
|---|---|---|---|---|
| low | 86,4±3,9 \| 87,2±4,1 | 83,6±11,8 \| 75,2±22,8 | 96 \| 82 | 0 \| 0 |
| medium | 89,2±1,6 \| 84,8±3,0 | 83,6±8,6 \| 76,8±17,9 | 96 \| 2 | 0 \| 0 |
| high | 82,8±4,3 \| 85,2±5,2 | 68,4±7,1 \| 69,6±9,4 | 74 \| 0 | 0 \| 0 |

### 5.3. Kết quả tổng hợp — Tỷ lệ va chạm (%)

| Mật độ | MAA2C | MAPPO | Greedy | Random |
|---|---|---|---|---|
| low | 13,6 \| 12,8 | 16,0 \| 21,6 | 4 \| 18 | 100 \| 100 |
| medium | 10,8 \| 15,2 | 16,0 \| 22,4 | 0 \| 98 | 100 \| 100 |
| high | 16,4 \| 14,8 | 31,2 \| 30,4 | 22 \| 100 | 100 \| 100 |

### 5.4. Phân tích 1 — So sánh thuật toán: MAA2C vượt trội và ổn định

Xét riêng chế độ **khiên ON** (môi trường tiêu chuẩn), MAA2C đạt 86–89% thành công ở cả ba mật độ với **độ lệch chuẩn rất nhỏ (1,6–4,3%)**. MAPPO thấp hơn (68–84%) và có **phương sai lớn hơn rõ rệt** (lên tới ±11,8% ở low). Ở mật độ cao, khoảng cách nới rộng: MAA2C 82,8% so với MAPPO 68,4%.

Nguyên nhân kỹ thuật: với cùng ngân sách 200k bước và mạng nhỏ, cập nhật trực tiếp của A2C (entropy cao 0,05, rollout ngắn) lại hội tụ ổn định hơn cho bài toán phối hợp 4 tác tử quy mô nhỏ này; trong khi PPO với cơ chế clip + nhiều epoch dễ rơi vào các cực tiểu cục bộ khác nhau tùy hạt giống → phương sai cao. Kết luận: **MAA2C là lựa chọn tốt nhất** cho kịch bản khảo sát, xét cả độ chính xác lẫn độ ổn định.

### 5.5. Phân tích 2 — A/B lưới an toàn: năng lực thật của chính sách

Đây là phát hiện trung tâm của đồ án. Khi **gỡ lưới an toàn**:

- **Greedy phụ thuộc hoàn toàn vào lưới.** Có khiên, Greedy đạt 96% (low/medium) — nhìn ngang ngửa hoặc hơn RL. Nhưng khi tắt khiên, Greedy sụp theo mật độ: **82% (low) → 2% (medium) → 0% (high)**, va chạm tương ứng tăng tới 98–100%. Nghĩa là Greedy không *biết* lái an toàn — nó tăng tốc/nhập làn bất chấp và **để cơ chế bảo vệ của môi trường gánh phần tránh va chạm**.
- **Chính sách học nội hóa an toàn nên bền.** Cùng điều kiện gỡ lưới, MAA2C gần như không đổi (low 86→87%, medium 89→85%, high 83→85% — thậm chí tăng nhẹ ở high), MAPPO chỉ giảm vừa phải (medium 84→77%, high 68→70%). RL **giữ được hành vi tránh va chạm ngay cả khi không có lưới**, vì nó đã học điều đó từ tín hiệu phần thưởng.

→ Khi đặt mọi chính sách vào **cùng điều kiện không-lưới-an-toàn** (sát thực tế triển khai hơn), **chính sách học thắng áp đảo baseline tham lam**: ở medium, MAA2C 85% so với Greedy 2%; ở high, MAA2C 85% so với Greedy 0%. Đây chính là minh chứng định lượng cho **giá trị của việc học** so với luật tay tĩnh.

### 5.6. Phân tích 3 — Ảnh hưởng của mật độ

- **Tầm quan trọng của lưới an toàn tăng theo mật độ.** Greedy không khiên: low còn 82% (đường thưa vẫn dễ tìm khe), nhưng medium/high sụp về 2%/0%. Mật độ càng cao, "vay mượn an toàn" càng không còn chỗ dựa.
- **RL suy giảm nhẹ và có kiểm soát khi mật độ tăng.** MAPPO: 84%(low)→84%(medium)→68%(high); MAA2C giữ 82–89% xuyên suốt. Không có hiện tượng sụp đổ như baseline.
- **Random là sàn cố định:** 0% thành công, 100% va chạm ở mọi cấu hình — xác nhận môi trường không tầm thường.

### 5.7. Phân tích 4 — Độ ổn định (phương sai theo hạt giống)

Độ lệch chuẩn phản ánh độ tin cậy của thuật toán khi đổi hạt giống ngẫu nhiên:
- MAA2C: 1,6–5,2% ở mọi cấu hình — **rất ổn định**.
- MAPPO: 7,1–8,6% (khiên ON) và **nổ lên 17,9–22,8% khi khiên OFF** — một số hạt giống sập (ví dụ low-OFF có seed chỉ 30%). Lưới an toàn không chỉ ảnh hưởng kết quả trung bình mà còn giúp **ổn định quá trình huấn luyện** của PPO.

### 5.8. Giai đoạn 2 — End-to-end, sửa lỗi nền tảng và nội tâm hóa an toàn

**Bài toán nâng cấp.** Kết thúc episode ngay tại điểm nhập làn không đo được sóng lùi phía sau lẫn tai nạn hậu-nhập. Giai đoạn 2 yêu cầu merger **tự lái bằng RL tới cuối đường** (end-to-end): đường kéo dài 200 lên 240m (vùng sau điểm nhập gấp 3), mật độ cao 84 xe, episode kết thúc đúng tại cuối đường.

**Phát hiện kỹ thuật trung tâm — lỗi nuốt episode boundary.** Mọi lần huấn luyện e2e ban đầu đều sụp (90% xuống 10%) bất kể thuật toán (PPO/A2C) hay reward (chuẩn hóa/làm dày): explained_variance xấp xỉ 0, value_loss xấp xỉ 1e-6 — critic không học được gì. Công cụ tự viết `probe_reward_stream.py` (soi dòng reward/done thô SB3 nhận) chỉ ra nguyên nhân: xe RL tới cuối đường bị hủy **ngay trong cùng chu kỳ mô phỏng**, trong khi bộ thu thập GAMA-Python chạy *trước* xe trong chu kỳ, nên tín hiệu kết-thúc-episode và phần thưởng terminal **không bao giờ tới được** SB3; buffer sau đó chỉ còn quan sát/reward bằng 0 (black-death) làm gradient nhiễu phá nát chính sách. Sửa bằng **grace-die** (xe RL giữ lại 2 chu kỳ cho tầng thu thập đọc tín hiệu): critic học thật (ev 0,2–0,85) và cùng quy trình huấn luyện trước đó thất bại nay đạt **90–100%**.

**Reward giai đoạn 2.** (i) **Dense progress** (potential-based shaping, Ng et al. 1999): thưởng tiến-độ-tới-đích mỗi tick thay vì chỉ thưởng lớn một lần lúc nhập làn — returns mượt, credit assignment tức thì. (ii) **Phạt can-thiệp-khiên tỷ lệ**: trừ k nhân (ga policy yêu cầu trừ ga khiên cho phép) mỗi tick.

**Kết quả giai đoạn 2** (mật độ cao 84, e2e, 10 episode/seed):

| Model | Nguồn gốc | Thành công | Va chạm | Nhường (G2) | Không-khiên |
|---|---|---|---|---|---|
| Curriculum 2-stage | from scratch 600k | 90% | 10% | 41% | 30% |
| yt50-150k (hợp tác) | warmstart nhiều giai đoạn | 90% | 10% | **94%** | 10% |
| yt52-150k (phạt tỷ lệ k=5) | tinh chỉnh từ yt50 | **100%** | **0%** | thấp | **50%** |
| Greedy | luật tĩnh | 40% | 60% | 0% | 0% |

**Phân tích.** (1) *RL thắng áp đảo ở mật độ cao trong cùng lưới an toàn* — Greedy lao sớm không khớp tốc nên đâm 60%; RL học chọn khe + căn thời điểm + khớp tốc độ (zipper merge). (2) *Phổ hành vi hợp tác - tự chủ*: checkpoint giữa nhường chủ động 94% số lần giảm tốc khi merger gần (metric "%nhường-hướng-merger" tự thiết kế, loại nhiễu histogram); checkpoint muộn merger tự tìm khe, không cần nhường vẫn 100% — trade-off hợp tác/lưu lượng đo được và chọn bằng checkpoint. (3) *Ablation khiên* (`--no-shield`): hành động rời rạc 5 mức không biểu diễn nổi điều-tốc-liên-tục nên policy ủy thác phần này cho khiên IDM-cap (lái trần còn 10–30%); chuyển phạt can-thiệp từ nhị phân -1 (chỉ kích pha khẩn) sang **tỷ lệ theo mức cắt ga** tạo gradient cho cả pha "cắt nhẹ hằng ngày", cho chuỗi 10% (binary) - 30% (k=1) - **50%** (k=5) không-khiên trong khi giữ 100% có-khiên: kỹ năng an toàn **nội tâm hóa dần qua thiết kế reward**. Mức lệ thuộc khiên đo bằng "số lần khiên can thiệp/episode" (merger giảm 9,9 xuống 0,7).

**So sánh thuật toán ở giai đoạn 2 (PPO vs A2C, cùng curriculum + cùng seed, môi trường đã sửa).** Kết quả lật lại thứ hạng của giai đoạn 1 — và quan trọng hơn, chỉ ra rằng **thứ hạng phụ thuộc chế độ đánh giá**:

| Gate giai đoạn 1 (nhập làn) | Deterministic (argmax) | Stochastic (lấy mẫu) |
|---|---|---|
| MAPPO | **100%** | ~100% |
| MAA2C (n_steps=64) | 0% | 60% |
| MAA2C (n_steps=256) | 0% | 60% |
| MAA2C (+normalize_advantage) | 0% | 30% |
| MAA2C (+lr chuẩn 7e-4, mài nhọn entropy thấp) | 0% | **90%** |

MAA2C *có học* (72% trong huấn luyện, 60-70% khi đánh giá lấy mẫu — kể cả ở giai đoạn 2 e2e với 44% nhường) nhưng mắc **deterministic collapse**: phân phối hành động không bao giờ "nhọn" về hành động nhập làn — argmax chỉ chọn giữ/tăng tốc, nên chạy chế độ tất định là 0%. Đối chứng then chốt: explained variance của critic **xấp xỉ 0 ở cả hai thuật toán** trong giai đoạn 1, vậy mà MAPPO vẫn đạt 100% argmax — chứng tỏ critic tốt không phải điều kiện cần; yếu tố quyết định là **cơ chế cập nhật trọn gói của PPO** (nhiều epoch trên mỗi batch + clipping + chuẩn hóa advantage theo minibatch). Tách từng thành phần đắp sang A2C (rollout dài hơn; bật normalize_advantage) đều không đủ. Phát hiện này cũng lý giải số liệu hai giai đoạn: "MAA2C 83% vượt MAPPO" ở giai đoạn 1 đo bằng **stochastic eval**, còn "MAPPO 100% vượt MAA2C" ở giai đoạn 2 đo bằng **argmax** — không mâu thuẫn, chỉ khác thước đo. Đặc biệt, tinh chỉnh MAA2C với learning rate chuẩn (7e-4) + entropy thấp nâng chế độ lấy mẫu lên 90% — gần MAPPO — nhưng argmax vẫn 0% (histogram nhọn 100% về tăng-tốc): bản chất là bài toán "hành-động-hiếm-sống-còn" — tăng tốc đúng ở hầu hết các bước nên trở thành mode; hành động nhập làn chỉ cần ở vài bước và chỉ nổi lên qua lấy mẫu. MAPPO đẩy được xác suất nhập làn vượt 0,5 tại đúng các trạng thái quyết định nhờ ~23.400 bước gradient (nhiều epoch × minibatch) so với 293–1.172 của MAA2C trên cùng ngân sách môi trường — chênh lệch 20–80 lần đã định lượng. Kết luận cho triển khai: MAPPO cho chính sách nhọn chạy tất định được (deployable); MAA2C đạt 90% nhưng chỉ ở chế độ lấy mẫu, kèm tính ngẫu nhiên hành vi.

**Tái lập.** `train_curriculum.sh` huấn luyện từ đầu 2 giai đoạn (học nhập làn, rồi học e2e + hợp tác; 600k bước, ~3 giờ) không sửa mã nguồn: giai đoạn 1 đạt 100% nhập làn from-scratch, giai đoạn 2 đạt 90% e2e — kết quả không phụ thuộc chuỗi huấn luyện lịch sử. Đường cong học + biểu đồ sinh tự động (`rl/make_thesis_plots.py`).

### 5.9. Sản phẩm

Với mỗi cấu hình: 10 mô hình tốt nhất (.zip), các tệp CSV chỉ số (merging + highway), biểu đồ PNG (success/collision/throughput/shockwave/radar/boxplot…), bảng so sánh và bảng LaTeX — tổ chức trong thư mục theo nhãn lần chạy (`thesis_<scenario>_shield<on/off>`).

---

## CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận

Đồ án đã hoàn thành toàn bộ mục tiêu: xây dựng môi trường mô phỏng nhập làn trên GAMA, hình thức hóa bài toán dưới dạng MDP đa tác tử, cài đặt huấn luyện MAPPO/MAA2C theo kiến trúc CTDE bằng Stable-Baselines3, và đánh giá so sánh định lượng trên ma trận mật độ × lưới an toàn với hai baseline không học.

Bốn kết luận chính:
1. **Việc học mang lại năng lực an toàn nội tại mà luật tham lam tĩnh không có** — nhất quán ở cả hai giai đoạn: giai đoạn 1 (A/B lưới an toàn) Greedy gỡ khiên sụp về 0% còn RL giữ 70–85%; giai đoạn 2 (e2e, mật độ cao, cùng khiên) RL đạt 90–100% còn Greedy đâm 60%.
2. **Hệ đạt đủ ba mục tiêu của đề cương ở chế độ end-to-end**: nhập làn an toàn (90–100%), highway phối hợp nhường chủ động (94% số lần giảm tốc hướng-merger ở checkpoint hợp tác), và đo/kiểm soát sóng lùi trọn hành trình; đồng thời phát hiện **phổ hành vi hợp tác - tự chủ** theo trục huấn luyện (zipper merge không cần nhường vẫn 100%).
3. **Đường ống huấn luyện chỉ đáng tin khi tín hiệu episode toàn vẹn**: lỗi nuốt episode-boundary ở tầng cầu nối làm critic mù (ev xấp xỉ 0) khiến mọi cải tiến thuật toán/reward vô hiệu cho tới khi sửa đúng tầng môi trường — bài học "chẩn đoán hạ tầng trước, thuật toán sau" (kèm công cụ probe tự viết).
4. **Mức lệ thuộc lưới an toàn đo được và giảm được**: ablation không-khiên + phạt can-thiệp tỷ lệ nâng năng lực lái trần từ 10% lên 50% mà không mất hiệu năng có-khiên — hướng khả thi tới chính sách tự an toàn hoàn toàn.

### 6.2. Hạn chế

- Mô phỏng tinh giản (action rời rạc + waypoint) — kết luận giới hạn trong phạm vi MDP đã định nghĩa.
- Chính sách dùng chung cho hai vai trò có ngữ nghĩa khác nhau — lựa chọn thiết kế đơn giản, không tương đương hai chính sách chuyên biệt.
- Một instance GAMA (không vector hóa song song) → thời gian huấn luyện dài, nhất là ở mật độ cao.
- Thí nghiệm A/B chỉ bật/tắt hai cơ chế lưới an toàn chính (gate nhập làn + shield car-following); các cơ chế chống-kẹt/hồi-sinh khác luôn bật cho mọi chính sách (đảm bảo công bằng nhưng không phải môi trường "trần trụi" hoàn toàn).
- Huấn luyện RL chạy trên một cấu hình giao thông cố định mỗi lần reset trong vòng train (chỉ đánh giá dùng giao thông đa dạng theo hạt giống); tái lập một phần do RNG nội bộ GAMA.

### 6.3. Hướng phát triển

- Khảo sát các phương pháp MARL nâng cao có phân rã giá trị (QMIX, value decomposition) và so với CTDE hiện tại.
- Bật và khảo sát phần thưởng hợp tác (cooperative reward) một cách có kiểm soát để đo lợi ích phối hợp tường minh.
- Mở rộng sang **hành động liên tục** (ga trong [-b, +a]) — ablation chỉ ra đây là nút thắt khiến policy ủy thác điều-tốc-tinh cho khiên; kết hợp **shield-annealing** (giảm dần khiên trong huấn luyện) hướng tới tự an toàn hoàn toàn.
- Vector hóa nhiều instance GAMA song song để rút ngắn thời gian huấn luyện và mở rộng quy mô thực nghiệm.
- Đưa giao thông đa dạng theo hạt giống vào cả vòng huấn luyện (không chỉ đánh giá) để tăng khả năng tổng quát hóa.

---

## TÀI LIỆU THAM KHẢO

1. Le Nguyen Tuan Thanh (2023). *Multi-agent reinforcement learning for traffic congestion on one-way multi-lane highways*. Journal of Information and Telecommunication, 7:3, 255–269.
2. Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., Wu, Y. (2021). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)*. NeurIPS. https://arxiv.org/abs/2103.01955
3. Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., Mordatch, I. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)*. NeurIPS.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. https://arxiv.org/abs/1707.06347
5. Terry, J.K., Black, B., Grammel, N., et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning*. NeurIPS. https://arxiv.org/abs/2009.14471
6. Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR, 22(268), 1–8.
7. Taillandier, P., Gaudou, B., Grignard, A., et al. (2019). *Building, composing and experimenting complex spatial models with the GAMA platform*. GeoInformatica, 23(2), 299–322. https://gama-platform.org
8. Ng, A.Y., Harada, D., Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping*. ICML.
9. Chen, D., Hajidavalloo, M.R., Li, Z., et al. (2023). *Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic*. IEEE T-ITS. https://arxiv.org/abs/2105.05701
10. Treiber, M., Hennecke, A., Helbing, D. (2000). *Congested traffic states in empirical observations and microscopic simulations (IDM)*. Physical Review E, 62(2).
11. Kesting, A., Treiber, M., Helbing, D. (2007). *General lane-changing model MOBIL for car-following models*. Transportation Research Record.

---

*Báo cáo phản ánh trạng thái cuối của mã nguồn: giai đoạn 1 (ma trận mật độ × lưới an toàn, preset `thesis`) + giai đoạn 2 (end-to-end, đường 240m, mật độ cao 84). Số liệu chi tiết trong `outputs/logs/`; biểu đồ `outputs/plots/` (sinh bởi `rl/make_thesis_plots.py`).*
