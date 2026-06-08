# PLAN V1 — Metric "sóng lùi" đúng nghĩa (Cách 3) — LÀM SAU

> Trạng thái: **chưa thực hiện**. Cách 1 (hạ `shockwave_index` thành phụ) + Cách 2 (thêm `mainline_mean_speed`) **đã làm xong**. File này mô tả chi tiết Cách 3 để thực hiện khi cần một thước đo "sóng lùi" theo đúng định nghĩa traffic engineering.

## 1. Vì sao cần Cách 3

`shockwave_index = std/mean` (CV tốc độ) **không phản ánh gridlock**: khi kẹt, mọi xe chậm đều nhau → std nhỏ → CV nhỏ ("êm giả"). Đã hạ nó thành chỉ số phụ và dùng `mainline_mean_speed` (thấp = kẹt) + `completion_rate` làm thước đo chính.

Tuy nhiên "sóng lùi" (stop-and-go wave / backward-propagating braking) là **hiện tượng động** (xe phanh gấp → xe sau phanh gấp → lan ngược). `mainline_mean_speed` đo *mức* kẹt nhưng không đo *tính dao động/lan truyền*. Cách 3 bổ sung metric đo trực tiếp hiện tượng này — cần khi hội đồng đòi đúng thuật ngữ "sóng lùi".

## 2. Metric đề xuất (2 chỉ số bổ sung nhau)

Gridlock có 2 dạng, cần 2 metric để phân biệt:

| Metric | Đo gì | Gridlock dao động | Đứng im hoàn toàn |
|---|---|---|---|
| **braking_event_rate** | Số lần phanh gấp / (xe×bước) | CAO | thấp |
| **time_stopped_frac** | Tỷ lệ thời gian xe đứng (speed < ngưỡng) | trung bình | CAO |

→ Dùng **cả hai**: stop-and-go thật sự = braking_event_rate cao; kẹt đứng = time_stopped_frac cao. Free-flow = cả hai thấp.

## 3. Cài đặt trong GAML (`models/Main_Traffic.gaml`)

### 3.1. Thêm field cho `car` (theo dõi tốc độ bước trước)
```gaml
float prev_speed_sw <- 0.0;   // tốc độ cycle trước, để tính giảm tốc
```
Cập nhật cuối `reflex behave` (hoặc cuối cycle): `prev_speed_sw <- speed;`

### 3.2. Thêm global counters (cạnh sw_n/sw_mean ~dòng 102–105)
```gaml
int   braking_events     <- 0;    // tổng lần phanh gấp (mainline vùng merge)
int   stopped_ticks      <- 0;    // tổng tick xe đứng (speed < ngưỡng)
int   sw_car_ticks       <- 0;    // mẫu (xe × tick) để chuẩn hoá
float hard_brake_thresh  <- deceleration * 0.6;   // ngưỡng phanh gấp (tune)
float stopped_speed_thresh <- 0.1;                // ngưỡng "đứng" (× speed_max)
```

### 3.3. Đếm trong `reflex sample_shockwave` (~dòng 1053, cùng vòng lặp lấy mẫu)
Trong loop xe mainline vùng merge (điều kiện `physical_lane = number_of_lanes-1` + vùng merge đã có sẵn), thêm:
```gaml
sw_car_ticks <- sw_car_ticks + 1;
float drop <- c_car.prev_speed_sw - c_car.speed;          // giảm tốc trong 1 sample window
if (drop > hard_brake_thresh) { braking_events <- braking_events + 1; }
if (c_car.speed < stopped_speed_thresh) { stopped_ticks <- stopped_ticks + 1; }
```
> Lưu ý: `sample_shockwave` chạy mỗi 5 cycle (`sw_due`) — `drop` là chênh tốc giữa 2 lần sample. Nếu muốn nhạy hơn, đếm mỗi cycle (tách reflex riêng `every(1 #cycles)`), nhưng tốn hơn. Bắt đầu với mỗi-5-cycle cho đồng bộ sw.

### 3.4. Reset cùng chỗ reset sw (dòng ~162 và ~913)
```gaml
braking_events <- 0;  stopped_ticks <- 0;  sw_car_ticks <- 0;
```

### 3.5. Expose trong `get_episode_info` (~dòng 3932)
```gaml
"braking_event_rate":: (sw_car_ticks > 0 ? braking_events / sw_car_ticks : 0.0),
"time_stopped_frac"::  (sw_car_ticks > 0 ? stopped_ticks  / sw_car_ticks : 0.0),
```

## 4. Plumb qua Python (giống hệt cách đã làm cho mainline_mean_speed)

1. **`rl/metrics.py`** — `EpisodeMetric`: thêm 2 field
   ```python
   braking_event_rate: float = 0.0
   time_stopped_frac: float = 0.0
   ```
   `build_episode_metric`: thêm
   ```python
   braking_event_rate=_as_float(info.get("braking_event_rate"), 0.0),
   time_stopped_frac=_as_float(info.get("time_stopped_frac"), 0.0),
   ```
2. **`rl/train_marl.py`** `MARLEpisodeCSVCallback` (merging row): thêm 2 dòng `...=float(info.get(...))`.
3. **`rl/evaluate_marl.py`** (chỗ `rows.append(EpisodeMetric(...))`): thêm 2 dòng từ `m0_info`.
4. **`rl/plots.py`** `ensure_metric_columns` defaults: thêm `"braking_event_rate": 0.0, "time_stopped_frac": 0.0`.
5. **`rl/analysis.py`** `build_comparison_table`: thêm `avg_braking_rate_*`, `avg_stopped_frac_*` (mean/std/ci95). Thêm 2 row vào LaTeX + (tùy) display_cols.

## 5. Kiểm chứng (định nghĩa "đúng" nếu)
- **Greedy medium (gridlock)**: `time_stopped_frac` CAO (xe đứng nhiều), `braking_event_rate` trung bình–cao.
- **RL medium (flow)**: cả hai THẤP (xe chạy đều).
- **Low (cả hai)**: thấp.
→ Nếu phân biệt được greedy-medium vs RL-medium rõ ràng → metric đạt. Nếu không → tune `hard_brake_thresh` / `stopped_speed_thresh` / tần suất sample.

## 6. Chạy lại
Sau khi cài: chạy lại `short --scenario low` + `short --scenario medium` (và medium nhiều seed nếu cần) để populate cột mới, rồi `analysis.py` dựng bảng.

## 7. Rủi ro / lưu ý
- `prev_speed_sw` phải cập nhật **mọi cycle** kể cả khi không sample, nếu không `drop` sai. Cân nhắc update trong `behave` (chạy mỗi cycle) thay vì trong sampler.
- Sample mỗi-5-cycle làm `drop` là chênh qua 5 cycle → ngưỡng phanh gấp phải scale theo (≈ `deceleration * 5 * hệ_số`). Tune kỹ.
- Counter là **toàn cục theo cửa sổ episode** (giống sw) → reset khi respawn/merge. Giá trị trong info phản ánh giai đoạn giữa 2 lần reset.
- Đây là metric **mức hệ thống** (mainline), gắn vào row `merging_0` (như shockwave/mainline_mean_speed hiện tại).

## 8. Định nghĩa thành công của Plan V1
Báo cáo có một thước đo **mang tên "sóng lùi / stop-and-go" bảo vệ được** (braking_event_rate + time_stopped_frac), phân biệt rõ gridlock vs flow, thay cho CV gây hiểu nhầm — khớp mục tiêu đề tài "giảm thiểu ùn tắc sóng lùi".

---

# ⚠️ QUY ƯỚC CODE BẮT BUỘC khi sửa `models/Main_Traffic.gaml`

> **Mọi thay đổi GAML (Cách 3, fix highway, reward...) PHẢI tuân theo phong cách hiện tại của file để tránh lỗi runtime.** Đây là các bẫy đã được tài liệu hoá rải rác trong file (GAMA 2025.6.x, chạy headless):

1. **KHÔNG khai báo biến local trong nested `if` ở hot-path** → gây lỗi "TempVariable". Thay bằng **field scalar khai báo ở cấp species** rồi gán/tái dùng (xem các field `m2_*`, `sw_*_cache`, `reward_cache` đã có). Mẫu: `calculate_merging_reward` dùng `m2_merge_gap_front`, `m2_urgency`...
2. **KHÔNG `do action(...)` trong reflex** (GAMA 2025.6.4: `DoStatement.getContext` NPE khi `getAgent()` null). Logic nạp action dùng **global reflex + field** (xem `apply_pz_python_each_cycle`).
3. **KHÔNG viết helper action có `return` biến cục bộ rồi gọi trong hot-path** — inline hoặc dùng field precompute.
4. **Dùng `euclid_point_dist(a,b)` thay `distance_to(point,point)`** (topology nil → NPE trên headless).
5. **Guard nil/dead khắp nơi**: `if (x != nil)`, `if (not dead(c_car))`, `try { ... } catch {}` quanh truy cập agent động (xem `sample_shockwave`, `filter_collision_neighbors`).
6. **Dùng nested `if` thay vì điều kiện ghép `and`** trong hot-path (giảm TempVariable; nhất quán với style hiện tại).
7. **Mỗi feature bật/tắt bằng cờ `enable_*`** (mặc định giữ hành vi hiện tại = giá trị đang chạy thesis, để không phá kết quả cũ).
8. **Thống kê chạy dùng Welford online** (sw_n/sw_mean/sw_M2) — KHÔNG lưu toàn bộ mẫu.
9. **Reward là single source of truth trong GAML**; Python chỉ đọc `reward_val`. Thêm term reward → sửa trong `calculate_reward` / `calculate_merging_reward`, KHÔNG shaping phía Python.
10. **Reset counter mới ở CẢ 2 chỗ reset episode** (`detect_gama_sim_reset` ~dòng 131 và respawn ~dòng 913) — giống sw_n.
11. Sau MỌI sửa GAML: **smoke test** (`python rl/smoke_test_env.py`) + kiểm cân bằng `{}` + chạy `short` 1 lần trước khi tin.

→ Khi triển khai bất kỳ mục nào dưới đây, **đối chiếu reward/metric hiện có làm khuôn mẫu** (copy pattern, không phát minh style mới).

---

# 9. BUG LOG

## 9.1. [ĐÃ FIX] eval không ghi metrics xe highway
- **Triệu chứng**: CSV `*_eval_highway.csv` ghi `mean_speed=0.00, outcome=unknown` toàn bộ.
- **Nguyên nhân**: episode kết thúc theo `merging_0` khi highway còn "running" → `final_info[highway]` rỗng → ghi default.
- **Fix** (`rl/evaluate_marl.py`, trước vòng ghi `highway_stats`): thêm fallback `last_infos` cho highway (giống `merging_0` đã có).
- **Verified**: highway `mean_speed=0.129`, `outcome=running` (thay vì 0/unknown). ✅
- **Lưu ý**: CSV highway từ các run TRƯỚC fix vẫn là 0/unknown → cần chạy lại mới populate.

---

# 10. FIX HIGHWAY RL (Option B) — LÀM SAU, cân nhắc kỹ

## 10.1. Chẩn đoán (đã có)
Eval medium PPO deterministic + `--log-actions` (đọc index theo semantics HIGHWAY: 0=tăng,1=giữ,**2=giảm**,3=trái,4=phải):
- `highway_0`: giữ 93.6% + giảm 5.7%
- `highway_1`: **giảm tốc 98%** → phanh liên tục → bò ~0
- `highway_2`: giữ 62% + giảm 38%
- `hw_collisions = 2/3` mỗi episode (dù phanh nhiều vẫn đâm).

→ **2 vấn đề**: (a) highway phanh thừa → mainline chậm; (b) vẫn va chạm → policy highway kém thật sự. merging_0 thì hoàn hảo (accel 100%, success).

## 10.2. Nguyên nhân nghi ngờ
- Thưởng tốc độ highway **quá yếu** (`speed × 0.04`) so với phạt bảo thủ (TTC penalty gap<20m, low-speed −0.05, accel-khi-gần −0.5) → agent học "chậm = an toàn".
- **Coop/flow reward TẮT** (`enable_marl_coop_reward=false`) → không có động cơ giữ flow. (Bản coop cũ bị bẫy "đứng yên farm +0.3" → đã tắt; cần redesign chứ không bật lại nguyên trạng.)
- Highway có thể bị **under-trained** (reward dồn cho merging).

## 10.3. Hướng sửa (thử từng cái MỘT, có cờ enable_)
Theo đúng quy ước ở trên (sửa trong `calculate_reward`, dùng field precompute, guard nil):
1. **Tăng thưởng giữ tốc** highway: hệ số `speed × 0.04` → cao hơn (vd 0.08–0.12), hoặc thêm term thưởng khi `speed >= 0.7×speed_max`.
2. **Giảm độ gắt bảo thủ**: TTC penalty chỉ kích hoạt khi gap<12m (thay vì <20m) / giảm hệ số 1.2.
3. **(redesign) Flow reward dựa kết quả hệ thống** thay coop-vị-trí: thưởng theo `mainline_mean_speed` (sw_mean đã có) hoặc team-reward — tránh bonus per-tick theo vị trí (nguồn bẫy "đứng yên").
4. Cải thiện tín hiệu **tránh va chạm** cho highway (vì vẫn đâm 2/3): kiểm collision-neighbor scan + obs radar đủ thông tin chưa.

## 10.4. Quy trình + chi phí (đặt kỳ vọng đúng)
- Sửa 1 thứ → retrain 1 seed (~25–30') → eval → đo `mainline_speed` + `hw_collisions` + kiểm **merging KHÔNG bị phá**.
- **Iterative**: thực tế 3–8 vòng; train 1-instance GAMA chậm; **không đảm bảo hội tụ**.
- Ước tính: **nửa ngày → vài ngày**. Rủi ro: đổi reward highway có thể phá merging → phải tune cả hai.
- **Time-box**: thử 2–3 vòng; không cải thiện rõ → quay về Option A (báo cáo như limitation).

## 10.5. Option A (mặc định nếu không đủ thời gian)
Báo cáo trung thực: "merging_0 đạt nhập làn ở mật độ cao nơi greedy gridlock; xe highway RL còn bảo thủ (phanh thừa → giảm flow) và chưa an toàn hoàn toàn (collision) — hạn chế đã định lượng, hướng cải tiến = flow-aware reward (mục 10)." Không tốn thêm train.

---

# 11. GHI CHÚ BÁO CÁO: stochastic vs deterministic
- **Deterministic** (argmax) = chế độ **deploy thật** (xe luôn chọn nước tốt nhất; SB3 default eval). Số đẹp & hợp lệ — dùng làm "năng lực policy khi triển khai".
- **Stochastic** (sampling) = đo **độ bền/bất định**; khắc nghiệt hơn (medium PPO stochastic từng thấy collision cao trong 5-ep test nhỏ — cần multi-seed × 50ep mới đáng tin).
- **Khung báo cáo đề xuất**: deterministic làm headline + stochastic làm mục "độ bền/robustness". Bộ thesis `low` đã có stochastic 5-seed (84%/71%). Nếu cần stochastic cho medium → phải train thêm seed cho medium + eval stochastic.

---

# 12. ĐIỀU TRA HIGHWAY-SLOW — KẾT QUẢ & REFRAME (đã chạy thực nghiệm)

## 12.1. Hành trình
1. **Triệu chứng**: deterministic eval → highway "giảm tốc 100%" → mainline chậm (0.18) → tưởng "RL ích kỷ / highway lỗi".
2. **Stochastic eval lật lại**: highway KHÔNG collapse — phân phối lành mạnh (≈75% giảm / 18% giữ / 5% tăng). Deterministic chỉ lấy MODE → "100%" giả. (Bài học: **luôn xem phân phối stochastic, đừng tin argmax deterministic** cho phân tích hành vi.)
3. **Thử fix** (3 cấu hình, a2c, có eval stochastic + select-best-checkpoint):

| Cấu hình | Merging success | mainline_sp | highway hành vi |
|---|---|---|---|
| Baseline shared (thesis) | **84%** (low, 5-seed) | ~0.18 | nhường (giảm tốc nhiều) |
| ① role-cond actor riêng | **80%** (low, 1-seed) | ~0.37 | nhường, ≈ baseline |
| ①+② flow reward + ③ phạt giảm-tốc-thừa | **0%** (low) / merge medium vẫn 100% det | 0.64 | giữ tốc 70%, hết nhường |

## 12.2. REFRAME (kết luận cốt lõi)
**"Highway chậm" KHÔNG phải lỗi — đó là HÀNH VI NHƯỜNG ĐƯỜNG HỢP TÁC (cooperative yielding) mà MARL đã học được, và nó CHÍNH LÀ thứ enable merging 84%.**
- Ép highway giữ tốc (②③) → highway hết nhường → xe ramp không có gap → **merge sụp 84%→0%**.
- Mainline_sp thấp (0.18) ở vùng merge = **cái giá vật lý tất yếu** của việc mở gap để nhập làn trong dòng đông.
- → Đúng tinh thần bài báo tham chiếu: **cooperation (nhường) giảm xung đột**. Đây là **kết quả TÍCH CỰC** cho luận văn, không phải defect cần sửa.

## 12.3. Quyết định
- **Revert ①②③** về baseline thesis (shared policy, flow OFF) — giữ bộ số 84%/71% đã validate + tương thích model cũ.
- **GIỮ các bổ sung hữu ích** (đã merge vào code, không phá gì): `completion_rate`, `mainline_mean_speed` (metric), fix eval-highway-metrics, preset `fast`, bỏ chế độ Manual. Code flow-reward (②③) giữ dạng **hook tắt sẵn** (`enable_highway_flow_reward=false`) cho future work.

## 12.4. Future work (cải tiến THẬT, nếu muốn)
Muốn highway VỪA giữ flow VỪA enable merge (Pareto tốt hơn) cần **cooperative team-reward đúng cách**:
- Highway **chỉ nhường khi có xe ramp cần gap gần đó** (không phanh thừa khắp nơi), giữ tốc còn lại.
- Reward theo **kết quả hệ thống** (merge success + flow), không theo vị trí (tránh bẫy "đứng yên farm").
- Có thể kèm ① (role-cond actor riêng) để 2 vai trò học độc lập.
- Chi phí: nhiều vòng train 200k + multi-seed — substantial, để sau. Hook đã sẵn (cờ + role-cond đã chứng minh chạy được).

## 12.5. Bài học phương pháp
- **Stochastic vs deterministic eval** cho hành vi rất khác → phân tích hành vi phải dùng stochastic.
- **fast 60k KHÔNG đo được highway** (merging học nhanh, highway học chậm ~200k) → tune highway phải ở ≥200k.
- Trade-off **flow ↔ merge-safety** là cốt lõi của bài toán → không có free lunch bằng tweak reward đơn giản.
