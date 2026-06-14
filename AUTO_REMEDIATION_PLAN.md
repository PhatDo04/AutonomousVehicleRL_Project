# Kế hoạch tự khắc phục yt45 (autonomous — user đang nghỉ)

Bối cảnh: yt45 = train trên đường dài (road 320, exit 320.38, nb_cars_max 112) + **khiên ngang**
(gap-gate đổi làn RL cả 2 hướng) + rl-postmerge (e2e). Warmstart yt43 250k. max-steps 800, window 700.
3 GOAL: (1) merge an toàn, (2) highway nhường-phối-hợp, (3) hạn chế sóng lùi.

## Tiêu chí đánh giá (best checkpoint trong 100/150/200/250/300k)
- **PASS**: success ≥ 75% & collision ≤ 15% & nhường-merger > 5% & mean_speed > 0.30 (không freeze)
  & merger thật sự đi tới ~320 (merge_x hợp lý, không timeout cao).
- **PARTIAL**: success ≥ 60% & collision ≤ 25% & không freeze.
- **FAIL**: success < 60% HOẶC collision > 25% HOẶC freeze (speed < 0.20) HOẶC nhường = 0%.

## Cây khắc phục (nếu FAIL) — chẩn theo TRIỆU CHỨNG
QUY TẮC AN TOÀN: sửa GAML CHỈ khi KHÔNG có train nào đang chạy (GAMA reload GAML mỗi episode →
sửa lúc train = hỏng). Trình tự: stop train+monitor → restart_headless → edit GAML → commit → train mới.
KHÔNG để Claude co-author trong commit (memory). Commit GAML thường xuyên (memory: tránh mất).

1. **COLLAPSE** (success < 40%, timeout cao): post-merge cruise 245 đơn-vị quá khó / window 700 quá dài.
   - Fix 1a: chọn checkpoint SỚM hơn nếu nó tốt hơn (đôi khi 150/200k > 300k do catastrophic forgetting).
   - Fix 1b: train thêm timesteps (300k→500k) warmstart từ checkpoint tốt nhất.
   - Fix 1c (GAML): rút đường lại 320→260 (post-merge 80 đơn-vị, vẫn dài hơn cũ 20) nếu 320 quá khó.

2. **HIGH COLLISION** (>25%) dù success ổn: nguồn breakdown-reveal + lateral.
   - Fix 2a (GAML): xe hỏng GIẢM TỐC DẦN thay vì speed→0 tức thì (line ~2037: thay `speed<-0.0` bằng
     phanh dần qua vài tick) → xe sau kịp phanh, hết pileup. ĐÂY là fix chính cho breakdown-reveal.
   - Fix 2b (GAML): siết gap-gate khiên ngang 8/6 → 12/10 (đổi làn an toàn hơn).

3. **FREEZE** (speed < 0.20): intervention penalty quá nặng trên đường dài (giống yt41).
   - Fix 3a (GAML): giảm intervention penalty -1 → -0.5 ở CẢ merger (line ~4516) LẪN highway (~4386).

4. **nhường = 0%** (không phối hợp, goal-2): highway không gặp/không nhường merger.
   - Fix 4a: kiểm vị trí highway tại merge-time (đã xác nhận spawn 35/46/57 bảo toàn) — nếu vẫn 0,
     có thể do early-merge (~x75) khiến highway đã ở phía trước → cân nhắc giảm hw speed spawn hoặc
     spawn highway upstream hơn. CẨN THẬN: đây là thay đổi tinh, đo %nhường (KHÔNG histogram).

## Mô hình tốt nhất hiện có (fallback nếu mọi remediation thất bại)
- goal2_best_550k.zip (terminate-at-merge, 52-58% nhường) — đường CŨ 200.
- yt43_hwshield 250k (83%/0% short road).
- Nếu yt45 + remediation đều tệ: giữ yt43 250k làm baseline, báo user cần can thiệp thủ công.

## Nhật ký tiến trình (auto cập nhật mỗi lần thức dậy)

### 00:48 — yt45 xong (300k), eval mọi checkpoint → FAIL (collapse)
| ckpt | success | coll | timeout | speed | nhường |
|---|---|---|---|---|---|
| 100k | 10% | 0% | 90% | 0.093 | 20% |
| 150k | 10% | 10% | 80% | 0.132 | 19% |
| 200k | 10% | 0% | 90% | 0.093 | 24% |
| 250k | 10% | 40% | 50% | 0.234 | 15% |
| 300k | 20% | 70% | 10% | 0.295 | 37% |

**Chẩn:** merger MERGE OK (failed_merge=0%, merge_x~71) nhưng KHÔNG hoàn thành cruise post-merge dài (71→320=249 đơn vị).
Sớm = freeze/bò (0.09→timeout 90%); muộn = liều→đâm 70%. **Nguyên nhân vật lý:** ở equilibrium ~0.18, 249 đơn vị
cần ~1380 tick > max-steps 800 → TIMEOUT chắc chắn. Slow-highway-equilibrium (memory) + step-budget quá ngắn.

**Round 1 (đang chạy):** diagnostic re-eval 250k @ max-steps 1500, window 1400 — xem timeout có thuần step-budget
không (nếu success tăng mạnh → fix = tăng max-steps; nếu vẫn thấp → freeze thật, cần reward-progress / rút đường).
Đây là chẩn RẺ trước khi retrain/sửa-GAML.

### 01:00 — Round 1 kết quả: KHÔNG phải step-budget, là FREEZE thật
250k @1500 steps: success 20% (chỉ +10%), **timeout VẪN 60%**, reward timeout cao (221). exit_rate highway = 0%
ở mọi eval → highway bò trong làn merge không tới cuối → chặn merger → gridlock-bò toàn hệ. Đây là
slow-highway-equilibrium (memory: cố ý, KHÔNG sửa bằng reward — code ghi chú reward-tuning gây gridlock).
→ Không đánh nhau với crawl. Đường 320 quá dài cho tốc-bò.

### 01:05 — Round 2: RÚT đường 320→240 + diagnostic yt43 250k run-to-end
Sửa GAML road_length 240, exit 240.38, nb_cars_max 84 (giữ mật độ), camera. Commit sạch. Restart headless.
Chạy diagnostic yt43 250k (nền 83% trên road 200) @ road 240, max-steps 1300, window 1250 (run-to-end).
NẾU yt43 chạy run-to-end @240 ổn (success cao, không gridlock) → retrain yt46 @240 (warmstart yt43 250k).
NẾU yt43 cũng collapse @240 run-to-end → run-to-end requirement tự nó là vấn đề → WebSearch + cân nhắc
window-based termination (như yt43 gốc 83%) thay vì run-to-end, báo user design tension khi họ quay lại.

### 01:07 — KẾT QUẢ: run-to-end là thủ phạm, KHÔNG phải đường/model. WINDOW-based = PASS
- yt43 250k @ road240 **run-to-end** (window 1250): success 20%, timeout 60% → CŨNG collapse. exit_rate hw 0% = gridlock.
- WebSearch xác nhận: Chen 2021 (nền tảng) + field dùng **fixed-horizon/window**, KHÔNG run-to-end-đường-tuỳ-ý.
- yt43 250k @ road240 **window 400** (post-merge continue 400 tick, KHÔNG run-to-end):
  **success 87.5% / collision 12.5% / timeout 0% / nhường-merger 28%** → ✅ **PASS đủ 3 goal**.
- → Window 400 vẫn đo sóng lùi (400 tick) + bắt tai nạn post-merge (đúng yêu cầu cốt lõi user phản đối terminate-AT-merge), MÀ không gridlock. Run-to-end bất khả thi do slow-equilibrium gridlock (đã chứng minh trên MỌI model).

**ĐÃ KHÓA FALLBACK:** `outputs/models/thesis_candidate_r240_win400.zip` (= yt43 250k). PASS sẵn sàng.

### 01:08 — Round 3: retrain yt46 train-ĐÚNG-config để cải thiện
yt46: road 240 + window 400 + khiên ngang, warmstart yt43 250k, max-steps 600, 300k, ckpt 50k (pid 1261, FPS 83 ~60ph).
Monitor (pid 1266) eval mỗi ckpt. Mục tiêu: ≥ fallback (87.5%/12.5%/0%TO/28%nhường). Nếu yt46 ≥ fallback → dùng yt46;
nếu không → giữ fallback yt43. Hai đường đều PASS.

### 01:36 — yt46 COLLAPSE (catastrophic forgetting) → kill, giữ fallback
yt46 (warmstart yt43 250k, train ĐÚNG config road240/win400): 100k/150k/200k đều success 10% (timeout 80% /
collision 60%). merge_x dịch 67→88-91 (merge muộn). → rl-postmerge training TỰ PHÁ nền tốt (memory cảnh báo
e2e collapse). yt43 250k (warmstart, KHÔNG train thêm) = 87.5% NHƯNG +200k training → 10%. **Train thêm HẠI.**
→ Killed yt46 (1261) + monitor (1266) lúc 237k (đừng đốt compute, đã có PASS). **GIỮ fallback yt43 250k.**
KẾT LUẬN: yt43 250k là điểm tối ưu; e2e training tiếp tục làm collapse. Fallback = MODEL THESIS cuối.

### 01:37 — Verify fallback multi-seed (chắc PASS không phải may seed 0)
Eval thesis_candidate_r240_win400.zip @ seed 1 + seed 2 (road240/win400). Nếu success vẫn ~75-87% → PASS bền,
HOÀN TẤT. Đây là deliverable cuối cho user.

### 01:50 — SEED-FRAGILITY phát hiện
Fallback yt43 250k @ road240/win400: seed0=87.5%, **seed1=30%** (collision 30%/timeout 40%). Variance LỚN
(memory yt33 đã ghi seed-fragility y hệt). → PASS seed0 KHÔNG bền. Đang chạy seed2 (15 eps) để có TB thật.
Kế: nếu TB seed0/1/2 thấp (~50%) → (a) quét NHANH các checkpoint yt43 KHÁC (200k/300k) ở seed1 (RẺ, không
retrain) tìm checkpoint bền hơn; (b) nếu không có → report HONEST cho user: approach window-400 đúng + chạy
được nhưng model seed-fragile; cần curriculum/train ổn định hơn (memory: warmstart ổn định seed) — để user quyết.
KHÔNG retrain mù.

### 01:58 — Quét checkpoint + test IDM-post-merge
- yt43 200k@seed1 = 10%, 300k@seed1 = 10% → đều TỆ HƠN 250k(30%). Fragility CỐ HỮU (mọi ckpt cùng run).
  250k là best yt43. TB seed0/1/2 ~50% success / ~25% collision (nhường 23-35% — goal-2 ổn).
- TEST quyết định: yt43 250k @ seed1/2 WINDOW 400 **KHÔNG rl-postmerge** (IDM lái post-merge, collision-free).
  Giả thuyết: fragility/collapse do RL-điều-khiển-post-merge (e2e khó). Nếu IDM-post-merge robust hơn nhiều
  → hướng khả thi: RL-merge + IDM-cruise (vẫn đo sóng lùi post-merge, collision-free). User từng muốn RL-postmerge
  nhưng nếu nó bất ổn cố hữu thì IDM-cruise là fallback hợp lý + citable. (Chờ kết quả.)

### 02:05 — KẾT LUẬN AUTONOMOUS (dừng remediation, chờ user quyết)
IDM-post-merge: seed1 = 30%/30%/40%, seed2 = 30%/40%/30% — **GIỐNG HỆT rl-postmerge** → fragility KHÔNG do
post-merge-controller mà ở **kịch bản merge tại từng seed** (seed khó: gridlock-timeout + va chạm). seed0 dễ (87.5%).

**ĐÃ VÉT CẠN phương án an toàn:**
| Phương án | Kết quả |
|---|---|
| Window-400 (vs run-to-end) | ✅ hết gridlock hệ thống, đúng chuẩn Chen 2021 |
| yt43 250k (best ckpt) | ~50% success TB (seed0 87.5/seed1 30/seed2 33), nhường 23-35% |
| Retrain yt45 (320), yt46 (240) | ❌ COLLAPSE (catastrophic forgetting, e2e bất ổn) |
| Checkpoint 200k/300k | ❌ tệ hơn 250k |
| IDM-post-merge | ❌ cùng fragility |

**BỨC TƯỜNG:** seed-fragility cố hữu + e2e training collapse. Vượt qua cần DESIGN DECISION (user):
1. **Giảm mật độ** (low/medium thay vì high=84) → seed dễ hơn, ít gridlock → success cao + bền hơn. (RẺ, đáng thử nhất.)
2. **Curriculum stabilization** train ổn định seed (memory: warmstart nền-ổn-định) — nhưng e2e collapse nên rủi ro.
3. **Chấp nhận ~50%** + báo cáo honest (RL merge an toàn 50%, nhường 23-35%, baseline so sánh).
→ DỪNG retrain liều (đúng lệnh user). Model tốt nhất giữ: outputs/models/thesis_candidate_r240_win400.zip (yt43 250k).
Mọi process train/monitor đã kill. Headless 1002 để lại cho user eval/demo. CHỜ USER.

### 02:15 — Medium density KHÔNG cứu (TB ~47%, timeout 30-40%) → phát hiện ARTIFACT max-steps
Medium(54): seed0 50%/seed1 40%/seed2 50% — biến thiên thấp hơn nhưng timeout 30-40% MỌI seed.
INSIGHT: window 400 + max-steps 600 → merger merge SAU step 200 thì merge_step+400 > 600 → timeout (KHÔNG
phải model tệ, là thiếu chỗ cho cửa sổ). seed0-high 87.5% vì merge sớm. → TEST: high(84) + **max-steps 800**
+ window 400, seeds 0/1/2. Nếu timeout→success → fragility chỉ là artifact eval-param → model THỰC SỰ tốt.
Revert nb_cars_max về 84. (Chờ kết quả b1ki3ytll.)

### 02:25 — max-steps 800 KHÔNG cứu timeout → fragility là THẬT (không artifact). KẾT THÚC autonomous.
yt43 250k @ high(84)/win400/max-steps800: seed0 70%/30%coll/0%TO, seed1 30%/30%/40%TO, seed2 30%/40%/30%TO.
→ Timeout seed1/2 KHÔNG chuyển thành success dù 800 steps → merger thật sự KẸT (gridlock/freeze) ở seed khó.
**TB thật ~43% success / ~33% collision** (seed0 70% là dễ nhất, không phải 87.5% — đó là mẫu 8-ep). Fragility CỐ HỮU.
Dọn sạch: kill yt44 zombie (885, "kill 883" chỉ giết bash wrapper) + monitor strays; GAML restored sạch (road240/84).

### 09:10 — C (đào collapse) KẾT LUẬN: value-failure MÃN TÍNH, RL fragile
- Cả PPO(yt45/46) + A2C(yt47) + norm-reward(yt48) đều explained_variance≈0 + value_loss≈0 → critic KHÔNG học.
- **KEY: yt43 (model 43% tốt nhất) CŨNG e2e(win200) + CŨNG ev≈0** → ev≈0 là MÃN TÍNH, không phân biệt OK vs collapse.
- norm-reward (VecNormalize) THẤT BẠI: yt48 100k = 10%/60-70% collapse, ev vẫn ≈0. → reward-scale KHÔNG phải gốc.
- Khác biệt yt43-OK(43%) vs yt46-collapse(10%) = **CONFIG TRAIN**: yt43=win200/density70/road200/KHÔNG-shield (dễ,
  warmstart+gradient đáp xuống 43%); yt46+=win400/density84/+lateral-shield (khó hơn → policy random-walk xuống 10%).
- Bản chất: ev≈0 → gradient variance cao (≈REINFORCE no-baseline) → config dễ landing 43%, config khó trôi xuống.
- Fix THẬT value-failure = RESEARCH (dense reward-shaping merge / critic tốt hơn / MARL algo có credit-assignment
  như QMIX). D (thêm reward) VÔ DỤNG khi ev≈0 (critic không học nổi reward mới).
- **yt43 250k (~43%, nhường 23-42%, decel thấp ~65-400 = ít crawl) là TRẦN thực tế.** Crawl nặng chỉ ở model collapse.

### 09:47 — dense-reward (yt49): ev leo đầu (0.087@3k, value_loss 500) NHƯNG relapse ≈0 cuối (value_loss 1e-7)
Dense progress reward cho critic tín hiệu GIAI ĐOẠN ĐẦU (value_loss 500, ev dương) nhưng tới 200k returns lại
near-constant (value_loss 1e-7, ev 0) → policy converged (tốt?) hoặc collapse lại. Eval yt49 100k+200k seed0
đang chạy để quyết. Nếu success giữ ≥43% → dense work (ev=0 cuối là converged lành tính); nếu 10% → dense
chưa đủ → C-fix khó hơn dự kiến.

### 10:55 — dense-reward (yt49) CŨNG COLLAPSE: 100k=20%/50%, 200k=10%/20%/70%TO
Dense cho critic tín hiệu ĐẦU train (ev 0.087, value_loss 500) nhưng khi policy trượt vào crawl thì progress
gần-hằng-số → value relapse ≈0 → collapse như cũ. BẢNG TỔNG KẾT 6 retrain từ yt43 250k:
| run | thay đổi | kết quả |
|---|---|---|
| yt45 | road 320 | collapse 10-20% |
| yt46 | road 240 + win400 + lateral-shield | collapse 10% |
| yt47 | A2C | collapse 10% |
| yt48 | norm-reward (VecNormalize) | collapse 10% |
| yt49 | DENSE progress + sparse 200→50 | collapse 10-20% |
→ **MỌI retrain trên config mới đều phá policy warmstart** (43%→10%), bất kể algo/scale/density-reward.
Nguyên nhân sâu: critic chưa bao giờ học được (ev≈0 mãn tính, kể cả yt43) → gradient ≈ REINFORCE variance cao
→ "train thêm" = random-walk khỏi điểm tốt. yt43 đáp 43% là nhờ config dễ hơn (win200/road200/no-shield).
Untried còn lại: from-scratch + dense (cần ~1-2M steps + curriculum như Chen 2021 = NHIỀU GIỜ, không chắc).

### 11:30 — 💥 ROOT CAUSE THẬT (probe + đọc mã nguồn): die_if_out_of_road NUỐT episode boundary
PROBE (rl/probe_reward_stream.py — log dòng reward THÔ SB3 thấy): merge bonus +50.9 CÓ đến (reward stream OK)
NHƯNG **episodes=0 trong 600 steps** — done=True KHÔNG BAO GIỜ đến SB3! Khớp CSV train: 2 episodes/300k steps.
CƠ CHẾ: die_if_out_of_road (GAML ~1631) set is_done=true rồi `do die` NGAY CÙNG CYCLE. petz_collect_tick chạy
TRƯỚC car trong cycle → cycle sau tìm xe thì ĐÃ CHẾT (nil) → termination + reward terminal KHÔNG sang Python.
Pad-wrapper luôn giữ agents=4 → tín hiệu dự phòng (_merging_just_died signal-2) cũng mù. Sau đó black_death
bơm obs=0/reward=0 VÔ TẬN → buffer toàn zeros → returns hằng → ev≈0/value_loss≈1e-6 → gradient rác → collapse.
GIẢI THÍCH TRỌN: yt43 (win200 < road-end t311) terminal qua window-path (KHÔNG die, grace 2 tick) → DELIVERED
→ học được 43%. yt45-49 (road-end ĐẾN TRƯỚC window) → mọi episode success bị nuốt → collapse. Highway exit
cũng die câm → exit_rate=0% mọi eval. A2C/norm/dense fail chung vì bug ở ENV.
FIX: RL agents KHÔNG `do die` ngay trong die_if_out_of_road — set terminal + để behave dead_timer (grace 2
tick) giết xe → collect kịp giao termination. Highway exit giờ có terminal "exited". Probe-2 đang verify.

### 12:15 — 🎉 yt50 (env đã fix) PHÁ TRẦN: 90-100% success / 0-10% coll / NHƯỜNG 94-97% / hw exit 40%+
ev train giữ 0.2-0.85 suốt (critic học thật, lần đầu). yt50 150k seed0: 90%/10%/TO 0%/nhường 97%/hw-exit 43.3%/
hw-coll 3.3%; seed1: 100%/0%/nhường 94%/hw-exit 40%/hw-coll 0%. Highway avg_reward -42→+52. Chuỗi nhân-quả
xác nhận trọn: die-ngay nuốt boundary → fix → critic học (ev 0.85) → policy học thật → CẢ 3 GOAL.
Đang eval 150k seed2 + 200k/300k seed0 để chọn best checkpoint làm thesis model.

### 13:07 — CHỐT: yt50 150k = MODEL THESIS → outputs/models/thesis_e2e_yt50_150k.zip
150k robust 3 seeds: 90/100/90% success, 0-10% coll, nhường 94-97%, hw exit 37-43%, hw coll 0-7%.
200k dip (60%); 300k: 100%/0% NHƯNG nhường 0% (highway học không-bao-giờ-phanh = mất goal-2, catastrophic
forgetting cuối train — đúng pattern memory). → 150k là điểm cân bằng 3 goal. Greedy baseline (3 seeds,
cùng config road240/e2e) đang chạy làm bảng so sánh thesis.

### 15:00-15:55 — SỰ CỐ DENSITY + TRUE-HIGH RE-EVAL + ABLATION
- **SỰ CỐ:** GUI-play patch nb_cars_max=54 (medium) lẫn vào commit → yt50 train + mọi eval "hôm nay" chạy MEDIUM.
  User nghi "high mà dễ quá" → lộ. Khôi phục 84 (commit cb42ed8). BÀI HỌC: verify nb_cars_max trước mọi train/eval.
- **TRUE-HIGH 84 seed0:** RL-150k 90%/10% (nhường 94%, decel 357, hw-coll 0%); RL-250k 90%/10% (nhường 0, ỷ-khiên
  merger 4.0 thấp nhất); GREEDY 40%/60%coll (đâm ngay t≈41-43) → **kịch bản KHÔNG dễ, RL ≫ greedy, phân biệt rõ.**
- **yt50 GENERALIZE:** train@54 vẫn 90%@84, tự tăng nhường (decel 124→357) theo mật độ.
- **G3 @84 (seed1):** RL-150k: 70%succ, CV .637, mainline .464 | RL-250k: 80%, .569, .498 | greedy: 70%, .449, .629
  (greedy nhanh hơn ~25% nhưng seed0 đâm 60% — "nhanh+chết" vs "chậm+sống"). Variance seed đáng kể @ high.
- **Cơ chế 250k merge-không-cần-nhường:** zipper merge — merge muộn hơn (x̄ 71.8 vs 65.2), căn pha khe tự nhiên
  giữa các cụm IDM + khớp tốc độ → NPC sau chỉ nhả ga nhẹ. Greedy fail vì lao sớm không khớp tốc.
- **ABLATION NO-SHIELD đang chạy** (cờ pz_no_shield, commit 1c5f4e1): RL nội-tâm-hóa an toàn hay ỷ khiên?
  Dự đoán ghi trước: 250k dip nhẹ 75-85%, 150k giữ, greedy sập <20%.

### 16:23 — ABLATION NO-SHIELD: khiên là BỘ PHẬN CHỊU LỰC (dự đoán sai, ghi nhận trung thực)
| @84 seed0 | Có khiên | KHÔNG khiên |
|---|---|---|
| RL-150k | 90%/10% | 10%/90% |
| RL-250k | 90%/10% | 20%/80% |
| Greedy | 40%/60% | 0%/100% |
- IDM-cap hoạt động MỌI TICK (điều tốc liên tục), không chỉ pha khẩn → RL học phân vai "policy=ý định,
  cap=tinh chỉnh" → rút cap thì merger đâm đuôi (Lazy-Agent đúng cảnh báo user; phạt -1 chỉ áp pha khẩn).
- Thứ bậc GIỮ NGUYÊN mọi chế độ: có khiên RL 90 ≫ greedy 40 (cùng khiên = giá trị HỌC); không khiên RL 10-20 > greedy 0.
- **Highway RL gần TỰ an toàn không khiên (coll 3.3-6.7%, vẫn nhường 98%)** — nội tâm hóa tốt; chỉ merger post-merge lệ thuộc.
- Khung thesis: hệ đề xuất = RL + shield (như Chen 2021); ablation tách "chất lượng quyết định" khỏi "khiên";
  hạn chế ghi thẳng: merger chưa nội tâm hóa điều tốc liên tục → future work: phạt theo mức-cap / shield-annealing.

## 🏁 KẾT LUẬN CUỐI (16:30) — BỘ SỐ LIỆU THESIS HOÀN CHỈNH @ TRUE HIGH 84
**Model:** `outputs/models/thesis_e2e_yt50_150k.zip` (hợp tác, nhường 94-97%) + `checkpoints/yt50_fixedenv/yt50_fixedenv_250000_steps.zip` (tự chủ, zipper).
**Bảng chính (seed0, 10 eps):** RL-150k 90%/10%, nhường 94%, hw-coll 0% | RL-250k 90%/10%, nhường 0%, ỷ-khiên min | Greedy 40%/60%.
**G3 (seed1):** mainline RL .46-.50 vs greedy .63 (nhanh hơn ~25% nhưng seed0 đâm 60% — "nhanh+chết" vs "chậm+sống").
**Variance seed @ high:** RL-150k 70-90%, 250k 80-90%, greedy 40-70% — báo cáo khoảng, không cherry-pick.
**Chuỗi phát hiện kỹ thuật:** (1) die-cùng-cycle nuốt episode boundary (root cause collapse, commit 0331ccc);
(2) dense progress reward + ev 0.85; (3) GUI-patch density lẫn commit (cb42ed8); (4) metric ỷ-khiên; (5) ablation no-shield.

## ⛔ TÓM TẮT CHO USER (ĐỌC TRƯỚC) — autonomous đã vét cạn, cần bạn quyết (LỖI THỜI — xem 12:15+13:07: ĐÃ GIẢI QUYẾT)
**ĐÃ LÀM (commit sạch, không Claude co-author):**
1. Khiên NGANG đổi làn RL (gap-gate 2 hướng, chỉ RL; NPC giữ MOBIL citable) — fix bug bạn phát hiện (đổi làn không gap → đâm).
2. Khiên IDM-cap + intervention penalty cho highway RL (đối xứng merger).
3. Thử nới đường 320 (collapse) → rút 240 (post-merge ~60 đơn-vị tại merge_x, density 84).

**PHÁT HIỆN LỚN:**
- **Run-to-end ("đi tới cuối đường") BẤT KHẢ THI** với slow-highway-equilibrium → gridlock (highway bò dồn, exit_rate 0%, merger kẹt). Chứng minh trên MỌI model. WebSearch: Chen 2021 + field dùng **window/horizon**, không run-to-end. → Chuyển **window-400** (chạy tiếp 400 tick sau merge: vẫn đo sóng lùi + bắt tai nạn post-merge — đúng yêu cầu cốt lõi của bạn).
- **Model seed-fragile:** yt43 250k (best) = TB **~43% success / ~33% collision / nhường 29-42%** (seed0 dễ 70%, seed1/2 ~30%).
- **Đã LOẠI TRỪ (test rẻ, không retrain):** density medium ✗, checkpoint 200k/300k ✗ (tệ hơn), IDM-post-merge ✗ (cùng fragility), max-steps 800 ✗. **Retrain yt45/yt46 → COLLAPSE** (e2e training catastrophic forgetting — train thêm HẠI).

**CẦN BẠN QUYẾT (mình không tự làm vì rủi ro/đụng memory):**
1. **Breakdown-reveal fix** (Fix 2a chưa thử): xe hỏng giảm-tốc-dần thay speed→0 tức thì → giảm va chạm pileup (~33% collision có thể có phần từ đây). GAML dynamics, ít rủi ro, NHƯNG cần retrain mới hưởng đủ.
2. **Curriculum stabilization** chống seed-fragility (memory: warmstart nền-ổn-định) — nhưng e2e đang collapse, rủi ro.
3. **Chấp nhận ~43%** + báo cáo honest (RL nhường 29-42% là goal-2 tốt; so baseline greedy/base_rule chưa chạy ở config này).

**Model tốt nhất:** `outputs/models/thesis_candidate_r240_win400.zip` (= yt43 250k). Headless port 1002 còn sống cho demo.
**Eval lại:** `./.venv/Scripts/python.exe rl/evaluate_marl.py --algo ppo --model outputs/models/thesis_candidate_r240_win400.zip --episodes 10 --seed 0 --port 1002 --max-episode-steps 800 --eval-continue --postmerge-window 400 --rl-postmerge`

**CHO USER khi quay lại (design decision):** run-to-end ("đi tới cuối đường") BẤT KHẢ THI với slow-highway-
equilibrium (gridlock, đã chứng minh + đúng lý thuyết Chen 2021 dùng horizon). Đã chuyển sang window-400
(post-merge 400 tick — vẫn đo sóng lùi + bắt tai nạn sau merge). Nếu user vẫn muốn run-to-end thật: cần
GIẢM MẬT ĐỘ mạnh (low density, hết gridlock) HOẶC sửa crawl-equilibrium (rủi ro, memory cấm) — cần user quyết.

## 🌙 ĐÊM 11/6 — PROPSHIELD (phạt tỷ lệ cắt ga) + THESIS CURRICULUM RUN
- yt51 k=1.0 (200k, warmstart yt50-150k): có-khiên 80%/nhường 93% (nền giữ), NO-shield 10→30% (đúng hướng, yếu — clip 0.04/tick bị progress 0.42/tick lấn).
- yt52 k=5.0: @150k có-khiên 100%/0%, ỷ-khiên merger 0.7/ván, NO-SHIELD 50% (×5 baseline) — ĐẠT nhánh thành công. @200k: nhường hồi 95% (decel 219) nhưng no-shield 30% — trade-off nội-tâm-hóa ↔ nhường theo checkpoint.
- Commits: 9360f67 (prop penalty), 03e0a51 (k=5.0). Số liệu narrative thesis: binary→prop = 10%→50% no-shield.
- 01:58 LAUNCH thesis-run: train_curriculum.sh 1001 0 (from scratch 2-stage, code k=5) — full learning curve.
  Stage1 ~80ph → gate-eval ≥70% merge; Stage2 ~80ph → sweep checkpoint (cân bằng success+nhường+no-shield).

## 🏁 SÁNG 11/6 — THESIS CURRICULUM RUN HOÀN TẤT (from-scratch, code k=5)
- GATE Stage1 (300k, terminate-at-merge): **100% merge, 0% collision, from scratch** — curriculum stage 1 hoàn hảo.
- Sweep Stage2 (e2e): 100k=40%, **150k=90%/10% (nhường 41%, ỷ-khiên 3.4)** ← BEST, 200k=30%, 250k=60%, 300k=60%/nhường 0%.
- curr-150k bổ sung: có-khiên seed1=70%/30% (variance seed như mọi model); NO-shield=30% (giữa yt50 10% và yt52 50% —
  from-scratch 300k chưa đủ nội-tâm-hóa bằng yt52 fine-tune; hợp lý).
- **CHỐT: outputs/models/thesis_curriculum_final.zip** (= curr_stage2_e2e 150k).

### BẢNG SO SÁNH CUỐI (@ high 84, seed0, có khiên | no-shield)
| Model | Lineage | Success | Nhường | No-shield |
|---|---|---|---|---|
| thesis_curriculum_final (150k S2) | FROM SCRATCH 600k tổng | 90% (70% s1) | 41% | 30% |
| thesis_e2e_yt50_150k | warmstart nhiều đời | 90% (70-90%) | 94% | 10% |
| yt52-150k propshield | yt50 + k=5 fine-tune | 100% | ít (decel 7) | **50%** |
| greedy | rule-based | 40% | 0% | 0% |
→ Trả lời 'train thesis từ đầu có đạt như hiện tại?': **CÓ về G1 (90%≈90%), MỘT PHẦN về G2 (41% vs 94% — nhường
cần thêm steps/curriculum giai đoạn 3), no-shield giữa đường (30%)**. Learning curve FULL 2 stage đã có:
curr_stage1_merge_episodes.csv + curr_stage2_e2e_episodes.csv + tensorboard/curr_* → đủ vẽ biểu đồ thesis.

## 🔬 11/6 CHIỀU-TỐI — SO SÁNH THUẬT TOÁN PPO vs A2C (code đã fix, công bằng)
User phát hiện "A2C 0% vô lý" → đào ra DETERMINISTIC COLLAPSE + chuỗi thí nghiệm:

| Gate Stage-1 (cùng curriculum, seed 0) | argmax | stochastic | ev train |
|---|---|---|---|
| PPO | **100%** | ~100% | ≈0 (!) |
| A2C n_steps=64 (gốc) | 0% (TO 70-100%) | 60% | ≈0 |
| A2C n_steps=256 (fix C, b5ab469) | 0% | 60% | ≈0 |
| A2C +normalize_advantage (fix D, 72787be) | 0% | 30% | ≈0 |
| Stage-2 e2e: PPO 90% vs A2C n64: argmax 10-40% / stoch 60-70% (nhường 34-44%) |

PHÁT HIỆN THEN CHỐT:
1. A2C HỌC ĐƯỢC (train-time stochastic 72%, eval stochastic 60-70%) nhưng KHÔNG BAO GIỜ nhọn
   về action-merge (histogram argmax: chỉ keep+accel, 0 lần action-3 là mode) = deterministic collapse.
2. ĐỐI CHỨNG QUYẾT ĐỊNH: PPO ev CŨNG ≈0 ở stage 1 mà argmax 100% → critic tốt KHÔNG phải điều
   kiện cần; yếu tố quyết định là CƠ CHẾ UPDATE TRỌN GÓI của PPO (nhiều epoch + clip + norm-adv
   theo minibatch). Tách riêng từng mảnh (rollout dài, norm-adv) đắp vào A2C đều KHÔNG đủ.
3. Giải mã lịch sử: "MAA2C 83% ≫ MAPPO" kỷ nguyên cũ = STOCHASTIC eval (kỷ nguyên đó eval stochastic)
   — không mâu thuẫn với "PPO 100% ≫ A2C 0%" argmax hiện tại. Hai kỷ nguyên đo 2 thứ khác nhau.
KẾT LUẬN ĐỀ CƯƠNG: PPO vượt trội về deployability (chính sách nhọn, chạy deterministic được);
A2C chỉ dùng được ở chế độ lấy mẫu (60-72%, kèm rủi ro hành vi ngẫu nhiên). Mọi số có nguồn log:
a2c_s1_gate/a2c_s1_mid/a2c_stoch_test/a2c_s2_sweep/a2c_v2_gate/a2c_v3_gate.log + curr_a2c_*_episodes.csv.

### 20:50 — V4 (fine-tune lr 7e-4 chuẩn SB3 + ent≈0, 100k): STOCHASTIC 60→90%! argmax vẫn 0 (histogram 100% accel)
GIẢI ĐẾN ĐÁY — bài toán "rare-critical-action": accel đúng ở hầu hết bước → policy nhọn về accel hợp lý cục bộ;
action-merge chỉ cần ở vài bước trong zone. Stochastic: p(3)~0.2-0.4 × ~50 bước → merge gần chắc chắn (90%).
Argmax: p(3) không vượt p(accel) tại bước quyết định → 0%. PPO thắng nhờ 23.400 gradient-steps đẩy p(3)>0.5
đúng chỗ; A2C 293-1172 steps không đủ (định lượng). KẾT: A2C cứu được 90% stochastic (fix B+E work);
deterministic-deployability là đặc quyền của multi-epoch update (PPO). DỪNG chuỗi A2C — trọn vẹn.
