// MARL-only (repo chinh): 4 RL agents merging_0 + highway_0/1/2, experiment TrafficMARLHeadless.
model HorizontalTraffic

global {
    int   number_of_lanes <- 3;
    float lane_width      <- 3.5;
    float road_length     <- 200.0;
    // Mainline: clip thap hon nguong die 1 chut de moi tick co the vuot nguong -> die (tranh ket tai bien).
    float road_mainline_exit_x <- 200.38;
    float road_mainline_clip_x  <- 200.58;
    float road_move_clip_x     <- 220.0; // chi dung trong nhanh merge_mode=1 (behave)
    float offset_y        <- 18.0;

    geometry shape <- rectangle(road_length, 100.0);

    float speed_max            <- 1.0;
    float speed_min            <- 0.0;
    float acceleration         <- 0.05;
    float deceleration         <- 0.1;
    // IDM (Intelligent Driver Model — Treiber 2000) cho xe NPC/HDV. Tham số scaled về đơn vị sim
    // (speed∈[0,1]/cycle, x theo mét), KHÔNG phải giây thật. time_headway theo cycle.
    float idm_time_headway     <- 2.5;   // T VỪA (e2e): khe followable post-merge (RL lái được, khiên hiếm fire → goal-3 của RL) NHƯNG vẫn cần nhường đôi lúc (goal-2). Cực-scarce (1.5) làm post-merge sập 90%. IDM collision-free mọi T>0.
    float idm_min_gap          <- 2.0;   // s0: bumper gap tối thiểu (mét)
    // MOBIL (Kesting/Treiber 2007) cho đổi làn HDV: an toàn + lợi ích + politeness.
    float idm_politeness       <- 0.3;   // p: mức quan tâm tới phanh ép lên xe sau làn đích
    float idm_b_safe           <- 0.2;   // b_safe: giảm tốc tối đa cho phép ép lên xe sau khi cắt vào
    float idm_lc_threshold     <- 0.12;  // ngưỡng lợi ích tối thiểu mới đổi làn (Kesting 2007 Δa_th; 0.01 quá thấp → NPC đổi làn vô cớ; nâng ~0.12 so b_safe=0.2 → chỉ đổi khi lợi ích thật)
    int   nb_cars_max          <- 70;   // GUI mac dinh nhe hon 45 de giam gridlock; headless co the tang qua tham so / Python
    float observation_distance <- 10.0;

    list<point> ramp_waypoints <- [];
    geometry    ramp_shape;
    list<geometry> ramp_surface_geoms <- [];
    list<geometry> ramp_joint_geoms <- [];
    list<geometry> ramp_joint_edge_geoms <- [];
    list<geometry> ramp_edge_geoms <- [];
    list<geometry> ramp_center_geoms <- [];
    list<geometry> accel_center_dash_geoms <- [];
    list<geometry> accel_outer_dash_geoms <- [];
    // Nen lane tang toc + vung gore (chi UI; khong doi ramp_waypoints / behave xe).
    list<geometry> ui_road_fill_geoms <- [];
    list<geometry> ui_accel_bottom_edge_geoms <- [];

    // Chi so waypoint dang huong toi diem ket thuc doan lane tang toc ngang (tru diem merge cuoi).
    // Tranh hard-code `4` vi length ramp_waypoints co the doi; invariant: 2 diem cuoi = {accel_end} -> {merge}.
    int ramp_accel_waypoint_ix <- 4;

    float bottom_lane_y;
    float accel_lane_y;
    float accel_start_x <- 48.0;
    float accel_end_x   <- 162.0;
    float merge_x       <- 180.0;
    // Khoảng cách spawn giữa 3 xe highway RL (làn nhập). 22m (cũ) → khe luôn đủ rộng để merger
    // slot vào mà 3 xe KHÔNG cần nhường → không học được hợp tác. Siết xuống để tạo "tường" thật,
    // buộc merger phải chờ/né hoặc highway RL phải giảm tốc mở khe. Dùng ở 2 chỗ: spawn RL + skip NPC.
    float hw_rl_spacing <- 11.0;

    float gap_front_max <- 7.0;
    float gap_front_min <- 8.0;
    float gap_rear_max  <- 6.0;
    float gap_rear_min  <- 6.0;

    // driving_policy / learning_rate / exp_rate: tham số cho xe NPC (non-RL) chạy trên cao tốc.
    // KHÔNG ảnh hưởng RL agents — merging_0 và highway_0/1/2 nhận action từ Python qua PzBridgeAgent.
    // Nếu GAMA-Python desync (timeout/disconnect), merging_0 fallback về action_rl = 1 (giữ tốc).
    // "Greedy" chỉ áp dụng cho xe NPC (non-RL) khi driving_policy = "Greedy".
    // Xe RL agent (merging_0, highway_0/1/2) LUÔN nhận action từ Python/SB3
    // thông qua PzBridgeAgent — biến này KHÔNG ảnh hưởng đến hành vi của chúng.
    string driving_policy  <- "Greedy";
    float  discount_factor <- 0.99;
    float  learning_rate   <- 0.001;
    float  exp_rate        <- 1.0;
    float  exp_decay       <- 0.005;

    float  collision_distance <- 3.5;
    // max_patience: giá trị khởi tạo cho trường patience của từng xe (đơn vị: counter nội bộ).
    // KHÔNG phải điều kiện terminate episode — max_episode_steps Python wrapper mới quyết định timeout.
    // 1 Python step = 1 GAMA cycle (bước mô phỏng). max_episode_steps=300 > max_patience=100 là ổn.
    int    max_patience       <- 100;
    float  observation_max    <- 100.0;

    string dashboard_policy <- "Python/SB3 model";
    bool   show_dashboard <- true;

    // Legacy restore flags: bat tung nhom logic cu co kiem soat, tranh bat dong loat gay NPE/timeout headless.
    bool enable_rl_respawn <- true;
    bool enable_highway_rl_respawn <- false;   // C: xe highway chạy hết đường thì RỜI, KHÔNG hồi sinh về x=0
                                               // (bỏ cảnh teleport→tăng tốc đâm đuôi). Merge sớm nên không mất tương tác nhường.
    bool enable_balance_traffic <- true;
    bool enable_collision_neighbor_scan <- true;
    // Mesh ramp (polygon + joint + vach giua): luon tinh mot lan trong global init — khong dung co bat/tat.
    bool enable_reward_collision_check <- true;
    bool enable_highway_ttc_reward <- true;
    bool enable_merging_gap_reward <- true;
    bool enable_marl_coop_reward <- true;   // PlanB: bật để khôi phục động lực NHƯỜNG, đối trọng flow reward
    // GLOBAL/REGIONAL reward coef (MARL4AV local+global): thưởng highway theo flow dòng chính (sw_mean).
    // Chống gridlock attractor (gridlock→sw_mean=0→mất thưởng). Đặt 0 để tắt.
    float global_flow_coef <- 0.3;
    // SVO/REGIONAL reward (MARL4AV — cơ chế cooperation chính): highway nhận PHẦN reward của merger
    // khi merger GẦN (trong merge zone). merger merge thành công (+200) gần highway nào → highway đó
    // hưởng social_coef×200 → việc NHƯỜNG (mở khe để merger merge gần mình) trở nên tự-lợi → goal 2.
    float idm_social_coef <- 0.25;   // GOAL 2 retry trên headless SẠCH (research-backed local+global flow reward)
    // Flow-aware highway reward (thử nghiệm Plan V1 §10): bù việc xe highway phanh thừa → giữ lưu lượng.
    // ON  = thưởng giữ tốc cao + TTC penalty nhẹ hơn (gap<12, coef 0.8).
    // OFF = baseline hiện tại (speed*0.04, TTC gap<20 coef 1.2). Để A/B sạch.
    // GHI CHÚ: bật ON phá merging (highway hết nhường → ramp đâm). highway chậm = cooperative yield. Giữ OFF.
    bool enable_highway_flow_reward <- true;   // PlanB: bật chống highway phanh-thừa/đứng-im (kết hợp oppcost + coop). Cảnh báo cũ "phá merging" là khi CHƯA có oppcost+coop — đo merge_step/collision để kiểm.
    // false = chi merge khi policy/heuristic goi action 3 + gap an toan (KPI eval co y nghia).
    bool enable_curriculum_merge_assist <- false;
    // CURRICULUM (#2): force_action3_only=true tắt auto-merge cuối ramp → success CHỈ qua action-3.
    // merge_gap_relax (cho merging_0): <1 nới gap dễ học action-3 (stage 1), =1 ngặt như thật (stage 2).
    bool  force_action3_only <- true;
    float merge_gap_relax    <- 1.0;
    // COMMIT-TO-MERGE (#B targeted): action-3 = CAM KẾT nhập (quyết định THÔ, cửa sổ rộng) → môi
    // trường thực thi merge ở tick gap-an-toàn kế tiếp + thưởng +200 NGAY tại tick merge. Biến quyết
    // định "tinh-thời-điểm" (không học nổi) thành "thô" (học được), trị đúng bài credit-assignment.
    bool  commit_to_merge    <- true;
    // AN TOÀN KHÔNG CẦN SHIELD: commit-to-merge LUÔN chờ gap THẬT an toàn mới thực thi (gate gap
    // thật được tách khỏi cờ shield) → merge tự an toàn dù enable_safety_shield=false.
    // 2026-06-07: TẮT gate (false) → bỏ lưới đỡ cho MỌI policy (greedy+RL công bằng). Merge thực thi
    // theo quyết định agent; sai khe → đâm thật. RL phải TỰ học merge đúng lúc (không dựa env crutch).
    bool  commit_gate_real   <- false;

    // KHIEN AN TOAN GAMA. Mac dinh OFF: theo huong "bo shield de RL hoc that". Merge van an toan
    // nho commit_gate_real (doc lap voi co nay). Rear-end cua highway phai tri bang REWARD + train lai
    // (phat car-following), KHONG che bang shield. A/B thuc nghiem: run_experiments --shield on/off.
    bool enable_safety_shield <- false;

    // FIX seed traffic: Python set bien nay (rl/gama_episode_reset.set_sim_seed) qua _execute_expression
    // SAU reset -> bootstrap reseed `seed <- pz_sim_seed` o SIMULATION-scope -> traffic bien thien theo
    // seed Python (reproducible + paired). -1 = chua set (giu hanh vi cu, khong reseed).
    float pz_sim_seed <- -1.0;
    // EVAL-ONLY post-merge: =1 → merger KHÔNG kết thúc tại merge mà chạy tiếp tới cuối làn (post-merge IDM)
    // → đo SÓNG LÙI trọn vẹn sau merge. CHỈ bật khi EVAL (Python set qua expression). Train giữ =0
    // (terminate-at-merge, ổn định — post-merge-continue lúc TRAIN làm merger né-merge, đã xác nhận yt22).
    float pz_eval_continue <- 0.0;
    float pz_postmerge_window <- 50.0;   // EVAL-continue: chạy tiếp pz_postmerge_window tick sau merge để đo đuôi sóng lùi rồi end success (tránh chạy tới cuối đường → gridlock do highway chậm + tích lũy xe)
    float pz_rl_postmerge <- 0.0;        // =1 → RL điều khiển merger SAU merge (end-to-end, thesis); =0 → IDM (cũ). Dùng kèm pz_eval_continue=1 + window lớn (chạy tới cuối) để đo trọn hành trình + merge-không-tai-nạn-sau.

    // Throughput counters: đếm tổng số xe ramp đã cố merge và số lần merge thành công.
    int total_ramp_attempts <- 0;   // Tăng khi xe ramp vào acceleration zone
    int total_merge_success <- 0;   // Tăng khi bất kỳ xe nào merge thành công

    // Cooperative reward: số cycle còn lại để highway RL nhận bonus (TTL), tránh phụ thuộc thứ tự reflex bool + reset.
    // Mỗi đầu cycle global reflex `apply_pz_python_each_cycle` giảm TTL — merge set TTL=3 → bao phủ ~2–3 cycle sau khi merge.
    int    recent_merge_coop_ticks <- 0;
    float  recent_merge_x          <- -1.0;   // Vị trí x merge gần nhất
    // Vị trí x của merger ĐANG trong accel-lane chờ nhập (cập nhật mỗi cycle qua reflex track_active_merger).
    // Dùng để THƯỞNG NHƯỜNG TRƯỚC merge: highway giảm tốc khi merger gần phía trước → mở khe. -1 = không có.
    float  active_merger_x         <- -1.0;

    // Xe hỏng: giới hạn số xe hỏng đồng thời trên làn chạy (port từ NetLogo damaged-nb-cars-inlane).
    int damaged_nb_cars_inlane <- 0;  // Số xe hỏng đang chiếm làn chạy
    int max_damaged_inlane     <- 3;  // Tối đa 3 xe hỏng cùng lúc

    // Shockwave index: đo độ bất ổn tốc độ xe mainline ở vùng merge.
    // Dùng Welford's online algorithm để tính mean và variance không cần lưu toàn bộ mẫu.
    // shockwave_index = std_dev / mean_speed (Coefficient of Variation) — thấp = êm, cao = sóng lùi.
    int    sw_n      <- 0;      // Số mẫu đã thu thập
    float  sw_mean   <- 0.0;   // Running mean (Welford)
    float  sw_M2     <- 0.0;   // Running sum of squared deviations (Welford)
    // SÓNG LÙI đúng chuẩn (literature: "emergency braking events"; plan_v1 Cách 3): đếm phanh gấp
    // ở dòng chính vùng merge / tổng (xe×mẫu). Đo TRONG episode (merger tiếp cận+merge gây phanh)
    // → KHÔNG cần post-merge-continue. braking_event_rate = braking_events / sw_car_ticks.
    int    braking_events    <- 0;
    int    sw_car_ticks      <- 0;
    float  hard_brake_thresh <- 0.15;   // drop tốc giữa 2 mẫu (5 cycle) > ngưỡng ⇒ 1 lần phanh gấp

    car selected_car <- nil;
    // Con tro tam khi nap pz_actions -> action_rl (global action, 1 loop) — tranh (c as car) lap trong bieu thuc -> TempVariable.
    car pz_action_target <- nil;
    // Khoa map copy sang string global truoc contains_key/at — giam TempVariableExpression trong if (GAMA 2025.6.4).
    string pz_apply_aid <- "";
    
    // Cache danh sách xe an toàn để tránh ConcurrentModificationException
    // khi Python gửi lệnh reset/tiêu hủy agent từ thread khác.
    list safe_cars <- [];
    bool initial_cars_created <- false;
    // Mirror global cho PettingZoo socket: Python eval biến global thay vì
    // `PzBridgeAgent[0]...`, tránh lỗi GAMA hiểu nhầm species index thành skill.
    list pz_agents <- [];
    list pz_possible_agents <- [];
    map pz_observation_spaces <- [];
    map pz_action_spaces <- [];
    map pz_observations <- [];
    map pz_infos <- [];
    map pz_actions <- [];
    map pz_data <- [];
    int prev_sim_cycle <- 0;
    bool pz_block_merging_respawn <- false;

    reflex detect_gama_sim_reset when: (cycle > 0) and every(1 #cycles) {
        // Guard cycle > 0: tick dau tien GAMA 2025.6.x co the chua kip set IScope.getRoot()
        // -> NPE khi set global. Bo qua tick 0 (khong can detect reset gi o luc moi khoi tao).
        try {
            if (cycle < prev_sim_cycle) {
                do hard_reset_world_for_new_episode();
            }
            prev_sim_cycle <- cycle;
        } catch {}
    }

    action hard_reset_world_for_new_episode {
        try {
            list kill_list <- [];
            if (safe_cars != nil) {
                loop c over: safe_cars {
                    if (c != nil) {
                        kill_list << c;
                    }
                }
            }
            loop kc over: kill_list {
                if (kc != nil) {
                    ask kc as car {
                        do die;
                    }
                }
            }
            safe_cars <- [];
            initial_cars_created <- false;
            total_ramp_attempts <- 0;
            total_merge_success <- 0;
            sw_n <- 0;
            sw_mean <- 0.0;
            sw_M2 <- 0.0;
            braking_events <- 0;
            sw_car_ticks <- 0;
            recent_merge_coop_ticks <- 0;
            recent_merge_x <- -1.0;
            damaged_nb_cars_inlane <- 0;
            pz_block_merging_respawn <- false;
        } catch {}
    }

    reflex update_safe_cars {
        // GAMA 2025.06.4 co the NPE AddStatement khi prune bang list cuc bo + `<<`
        // trong global reflex. Giu cache non-nil; cac diem dung safe_cars da guard nil/dead.
        if (safe_cars = nil) {
            safe_cars <- [];
        }
    }

    // NHUONG: tim merger dang trong accel-lane cho nhap -> set active_merger_x (x lon nhat = gan merge nhat).
    // Highway reward dung gia tri nay de thuong giam toc (mo khe) khi merger gan phia truoc.
    reflex track_active_merger {
        active_merger_x <- -1.0;
        if (safe_cars != nil) {
            loop c over: safe_cars {
                if (c != nil) {
                    car cc <- c as car;
                    if (not dead(cc)) {
                        if (cc.merge_mode = 1) {
                            if (cc.in_accel_zone) {
                                if (cc.move_next_x > active_merger_x) {
                                    active_merger_x <- cc.move_next_x;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 6.1 cleanup: bo N (xe trong safe_cars) trong tung episode.
    // Pattern counter (M5) tranh `cycle mod N` trong `when:` -> TempVariable.
    // Moi 50 ticks: rebuild `safe_cars`, chi giu xe non-nil va not dead.
    int  purge_tick <- 0;
    bool purge_due  <- false;
    list purge_new_cars <- [];

    reflex purge_tickup {
        purge_tick <- purge_tick + 1;
        purge_due <- false;
        if (purge_tick >= 50) {
            purge_tick <- 0;
            purge_due <- true;
        }
    }

    reflex purge_safe_cars when: purge_due {
        try {
            purge_new_cars <- [];
            if (safe_cars != nil) {
                loop c over: safe_cars {
                    if (c != nil) {
                        car c_car <- c as car;
                        if (c_car != nil) {
                            if (not dead(c_car)) {
                                purge_new_cars << c_car;
                            }
                        }
                    }
                }
            }
            safe_cars <- purge_new_cars;
        } catch {}
    }

    // PettingZoo: nap `pz_actions` -> `action_rl`. Chi dung reflex + field global `pz_action_target` — KHONG `do action(...)`
    // (GAMA 2025.6.4: DoStatement.getContext NPE khi getAgent() null trong reflex architecture).
    reflex apply_pz_python_each_cycle when: (cycle > 0) and every(1 #cycles) {
        // Guard cycle > 0: GAMA 2025.6.x scope race tai tick 0 -> NPE khi set global.
        try {
            if (recent_merge_coop_ticks > 0) {
                recent_merge_coop_ticks <- recent_merge_coop_ticks - 1;
            }
            if (pz_actions != nil) {
                loop c over: safe_cars {
                    // Per-iteration try/catch: GAMA 2025.6.x co the NPE o `contains_key` neu
                    // pz_actions bi Python set nil giua iteration. Outer try/catch khong bat
                    // duoc do NPE wrap qua BinaryOperator lambda. Wrap iteration de 1 xe loi
                    // khong stop ca loop.
                    try {
                        pz_action_target <- nil;
                        pz_apply_aid <- "";
                        if (c != nil) {
                            pz_action_target <- c as car;
                            if (pz_action_target != nil) {
                                if (not dead(pz_action_target)) {
                                    pz_apply_aid <- pz_action_target.rl_agent_id;
                                    if (pz_apply_aid != nil) {
                                        if (pz_apply_aid != "") {
                                            // Tranh `contains_key` (GAMA 2025.6.4 NPE).
                                            // Dung `at` truc tiep: tra nil neu key missing, khong NPE.
                                            unknown action_val <- pz_actions at pz_apply_aid;
                                            if (action_val != nil) {
                                                pz_action_target.action_rl <- int(action_val);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        pz_action_target <- nil;
                        pz_apply_aid <- "";
                    } catch {}
                }
            }
        } catch {}
    }

    // Dong bo observation/reward GAMA -> Python moi cycle do species petz_collect_tick
    // (action tick_sync_from_world) dam nhiem. Khong co action sync nao o cap global.

    // Diem tren polyline ramp theo arc-length (meter theo mang hinh — dung cho net dut deu trong init).
    action ramp_point_at_distance(float dist_along) type: point {
        if (ramp_waypoints = nil) { return nil; }
        if (length(ramp_waypoints) = 0) { return nil; }
        if (dist_along <= 0.0) {
            if (ramp_waypoints[0] != nil) {
                return ramp_waypoints[0];
            }
            return nil;
        }
        float d <- dist_along;
        loop i from: 0 to: length(ramp_waypoints) - 2 {
            if (ramp_waypoints[i] != nil) {
                if (ramp_waypoints[i + 1] != nil) {
                    point pw0 <- ramp_waypoints[i];
                    point pw1 <- ramp_waypoints[i + 1];
                    float sdx <- pw1.x - pw0.x;
                    float sdy <- pw1.y - pw0.y;
                    float sl <- sqrt(sdx * sdx + sdy * sdy);
                    if (sl > 0.0001) {
                        if (d <= sl) {
                            float tr <- d / sl;
                            return {pw0.x + sdx * tr, pw0.y + sdy * tr};
                        }
                        d <- d - sl;
                    }
                }
            }
        }
        int last_ix <- length(ramp_waypoints) - 1;
        if (last_ix >= 0) {
            if (ramp_waypoints[last_ix] != nil) {
                return ramp_waypoints[last_ix];
            }
        }
        return nil;
    }

    init {
        bottom_lane_y <- offset_y + (number_of_lanes - 1) * lane_width + lane_width / 2.0;
        accel_lane_y  <- offset_y + number_of_lanes * lane_width + lane_width / 2.0;

        // Quy dao ramp: x tang dan (khong lui x), cong vao tam lane tang toc -> thang -> cheo nhap lane (nhu line do).
        float ramp_dy_drop <- accel_lane_y - bottom_lane_y;
        float ramp_outer_y <- offset_y + number_of_lanes * lane_width + lane_width;
        ramp_waypoints <- [
            {12.0, ramp_outer_y - 0.15},
            {28.0, accel_lane_y + lane_width * 0.38},
            {40.0, accel_lane_y + lane_width * 0.10},
            {accel_start_x, accel_lane_y},
            {accel_end_x,   accel_lane_y},
            {accel_end_x + (merge_x - accel_end_x) * 0.38, accel_lane_y - ramp_dy_drop * 0.45},
            {accel_end_x + (merge_x - accel_end_x) * 0.72, accel_lane_y - ramp_dy_drop * 0.82},
            {merge_x, bottom_lane_y}
        ];

        ramp_accel_waypoint_ix <- 0;
        // Chi danh dau waypoint DAU lane tang toc (khong phai cuoi) — RL merge theo in_accel_zone, khong theo 1 diem cuoi.
        loop wi from: 0 to: length(ramp_waypoints) - 1 {
            if (ramp_waypoints[wi] != nil) {
                if (abs(ramp_waypoints[wi].x - accel_start_x) < 0.02) {
                    if (abs(ramp_waypoints[wi].y - accel_lane_y) < 0.02) {
                        ramp_accel_waypoint_ix <- wi;
                    }
                }
            }
        }
        if (ramp_accel_waypoint_ix <= 0) {
            if (length(ramp_waypoints) >= 6) {
                ramp_accel_waypoint_ix <- length(ramp_waypoints) - 4;
            } else if (length(ramp_waypoints) >= 2) {
                ramp_accel_waypoint_ix <- length(ramp_waypoints) - 2;
            }
        }

        // M4: cache scalar wp0 cho reflex spawn_ramp_cars (tranh `ramp_waypoints[0].x` -> TempVariable trong hot path).
        ramp_wp0_x <- 12.0;
        ramp_wp0_y <- ramp_outer_y - 0.15;
        if (length(ramp_waypoints) > 0) {
            if (ramp_waypoints[0] != nil) {
                ramp_wp0_x <- ramp_waypoints[0].x;
                ramp_wp0_y <- ramp_waypoints[0].y;
            }
        }
        ramp_wp0_ready <- true;

        // Khong dung polyline(ramp_waypoints) de tranh NPE getX/getY trong GAMA 2025.06.4.
        // ramp_shape giu lai de tuong thich, phan ve ramp se dung doan line theo waypoint trong display.
        ramp_shape <- nil;
        ramp_surface_geoms <- [];
        ramp_joint_geoms <- [];
        ramp_joint_edge_geoms <- [];
        ramp_edge_geoms <- [];
        ramp_center_geoms <- [];
        accel_center_dash_geoms <- [];
        accel_outer_dash_geoms <- [];
        ui_road_fill_geoms <- [];
        ui_accel_bottom_edge_geoms <- [];
        float gore_y_init <- offset_y + number_of_lanes * lane_width;

        // Nen ramp + bien + joint + vach giua: tinh mot lan o day (hinh hoc tinh sau init), giong y tuong road_segment.
        if (length(ramp_waypoints) > 1) {
                loop i from: 0 to: length(ramp_waypoints) - 2 {
                    if (ramp_waypoints[i] != nil) {
                        if (ramp_waypoints[i + 1] != nil) {
                            point p0 <- ramp_waypoints[i];
                            point p1 <- ramp_waypoints[i + 1];
                            float dx <- p1.x - p0.x;
                            float dy <- p1.y - p0.y;
                            float seg_len <- sqrt(dx * dx + dy * dy);
                            if (seg_len > 0.0001) {
                                // Chi coi la taper khi doan BAT DAU tu accel_end tro di (doan 48->162 van ve mesh).
                                bool seg_in_taper <- false;
                                if (p0.x >= accel_end_x - 0.5) {
                                    seg_in_taper <- true;
                                }
                                float half_w <- lane_width / 2.0;
                                if (seg_in_taper) {
                                    // Ribbon hep doc tam duong taper — tranh lem xam ra ngoai duong cheo trang.
                                    half_w <- lane_width * 0.30;
                                }
                                float nx <- -dy / seg_len;
                                float ny <- dx / seg_len;
                                point a <- {p0.x + nx * half_w, p0.y + ny * half_w, 0.01};
                                point b <- {p1.x + nx * half_w, p1.y + ny * half_w, 0.01};
                                point c <- {p1.x - nx * half_w, p1.y - ny * half_w, 0.01};
                                point d <- {p0.x - nx * half_w, p0.y - ny * half_w, 0.01};
                                float e1y_mid <- (a.y + b.y) / 2.0;
                                float e2y_mid <- (d.y + c.y) / 2.0;
                                float e1x_mid <- (a.x + b.x) / 2.0;
                                float e2x_mid <- (d.x + c.x) / 2.0;
                                // Vung taper: chi nen tam giac gore (ui_road_fill), khong mesh rong — tranh lem xam ngoai duong cheo.
                                if (not seg_in_taper) {
                                    if (polygon([a, b, c, d]) != nil) {
                                        ramp_surface_geoms << polygon([a, b, c, d]);
                                    }
                                }
                                // Chan cac edge "loi" len lane cao toc o doan taper (gay net cheo khong hop ly).
                                if (not seg_in_taper) {
                                if (not (
                                        (abs(e1y_mid - gore_y_init) < 0.25)
                                        and (e1x_mid >= accel_start_x - 1.0)
                                        and (e1x_mid <= merge_x + 1.0)
                                    )) {
                                    if (not (
                                            (e1x_mid >= accel_start_x - 1.0)
                                            and (e1y_mid < gore_y_init + lane_width * 0.60)
                                        )) {
                                    if (line([{a.x, a.y, 0.04}, {b.x, b.y, 0.04}]) != nil) {
                                        ramp_edge_geoms << line([{a.x, a.y, 0.04}, {b.x, b.y, 0.04}]);
                                    }
                                    }
                                }
                                if (not (
                                        (abs(e2y_mid - gore_y_init) < 0.25)
                                        and (e2x_mid >= accel_start_x - 1.0)
                                        and (e2x_mid <= merge_x + 1.0)
                                    )) {
                                    if (not (
                                            (e2x_mid >= accel_start_x - 1.0)
                                            and (e2y_mid < gore_y_init + lane_width * 0.60)
                                        )) {
                                    if (line([{d.x, d.y, 0.04}, {c.x, c.y, 0.04}]) != nil) {
                                        ramp_edge_geoms << line([{d.x, d.y, 0.04}, {c.x, c.y, 0.04}]);
                                    }
                                    }
                                }
                                }
                                // Vien duoi ramp cong (truoc lane tang toc ngang); taper chi dung net cheo display.
                                if (p1.x <= accel_start_x + 0.01) {
                                    if (e1y_mid >= e2y_mid) {
                                        if (e1y_mid > gore_y_init + 0.15) {
                                            if (line([{a.x, a.y, 0.05}, {b.x, b.y, 0.05}]) != nil) {
                                                ui_accel_bottom_edge_geoms << line([{a.x, a.y, 0.05}, {b.x, b.y, 0.05}]);
                                            }
                                        }
                                    } else {
                                        if (e2y_mid > gore_y_init + 0.15) {
                                            if (line([{d.x, d.y, 0.05}, {c.x, c.y, 0.05}]) != nil) {
                                                ui_accel_bottom_edge_geoms << line([{d.x, d.y, 0.05}, {c.x, c.y, 0.05}]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (length(ramp_waypoints) > 2) {
                    loop j from: 1 to: length(ramp_waypoints) - 2 {
                        if (ramp_waypoints[j - 1] != nil) {
                            if (ramp_waypoints[j] != nil) {
                                if (ramp_waypoints[j + 1] != nil) {
                                    point p_prev <- ramp_waypoints[j - 1];
                                    point pj <- ramp_waypoints[j];
                                    point pj3 <- {pj.x, pj.y, 0.01};
                                    point p_next <- ramp_waypoints[j + 1];
                                    float dx0 <- pj.x - p_prev.x;
                                    float dy0 <- pj.y - p_prev.y;
                                    float dx1 <- p_next.x - pj.x;
                                    float dy1 <- p_next.y - pj.y;
                                    float l0 <- sqrt(dx0 * dx0 + dy0 * dy0);
                                    float l1 <- sqrt(dx1 * dx1 + dy1 * dy1);
                                    if (l0 > 0.0001) {
                                        if (l1 > 0.0001) {
                                            float hwj <- lane_width / 2.0;
                                            float n0x <- -dy0 / l0;
                                            float n0y <- dx0 / l0;
                                            float n1x <- -dy1 / l1;
                                            float n1y <- dx1 / l1;
                                            point L0 <- {pj.x + n0x * hwj, pj.y + n0y * hwj, 0.01};
                                            point L1 <- {pj.x + n1x * hwj, pj.y + n1y * hwj, 0.01};
                                            point R0 <- {pj.x - n0x * hwj, pj.y - n0y * hwj, 0.01};
                                            point R1 <- {pj.x - n1x * hwj, pj.y - n1y * hwj, 0.01};
                                            bool pj_in_taper <- false;
                                            if (pj.x >= accel_end_x - 0.5) {
                                                pj_in_taper <- true;
                                            }
                                            // Dung 2 tam giac de tranh tu-cat quad tai khuc ngoat manh (gay den/nhap nhay).
                                            if (not pj_in_taper) {
                                                if (polygon([L0, L1, pj3]) != nil) {
                                                    ramp_joint_geoms << polygon([L0, L1, pj3]);
                                                }
                                                if (polygon([R0, R1, pj3]) != nil) {
                                                    ramp_joint_geoms << polygon([R0, R1, pj3]);
                                                }
                                            }
                                            float lenL <- sqrt((L1.x - L0.x) * (L1.x - L0.x) + (L1.y - L0.y) * (L1.y - L0.y));
                                            float lenR <- sqrt((R1.x - R0.x) * (R1.x - R0.x) + (R1.y - R0.y) * (R1.y - R0.y));
                                            // Khong ve joint-edge o taper/cuoi accel (x>=accel_end): gay net trang cheo "chi len" o giao gore.
                                            if (not pj_in_taper) {
                                                if (lenL < lane_width * 1.25) {
                                                    if (line([{L0.x, L0.y, 0.04}, {L1.x, L1.y, 0.04}]) != nil) {
                                                        ramp_joint_edge_geoms << line([{L0.x, L0.y, 0.04}, {L1.x, L1.y, 0.04}]);
                                                    }
                                                }
                                                if (lenR < lane_width * 1.05) {
                                                    if (line([{R0.x, R0.y, 0.04}, {R1.x, R1.y, 0.04}]) != nil) {
                                                        ramp_joint_edge_geoms << line([{R0.x, R0.y, 0.04}, {R1.x, R1.y, 0.04}]);
                                                    }
                                                } else {
                                                    if (pj.x < accel_start_x - 1.0) {
                                                        if (line([{R0.x, R0.y, 0.04}, {R1.x, R1.y, 0.04}]) != nil) {
                                                            ramp_joint_edge_geoms << line([{R0.x, R0.y, 0.04}, {R1.x, R1.y, 0.04}]);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                float rp_total <- 0.0;
                loop ir from: 0 to: length(ramp_waypoints) - 2 {
                    if (ramp_waypoints[ir] != nil) {
                        if (ramp_waypoints[ir + 1] != nil) {
                            point pr0 <- ramp_waypoints[ir];
                            point pr1 <- ramp_waypoints[ir + 1];
                            float rdx <- pr1.x - pr0.x;
                            float rdy <- pr1.y - pr0.y;
                            rp_total <- rp_total + sqrt(rdx * rdx + rdy * rdy);
                        }
                    }
                }
                float dash_w <- 2.5;
                float gap_w <- 3.5;
                float step_full <- dash_w + gap_w;
                int max_center_dashes <- 320;
                loop kd from: 0 to: max_center_dashes {
                    float s_run <- kd * step_full;
                    if (s_run + 0.001 < rp_total) {
                        point qc0 <- ramp_point_at_distance(s_run);
                        point qc1 <- ramp_point_at_distance(min(rp_total, s_run + dash_w));
                        if (qc0 != nil) {
                            if (qc1 != nil) {
                            // Chi ve vach giua trong phan lane gia toc "on dinh";
                            // dung truoc doan taper nhap lan de tranh net cheo vo ly vao lane chinh.
                            if (qc0.y > gore_y_init + 0.2) {
                                if (qc1.y > gore_y_init + 0.2) {
                                    // Chi giu phan vach giua cua doan ramp cong; lane gia toc se co vach ngang rieng.
                                    if (qc0.x < accel_start_x - 0.5) {
                                        if (qc1.x < accel_start_x - 0.5) {
                                            if (line([{qc0.x, qc0.y, 0.03}, {qc1.x, qc1.y, 0.03}]) != nil) {
                                                ramp_center_geoms << line([{qc0.x, qc0.y, 0.03}, {qc1.x, qc1.y, 0.03}]);
                                            }
                                        }
                                    }
                                }
                            }
                            }
                        }
                    }
                }
            }

        float accel_dash_len_init <- 2.5;
        float accel_period_init <- 6.0;
        int accel_k_max <- int(road_length / accel_period_init) + 2;
        float outer_y_init <- offset_y + number_of_lanes * lane_width + lane_width;
        float accel_center_y_init <- gore_y_init + lane_width / 2.0;

        // Nen lane tang toc ngang (48..162) + tam giac gore (tren duong cheo, khong dung hinh thang lon).
        if (polygon([
            {accel_start_x, gore_y_init, 0.008},
            {accel_end_x, gore_y_init, 0.008},
            {accel_end_x, outer_y_init, 0.008},
            {accel_start_x, outer_y_init, 0.008}
        ]) != nil) {
            ui_road_fill_geoms << polygon([
                {accel_start_x, gore_y_init, 0.008},
                {accel_end_x, gore_y_init, 0.008},
                {accel_end_x, outer_y_init, 0.008},
                {accel_start_x, outer_y_init, 0.008}
            ]);
        }
        // Tam giac gore: 3 dinh trung khop net cheo; goc merge lui nhe de khong lem xam ra ngoai.
        float gore_tri_mx <- merge_x - 0.40;
        float gore_tri_my <- gore_y_init;
        if (polygon([
            {accel_end_x, gore_y_init, 0.008},
            {gore_tri_mx, gore_tri_my, 0.008},
            {accel_end_x, outer_y_init, 0.008}
        ]) != nil) {
            ui_road_fill_geoms << polygon([
                {accel_end_x, gore_y_init, 0.008},
                {gore_tri_mx, gore_tri_my, 0.008},
                {accel_end_x, outer_y_init, 0.008}
            ]);
        }

        // Nen lan khan cap: dai tu (offset_y - lane_width) den offset_y, mau long duong cao toc.
        float rescue_lane_y_init <- offset_y - lane_width;
        if (polygon([
            {0.0, rescue_lane_y_init, 0.008},
            {road_length, rescue_lane_y_init, 0.008},
            {road_length, offset_y, 0.008},
            {0.0, offset_y, 0.008}
        ]) != nil) {
            ui_road_fill_geoms << polygon([
                {0.0, rescue_lane_y_init, 0.008},
                {road_length, rescue_lane_y_init, 0.008},
                {road_length, offset_y, 0.008},
                {0.0, offset_y, 0.008}
            ]);
        }

        // Tam tat net dut precompute de tranh loi geometry runtime; se co cac net co so ve truc tiep o display.

        loop i from: 0 to: number_of_lanes - 1 {
            create road_segment {
                lane_index <- i;
                float cy   <- offset_y + (i * lane_width) + (lane_width / 2.0);
                location   <- {road_length / 2.0, cy};
                shape      <- rectangle(road_length, lane_width);
                color      <- rgb(80, 80, 80);
            }
        }
        create PzBridgeAgent number: 1 {
            // MARL: 1 xe nhập làn (merging_0) + 3 xe cao tốc làn dưới (highway_0/1/2) cùng học chính sách.
            possible_agents    <- ["merging_0", "highway_0", "highway_1", "highway_2"];
            agents             <- copy(possible_agents);
            observations       <- [];
            rewards            <- [];
            terminations       <- [];
            truncations        <- [];
            infos              <- [];
            actions            <- [];
            data               <- [];
            // Tất cả agent dùng cùng 15D observation space để SuperSuit có thể dùng shared policy.
            observation_spaces <- [
                "merging_0"  :: ["type"::"Box", "low"::0.0, "high"::1.0, "shape"::[15], "dtype"::"float"],
                "highway_0"  :: ["type"::"Box", "low"::0.0, "high"::1.0, "shape"::[15], "dtype"::"float"],
                "highway_1"  :: ["type"::"Box", "low"::0.0, "high"::1.0, "shape"::[15], "dtype"::"float"],
                "highway_2"  :: ["type"::"Box", "low"::0.0, "high"::1.0, "shape"::[15], "dtype"::"float"]
            ];
            // 5 action dùng chung: merging_0=(0 giảm,1 giữ,2 tăng,3 nhập làn,4 chờ); highway=(0 tăng,1 giữ,2 giảm,3 rẽ trái,4 rẽ phải).
            action_spaces      <- [
                "merging_0"  :: ["type"::"Discrete", "n"::5],
                "highway_0"  :: ["type"::"Discrete", "n"::5],
                "highway_1"  :: ["type"::"Discrete", "n"::5],
                "highway_2"  :: ["type"::"Discrete", "n"::5]
            ];
            // Bootstrap payload de Python reset dau tien luon thay du agent keys, tranh RuntimeError got [].
            observations <- [
                "merging_0"::list_with(15, 0.0),
                "highway_0"::list_with(15, 0.0),
                "highway_1"::list_with(15, 0.0),
                "highway_2"::list_with(15, 0.0)
            ];
            rewards <- [
                "merging_0"::0.0,
                "highway_0"::0.0,
                "highway_1"::0.0,
                "highway_2"::0.0
            ];
            terminations <- [
                "merging_0"::false,
                "highway_0"::false,
                "highway_1"::false,
                "highway_2"::false
            ];
            truncations <- [
                "merging_0"::false,
                "highway_0"::false,
                "highway_1"::false,
                "highway_2"::false
            ];
            infos <- [
                "merging_0"::["outcome"::"bootstrapping", "success"::false, "collision"::false, "failed_merge"::false],
                "highway_0"::["outcome"::"bootstrapping", "success"::false, "collision"::false, "failed_merge"::false],
                "highway_1"::["outcome"::"bootstrapping", "success"::false, "collision"::false, "failed_merge"::false],
                "highway_2"::["outcome"::"bootstrapping", "success"::false, "collision"::false, "failed_merge"::false]
            ];
            data <- [
                "Observations"::observations,
                "Rewards"::rewards,
                "Terminations"::terminations,
                "Truncations"::truncations,
                "Infos"::infos
            ];
            pz_possible_agents <- possible_agents;
            pz_agents <- agents;
            pz_observation_spaces <- observation_spaces;
            pz_action_spaces <- action_spaces;
            pz_observations <- observations;
            pz_infos <- infos;
            pz_data <- data;
        }
        create petz_apply_tick number: 1 {}
        create petz_collect_tick number: 1 {}
        // Dong bo global maps cho socket (Discrete map doi khi None qua bridge neu chi gan trong species init).
        pz_action_spaces <- [
            "merging_0"::["type"::"Discrete", "n"::5],
            "highway_0"::["type"::"Discrete", "n"::5],
            "highway_1"::["type"::"Discrete", "n"::5],
            "highway_2"::["type"::"Discrete", "n"::5]
        ];
        // Khong sync bridge trong init de tranh context local chua on dinh (NPE this.local).
    }

    // Trì hoãn create car sang reflex đầu tiên để chắc chắn simulation scope đã sẵn population.
    reflex bootstrap_initial_cars when: ((not initial_cars_created) and (cycle > 0)) {
        // FIX seed traffic: gama_client_wrapper set `seed <- X` o EXPERIMENT-scope SAU reload ->
        // KHONG reseed RNG cua simulation dang chay -> traffic giong het moi episode. Ep reseed o
        // SIMULATION-scope (reflex nay) ngay TRUOC spawn -> traffic bien thien theo seed Python truyen
        // vao (reproducible). Chi headless (dashboard="Python/SB3 model"); GUI demo KHONG dinh.
        if (dashboard_policy = "Python/SB3 model" and pz_sim_seed >= 0.0) {
            seed <- pz_sim_seed;
        }
        // --- 1. KHỞI TẠO XE TRÊN CAO TỐC CHÍNH ---
        int cars_per_lane_boot <- int(nb_cars_max / number_of_lanes);
        float spacing_boot <- road_length / (cars_per_lane_boot + 1);
        loop i from: 0 to: number_of_lanes - 1 {
            loop j from: 0 to: cars_per_lane_boot - 1 {
                // Né vị trí dành cho 3 xe RL highway (làn đáy, x=8/30/52):
                // nếu spawn NPC đè lên RL → start là va chạm → NPC biến mất. Bỏ qua slot trùng.
                float nx_boot  <- 10.0 + j * spacing_boot;
                bool  skip_npc <- false;
                if (i = number_of_lanes - 1) {
                    loop hw_k from: 0 to: 2 {
                        if (abs(nx_boot - (35.0 + hw_k * hw_rl_spacing)) < 9.0) { skip_npc <- true; }
                    }
                }
                if (not skip_npc) {
                    create car number: 1 {
                        speed              <- rnd(0.4, speed_max);
                        color              <- rgb(200 + rnd(55), 200 + rnd(55), 200 + rnd(55));
                        is_merging         <- false;
                        in_accel_zone      <- false;
                        current_lane_index <- i;
                        target_lane_index  <- i;
                        location           <- {nx_boot + rnd(-3.0, 3.0), offset_y + (i * lane_width) + (lane_width / 2.0)};
                        move_next_x        <- location.x;
                        move_next_y        <- location.y;
                        direction          <- 1;
                        heading            <- 0.0;
                        is_rl_agent        <- false;
                        rl_agent_id        <- "";
                    }
                }
            }
        }

        // --- 2. KHỞI TẠO merging_0 TRÊN NHÁNH NHẬP LÀN (đầu ramp_waypoints, 1 xe RL) ---
        if (length(ramp_waypoints) >= 2) {
            point wp0_boot <- ramp_waypoints[0];
            point wp1_boot <- ramp_waypoints[1];
            if (wp0_boot != nil) {
                if (wp1_boot != nil) {
                    create car number: 1 {
                        is_merging         <- true;
                        merge_mode         <- 1;
                        terminal_reason    <- "running";
                        is_done            <- false;
                        merge_success      <- false;
                        merge_reward_given <- false;
                        merge_bonus_tick   <- false;
                        failed_merge       <- false;
                        collision_event    <- false;
                        episode_step       <- 0;
                        merge_step         <- 0;
                        // merging_0 xuat phat dau ramp (wp0), khong spawn giua accel — dung scenario nhap lan thuc te.
                        location           <- {wp0_boot.x, wp0_boot.y};
                        move_next_x        <- wp0_boot.x;
                        move_next_y        <- wp0_boot.y;
                        in_accel_zone      <- false;
                        waypoint_index     <- 1;
                        heading            <- atan2(wp1_boot.y - wp0_boot.y, wp1_boot.x - wp0_boot.x);
                        speed              <- rnd(0.25, 0.45);
                        color              <- #magenta;
                        direction          <- 1;
                        current_lane_index <- number_of_lanes - 1;
                        target_lane_index  <- number_of_lanes - 1;
                        is_rl_agent        <- true;
                        rl_agent_id        <- "merging_0";
                    }
                }
            }
        }

        // --- 3. KHỞI TẠO 6 XE RL TRÊN LÀN DƯỚI CÙNG (tăng penetration cho goal 2) ---
        list<string> hw_ids_boot <- ["highway_0", "highway_1", "highway_2"];
        float bottom_y_boot <- offset_y + (number_of_lanes - 1) * lane_width + lane_width / 2.0;
        loop hw_k from: 0 to: 2 {
            // NHƯỜNG: spawn UPSTREAM (x=8/30/52, trước/đầu accel_start=48) + tốc CHẬM hơn → xe RL highway
            // còn ở trong vùng merge ĐÚNG LÚC merger (xuất phát ramp, chậm) tới → có cơ hội học nhường.
            // (Trước: 50/100/150 + nhanh → trôi qua/ra khỏi đường trước khi merger đến → "đi quá".)
            float hw_x_boot <- 35.0 + hw_k * hw_rl_spacing;
            create car number: 1 {
                is_merging         <- false;
                in_accel_zone      <- false;
                current_lane_index <- number_of_lanes - 1;
                target_lane_index  <- number_of_lanes - 1;
                location           <- {hw_x_boot, bottom_y_boot};
                move_next_x        <- hw_x_boot;
                move_next_y        <- bottom_y_boot;
                direction          <- 1;
                heading            <- 0.0;
                speed              <- rnd(0.35, 0.55);
                color              <- #cyan;
                is_rl_agent        <- true;
                rl_agent_id        <- hw_ids_boot[hw_k];
            }
        }
        initial_cars_created <- true;
        // Khong sync bridge ngay sau bootstrap; petz_collect_tick se dong bo o cycle on dinh.
    }

    // distance_to(point,point) trong reflex global đôi khi NPE (topology nil) trên headless — dùng khoảng cách Euclid 2D.
    // each.location có thể nil (xe vừa tạo/hủy) — kiểm tra trước khi .x/.y.
    // M4: dynamic spawn xe NPC ramp dung counter global, KHONG dung `cycle mod ...` trong `when:`
    // de tranh TempVariable trong condition. Counter tang moi cycle qua mot reflex don gian.
    float ramp_wp0_x <- 0.0;
    float ramp_wp0_y <- 0.0;
    bool  ramp_wp0_ready <- false;
    int   spawn_ramp_tick <- 0;
    bool  spawn_ramp_due <- false;
    bool  spawn_ramp_safe <- false;
    float spawn_ramp_dx <- 0.0;
    float spawn_ramp_dy <- 0.0;
    float spawn_ramp_dist_sq <- 0.0;
    int   spawn_ramp_count <- 0;       // Phase 6.2: dem so xe ramp dang active
    int   spawn_ramp_cap   <- 3;        // Cap toi da xe ramp NPC active cung luc (giam tu 5 de tranh ket ramp)

    // Reflex 1: counter tick + set flag due moi 25 tick. Khong dung loop, khong dung binary phuc tap.
    // Phase 6.2: BAT LAI (bo `when: false`) sau khi them cleanup (`die_if_out_of_road` +
    // `purge_safe_cars`). N tu day duoc bo: bootstrap 16 + ramp_cap 3 = max 19 xe.
    reflex spawn_ramp_tickup {
        spawn_ramp_tick <- spawn_ramp_tick + 1;
        spawn_ramp_due <- false;
        if (spawn_ramp_tick >= 3) {
            spawn_ramp_tick <- 0;
            if (ramp_wp0_ready) {
                spawn_ramp_due <- true;
            }
        }
    }

    // Reflex 2: spawn khi `spawn_ramp_due`, co cho an toan quanh wp0, va chua het cap.
    // Phase 6.2: them count `spawn_ramp_count` trong cung loop -> khong cost O(N) them.
    reflex spawn_ramp_cars when: spawn_ramp_due {
        try {
            spawn_ramp_count <- 0;
            spawn_ramp_safe <- true;
            if (safe_cars != nil) {
                loop c over: safe_cars {
                    if (c != nil) {
                        car c_car <- c as car;
                        if (c_car != nil) {
                            if (not dead(c_car)) {
                                if (c_car.merge_mode = 1) {
                                    spawn_ramp_count <- spawn_ramp_count + 1;
                                    if (c_car.move_next_x > 0.0) {
                                        spawn_ramp_dx <- c_car.move_next_x - ramp_wp0_x;
                                        spawn_ramp_dy <- c_car.move_next_y - ramp_wp0_y;
                                        spawn_ramp_dist_sq <- spawn_ramp_dx * spawn_ramp_dx + spawn_ramp_dy * spawn_ramp_dy;
                                        if (spawn_ramp_dist_sq < 225.0) {
                                            spawn_ramp_safe <- false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (spawn_ramp_safe) {
                if (spawn_ramp_count < spawn_ramp_cap) {
                    create car number: 1 {
                        is_merging         <- true;
                        merge_mode         <- 1;
                        in_accel_zone      <- false;
                        waypoint_index     <- 1;
                        location           <- {ramp_wp0_x, ramp_wp0_y};
                        move_next_x        <- ramp_wp0_x;
                        move_next_y        <- ramp_wp0_y;
                        heading            <- 0.0;
                        speed              <- rnd(0.30, 0.55);
                        color              <- #orange;
                        direction          <- 1;
                        current_lane_index <- number_of_lanes - 1;
                        target_lane_index  <- number_of_lanes - 1;
                        is_rl_agent        <- false;
                        rl_agent_id        <- "";
                    }
                }
            }
        } catch {}
    }

    // Tránh empty(car where ...) và empty(nil): một số bản GAMA trả nil / không rút gọn or → UnaryOperator.empty NPE.
    reflex respawn_rl_merging_agent when: enable_rl_respawn {
        try {
            // GUI/Heuristic: không có Python reset nên pz_block_merging_respawn sẽ kẹt true
            // sau khi merging_0 success. Tự động clear flag khi không ở Python mode.
            if (pz_block_merging_respawn) {
                if (dashboard_policy != "Python/SB3 model") {
                    pz_block_merging_respawn <- false;
                } else {
                    return;
                }
            }
            if (not initial_cars_created) { return; }
            // Sinh merging_0 moi ngay khi khong con xe RL ramp song (highway RL chay lap rieng).
            // Dùng safe_cars thay vì loop trực tiếp lên population car để tránh CME
            bool merging_slot_taken <- false;
            if (safe_cars != nil) {
                loop c over: safe_cars {
                    if (c != nil) {
                        car c_car <- c as car;
                        if (c_car != nil) {
                            if (not dead(c_car)) {
                                if (c_car.rl_agent_id = "merging_0") {
                                    merging_slot_taken <- true;
                                }
                            }
                        }
                    }
                }
            }
            if (ramp_waypoints != nil) {
                if (length(ramp_waypoints) > 0) {
                    if (not merging_slot_taken) {
                        point wp0 <- ramp_waypoints[0];
                        if (wp0 != nil) {
                            bool is_safe <- true;
                            if (safe_cars != nil) {
                                loop c over: safe_cars {
                                    if (c != nil) {
                                        car c_car <- c as car;
                                        if (c_car != nil) {
                                            if (not dead(c_car)) {
                                                if (c_car.location != nil) {
                                                    if (c_car.is_merging) {
                                                        if (sqrt((c_car.location.x - wp0.x) * (c_car.location.x - wp0.x) + (c_car.location.y - wp0.y) * (c_car.location.y - wp0.y)) < 15.0) {
                                                            is_safe <- false;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (is_safe) {
                                // Reset metric tích lũy toàn simulation → throughput / shockwave trong info phản ánh đúng "cửa sổ" episode merging_0.
                                total_ramp_attempts      <- 0;
                                total_merge_success      <- 0;
                                sw_n                     <- 0;
                                sw_mean                  <- 0.0;
                                sw_M2                    <- 0.0;
                                braking_events           <- 0;
                                sw_car_ticks             <- 0;
                                recent_merge_coop_ticks  <- 0;
                                recent_merge_x           <- -1.0;
                                create car number: 1 {
                                    is_merging         <- true;
                                    merge_mode         <- 1;
                                    terminal_reason    <- "running";
                                    is_done            <- false;
                                    merge_success      <- false;
                                    failed_merge       <- false;
                                    collision_event    <- false;
                                    episode_step       <- 0;
                                    merge_step         <- 0;
                                    speed_sum          <- 0.0;
                                    cumulative_reward  <- 0.0;
                                    min_front_gap      <- 999.0;
                                    min_rear_gap       <- 999.0;
                                    stuck_counter      <- 0;
                                    action_rl          <- 1;
                                    // Respawn dau ramp (cung vi tri bootstrap) — agent di het doan cong + accel roi merge.
                                    location           <- {wp0.x, wp0.y};
                                    move_next_x        <- wp0.x;
                                    move_next_y        <- wp0.y;
                                    in_accel_zone      <- false;
                                    waypoint_index     <- 1;
                                    if (length(ramp_waypoints) > 1) {
                                        if (ramp_waypoints[1] != nil) {
                                            point w1r <- ramp_waypoints[1];
                                            heading   <- atan2(w1r.y - wp0.y, w1r.x - wp0.x);
                                        }
                                    }
                                    speed              <- rnd(0.25, 0.45);
                                    color              <- #magenta;
                                    direction          <- 1;
                                    current_lane_index <- number_of_lanes - 1;
                                    target_lane_index  <- number_of_lanes - 1;
                                    is_rl_agent        <- true;
                                    rl_agent_id        <- "merging_0";
                                }
                                // Khong goi sync bridge o day de tranh local context null.
                            }
                        }
                    }
                }
            }
        } catch {}
    }

    // MARL: Tái tạo xe highway RL. every(3) thay vì every(1) giảm overhead mỗi cycle.
    // 3 cycles trễ tái tạo là không đáng kể so với episode dài 300 steps.
    reflex respawn_highway_rl_agents when: enable_highway_rl_respawn {
        try {
            if (not initial_cars_created) { return; }
            list<string> hw_ids <- ["highway_0", "highway_1", "highway_2"];
            float bottom_y <- offset_y + (number_of_lanes - 1) * lane_width + lane_width / 2.0;
            // CHỈ respawn 1 xe / tick: safe_cars là snapshot cache → xe vừa tạo chưa có trong list,
            // nếu tạo >=2 xe cùng tick chúng chồng lên x=0 → đâm nhau → die → loop hồi sinh liên tục.
            bool did_respawn <- false;
            // Dùng safe_cars thay vì loop trực tiếp lên population car để tránh CME
            loop hid over: hw_ids {
                bool hw_alive <- false;
                if (safe_cars != nil) {
                    loop c over: safe_cars {
                        if (c != nil) {
                            car c_car <- c as car;
                            if (c_car != nil) {
                                if (not dead(c_car)) {
                                    if (c_car.rl_agent_id = hid) {
                                        hw_alive <- true;
                                    }
                                }
                            }
                        }
                    }
                }
                if (not hw_alive) {
                    bool is_safe <- true;
                    if (safe_cars != nil) {
                        loop c over: safe_cars {
                            if (c != nil) {
                                car c_car <- c as car;
                                if (c_car != nil) {
                                    if (not dead(c_car)) {
                                        if (c_car.location != nil) {
                                            // physical_lane: tranh "ma xe" khi xe vua action 3/4 sang lane n-1.
                                            if (c_car.physical_lane = number_of_lanes - 1) {
                                                if (not c_car.is_merging) {
                                                    // Cần runway trống >=18u (8u quá ngắn → respawn đâm xe chậm phía trước).
                                                    if (c_car.location.x < 18.0) {
                                                        is_safe <- false;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (is_safe and not did_respawn) {
                        create car number: 1 {
                            is_merging         <- false;
                            in_accel_zone      <- false;
                            current_lane_index <- number_of_lanes - 1;
                            target_lane_index  <- number_of_lanes - 1;
                            location           <- {0.0, bottom_y};
                            move_next_x        <- 0.0;
                            move_next_y        <- bottom_y;
                            direction          <- 1;
                            heading            <- 0.0;
                            speed              <- 0.35;   // spawn chậm để không lao vào xe chậm phía trước
                            color              <- #cyan;
                            is_rl_agent        <- true;
                            rl_agent_id        <- hid;
                        }
                        did_respawn <- true;   // chặn tạo xe thứ 2 cùng tick (tránh chồng x=0 → đâm → loop)
                        // Khong goi sync bridge o day de tranh local context null.
                    }
                }
            }
        } catch {}
    }

    // Shockwave: thu thap toc do xe MAINLINE o lan duoi (bottom lane vat ly) trong vung merge.
    // Welford online mean/variance (sw_n, sw_mean, sw_M2) - khong luu toan bo mau.
    // M5: dung `move_next_x` (scalar precomputed) thay vi `location.x`; dung `merge_mode != 1`
    // thay vi `not is_merging` cho hot path stability. Scratch fields `sw_delta_*` (global)
    // tranh local trong nested if -> TempVariable. Trigger moi 5 cycle thong qua counter
    // (KHONG dung `cycle mod ...` trong when -> tranh TempVariable trong condition).
    float sw_delta_cache <- 0.0;
    float sw_delta2_cache <- 0.0;
    int   sw_tick <- 0;
    bool  sw_due <- false;

    reflex sw_tickup {
        sw_tick <- sw_tick + 1;
        sw_due <- false;
        if (sw_tick >= 5) {
            sw_tick <- 0;
            sw_due <- true;
        }
    }

    reflex sample_shockwave when: sw_due {
        try {
            if (safe_cars != nil) {
                loop c over: safe_cars {
                    if (c != nil) {
                        car c_car <- c as car;
                        if (c_car != nil) {
                            if (not dead(c_car)) {
                                if (c_car.merge_mode != 1) {
                                    if (not c_car.is_merging_transition) {
                                        // physical_lane: lay sample xe THUC SU dang vat ly o lane n-1.
                                        if (c_car.physical_lane = number_of_lanes - 1) {
                                            if (c_car.move_next_x >= accel_start_x - 20.0) {
                                                if (c_car.move_next_x <= merge_x + 30.0) {
                                                    sw_n <- sw_n + 1;
                                                    sw_delta_cache <- c_car.speed - sw_mean;
                                                    sw_mean <- sw_mean + sw_delta_cache / sw_n;
                                                    sw_delta2_cache <- c_car.speed - sw_mean;
                                                    sw_M2 <- sw_M2 + sw_delta_cache * sw_delta2_cache;
                                                    // PHANH GẤP: drop tốc so mẫu trước > ngưỡng ⇒ 1 braking event.
                                                    // (`ask c_car` đã xác nhận VÔ CAN với gridlock — thủ phạm là
                                                    // headless degrade, không phải dòng này; re-add để đo braking đúng.)
                                                    sw_car_ticks <- sw_car_ticks + 1;
                                                    if (c_car.prev_speed_sw >= 0.0) {
                                                        if (c_car.prev_speed_sw - c_car.speed > hard_brake_thresh) {
                                                            braking_events <- braking_events + 1;
                                                        }
                                                    }
                                                    ask c_car { prev_speed_sw <- speed; }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch {}
    }

    // Demo chậm trên GUI: dùng thanh tốc độ toolbar GAMA hoặc Python --step-delay (không pause ms trong GAML).

    int  balance_tick <- 0;
    bool balance_due <- false;

    reflex balance_traffic_tickup {
        balance_due <- false;
        if (enable_balance_traffic) {
            balance_tick <- balance_tick + 1;
            if (balance_tick >= 10) {
                balance_tick <- 0;
                balance_due <- true;
            }
        }
    }

    reflex balance_traffic when: balance_due {
        int total_cars <- 0;
        if (safe_cars != nil) {
            total_cars <- length(safe_cars);
        }
        if (total_cars < nb_cars_max) {
            int spawn_lane <- rnd(0, number_of_lanes - 1);
            float spawn_y  <- offset_y + (spawn_lane * lane_width) + (lane_width / 2.0);
            bool is_safe   <- true;
            // Dùng safe_cars để tránh CME
            loop c over: safe_cars {
                if (c != nil) {
                    car c_car <- c as car;
                    if (c_car != nil) {
                        if (not dead(c_car)) {
                            if (c_car.move_next_x >= 0.0) {
                                if (c_car.current_lane_index = spawn_lane) {
                                    if (c_car.move_next_x < 22.0) {
                                        is_safe <- false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (is_safe) {
                create car number: 1 {
                    speed  <- rnd(0.5, speed_max);
                    color  <- rgb(200 + rnd(55), 200 + rnd(55), 200 + rnd(55));
                    is_merging         <- false;
                    in_accel_zone      <- false;
                    current_lane_index <- spawn_lane;
                    target_lane_index  <- spawn_lane;
                    location           <- {0.0, spawn_y};
                    move_next_x        <- 0.0;
                    move_next_y        <- spawn_y;
                    direction          <- 1;
                    heading            <- 0.0;
                    is_rl_agent        <- false;
                    rl_agent_id        <- "";
                }
            }
        }
    }

    action build_road {
        loop i from: 0 to: number_of_lanes - 1 {
            create road_segment {
                lane_index <- i;
                float cy   <- offset_y + (i * lane_width) + (lane_width / 2.0);
                location   <- {road_length / 2.0, cy};
                shape      <- rectangle(road_length, lane_width);
                color      <- rgb(80, 80, 80);
            }
        }
    }

    action create_initial_cars {
        // --- 1. KHỞI TẠO XE TRÊN CAO TỐC CHÍNH (Code cũ của bạn) ---
        int   cars_per_lane <- int(nb_cars_max / number_of_lanes);
        float spacing       <- road_length / (cars_per_lane + 1);

        loop i from: 0 to: number_of_lanes - 1 {
            loop j from: 0 to: cars_per_lane - 1 {
                create car number: 1 {
                    speed              <- rnd(0.4, speed_max);
                    color              <- rgb(200 + rnd(55), 200 + rnd(55), 200 + rnd(55));
                    is_merging         <- false;
                    in_accel_zone      <- false;
                    current_lane_index <- i;
                    target_lane_index  <- i;
                    location           <- {10.0 + j * spacing + rnd(-3.0, 3.0), offset_y + (i * lane_width) + (lane_width / 2.0)};
                    move_next_x        <- location.x;
                    move_next_y        <- location.y;
                    direction          <- 1;
                    heading            <- 0.0;
                    // Xe trên cao tốc giữ vai trò môi trường giao thông nền, chưa phải agent học tăng cường.
                    is_rl_agent        <- false;
                    rl_agent_id        <- "";
                }
            }
        }

        // --- 2. THÊM MỚI: KHỞI TẠO XE TRÊN NHÁNH NHẬP LÀN (ON-RAMP) ---
        int nb_ramp_cars <- 3; // Số lượng xe ban đầu trên nhánh (bạn có thể tùy chỉnh)
        if (length(ramp_waypoints) >= 2) {
        point wp0 <- ramp_waypoints[0];
        point wp1 <- ramp_waypoints[1];

        if (wp0 != nil) {
        if (wp1 != nil) {
        loop k from: 0 to: nb_ramp_cars - 1 {
            create car number: 1 {
                is_merging         <- true;
                merge_mode         <- 1;
                in_accel_zone      <- false;
                waypoint_index     <- 1; // Hướng mục tiêu là điểm thứ 2 của ramp
                
                // Trải đều vị trí xe dọc theo đoạn thẳng từ wp0 đến wp1
                // Phải chia thực — k/nb kiểu int trong GAML có thể cho ratio luôn 0.
                float ratio <- (k * 1.0) / nb_ramp_cars;
                float lx <- wp0.x + ratio * (wp1.x - wp0.x);
                float ly <- wp0.y + ratio * (wp1.y - wp0.y);
                location <- {lx, ly};
                move_next_x        <- lx;
                move_next_y        <- ly;
                
                heading            <- atan2(wp1.y - ly, wp1.x - lx);
                speed              <- rnd(0.2, 0.4);          // Đi chậm hơn cao tốc một chút
                // Xe k = 0 là xe tự hành RL nên tô màu magenta; các xe nhập làn còn lại vẫn là xe nền màu cam.
                color              <- #orange;
                direction          <- 1;
                
                // Về mặt logic, xe nhập làn sẽ hướng tới làn dưới cùng của cao tốc
                current_lane_index <- number_of_lanes - 1;
                target_lane_index  <- number_of_lanes - 1;
                // Chỉ xe nhập làn đầu tiên được Python/PettingZoo điều khiển trong giai đoạn triển khai đầu.
                is_rl_agent        <- false;
                rl_agent_id        <- "";
                if (k = 0) {
                    color       <- #magenta;
                    is_rl_agent <- true;
                    rl_agent_id <- "merging_0";
                }
            }
        }
        }
        }
        }

        // --- 3. MARL: KHỞI TẠO 3 XE RL TRÊN LÀN DƯỚI CÙNG (highway_0/1/2) ---
        // Chọn 3 vị trí trải đều trong đoạn từ 1/4 đến 3/4 đường để bao phủ vùng merge.
        list<string> hw_ids <- ["highway_0", "highway_1", "highway_2"];
        float bottom_y <- offset_y + (number_of_lanes - 1) * lane_width + lane_width / 2.0;
        loop hw_k from: 0 to: 2 {
            float hw_x <- road_length * (0.25 + hw_k * 0.25);
            create car number: 1 {
                is_merging         <- false;
                in_accel_zone      <- false;
                current_lane_index <- number_of_lanes - 1;
                target_lane_index  <- number_of_lanes - 1;
                location           <- {hw_x, bottom_y};
                move_next_x        <- hw_x;
                move_next_y        <- bottom_y;
                direction          <- 1;
                heading            <- 0.0;
                speed              <- rnd(0.5, speed_max);
                color              <- #cyan;
                is_rl_agent        <- true;
                rl_agent_id        <- hw_ids[hw_k];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
species road_segment {
    int lane_index;
    rgb color;

    aspect default {
        draw shape color: color;
        if (lane_index < number_of_lanes - 1) {
            float dy <- location.y + lane_width / 2.0;
            float dx <- 0.0;
            loop while: dx < road_length {
                draw line([{dx, dy, 0.05}, {dx + 2.5, dy, 0.05}]) color: #white width: 1.5;
                dx <- dx + 6.0;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// PettingZoo / gama-pettingzoo: PzBridgeAgent[0] + tick species (tạo trước/sau car để áp dụng hành động rồi thu thập sau bước xe).
species PzBridgeAgent {
    list agents;
    list possible_agents;
    map          observation_spaces;
    map          action_spaces;
    map          observations;
    map          rewards;
    map          terminations;
    map          truncations;
    map          infos;
    map          actions;
    map          data;

    init {
        if (possible_agents = nil) { possible_agents <- []; }
        if (agents = nil) { agents <- []; }
        if (observations = nil) { observations <- []; }
        if (rewards = nil) { rewards <- []; }
        if (terminations = nil) { terminations <- []; }
        if (truncations = nil) { truncations <- []; }
        if (infos = nil) { infos <- []; }
        if (actions = nil) { actions <- []; }
        if (data = nil) { data <- []; }
    }

    // Species PzBridgeAgent chi giu lai de init block (global) bootstrap pz_observation_spaces /
    // pz_action_spaces / pz_observations truoc khi petz_collect_tick.tick_sync_from_world chay tu cycle 1.

    aspect default {}
}

species petz_apply_tick {
    // Logic nap pz_actions da chuyen sang global reflex `apply_pz_python_each_cycle` (on dinh headless).
    reflex apply_python_actions when: false {
    }
    aspect default {}
}

species petz_collect_tick {
    action tick_default_info(string outcome) type: map {
        return ["outcome"::outcome, "success"::false, "collision"::false, "failed_merge"::false];
    }

    action tick_find_rl_car(string aid) type: car {
        if (aid = nil) { return nil; }
        if (safe_cars = nil) { return nil; }
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (c_car != nil) {
                    try {
                        if (not dead(c_car)) {
                            if (c_car.rl_agent_id = aid) {
                                return c_car;
                            }
                        }
                    } catch {}
                }
            }
        }
        return nil;
    }

    action tick_sync_from_world {
        map pz_rewards <- [
            "merging_0"::0.0,
            "highway_0"::0.0, "highway_1"::0.0, "highway_2"::0.0
        ];
        map pz_terminations <- [
            "merging_0"::false,
            "highway_0"::false, "highway_1"::false, "highway_2"::false
        ];
        map pz_truncations <- [
            "merging_0"::false,
            "highway_0"::false, "highway_1"::false, "highway_2"::false
        ];
        pz_infos <- [
            "merging_0"::tick_default_info("running"),
            "highway_0"::tick_default_info("running"), "highway_1"::tick_default_info("running"), "highway_2"::tick_default_info("running")
        ];
        pz_observations <- [
            "merging_0"::list_with(15, 0.0),
            "highway_0"::list_with(15, 0.0), "highway_1"::list_with(15, 0.0), "highway_2"::list_with(15, 0.0)
        ];
        pz_agents <- [];

        car rc0 <- tick_find_rl_car("merging_0");
        if (rc0 != nil) {
            pz_rewards << "merging_0"::rc0.reward_val;
            pz_terminations << "merging_0"::rc0.is_done;
            pz_observations << "merging_0"::rc0.get_merging_state();
            pz_infos << "merging_0"::rc0.get_episode_info();
            if (not rc0.is_done) { pz_agents << "merging_0"; }
        }

        // REGIONAL/SVO: lấy reward + vị trí merger để chia phần cho highway gần.
        float mrg_rwd <- 0.0;
        float mrg_x   <- -1.0;
        if (rc0 != nil) {
            mrg_rwd <- rc0.reward_val;
            if (rc0.location != nil) { mrg_x <- rc0.move_next_x; }
        }

        car rc1 <- tick_find_rl_car("highway_0");
        if (rc1 != nil) {
            float reg1 <- rc1.reward_val;
            if (mrg_x >= 0.0 and rc1.location != nil) { if (abs(rc1.move_next_x - mrg_x) < 35.0) { reg1 <- reg1 + idm_social_coef * mrg_rwd; } }
            pz_rewards << "highway_0"::reg1;
            pz_terminations << "highway_0"::rc1.is_done;
            pz_observations << "highway_0"::rc1.get_current_state();
            pz_infos << "highway_0"::rc1.get_episode_info();
            if (not rc1.is_done) { pz_agents << "highway_0"; }
        }

        car rc2 <- tick_find_rl_car("highway_1");
        if (rc2 != nil) {
            float reg2 <- rc2.reward_val;
            if (mrg_x >= 0.0 and rc2.location != nil) { if (abs(rc2.move_next_x - mrg_x) < 35.0) { reg2 <- reg2 + idm_social_coef * mrg_rwd; } }
            pz_rewards << "highway_1"::reg2;
            pz_terminations << "highway_1"::rc2.is_done;
            pz_observations << "highway_1"::rc2.get_current_state();
            pz_infos << "highway_1"::rc2.get_episode_info();
            if (not rc2.is_done) { pz_agents << "highway_1"; }
        }

        car rc3 <- tick_find_rl_car("highway_2");
        if (rc3 != nil) {
            float reg3 <- rc3.reward_val;
            if (mrg_x >= 0.0 and rc3.location != nil) { if (abs(rc3.move_next_x - mrg_x) < 35.0) { reg3 <- reg3 + idm_social_coef * mrg_rwd; } }
            pz_rewards << "highway_2"::reg3;
            pz_terminations << "highway_2"::rc3.is_done;
            pz_observations << "highway_2"::rc3.get_current_state();
            pz_infos << "highway_2"::rc3.get_episode_info();
            if (not rc3.is_done) { pz_agents << "highway_2"; }
        }


        pz_data <- [
            "Observations"::pz_observations,
            "Rewards"::pz_rewards,
            "Terminations"::pz_terminations,
            "Truncations"::pz_truncations,
            "Infos"::pz_infos
        ];
    }

    reflex push_outputs when: (cycle > 0) and every(1 #cycles) {
        // Guard cycle > 0: GAMA 2025.6.x scope race tai tick 0.
        if (initial_cars_created) {
            if (safe_cars != nil) {
                if (length(safe_cars) > 0) {
                    do tick_sync_from_world;
                }
            }
        }
    }

    aspect default {}
}

// ─────────────────────────────────────────────────────────────
// MovingSkill được loại bỏ để tránh MovingSkill.setLocation NPE khi topology = nil trong headless socket mode.
// speed và heading được khai báo thủ công bên dưới — chỉ có hai biến này cần từ MovingSkill.
species car {
    init {
        if (safe_cars = nil) {
            safe_cars <- [];
        }
        safe_cars << self;
        // Init physical_lane theo current_lane_index khi spawn — tick dau sau spawn, behave chua chay
        // -> neu mac dinh -1 thi filter "an" xe -> xe khac coi nhu lane trong -> dam vao.
        // Mainline xe spawn voi Y dat tam lane => physical_lane = current_lane_index. Ramp/merging => -1.
        if (merge_mode != 1) {
            physical_lane <- current_lane_index;
        } else {
            physical_lane <- -1;
        }
    }

    // Phase 6.1 cleanup: xe mainline ra khoi doan ve / lech Y qua xa -> die.
    // (Truoc day chi x>210 + clip x toi 220 -> xe nam tren nen xanh nhung khong bien mat.)
    reflex die_if_out_of_road {
        if (merge_mode = 1) {
            if (move_next_x > road_move_clip_x) {
                do die;
            }
        } else {
            if (move_next_x > road_mainline_exit_x) {
                if (rl_agent_id = "merging_0") {
                    if (terminal_reason = "running") {
                        if (merge_success) {
                            // Nhap lan thanh cong va chay den cuoi cao toc.
                            terminal_reason <- "success";
                            is_done <- true;
                        } else if (failed_merge) {
                            // Forced-merge len mainline nhung khong nhap lan duoc: failed_merge.
                            terminal_reason <- "failed_merge";
                            // force_action3_only: phạt nặng -150 (vs -50) để "không merge" thành thảm họa.
                            reward_val <- (force_action3_only ? -150.0 : -50.0);
                            is_done <- true;
                            cumulative_reward <- cumulative_reward + reward_val;
                        }
                    }
                }
                do die;
            }
            if (move_next_y > 0.0) {
                float y_lo_die <- offset_y - lane_width * 0.9;
                float y_hi_die <- offset_y + number_of_lanes * lane_width + lane_width * 0.9;
                if (move_next_y < y_lo_die) {
                    do die;
                }
                if (move_next_y > y_hi_die) {
                    do die;
                }
            }
        }
    }

    int   current_lane_index;
    int   target_lane_index;
    int   direction;
    rgb   color;
	bool is_merging_transition <- false;
    bool  is_merging      <- false;
    int   merge_mode      <- 0;
    bool  in_accel_zone   <- false;
    bool  was_in_accel_prev <- false;  // Tracking prev tick state cho one-shot bonus "vao zone".
    int   waypoint_index  <- 0;
    bool  is_rl_agent     <- false;
    string rl_agent_id     <- "";

    float speed   <- 0.0;   // thủ công thay MovingSkill
    float heading <- 0.0;   // thủ công thay MovingSkill

    float car_length    <- 3.5;
    float car_width     <- 1.7;
    int   patience      <- 100;
    int   stuck_counter <- 0;
    // Anti-deadlock counter cho xe mainline (tach khoi stuck_counter dung cho NPC ramp force-merge).
    int   mainline_stuck_counter <- 0;
    // Physical lane theo Y thuc te (-1 = ngoai lane / dang chuyen). Khac current_lane_index (update tuc thi
    // khi action 3/4) — dung trong get_car_ahead/behind de tranh "ma xe": xe da doi current_lane_index nhung
    // vat ly van o lane cu, neu filter theo current_lane_index thi xe sau coi nhu lane cu trong -> dam vao.
    int   physical_lane <- -1;
    // Cooldown sau khi chuyen lan — tranh xe "lac": heuristic / anti-deadlock spam lane change moi tick.
    int   lane_change_cooldown <- 0;

    float       reward_val <- 0.0;
    list<float> state;
    list<float> next_state;
    int         action_rl <- -1;
    float       action_penalty <- 0.0;
    float       obs_speed <- 0.5;
    float       obs_lane <- 0.0;
    float       obs_x <- 0.0;
    float       obs_action <- 0.5;
    float       obs_patience <- 1.0;
    float       obs_progress <- 0.0;
    float       obs_dist_to_merge <- 1.0;
    float       obs_lateral <- 1.0;
    float       obs_in_accel <- 0.0;
    float       obs_terminal <- 0.0;
    float       obs_gap_safe <- 0.0;
    float       obs_target_front_gap <- 1.0;
    float       obs_target_rear_gap <- 1.0;
    float       obs_front_gap_raw <- 999.0;
    float       obs_rear_gap_raw <- 999.0;
    float       obs_front_speed_raw <- 1.0;   // persist tốc độ xe dẫn làn đích cho merger speed-match
    float       obs_scan_dx <- 0.0;
    float       obs_required_front <- 0.0;
    float       obs_required_rear <- 0.0;
    float       obs_merge_len <- 1.0;
    float       move_next_x <- 0.0;
    float       move_next_y <- 0.0;
    // Collision detection M1: tinh trong compute_nearest_collision, danh cho `reflex behave` check terminal.
    float       nearest_collision_dist <- 1.0E9;
    float       coll_scan_dx <- 0.0;
    float       coll_scan_dy <- 0.0;
    float       coll_scan_dist <- 0.0;
    // Xe hỏng phase 1: scratch (tránh `bool`/`float` local trong nested if → TempVariable NPE headless 2025.6.x).
    bool        tow_blocked_flag   <- false;
    bool        should_drift_flag  <- false;   // scratch for handle_mainline_breakdown
    float       drift_fwd_cache    <- 0.0;     // scratch: tiến x khi trôi lề
    float       tow_rescue_y_tgt   <- 0.0;
    // M2: precompute "xe gan nhat phia truoc cung lane vat ly" cho TTC penalty trong calculate_reward.
    float       ahead_gap_dx <- 1.0E9;
    float       ahead_speed_other <- 0.0;
    // M3b RADAR 15D: precompute gap+speed 6 zones quanh self + 1 merger anchor.
    // INLINE compute trong reflex behave scan loop -> dung trong get_current_state.
    // Khong dung `do action()` (DoStatement.getContext NPE).
    float       coll_scan_dy_raw <- 0.0;
    float       coll_scan_dx_raw <- 0.0;
    float       behind_gap_dx     <- 1.0E9;
    float       behind_speed_other<- 0.0;
    float       ahead_gap_left    <- 1.0E9;
    float       ahead_speed_left  <- 0.0;
    float       behind_gap_left   <- 1.0E9;
    float       behind_speed_left <- 0.0;
    float       ahead_gap_right   <- 1.0E9;
    float       ahead_speed_right <- 0.0;
    float       behind_gap_right  <- 1.0E9;
    float       behind_speed_right<- 0.0;
    float       merger_gap_x      <- 1.0E9;
    float       merger_speed      <- 0.0;
    // M2: scratch field cho calculate_merging_reward (tranh declare local trong nested if -> TempVariable).
    float       m2_merge_gap_front <- 0.0;
    float       m2_accel_len <- 1.0;
    float       m2_urgency <- 0.0;
    // M3: scratch + radar fields cho highway agents (lane change + 15D observation).
    float       m3_target_y <- 0.0;
    float       m3_lane_blend <- 0.3;
    float       m3_step_y <- 0.0;
    float       m3_dy <- 0.0;
    float       m3_dx <- 0.0;
    float       m3_lat_diff <- 0.0;
    float       prev_move_tick_x <- 0.0;
    float       prev_move_tick_y <- 0.0;
    // RL ramp: snapshot vao dau tick — action merge (3) chi hop le khi da o vung accel theo waypoint.
    bool        merge_rl_accel_gate <- false;
    // Radar 6 huong cho highway: same/left/right lane, ahead/behind, va merger.
    float       obs_dist_ahead_same <- 1.0;
    float       obs_speed_ahead_same <- 1.0;
    float       obs_dist_ahead_left <- 1.0;
    float       obs_speed_ahead_left <- 1.0;
    float       obs_dist_behind_left <- 1.0;
    float       obs_speed_behind_left <- 1.0;
    float       obs_dist_ahead_right <- 1.0;
    float       obs_speed_ahead_right <- 1.0;
    float       obs_dist_behind_right <- 1.0;
    float       obs_speed_behind_right <- 1.0;
    float       obs_dist_merger <- 1.0;
    float       obs_speed_merger <- 1.0;
    float       info_mean_speed <- 0.0;
    float       info_throughput <- 0.0;
    float       info_sw_variance <- 0.0;
    float       info_sw_std <- 0.0;
    float       info_shockwave_index <- 0.0;
    float       reward_cache <- 0.0;
    int         episode_step <- 0;
    int         merge_step   <- 0;   // Step cụ thể khi xe thực hiện merge thành công (≠ episode_step)
    float       cumulative_reward <- 0.0;
    float       speed_sum <- 0.0;
    float       min_front_gap <- 999.0;
    float       min_rear_gap <- 999.0;
    string      terminal_reason <- "running";
    bool        merge_success <- false;
    bool        merge_reward_given <- false;   // one-shot: +200 chỉ trao 1 lần tại tick merge
    bool        merge_bonus_tick   <- false;   // transient: cờ cộng +200 sau calculate_reward (tick merge)
    float       prev_speed_sw <- -1.0;         // tốc độ mẫu shockwave trước (-1=chưa mẫu) — đếm phanh gấp
    // VERIFY goal-2: phân biệt highway giảm tốc VÌ NHƯỜNG MERGER vs VÌ XE CÙNG CỤM (car-following).
    int         dbg_decel_total <- 0;          // tổng lần chọn decel (action 2)
    int         dbg_decel_mrg   <- 0;          //   có merger sát <35m phía trước = nhường thật
    int         dbg_decel_ahead <- 0;          //   có xe sát <12m phía trước = phanh-cụm (không phải nhường)
    float       dbg_merge_x_pos <- -1.0;       // vị trí x merger TẠI LÚC merge (căn đặt RL highway)
    bool        shield_intervened <- false;    // KHIÊN IDM-cap đã ép phanh > RL muốn (tick này) → phạt intervention cost (RL không ỷ lại)
    bool        merge_committed <- false;   // commit-to-merge: đã bấm action-3 cam kết nhập, chờ gap an toàn
    bool        collision_event <- false;
    bool        failed_merge <- false;
	
    // ── Xe hỏng: hệ thống 2 phase (port logic NetLogo) ──────────────────────────────────
    bool  is_damaged             <- false;
    bool  damaged_in_rescue_lane <- false;   // true = đã sang làn khẩn cấp (phase 2)
    int   damaged_inlane_timer   <- 0;       // Phase 1: đếm tick đứng trong làn chạy
    int   damaged_inrescue_timer <- 0;       // Phase 2: đếm tick ở làn khẩn cấp
    int   damaged_tow_blocked_ticks <- 0;    // Phase tow: bi kẹt xe xung quanh -> ép kéo sau timeout
    int   damaged_inlane_max     <- 120;     // Phase 1: dừng trong làn (NetLogo) — 120 tick đủ nhìn thấy nút cổ chai
    // Phase 2: 80 -> 40 tick (~7s realtime). Rut ngan thoi gian "sua xe" de rejoin
    // xuat hien thuong xuyen, de quan sat khi demo. NetLogo dung 2700 tick (rat lau) vi
    // ho coi day la realistic simulation; minh uu tien observability.
    int   damaged_inrescue_max   <- 40;
    // Theo NetLogo reference (`prob-damaged-car = 0.00002`), GAMA 0.001 cao gap 50x -> ~5.4 lan hong/episode
    // -> gridlock thuong xuyen. Giam xuong 0.0003 (~1.6 lan/episode) — du demo shockwave/bottleneck
    // ma khong ket dai. KHONG dat 0.00002 vi muon giu noise du de RL hoc xu ly damaged.
    float prob_damaged_car       <- 0.0003;  // Xác suất hỏng / cycle (chỉ NPC cao tốc)
    // 0.015 -> 0.003: voi rescue_max=40, P(die before rejoin) = 1 - 0.997^40 ~ 11%
    // (truoc: 1 - 0.985^80 ~ 70%). ~90% xe se rejoin thay vi tow di — dung tinh than NetLogo
    // (prob-remove-car=0.00005), khong de xe hong "bien mat" lang im.
    float prob_remove_damaged    <- 0.003;   // Xác suất biến mất (xe được kéo đi) ở lề

    bool is_done <- false;
    int  dead_timer <- 0;  // đếm cycle trước khi xóa xe sau terminal (tăng để Python/MARL kịp đọc reward terminal)

    // M1 collision detection (scalar): tinh `nearest_collision_dist` bang scan `safe_cars`
    // theo dx scalar; KHONG return list temp, KHONG truyen point qua helper de tranh
    // TempVariable / GamaPoint.getX nil trong GAMA 2025.06.4 headless.
    // Quy uoc: va cham VAT LY xay ra khi 2 xe co `|dy| < lane_width * 0.7` (cung lane vat ly)
    // VA `|dx| < collision_distance`. Khong dung `current_lane_index` cho semantics nay vi
    // ramp cars / merging cars co the cung `current_lane_index = number_of_lanes - 1`
    // ("target lane sau merge") nhung dang o tren ramp (Y khac highway).
    // Bo qua `episode_step <= 2` de tranh false positive khi xe chua tach.
    action compute_nearest_collision {
        nearest_collision_dist <- 1.0E9;
        if (episode_step <= 2) { return; }
        if (move_next_x <= 0.0) { return; }
        if (move_next_y <= 0.0) { return; }
        if (safe_cars = nil) { return; }
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (c_car != nil) {
                    if (not dead(c_car)) {
                        if (c_car != self) {
                            if (c_car.move_next_x > 0.0) {
                                if (c_car.move_next_y > 0.0) {
                                    if (c_car.terminal_reason = "running") {
                                        coll_scan_dy <- abs(c_car.move_next_y - move_next_y);
                                        if (coll_scan_dy < lane_width * 0.7) {
                                            coll_scan_dx <- abs(c_car.move_next_x - move_next_x);
                                            if (coll_scan_dx < nearest_collision_dist) {
                                                nearest_collision_dist <- coll_scan_dx;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // NPC cao tốc: port lại logic xe hỏng từ Old Model để animation mượt.
    // - Không đợi timer cứng: xe bắt đầu trôi sang lề khi speed < 0.2 (giống Old Model).
    // - Khi trôi: update cả X lẫn Y đồng thời để heading có phương chính xác.
    // - Giữ guard nil và nested-if (tránh NPE GAMA 2025.x headless).
    action handle_mainline_breakdown {
        if (location = nil) { return; }
        tow_rescue_y_tgt <- offset_y - (lane_width / 2.0);

        if (not damaged_in_rescue_lane) {
            // ── FAILSAFE: Phase 1 -> Phase 2 transition gap fix ─────────────────
            // Bug: drift kich hoat khi Y > tgt+0.05 (>16.30), Phase 2 entry khi Y <= tgt
            // (<=16.25). Khoang (16.25, 16.30] la vung chet: drift dung, Phase 2 khong bat.
            // Voi drift step 0.08, xe co the roi vao day va ket vinh vien voi heading=0
            // -> visually nam tren rescue lane nhung timer Phase 2 khong start -> khong rejoin.
            // Diagnostic da xac nhan: 3 cars stuck @ y=16.27, 16.29 voi timer_inlane > 14000.
            // Snap Y va force transition khi da du gan.
            if (move_next_y <= tow_rescue_y_tgt + 0.10) {
                move_next_y <- tow_rescue_y_tgt;
                heading <- 0.0;
                damaged_in_rescue_lane <- true;
                damaged_inrescue_timer <- 0;
                damaged_tow_blocked_ticks <- 0;
                current_lane_index <- -1;
                target_lane_index  <- -1;
                if (damaged_nb_cars_inlane > 0) {
                    damaged_nb_cars_inlane <- damaged_nb_cars_inlane - 1;
                }
                return;
            }

            // ── PHASE 1a: Giảm tốc dần theo quán tính ───────────────────────────
            if (speed > 0.0) {
                speed <- max(0.0, speed - deceleration * 0.5);
            }
            damaged_inlane_timer <- damaged_inlane_timer + 1;

            // ── PHASE 1b: Trôi sang lề khi đã đủ chậm ────────────────────────────
            // Port Old Model: dùng điều kiện speed thực tế (< 0.2) thay vì timer cứng 120 tick.
            // Fallback: nếu vẫn chạy chậm quá lâu (80 tick), vẫn bắt đầu trôi.
            // Dùng field should_drift_flag thay vì bool local → tránh TempVariable NPE GAMA 2025.x.
            should_drift_flag <- false;
            if (speed < 0.2) {
                if (move_next_y > tow_rescue_y_tgt + 0.05) {
                    should_drift_flag <- true;
                }
            } else {
                if (damaged_inlane_timer >= 80) {
                    if (move_next_y > tow_rescue_y_tgt + 0.05) {
                        should_drift_flag <- true;
                    }
                }
            }

            if (should_drift_flag) {
                // Kiểm tra "gương chiếu hậu" trước khi trôi (giống Old Model clear_to_drift).
                tow_blocked_flag <- false;
                if (safe_cars != nil) {
                    loop c over: safe_cars {
                        if (c != nil) {
                            car c_car <- c as car;
                            if (c_car != nil) {
                                if (not dead(c_car)) {
                                    if (c_car != self) {
                                        if (not c_car.is_damaged) {
                                            coll_scan_dy <- move_next_y - c_car.move_next_y;
                                            coll_scan_dx <- abs(c_car.move_next_x - move_next_x);
                                            if (coll_scan_dy > -0.5) {
                                                if (coll_scan_dy < lane_width * 1.5) {
                                                    if (coll_scan_dx < car_length * 1.2) {
                                                        tow_blocked_flag <- true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (not tow_blocked_flag) {
                    damaged_tow_blocked_ticks <- 0;
                    // Drift VUONG GOC voi duong (match NetLogo `set heading 0/180; forward`).
                    // Truoc day update ca X+Y voi drift_fwd_cache=0.02 -> goc motion ~76° -> visual
                    // "cheo cheo" trong ky vi xe khong vuong goc khi roi lane sang rescue lane.
                    // Bo X step -> towards() cho ra heading +-90° (perpendicular to road).
                    move_next_y <- move_next_y - 0.08;
                    heading <- towards(
                        {move_next_x, move_next_y + 0.08},
                        {move_next_x, move_next_y}
                    );
                    if (move_next_y <= tow_rescue_y_tgt) {
                        move_next_y <- tow_rescue_y_tgt;
                        heading     <- 0.0;
                        damaged_in_rescue_lane <- true;
                        damaged_inrescue_timer <- 0;
                        damaged_tow_blocked_ticks <- 0;
                        // Sua bug "bong ma" lane cuu ho: lane cuu ho o y = offset_y - lane_width/2,
                        // KHONG nam trong range lane index 0..n_lanes-1. Neu giu current_lane_index cu,
                        // get_car_ahead/get_car_behind van loc xe nay theo lane vat ly cu -> xe phia sau
                        // tuong xe hong van chan duong -> ket. Danh dau -1 = "khong trong lan nao".
                        current_lane_index <- -1;
                        target_lane_index  <- -1;
                        if (damaged_nb_cars_inlane > 0) {
                            damaged_nb_cars_inlane <- damaged_nb_cars_inlane - 1;
                        }
                    }
                } else {
                    // Có xe bên hông → thẳng lái chờ thời cơ (Old Model)
                    heading <- 0.0;
                    damaged_tow_blocked_ticks <- damaged_tow_blocked_ticks + 1;
                    // Force bước manh hon sau 20 tick ket (truoc: 0.05/60 ticks = 0.83mm/tick TB,
                    // 3.5m drift can ~4200 ticks neu blocked toan thoi gian -> gridlock keo dai).
                    // Doi sang 0.15/20 ticks = 7.5mm/tick TB -> ~470 ticks de drift het = 9x nhanh hon.
                    // Tradeoff: xe ben canh co the cam thay bi "huch" hon, nhung do la ban chat anti-deadlock.
                    if (damaged_tow_blocked_ticks > 20) {
                        move_next_y <- move_next_y - 0.15;
                        damaged_tow_blocked_ticks <- 0;
                    }
                }
            } else {
                heading <- 0.0;
            }
        } else {
            // ── PHASE 2: Xe ở làn khẩn cấp — dừng hẳn, chờ kéo đi ─────────────
            speed   <- 0.0;
            heading <- 0.0;
            damaged_inrescue_timer <- damaged_inrescue_timer + 1;
            if (flip(prob_remove_damaged)) {
                do die;
                return;
            }
            if (damaged_inrescue_timer >= damaged_inrescue_max) {
                // REJOIN GRADUAL (port NetLogo move-to-target-lane):
                // - KHONG teleport Y, KHONG gap-check ad-hoc.
                // - Set current_lane_index=0 + is_merging_transition=true; Y van con o rescue lane
                //   (~offset_y - lane_width/2). m3_lat_diff = lane_width (~3.5m) -> Y-blend code o
                //   behave (~line 2143-2192) tu nhich Y len lane 0 qua ~13 tick voi avoidance san co.
                // - Safety shield M3c (line ~2029) van cap speed neu ahead_gap_dx qua nho -> tranh
                //   dam vao xe lane 0 trong qua trinh nhich len.
                // - Truoc day teleport + gap-check car_length*2 lam xe ket vinh vien o rescue lane
                //   khi traffic dac (lane 0 luon co xe trong 8m) -> chi thoat duoc khi bi tow.
                is_damaged             <- false;
                damaged_in_rescue_lane <- false;
                damaged_inlane_timer   <- 0;
                damaged_inrescue_timer <- 0;
                color                  <- rgb(200 + rnd(55), 200 + rnd(55), 200 + rnd(55));
                current_lane_index     <- 0;
                target_lane_index      <- 0;
                is_merging_transition  <- true;
                heading                <- 0.0;
                // Speed du de Y-blend (line ~2149 yeu cau speed > 0) nhung khong qua nhanh.
                speed                  <- rnd(0.25, 0.5);
            }
        }
    }

    reflex behave {
        // Headless batch ổn định: không gọi highway_rl_move (logic sự cố đã gom vào handle_mainline_breakdown).
        if (terminal_reason != "running") {
            speed <- 0.0;
            dead_timer <- dead_timer + 1;
            // Vai tick de Python/MARL doc termination + reward, roi huy xe de respawn_rl_merging_agent mo slot.
            if (dead_timer > 2) {
                do die;
            }
            return;
        }

        dead_timer <- 0;

        merge_rl_accel_gate <- false;

        // Tick xuong cooldown lane change (chong xe "lac" do spam chuyen lan moi tick).
        if (lane_change_cooldown > 0) {
            lane_change_cooldown <- lane_change_cooldown - 1;
        }

        prev_move_tick_x <- move_next_x;
        prev_move_tick_y <- move_next_y;

        // Sự cố ngẫu nhiên: chỉ xe NPC cao tốc (không RL, không đang trên ramp merge).
        // Cap: không vượt quá max_damaged_inlane xe hỏng đồng thời (port NetLogo damaged-nb-cars-inlane).
        if (merge_mode != 1) {
            if (rl_agent_id = "") {
                if (not is_damaged) {
                    if (flip(prob_damaged_car)) {
                        if (damaged_nb_cars_inlane < max_damaged_inlane) {
                            is_damaged             <- true;
                            damaged_in_rescue_lane <- false;
                            damaged_inlane_timer   <- 0;
                            damaged_inrescue_timer <- 0;
                            damaged_tow_blocked_ticks <- 0;
                            color                  <- #red;
                            speed                  <- 0.0;
                            damaged_nb_cars_inlane <- damaged_nb_cars_inlane + 1;
                        }
                    }
                } else {
                    do handle_mainline_breakdown();
                    location <- {move_next_x, move_next_y};
                    episode_step <- episode_step + 1;
                    return;
                }
            }
        }

        // Penalty theo action chi ap dung cho tick hien tai, khong tich luy qua episode.
        action_penalty <- 0.0;

        // Dashboard / socket: `rl_merging_behavior` khong con goi trong behave — gan action_rl o day.
        // Python/SB3: global reflex `apply_pz_python_each_cycle` da nap tu `pz_actions`; neu thieu key thi fallback keep.
        if (dashboard_policy != "Python/SB3 model") {
            if (rl_agent_id != "") {
                if (rl_agent_id = "merging_0") {
                    if (dashboard_policy = "Heuristic") {
                        // Phan biet ramp (merge_mode=1) vs sau merge (merge_mode=0):
                        //   - Tren ramp: dung heuristic_merging (action 3 = MERGE, gap check).
                        //   - Sau merge: dung heuristic_highway (action 3/4 = lane change mainline).
                        // Truoc day luon dung heuristic_merging -> action 3 semantic sai sau merge,
                        // anti-deadlock cung khong phu hop -> xe drift sang lan khac vo nghia gan cuoi duong.
                        if (merge_mode = 1) {
                            action_rl <- get_heuristic_merging_action();
                        } else {
                            // SUA BUG SEMANTIC MISMATCH: heuristic_highway tra ve highway-semantic
                            // (0=Acc, 2=Dec) nhung speed code cua merging_0 (line ~1929) van interpret
                            // 0=Dec, 2=Acc (consistency voi Python action space). Khong swap -> sau
                            // force-merge xe stuck speed=0 vinh vien (heuristic thay duong thoang ->
                            // tra 0=Acc -> speed code GIAM toc -> deadlock cao toc).
                            // Giu nguyen 1 (Keep), 3 (Lane left), 4 (Lane right) — chung khop ca 2 ngu canh.
                            int hw_act <- get_heuristic_highway_action();
                            if (hw_act = 0) {
                                action_rl <- 2;
                            } else if (hw_act = 2) {
                                action_rl <- 0;
                            } else {
                                action_rl <- hw_act;
                            }
                        }
                    }
                } else {
                    if (dashboard_policy = "Heuristic") {
                        action_rl <- get_heuristic_highway_action();
                    }
                }
            } else {
                // NPC = HDV: đổi làn theo MOBIL (tốc độ theo IDM ở khối dưới). action_rl chỉ mang
                // tín hiệu lane-change (3/4) cho khối lane-change; longitudinal do IDM quyết định.
                action_rl <- mobil_decision();
            }
        } else {
            if (rl_agent_id != "") {
                if (action_rl < 0) {
                    action_rl <- 1;
                }
            } else {
                // NPC = HDV (Python mode): đổi làn theo MOBIL; tốc độ theo IDM (khối dưới).
                action_rl <- mobil_decision();
            }
        }

        shield_intervened <- false;   // reset mỗi tick; post-merge block set true nếu KHIÊN ép phanh
        // merging_0: 0 giam, 2 tang. merge_mode=1: toc do trong rl_merging_behavior.
        // Sua bug "ket toc do=0": action 1 (keep_speed) khi speed=0 -> giu 0 vinh vien.
        // Kick-start nhe (+acceleration) khi speed < 0.05 de xe co the recover. Safety shield
        // M3c phia duoi (ahead_gap_dx < 12) van ep giam toc lai neu xe truoc qua gan -> khong va cham.
        if (rl_agent_id = "merging_0") {
            if (merge_mode != 1) {
                // POST-MERGE control. pz_rl_postmerge=1 → RL điều khiển TIẾP (end-to-end, thesis: đo
                // sóng lùi trọn hành trình + merge-không-tai-nạn-sau là thành công thật). merging_0
                // longitudinal: action 0=giảm, 2=tăng, khác=giữ (kick-start nếu quá chậm). Chống collapse
                // (RL-post-merge từng làm né-merge) bằng CURRICULUM warmstart từ nền biết-merge + phạt
                // collision (calculate_reward) + shield nhẹ chống đâm thảm. pz_rl_postmerge=0 → IDM (cũ).
                if (pz_rl_postmerge >= 0.5) {
                    // RL chọn gia tốc (0=giảm, 2=tăng, khác=giữ + kick-start). KHIÊN = cap theo IDM-safe:
                    // applied = min(rl_acc, idm_safe) → không bao giờ tăng tốc đâm đuôi (collision-free).
                    // Khi khiên cắt RL (ép phanh > RL muốn) → shield_intervened=true → PHẠT trong reward
                    // (intervention cost) → RL không ỷ lại, học tự phanh-từ-xa/điều-tốc-mượt (goal-3 của RL).
                    float rl_acc_pm <- 0.0;
                    if (action_rl = 0) { rl_acc_pm <- 0.0 - deceleration; }
                    else if (action_rl = 2) { rl_acc_pm <- acceleration; }
                    else { if (speed < 0.05) { rl_acc_pm <- acceleration; } }
                    float safe_acc_pm <- idm_acc(speed, ahead_gap_dx, ahead_speed_other);
                    float applied_pm <- min(rl_acc_pm, safe_acc_pm);
                    // Intervention CHỈ tính khi KHẨN THẬT: RL muốn ĐI (accel/keep, rl_acc≥0) NHƯNG khiên
                    // ép PHANH thật (applied<-0.05) = "khiên chiếm quyền". KHÔNG tính lúc theo-xe bình
                    // thường (cap nhẹ accel) → tránh phạt oan làm merger đóng băng (yt41: -5 quá nặng → freeze).
                    if (rl_acc_pm >= -0.001 and applied_pm < -0.05) { shield_intervened <- true; }
                    speed <- min(speed_max, max(0.0, speed + applied_pm));
                } else {
                    speed <- min(speed_max, max(0.0, speed + idm_acc(speed, ahead_gap_dx, ahead_speed_other)));
                }
            }
        } else if (rl_agent_id != "") {
            // RL highway longitudinal: 0=accel, 2=decel, khác=keep. KHIÊN IDM-cap (đối xứng merger):
            // applied = min(rl_acc, idm_safe) → collision-free → HẾT hiện tượng highway tăng tốc đâm chết.
            // Intervention (RL muốn đi NHƯNG khiên ép phanh) → phạt trong reward (không ỷ lại).
            float rl_acc_hw <- 0.0;
            if (action_rl = 0) { rl_acc_hw <- acceleration; }
            else if (action_rl = 2) {
                rl_acc_hw <- 0.0 - deceleration;
                dbg_decel_total <- dbg_decel_total + 1;
                if (active_merger_x >= 0.0 and move_next_x < active_merger_x and (active_merger_x - move_next_x) < 35.0) {
                    dbg_decel_mrg <- dbg_decel_mrg + 1;
                }
                if (ahead_gap_dx < 12.0) { dbg_decel_ahead <- dbg_decel_ahead + 1; }
            } else { if (speed < 0.05) { rl_acc_hw <- acceleration; } }
            float safe_acc_hw <- idm_acc(speed, ahead_gap_dx, ahead_speed_other);
            float applied_hw <- min(rl_acc_hw, safe_acc_hw);
            if (rl_acc_hw >= -0.001 and applied_hw < -0.05) { shield_intervened <- true; }
            speed <- min(speed_max, max(0.0, speed + applied_hw));
        } else {
            // NPC cars execution — HDV: longitudinal = IDM (mượt, collision-free, KHÔNG nhường
            // chủ động merger). action_rl (MOBIL) chỉ dùng cho lane-change ở khối dưới.
            if (merge_mode != 1) {
                speed <- min(speed_max, max(0.0, speed + idm_acc(speed, ahead_gap_dx, ahead_speed_other)));
            } else {
                if (speed < 0.05) {
                    speed <- min(speed_max, speed + acceleration);
                } else {
                    speed <- min(speed_max, max(speed_min, speed));
                }
            }
        }

        // M3: Lane change cho highway agents
        // action 3 = lane left (giam lane_index), 4 = lane right (tang). Hot path INLINE,
        // khong goi `do execute_lane_change()` de tranh DoStatement.getContext NPE.
        // Lane change cho xe đã ở cao tốc (merge_mode != 1). Xe ramp đã merge (merge_success) thì khóa:
        // action-3 với merging = "merge" nên không để handler này diễn giải thành lane-left rời làn liền kề.
        if (merge_mode != 1 and not merge_success) {
            // Cooldown gate: trong N tick sau lane change, action 3/4 KHONG thuc thi (chong xe "lac").
            // ChỈ áp cho NPC + heuristic — RL Python mode KHONG cooldown de policy tu hoc "khong spam".
            // Tranh anh huong training: RL agent van nhan tin hieu reward/penalty day du moi tick.
            bool can_lane_change <- true;
            if (lane_change_cooldown > 0) {
                if (dashboard_policy != "Python/SB3 model") {
                    can_lane_change <- false;
                } else if (rl_agent_id = "") {
                    // NPC: van bi cooldown du o Python mode
                    can_lane_change <- false;
                }
            }
            if (can_lane_change) {
                if (action_rl = 3) {
                    if (current_lane_index > 0) {
                        // KHIÊN NGANG (lateral shield): CHỈ đổi làn khi gap làn TRÁI đủ an toàn
                        // (ahead_gap_left>8, behind_gap_left>6 — khớp ngưỡng force-escape anti-deadlock).
                        // Thiếu gap (kể cả xe HỎNG đứng im trong làn trái) → CHẶN đổi làn + intervention
                        // penalty (shield_intervened → -1) để RL học gap-awareness. Khiên M3c (dưới) chỉ
                        // chặn DỌC (cap tốc độ làn hiện tại), KHÔNG chặn move NGANG → trước đây RL đổi-làn
                        // vào gap hẹp = đâm xe/đâm HDV-hỏng (shield bất lực với va chạm ngang).
                        if (ahead_gap_left > 8.0 and behind_gap_left > 6.0) {
                            // Phạt đổi làn cơ bản -0.3 (nhẹ → vẫn cho phép đổi làn NÉ một lần dưới argmax,
                            // tránh xe chỉ phanh rồi rear-end). Nếu đang còn trong cooldown lần đổi trước
                            // (lane_change_cooldown>0) → ĐỔI LÀN LẶP = "đánh lái trái-phải": phạt nặng -2.1
                            // để giết flip-flop (RL Python ko bị engine gate cooldown). Nhắm trúng spam.
                            if (lane_change_cooldown > 0) {
                                action_penalty <- action_penalty - 2.1;
                            } else {
                                action_penalty <- action_penalty - 0.3;
                            }
                            target_lane_index <- current_lane_index - 1;
                            current_lane_index <- target_lane_index;
                            lane_change_cooldown <- 25;
                            is_merging_transition <- true;
                        } else {
                            shield_intervened <- true;   // khiên ngang chặn đổi-làn-không-an-toàn
                        }
                    } else {
                        action_penalty <- action_penalty - 2.0;
                    }
                } else if (action_rl = 4) {
                    if (current_lane_index < number_of_lanes - 1) {
                        if (ahead_gap_right > 8.0 and behind_gap_right > 6.0) {
                            if (lane_change_cooldown > 0) {
                                action_penalty <- action_penalty - 2.1;
                            } else {
                                action_penalty <- action_penalty - 0.3;
                            }
                            target_lane_index <- current_lane_index + 1;
                            current_lane_index <- target_lane_index;
                            lane_change_cooldown <- 25;
                            is_merging_transition <- true;
                        } else {
                            shield_intervened <- true;   // khiên ngang chặn đổi-làn-không-an-toàn
                        }
                    } else {
                        action_penalty <- action_penalty - 2.0;
                    }
                }
            }
        }

        // M3c SAFETY SHIELD: policy MARL co the hoc "keep" lien tuc; neu khong co
        // car-following guard, highway agents rear-end xe phia truoc truoc khi reward
        // TTC kip sua. Dung ahead_gap_dx/ahead_speed_other tu scan tick truoc de ep
        // mainline car giam toc truoc movement, khong ap dung cho xe ramp merge_mode=1.
        // KHIEN OFF (A/B): tat shield M3c o headless -> xe dam duoi / di sat bat chap. GUI giu shield.
        // 2026-06-07: NPC giờ dùng IDM (tự phanh collision-free) → KHÔNG cần khiên M3c nữa.
        // RL agent vẫn không khiên ở Python mode → tự học phanh. Khiên chỉ còn cho GUI Heuristic.
        if (merge_mode != 1 and (enable_safety_shield or dashboard_policy != "Python/SB3 model")) {
            if (ahead_gap_dx < 12.0) {
                if (speed > ahead_speed_other) {
                    // Xe truoc gan nhu dung: neu van con > collision_distance theo dx thi cho "lun" nhe
                    // de tranh gridlock giua duong khi mật độ cao / Random policy.
                    float cap_follow <- ahead_speed_other;
                    if (ahead_speed_other < 0.12) {
                        // Khe tam xe ~3.5m; nguong qua chat (+0.12) -> cap_follow=0 -> ket dai truoc bien clip.
                        if (ahead_gap_dx > collision_distance + 0.02) {
                            cap_follow <- min(speed, 0.20 + 0.065 * (ahead_gap_dx - collision_distance));
                        }
                    }
                    speed <- min(speed, max(speed_min, cap_follow));
                }
                if (ahead_gap_dx < 6.0) {
                    if (ahead_gap_dx > collision_distance + 0.08) {
                        speed <- max(speed_min, speed - deceleration * 0.65);
                    } else {
                        speed <- max(speed_min, speed - deceleration * 2.0);
                    }
                }
            }
        }

        // ANTI-DEADLOCK: khi xe ket lau, thu CHUYEN LAN truoc, KHONG let cham (truoc day force speed=0.10
        // tao "let" lien tuc + xe sau coi nhu trong vi gap shield co dan).
        // Approach moi: counter > 30 tick + lane ben trong -> force chuyen lan. Neu khong co lane thoat,
        // chap nhan dung hang (van con force-lun nhe khi gap rong tu nhien).
        if (merge_mode != 1) {
            if (terminal_reason = "running") {
                if (speed < 0.035) {
                    mainline_stuck_counter <- mainline_stuck_counter + 1;
                    // Force-lun khi co the (gap tu rong) — pha gridlock kieu cu
                    if (ahead_gap_dx > collision_distance + 0.02) {
                        if (ahead_gap_dx < 25.0) {
                            speed <- 0.12;
                            mainline_stuck_counter <- 0;
                        }
                    }
                    // Lane-escape sau 30 tick ket cung: thu trai truoc (lane nhanh hon), roi phai
                    // CHECK cooldown — tranh spam force-escape gay lac xe.
                    // SKIP khi gan cuoi duong (move_next_x > road_length - 30): xe sap ra, lane change vo nghia
                    // — chi can force-lun de xe ra het, dung anh huong RL ramp dang chay den exit.
                    if (mainline_stuck_counter > 30) {
                        if (lane_change_cooldown <= 0) {
                            if (move_next_x < road_length - 30.0) {
                            if (current_lane_index > 0) {
                                if (ahead_gap_left > 8.0) {
                                    if (behind_gap_left > 6.0) {
                                        target_lane_index <- current_lane_index - 1;
                                        current_lane_index <- target_lane_index;
                                        mainline_stuck_counter <- 0;
                                        is_merging_transition <- true;
                                        lane_change_cooldown <- 25;
                                    }
                                }
                            }
                            if (mainline_stuck_counter > 30) {
                                if (current_lane_index < number_of_lanes - 1) {
                                    if (ahead_gap_right > 8.0) {
                                        if (behind_gap_right > 6.0) {
                                            target_lane_index <- current_lane_index + 1;
                                            current_lane_index <- target_lane_index;
                                            mainline_stuck_counter <- 0;
                                            is_merging_transition <- true;
                                            lane_change_cooldown <- 25;
                                        }
                                    }
                                }
                            }
                            } // close: if (move_next_x < road_length - 30.0)
                        }
                    }
                } else {
                    mainline_stuck_counter <- 0;
                }
            }
        }

        // Gan cuoi mainline: day nhe hang xe ve phia nguong die (tranh "mam xich" dung im xa bien clip).
        if (merge_mode != 1) {
            if (terminal_reason = "running") {
                if (move_next_x > road_length - 20.0) {
                    if (speed < 0.07) {
                        if (ahead_gap_dx > collision_distance + 0.02) {
                            if (ahead_gap_dx < 40.0) {
                                speed <- max(speed, 0.10);
                            }
                        }
                    }
                }
            }
        }

        // Movement: xe ramp (merge_mode=1) — ramp_behavior (RL: rl_merging_behavior).
        if (merge_mode = 1) {
            location <- {move_next_x, move_next_y};
            do ramp_behavior();
            move_next_x <- min(road_move_clip_x, max(0.0, location.x));
            move_next_y <- location.y;
            if (rl_agent_id != "") {
                // Sua bug: KHONG cong cumulative_reward / speed_sum o day —
                // rl_merging_behavior -> update_merging_metrics() da cong roi (line 3696-3697).
                // Cong them o day gay double-counting: mean_speed bao cao gap ~2x thuc te
                // (vd PPO mean_speed=1.12 > speed_max=1.0, greedy mean_speed=0.94 voi xe dung yen).
                return;
            }
        } else {
            move_next_x <- min(road_mainline_clip_x, max(0.0, move_next_x + speed));
            if (move_next_y <= 0.0) {
                move_next_y <- offset_y + (current_lane_index * lane_width) + (lane_width / 2.0);
            }
            // M3: Lateral Y movement cho xe KHONG dang merge_mode (tranh keo merging_0 tu ramp xuong highway tu dong).
            // Precompute scalar `m3_target_y`, `m3_lat_diff` truoc if -> tranh binary expr TempVariable.
            m3_target_y <- offset_y + (current_lane_index * lane_width) + (lane_width / 2.0);
            m3_lat_diff <- move_next_y - m3_target_y;
            if (m3_lat_diff < 0.0) {
                m3_lat_diff <- -m3_lat_diff;
            }
            if (m3_lat_diff > 0.1) {
                if (speed > 0.0) {
                    // Tang blend tu 0.2 -> 0.45: Y di chuyen nhanh gap 2x, lane change hoan tat ~70 cycle
                    // thay vi ~175 cycle. Tranh "ma xe o lane cu" (current_lane_index update tuc thi nhung
                    // vat ly Y di chuyen cham -> filter get_car_ahead mismatch -> xe sau dam vao).
                    m3_lane_blend <- 0.45;
                    if (is_merging_transition) {
                        if (rl_agent_id = "merging_0") {
                            m3_lane_blend <- 0.55;
                            if (m3_lat_diff > 0.35) {
                                speed <- min(speed, 0.42);
                            }
                        } else {
                            m3_lane_blend <- 0.55;
                        }
                    }
                    // Khi lane change escape (anti-deadlock), xe co the dang dung (speed gan 0).
                    // Ep speed tam thoi de Y di chuyen kip — neu khong xe ket Y mai mai o vi tri cu.
                    if (is_merging_transition) {
                        if (speed < 0.12) {
                            speed <- 0.20;
                        }
                    }
                    m3_step_y <- speed * m3_lane_blend;
                    if (move_next_y < m3_target_y) {
                        move_next_y <- min(m3_target_y, move_next_y + m3_step_y);
                    } else {
                        move_next_y <- max(m3_target_y, move_next_y - m3_step_y);
                    }
                }
            } else {
                move_next_y <- m3_target_y;
                is_merging_transition <- false;
            }
            if (rl_agent_id = "merging_0") {
                if (is_merging_transition) {
                    float lane_half_cap <- lane_width * 0.40;
                    if (move_next_y > m3_target_y + lane_half_cap) {
                        move_next_y <- m3_target_y + lane_half_cap;
                    }
                    if (move_next_y < m3_target_y - lane_half_cap) {
                        move_next_y <- m3_target_y - lane_half_cap;
                    }
                }
            }
        }
        location <- {move_next_x, move_next_y};

        if (merge_mode != 1) {
            float ddx_h <- move_next_x - prev_move_tick_x;
            float ddy_h <- move_next_y - prev_move_tick_y;
            // Chi rotate heading khi DANG lane change. Lateral movement nho (~0.02m do floating noise)
            // truoc day cung xoay xe -> visual "lac". Khi khong transition, force heading = 0 (thang).
            if (is_merging_transition) {
                if (abs(ddx_h) > 0.001) {
                    heading <- atan2(ddy_h, ddx_h);
                } else {
                    heading <- 0.0;
                }
            } else {
                heading <- 0.0;
            }
            // Cap nhat physical_lane theo Y vat ly thuc te.
            // Damaged car: phai phan biet phase 1 (van vat ly chiem lane cu) vs phase 2 (rescue lane = -1).
            // Bug truoc: damaged LUON -1 -> xe sau khong thay damaged phase 1 -> dam vao -> cascade brake -> ket vinh vien.
            if (is_damaged) {
                if (damaged_in_rescue_lane) {
                    physical_lane <- -1;
                } else {
                    // Phase 1 (drift): van chiem lane cu de xe sau phat hien va phanh kip.
                    physical_lane <- current_lane_index;
                }
            } else {
                if (move_next_y > 0.0) {
                    float lane_f_compute <- (move_next_y - offset_y - lane_width / 2.0) / lane_width;
                    int lane_int_compute <- int(lane_f_compute + 0.5);
                    if (lane_int_compute < 0) {
                        physical_lane <- -1;
                    } else if (lane_int_compute > number_of_lanes - 1) {
                        physical_lane <- -1;
                    } else {
                        // Chiem lane khi Y du gan tam (de tranh ghi nhan giua 2 lane khi dang chuyen).
                        float center_y_check <- offset_y + lane_int_compute * lane_width + lane_width / 2.0;
                        if (abs(move_next_y - center_y_check) < lane_width * 0.45) {
                            physical_lane <- lane_int_compute;
                        } else {
                            // Dang transitioning giua 2 lane: claim lane MUC TIEU (current_lane_index)
                            // de xe khac biet de tranh, khong "tang hinh" giua duong.
                            physical_lane <- current_lane_index;
                        }
                    }
                } else {
                    physical_lane <- -1;
                }
            }
        } else {
            physical_lane <- -1;
        }

        episode_step <- episode_step + 1;

        // M1+M2+M3b: cung 1 scan loop, compute:
        // - `nearest_collision_dist` (same physical lane, abs dx) -> terminal check.
        // - `ahead_gap_dx / ahead_speed_other`                    -> TTC penalty + obs.
        // - `behind_gap_dx / behind_speed_other`                  -> obs same lane behind.
        // - `ahead_gap_{left,right} / ahead_speed_{left,right}`   -> obs lane ben.
        // - `behind_gap_{left,right} / behind_speed_{left,right}` -> obs lane ben.
        // - `merger_gap_x / merger_speed`                          -> obs xe dang merge.
        // Lane classify theo |dy|: <0.5*lw=same, 0.5*lw..1.5*lw=adjacent (sign dy_raw),
        // >1.5*lw=ngoai radar. INLINE de tranh NPE DoStatement.getContext.
        nearest_collision_dist <- 1.0E9;
        ahead_gap_dx <- 1.0E9;
        ahead_speed_other <- 0.0;
        behind_gap_dx <- 1.0E9;
        behind_speed_other <- 0.0;
        ahead_gap_left <- 1.0E9;
        ahead_speed_left <- 0.0;
        behind_gap_left <- 1.0E9;
        behind_speed_left <- 0.0;
        ahead_gap_right <- 1.0E9;
        ahead_speed_right <- 0.0;
        behind_gap_right <- 1.0E9;
        behind_speed_right <- 0.0;
        merger_gap_x <- 1.0E9;
        merger_speed <- 0.0;
        if (episode_step > 2) {
            if (move_next_y > 0.0) {
                if (safe_cars != nil) {
                    loop c over: safe_cars {
                        if (c != nil) {
                            car c_car <- c as car;
                            if (c_car != nil) {
                                if (not dead(c_car)) {
                                    if (c_car != self) {
                                        if (c_car.move_next_x > 0.0) {
                                            if (c_car.move_next_y > 0.0) {
                                                if (c_car.terminal_reason = "running") {
                                                    coll_scan_dy_raw <- c_car.move_next_y - move_next_y;
                                                    coll_scan_dy <- coll_scan_dy_raw;
                                                    if (coll_scan_dy < 0.0) {
                                                        coll_scan_dy <- -coll_scan_dy;
                                                    }
                                                    coll_scan_dx_raw <- c_car.move_next_x - move_next_x;
                                                    coll_scan_dx <- coll_scan_dx_raw;
                                                    if (coll_scan_dx < 0.0) {
                                                        coll_scan_dx <- -coll_scan_dx;
                                                    }
                                                    // SAME LANE (|dy| < 0.5 * lane_width)
                                                    if (coll_scan_dy < lane_width * 0.5) {
                                                        if (coll_scan_dx < nearest_collision_dist) {
                                                            nearest_collision_dist <- coll_scan_dx;
                                                        }
                                                        if (coll_scan_dx_raw > 0.0) {
                                                            if (coll_scan_dx < ahead_gap_dx) {
                                                                ahead_gap_dx <- coll_scan_dx;
                                                                ahead_speed_other <- c_car.speed;
                                                            }
                                                        } else {
                                                            if (coll_scan_dx < behind_gap_dx) {
                                                                behind_gap_dx <- coll_scan_dx;
                                                                behind_speed_other <- c_car.speed;
                                                            }
                                                        }
                                                    }
                                                    // ADJACENT LANE (0.5 lw <= |dy| < 1.5 lw)
                                                    if (coll_scan_dy >= lane_width * 0.5) {
                                                        if (coll_scan_dy < lane_width * 1.5) {
                                                            if (coll_scan_dy_raw < 0.0) {
                                                                // LEFT (other Y smaller = lower lane index)
                                                                if (coll_scan_dx_raw > 0.0) {
                                                                    if (coll_scan_dx < ahead_gap_left) {
                                                                        ahead_gap_left <- coll_scan_dx;
                                                                        ahead_speed_left <- c_car.speed;
                                                                    }
                                                                } else {
                                                                    if (coll_scan_dx < behind_gap_left) {
                                                                        behind_gap_left <- coll_scan_dx;
                                                                        behind_speed_left <- c_car.speed;
                                                                    }
                                                                }
                                                            } else {
                                                                // RIGHT (other Y larger = higher lane index)
                                                                if (coll_scan_dx_raw > 0.0) {
                                                                    if (coll_scan_dx < ahead_gap_right) {
                                                                        ahead_gap_right <- coll_scan_dx;
                                                                        ahead_speed_right <- c_car.speed;
                                                                    }
                                                                } else {
                                                                    if (coll_scan_dx < behind_gap_right) {
                                                                        behind_gap_right <- coll_scan_dx;
                                                                        behind_speed_right <- c_car.speed;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    // MERGER (xe ramp, phia truoc theo x)
                                                    if (c_car.merge_mode = 1) {
                                                        if (coll_scan_dx_raw > 0.0) {
                                                            if (coll_scan_dx < merger_gap_x) {
                                                                merger_gap_x <- coll_scan_dx;
                                                                merger_speed <- c_car.speed;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (nearest_collision_dist < collision_distance) {
            reward_val <- -100.0;
            terminal_reason <- "collision";
            collision_event <- true;
            is_done <- true;
            merge_mode <- 0;
            cumulative_reward <- cumulative_reward + reward_val;
            speed_sum <- speed_sum + speed;
            return;
        }

        // RL ramp (merging_0): khi den cuoi duong cheo (merge_x) ma chua nhap lan,
        // KHONG terminate ngay — cho xe forced-merge len mainline va tiep tuc chay
        // den het cao toc (road_mainline_exit_x) roi moi ket thuc failed_merge.
        // NPC ramp (is_rl_agent=false): van terminate ngay tai merge_x cu.
        if (merge_mode = 1) {
            if (move_next_x >= merge_x - 1.0) {
                if (is_rl_agent) {
                    // Force merge len mainline: chuyen sang behave mainline mode,
                    // giu vi tri x hien tai va chuyen Y len bottom_lane_y.
                    failed_merge    <- true;
                    merge_success   <- false;
                    is_merging      <- false;
                    merge_mode      <- 0;
                    in_accel_zone   <- false;
                    is_merging_transition <- true;
                    current_lane_index    <- number_of_lanes - 1;
                    target_lane_index     <- number_of_lanes - 1;
                    if (ramp_waypoints != nil) {
                        waypoint_index <- length(ramp_waypoints);
                    }
                    // Tiep tuc chay tren mainline; se terminate khi den road_mainline_exit_x
                    // thong qua reflex die_if_out_of_road -> rl_agent failed_merge.
                    // Khong reward terminal ngay o day — cho xe chay het cao toc.
                } else {
                    reward_val <- -50.0;
                    terminal_reason <- "failed_merge";
                    failed_merge <- true;
                    merge_mode <- 0;
                    is_done <- true;
                    cumulative_reward <- cumulative_reward + reward_val;
                    speed_sum <- speed_sum + speed;
                    return;
                }
            }
        }

        // merging_0: success sau khi da nhap lane va chay gan het mainline (khong ket thuc ngay luc merge).
        if (rl_agent_id = "merging_0") {
            if (merge_success) {
                if (dbg_merge_x_pos < 0.0) { dbg_merge_x_pos <- move_next_x; }
                // +200 tại tick merge + KẾT THÚC ván (terminate-at-merge). REVERT post-merge-continue:
                // 3 lần thử continue (yt10/yt11 collapse train + yt9 mislabel eval) cho thấy nó đụng độ
                // kiến trúc episode merger-centric. Giữ terminate-at-merge = trạng thái chạy được (85%/0%).
                // Đo sóng lùi sau-merge cần giải pháp kiến trúc riêng (đang research) — KHÔNG ép ở đây.
                // TERMINATE-AT-MERGE (chuẩn). post-merge-continue ĐÃ BỎ HẲN: yt22 (fresh headless) xác nhận
                // nó làm merger né-merge (5% succ / 95% TO) — lỗi LOGIC thật, không phải headless. Goal 3 đo
                // bằng braking_event_rate (in-episode), không cần chạy tiếp sau merge.
                if (commit_to_merge and merge_mode != 1) {
                    if (pz_eval_continue < 0.5) {
                        // TRAIN: terminate-at-merge (ổn định).
                        reward_val <- 200.0;
                        terminal_reason <- "success";
                        is_done <- true;
                        pz_block_merging_respawn <- true;
                        cumulative_reward <- cumulative_reward + reward_val;
                        speed_sum <- speed_sum + speed;
                        return;
                    } else {
                        // EVAL: +200 one-shot, chạy tiếp CỬA SỔ CỐ ĐỊNH (pz_postmerge_window tick) rồi end
                        // success → đo SÓNG LÙI sau merge (đuôi sóng lan trong ~vài chục tick). KHÔNG chạy
                        // tới cuối đường (gây gridlock do highway chậm + tích lũy xe → đo bẩn).
                        if (not merge_reward_given) {
                            merge_reward_given <- true;
                            merge_bonus_tick   <- true;
                            pz_block_merging_respawn <- true;
                        }
                        if (episode_step - merge_step >= int(pz_postmerge_window)) {
                            terminal_reason <- "success";
                            is_done <- true;
                            pz_block_merging_respawn <- true;
                            cumulative_reward <- cumulative_reward + reward_val;
                            speed_sum <- speed_sum + speed;
                            return;
                        }
                    }
                }
            }
        }

        reward_val <- calculate_reward();
        if (merge_bonus_tick) {
            reward_val <- reward_val + 200.0;
            merge_bonus_tick <- false;
        }
        cumulative_reward <- cumulative_reward + reward_val;
        speed_sum <- speed_sum + speed;
    }

    // Khoảng cách Euclid 2D — tránh distance_to / at_distance (topology nil trên headless → NPE GamaPoint.getX/Y).
    action euclid_point_dist(point pa, point pb) type: float {
        if (pa = nil) { return 1.0E9; }
        if (pb = nil) { return 1.0E9; }
        float dx <- pa.x - pb.x;
        float dy <- pa.y - pb.y;
        return sqrt(dx * dx + dy * dy);
    }

    // GAMA 2025.x: and / or / ? : có thể không rút gọn → lồng if trước khi .location / [] / length() phụ thuộc nhánh khác.
    action safe_obs_gap_norm(car o, bool rear) type: float {
        if (o = nil) { return 1.0; }
        if (o.location = nil) { return 1.0; }
        if (location = nil) { return 1.0; }
        float obs_denom_sog <- max(1.0, observation_max);
        if (rear) {
            return min(1.0, (location.x - o.location.x) / obs_denom_sog);
        }
        return min(1.0, (o.location.x - location.x) / obs_denom_sog);
    }

    action safe_obs_speed_norm(car o) type: float {
        if (o = nil) { return 1.0; }
        return min(1.0, max(0.0, o.speed / max(0.001, speed_max)));
    }

    // GAMA 2025.x: bool tạm khai báo trong thân loop → NPE TempVariableExpression (ExecutionContext.local null). Gom điều kiện vào action gọi trên agent.
    action is_on_ramp_merge_path type: bool {
        if (is_merging) {
            return true;
        }
        if (is_merging_transition) {
            return true;
        }
        return false;
    }

    // Neighbor trong bán kính collision — không dùng at_distance (cũng đi qua topology).
    // Snapshot danh sách car trước loop để tránh CME khi xe được tạo/hủy cùng cycle.
    action filter_collision_neighbors type: list {
        list out <- [];
        if (not enable_collision_neighbor_scan) { return out; }
        if (safe_cars = nil) { return out; }
        if (terminal_reason != "running") { return out; }
        if (move_next_x <= 0.0) { return out; }
        if (move_next_y <= 0.0) { return out; }
        // Legacy restore: khong dung at_distance/topology/point helper; scan bounded tren scalar cache.
        // Action nay phuc vu cac action cu neu duoc goi lai, con hot path hien tai van dung scan inline.
        loop c over: safe_cars {
            if (length(out) < 4) {
                if (c != nil) {
                    car c_car <- c as car;
                    if (c_car != nil) {
                        if (c_car != self) {
                            if (not dead(c_car)) {
                                if (c_car.terminal_reason = "running") {
                                    if (c_car.move_next_x > 0.0) {
                                        if (c_car.move_next_y > 0.0) {
                                            coll_scan_dy <- c_car.move_next_y - move_next_y;
                                            if (coll_scan_dy < 0.0) {
                                                coll_scan_dy <- -coll_scan_dy;
                                            }
                                            if (coll_scan_dy < lane_width * 0.7) {
                                                coll_scan_dx <- c_car.move_next_x - move_next_x;
                                                if (coll_scan_dx < 0.0) {
                                                    coll_scan_dx <- -coll_scan_dx;
                                                }
                                                if (coll_scan_dx < collision_distance) {
                                                    out << c_car;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    action ramp_behavior {
        if (location = nil) { return; }
        // Nếu xe nhập làn là agent RL thì bỏ qua luật nhập làn thủ công để Python điều khiển trực tiếp.
        if (rl_agent_id != "") {
            do rl_merging_behavior();
            return;
        }

        float max_allowed_speed <- speed_max;
        float rb_ramp_creep_min <- 0.12;
        bool rb_on_curve_approach <- false;
        if (location != nil) {
            if (location.x < accel_start_x - 2.0) {
                rb_on_curve_approach <- true;
            }
        }

        // 1. Radar ramp: chi chan theo quang duong doc ramp (khong phai moi xe co x lon hon).
        // Xe RB dung im o mieng accel (cho gap mainline) khong duoc "khoa" ca doan cong phia sau.
        list car_snap_rb <- safe_cars;
        list blocking_ramp_cars <- [];
        loop c over: car_snap_rb {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            if (c_car.is_on_ramp_merge_path()) {
                                if (c_car.location.x > self.location.x) {
                                    bool rb_count_blocker <- true;
                                    if (rb_on_curve_approach) {
                                        if (c_car.in_accel_zone) {
                                            if (c_car.speed < 0.14) {
                                                rb_count_blocker <- false;
                                            }
                                        }
                                    }
                                    if (rb_count_blocker) {
                                        float rb_bdx <- c_car.location.x - location.x;
                                        float rb_bdy <- c_car.location.y - location.y;
                                        if (rb_bdx * rb_bdx + rb_bdy * rb_bdy < 484.0) {
                                            blocking_ramp_cars << c_car;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        car ramp_ahead <- nil;
        if (length(blocking_ramp_cars) > 0) {
            float best_dist_rb <- 1.0E9;
            loop rc over: blocking_ramp_cars {
                if (rc != nil) {
                    car rc_car <- rc as car;
                    if (rc_car.location != nil) {
                        if (euclid_point_dist(rc_car.location, self.location) < best_dist_rb) {
                            best_dist_rb <- euclid_point_dist(rc_car.location, self.location);
                            ramp_ahead <- rc_car;
                        }
                    }
                }
            }
        }
        
        // 2. Car-following tren ramp: khong ep speed=0 (gay ket o giao ramp/accel).
        if (ramp_ahead != nil) {
            if (ramp_ahead.location != nil) {
                float dist <- euclid_point_dist(location, ramp_ahead.location);
                if (dist <= car_length + 0.5) {
                    max_allowed_speed <- rb_ramp_creep_min;
                } else {
                    max_allowed_speed <- max(rb_ramp_creep_min, dist - (car_length + 1.5));
                }
            }
        }

        int accel_flag_prev <- 0;
        if (in_accel_zone) { accel_flag_prev <- 1; }
        in_accel_zone <- false;
        if (location != nil) {
            if (location.x >= accel_start_x) {
                if (location.x < accel_end_x) {
                    in_accel_zone <- true;
                }
            }
        }
        if (in_accel_zone) {
            if (accel_flag_prev = 0) {
                total_ramp_attempts <- total_ramp_attempts + 1;
            }
        }

        if (in_accel_zone) {
            speed <- min(speed_max, speed + acceleration);

            float accel_len <- accel_end_x - accel_start_x;
            float urgency   <- 0.0;
            if (accel_len > 0) {
                urgency <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len));
            }
            // Truoc day: req = gap_*_max - urgency*(...) -> tai dau accel can ~7m truoc + ~6m sau,
            // rat kho dat -> RB dung hang nhieu tick du tren mainline co khe "du mat".
            // Giu san duoi = gap_*_min khi urgency=1; thu nhe phan tuyen tinh phia dau accel.
            float ur_ease <- urgency + 0.22 * (1.0 - urgency);
            float req_front <- gap_front_min + (1.0 - ur_ease) * (gap_front_max - gap_front_min) * 0.42;
            float req_rear  <- gap_rear_min  + (1.0 - ur_ease) * (gap_rear_max  - gap_rear_min)  * 0.42;
            // Dau lane accel (urgency thap): chi can gap nhe de tiep tuc luot, tranh phanh dung ngay giao ramp.
            if (urgency < 0.12) {
                req_front <- gap_front_min * 0.85;
                req_rear  <- gap_rear_min * 0.85;
            }

            list bottom_lane_cars <- [];
            loop c over: car_snap_rb {
                if (c != nil) {
                    car c_car <- c as car;
                    if (not dead(c_car)) {
                        if (c_car != self) {
                            if (c_car.location != nil) {
                                if (not c_car.is_merging) {
                                    // physical_lane: tranh "ma xe".
                                    if (c_car.physical_lane = number_of_lanes - 1) {
                                        bottom_lane_cars << c_car;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            list leads <- [];
            list lags <- [];
            // Guarder nil trước khi truy cập .x — location có thể nil nếu xe vừa hủy.
            loop b over: bottom_lane_cars {
                if (b != nil) {
                    car b_car <- b as car;
                    if (b_car.location != nil) {
                        if (b_car.location.x >= self.location.x) { leads << b_car; }
                        else { lags << b_car; }
                    }
                }
            }
            // leads/lags chỉ chứa xe đã kiểm tra b.location != nil trong vòng lặp phía trên.
            car lead_car <- nil;
            if (length(leads) > 0) {
                float best_dx_lead <- 1.0E9;
                loop lc over: leads {
                    if (lc != nil) {
                        car lc_car <- lc as car;
                        if (lc_car.location != nil) {
                            if (lc_car.location.x - self.location.x < best_dx_lead) {
                                best_dx_lead <- lc_car.location.x - self.location.x;
                                lead_car <- lc_car;
                            }
                        }
                    }
                }
            }
            car lag_car <- nil;
            if (length(lags) > 0) {
                float best_dx_lag <- 1.0E9;
                loop lc over: lags {
                    if (lc != nil) {
                        car lc_car <- lc as car;
                        if (lc_car.location != nil) {
                            if (self.location.x - lc_car.location.x < best_dx_lag) {
                                best_dx_lag <- self.location.x - lc_car.location.x;
                                lag_car <- lc_car;
                            }
                        }
                    }
                }
            }

            float gap_front <- 999.0;
            float gap_rear  <- 999.0;
            if (lead_car != nil) {
                if (lead_car.location != nil) {
                    gap_front <- lead_car.location.x - self.location.x;
                }
            }
            if (lag_car != nil) {
                if (lag_car.location != nil) {
                    gap_rear <- self.location.x - lag_car.location.x;
                }
            }

            if (urgency >= 0.08) {
                if (lead_car != nil) {
                    if (gap_front < req_front) {
                        speed <- max(speed_min, speed - deceleration * 0.45);
                    } else {
                        if (lag_car != nil) {
                            if (gap_rear < req_rear) {
                                speed <- min(speed_max, speed + acceleration * 0.8);
                            }
                        }
                    }
                } else {
                    if (lag_car != nil) {
                        if (gap_rear < req_rear) {
                            speed <- min(speed_max, speed + acceleration * 0.8);
                        }
                    }
                }
            }

            if (urgency >= 0.08) {
                if (gap_front >= req_front) {
                    if (gap_rear >= req_rear) {
                        do execute_merge();
                        return;
                    }
                }
            }
            if (stuck_counter > 55) {
                if (gap_front > car_length * 0.95) {
                    if (gap_rear > car_length * 0.95) {
                        do execute_merge();
                        return;
                    }
                }
            }
            stuck_counter <- stuck_counter + 1;
            if (urgency < 0.20) {
                speed <- max(rb_ramp_creep_min, speed);
            }

        } else {
            speed <- max(rb_ramp_creep_min, min(speed_max * 0.7, speed + acceleration));
            stuck_counter <- 0;
        }

        // 3. ÉP GIỚI HẠN TỐC ĐỘ CUỐI CÙNG (Không cho phép vượt quá max_allowed_speed)
        speed <- min(speed, max_allowed_speed);
        speed <- max(rb_ramp_creep_min, speed);

        // Thực thi di chuyển trên nhánh (polyline — khớp đoạn chéo gore)
        if (ramp_waypoints != nil) {
            if (waypoint_index < length(ramp_waypoints)) {
                do step_along_ramp_polyline();
            }
            if (waypoint_index >= length(ramp_waypoints)) {
                do execute_merge();
            }
        }
    }

    // Di chuyen doc theo doan thang wp[i-1]->wp[i] (tranh cat goc / lech khoi doan cheo gore UI).
    action step_along_ramp_polyline {
        if (ramp_waypoints = nil) { return; }
        if (location = nil) { return; }
        int n_wp <- length(ramp_waypoints);
        if (n_wp < 2) { return; }
        int seg_ix <- waypoint_index;
        if (seg_ix < 1) { seg_ix <- 1; }
        if (seg_ix >= n_wp) { return; }
        point p0 <- ramp_waypoints[seg_ix - 1];
        point p1 <- ramp_waypoints[seg_ix];
        if (p0 = nil) { return; }
        if (p1 = nil) { return; }
        coll_scan_dx <- p1.x - p0.x;
        coll_scan_dy <- p1.y - p0.y;
        coll_scan_dist <- sqrt(coll_scan_dx * coll_scan_dx + coll_scan_dy * coll_scan_dy);
        if (coll_scan_dist < 0.01) { return; }
        float ux <- coll_scan_dx / coll_scan_dist;
        float uy <- coll_scan_dy / coll_scan_dist;
        float vx <- location.x - p0.x;
        float vy <- location.y - p0.y;
        float t_proj <- (vx * coll_scan_dx + vy * coll_scan_dy) / (coll_scan_dist * coll_scan_dist);
        if (t_proj < 0.0) { t_proj <- 0.0; }
        if (t_proj > 1.0) { t_proj <- 1.0; }
        move_next_x <- p0.x + t_proj * coll_scan_dx;
        move_next_y <- p0.y + t_proj * coll_scan_dy;
        float remain_seg <- (1.0 - t_proj) * coll_scan_dist;
        float step_along <- speed;
        if (seg_ix >= 5) {
            step_along <- min(step_along, 0.52);
        }
        if (step_along > remain_seg) { step_along <- remain_seg; }
        if (step_along < 0.0) { step_along <- 0.0; }
        location <- {move_next_x + ux * step_along, move_next_y + uy * step_along};
        move_next_x <- location.x;
        move_next_y <- location.y;
        heading <- atan2(uy, ux);
        if (euclid_point_dist(location, p1) < max(speed * 1.5, 0.75)) {
            waypoint_index <- seg_ix + 1;
        }
    }

    action sync_ramp_waypoint_ahead {
        if (ramp_waypoints = nil) { return; }
        if (location = nil) { return; }
        int n_wp <- length(ramp_waypoints);
        if (n_wp = 0) { return; }
        if (waypoint_index >= n_wp) { return; }
        point tp_ahead_chk <- ramp_waypoints[waypoint_index];
        if (tp_ahead_chk = nil) { return; }
        if (tp_ahead_chk.x > location.x + 0.5) { return; }
        loop k from: 0 to: n_wp - 1 {
            point wp_k <- ramp_waypoints[k];
            if (wp_k != nil) {
                if (wp_k.x > location.x + 1.0) {
                    waypoint_index <- k;
                    return;
                }
            }
        }
        waypoint_index <- n_wp - 1;
    }

    action rl_merging_behavior {
        if (location = nil) { return; }
        bool was_in_accel <- in_accel_zone;
        // Sua bug staleness: dung get_merging_state() de refresh obs_front_gap_raw / obs_rear_gap_raw / obs_gap_safe
        // truoc khi calculate_merging_reward doc cac field nay. Truoc day goi get_current_state() (radar highway 15D)
        // -> reward shaping doc obs tu tick TRUOC (set boi petz_collect_tick.push_outputs cuoi cycle).
        state <- get_merging_state();
        episode_step <- episode_step + 1;

        // Reset phạt tạm thời ở mỗi tick để reward chỉ phản ánh action vừa thực hiện.
        action_penalty <- 0.0;

        // Cập nhật cờ vùng tăng tốc theo toạ độ x (không chỉ 1 waypoint) — khớp heuristic Python.
        in_accel_zone <- false;
        if (location != nil) {
            if (location.x >= accel_start_x) {
                if (location.x < accel_end_x) {
                    in_accel_zone <- true;
                }
            }
        }
        if (in_accel_zone) {
            if (not was_in_accel) {
                total_ramp_attempts <- total_ramp_attempts + 1;
            }
        }

        // COMMIT-TO-MERGE: đã cam kết (bấm action-3 trước đó) + giờ gap an toàn → môi trường THỰC THI
        // merge ngay (mọi tick, không cần action-3 lại). Biến quyết định tinh-thời-điểm thành thô.
        if (commit_to_merge and merge_committed and merge_mode = 1) {
            if (merge_gate_open()) {
                do execute_merge();
                merge_success <- true;
                failed_merge  <- false;
                merge_step    <- episode_step;
            }
        }

        // Mặc định là giữ tốc nếu Python chưa kịp gửi action trong tick đầu tiên.
        if (action_rl < 0) {
            action_rl <- 1;
        }

        // Dashboard GUI: Heuristic = rule-based, con lai la Python/SB3.
        if (dashboard_policy = "Heuristic") {
            action_rl <- get_heuristic_merging_action();
        }

        // Action 0: giảm tốc để tạo khoảng trống với xe phía trước hoặc chờ gap an toàn.
        if (action_rl = 0) {
            speed <- max(speed_min, speed - deceleration);
        }
        // Action 1: giữ tốc, dùng khi agent muốn duy trì quỹ đạo hiện tại trên ramp.
        // Sua bug "ket toc do=0": neu speed=0, keep -> ket vinh vien. Floor rl_ramp_creep_min=0.12
        // o phia duoi van se nang len, nhung kick-start o day cho semantics nhat quan voi mainline.
        else if (action_rl = 1) {
            if (speed < 0.05) {
                speed <- min(speed_max, speed + acceleration);
            }
        }
        // Action 2: tăng tốc để bắt kịp tốc độ dòng chính trước khi nhập làn.
        else if (action_rl = 2) {
            speed <- min(speed_max, speed + acceleration);
        }
        // Action 3: nhap lan khi DA vao accel zone (x >= accel_start_x) hoac gan cuoi doan cheo.
        // Sua bug: truoc day gate `x >= merge_x-14` khien action 3 vo dung trong toan bo accel zone [48,162],
        // mau thuan voi heuristic / reward shaping (+2.5 cho action 3 + in_accel + gap_safe).
        else if (action_rl = 3) {
            // CHÚ Ý: reward/penalty cho action 3 được tính TẬP TRUNG trong calculate_merging_reward()
            // (gọi ở cuối reflex, dòng ~3065 qua `reward_val <- calculate_reward()`). KHÔNG ghi
            // reward_val ở đây vì nó sẽ bị ghi đè. Nhánh này chỉ thực thi side-effect merge + flag.
            if (location != nil) {
                if (in_accel_zone or location.x >= merge_x - 14.0) {
                    if (commit_to_merge) {
                        merge_committed <- true;   // CAM KẾT nhập — env thực thi ở tick gap-an-toàn kế tiếp
                    }
                    if (merge_gate_open()) {
                        do execute_merge();
                        merge_success <- true;
                        failed_merge <- false;
                        merge_step    <- episode_step;
                    }
                    // Gap không an toàn: không merge; phạt -2 do calculate_merging_reward xử lý.
                } else {
                    // Action 3 NGOÀI zone: penalty rất nhẹ (-0.01) — đủ để agent không spam
                    // action 3 trước khi vào zone. Đặt qua action_penalty (được reward cộng vào).
                    action_penalty <- action_penalty - 0.01;
                }
            }
        }
        // Action 4: chờ nhập làn, giảm tốc nhẹ để tránh lao tới cuối làn tăng tốc quá sớm.
        else if (action_rl = 4) {
            speed <- max(speed_min, speed - deceleration * 0.3);
            action_penalty <- action_penalty - 0.02;
        }

        // Car-following ramp (cung logic ramp_behavior RB): khong ep speed=0 -> ket o wp0 khi spawn thuc te.
        // ROOT CAUSE FIX cho policy collapse: floor cu 0.12 + ramp ~48m = 400 ticks de traverse,
        // VUOT QUA max_episode_steps=300 -> RL KHONG THE dat accel_zone du chon action nao -> mask
        // khong active -> argmax stuck Keep/Dec/Wait mai mai. Greedy thanh cong nho speed ~0.85.
        // Raise len 0.30 -> 48m / 0.30 = 160 ticks, du de vao zone va co thoi gian thu merge.
        float max_allowed_speed <- speed_max;
        float rl_ramp_creep_min <- 0.30;
        bool rl_on_curve_approach <- false;
        if (location != nil) {
            if (location.x < accel_start_x - 2.0) {
                rl_on_curve_approach <- true;
            }
        }
        list blocking_ramp_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            if (c_car.is_on_ramp_merge_path()) {
                                if (c_car.location.x > self.location.x) {
                                    bool rl_count_blocker <- true;
                                    if (rl_on_curve_approach) {
                                        if (c_car.in_accel_zone) {
                                            if (c_car.speed < 0.14) {
                                                rl_count_blocker <- false;
                                            }
                                        }
                                    }
                                    if (rl_count_blocker) {
                                        float rl_bdx <- c_car.location.x - location.x;
                                        float rl_bdy <- c_car.location.y - location.y;
                                        if (rl_bdx * rl_bdx + rl_bdy * rl_bdy < 484.0) {
                                            blocking_ramp_cars << c_car;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        car ramp_ahead <- nil;
        if (length(blocking_ramp_cars) > 0) {
            float best_dist_rm <- 1.0E9;
            loop rc over: blocking_ramp_cars {
                if (rc != nil) {
                    car rc_car <- rc as car;
                    if (rc_car.location != nil) {
                        if (euclid_point_dist(rc_car.location, self.location) < best_dist_rm) {
                            best_dist_rm <- euclid_point_dist(rc_car.location, self.location);
                            ramp_ahead <- rc_car;
                        }
                    }
                }
            }
        }

        if (ramp_ahead != nil) {
            if (ramp_ahead.location != nil) {
                float dist <- euclid_point_dist(location, ramp_ahead.location);
                if (dist <= car_length + 0.5) {
                    max_allowed_speed <- rl_ramp_creep_min;
                } else {
                    max_allowed_speed <- max(rl_ramp_creep_min, dist - (car_length + 1.5));
                }
            }
        }

        speed <- min(speed, max_allowed_speed);
        speed <- max(rl_ramp_creep_min, min(speed_max, speed));

        // Di chuyển dọc theo polyline ramp; RL chỉ điều khiển tốc độ và quyết định nhập làn.
        if (ramp_waypoints != nil) {
            if (location != nil) {
                do sync_ramp_waypoint_ahead();
            }
            if (waypoint_index < length(ramp_waypoints)) {
                do step_along_ramp_polyline();
            }
            if (waypoint_index >= length(ramp_waypoints)) {
                // CURRICULUM: force_action3_only tắt auto-merge cuối ramp → ép success qua action-3.
                if (not force_action3_only and merge_gate_open()) {
                    do execute_merge();
                    merge_success <- true;
                    failed_merge <- false;
                    merge_step    <- episode_step;
                }
            }
        }

        // Reward được tính sau khi xe đã di chuyển để phản ánh hậu quả của action.
        reward_val <- calculate_reward();

        // Kiểm tra va chạm hoặc thất bại do đi hết làn tăng tốc mà chưa nhập được.
        do check_merging_terminal_state();

        // Cập nhật metric sau terminal check để CSV nhận đúng reward/outcome cuối cùng.
        do update_merging_metrics();

        // next_state là observation sau action, dùng cho training loop kiểu RL chuẩn.
        // Bug #10 consistency: dung get_merging_state (cung loai voi state dau ham), khong phai radar highway.
        next_state <- get_merging_state();
    }

    action execute_merge {
        is_merging            <- false;
        merge_mode            <- 0;
        in_accel_zone         <- false;
        stuck_counter         <- 0;
        is_merging_transition <- true;
        current_lane_index    <- number_of_lanes - 1;
        target_lane_index     <- number_of_lanes - 1;
        speed                 <- min(speed_max, speed);
        patience              <- 100;
        if (ramp_waypoints != nil) {
            waypoint_index <- length(ramp_waypoints);
        }
        move_next_x <- location.x;
        move_next_y <- location.y;
        total_merge_success <- total_merge_success + 1;
        recent_merge_coop_ticks <- 3;
        recent_merge_x          <- location.x;
    }

    action highway_rl_move {
        if (location = nil) { return; }
        float lane_blend_hw <- 0.3;
        float step_y_hw <- 0.0;
        point steer_pt_hw <- {0.0, 0.0, 0.0};
        float rescue_y_hw <- offset_y - (lane_width / 2.0);
        float ox_hw <- 0.0;
        float oy_hw <- 0.0;
        float nx_hw <- 0.0;
        float ny_hw <- 0.0;
        // Xe hỏng: dùng cùng hệ 2-phase như behave (handle_mainline_breakdown).
        if (is_damaged) {
            do handle_mainline_breakdown();
            return;
        }

        state <- get_current_state();
        
        action_penalty <- 0.0;
        
        // 1. TÁC TỬ CHỌN HÀNH ĐỘNG (0: Tăng tốc, 1: Giữ nguyên, 2: Giảm tốc, 3: Rẽ Trái, 4: Rẽ Phải)
        if (rl_agent_id != "") {
            if (action_rl < 0) {
                action_rl <- 1;
            }
        } else {
            action_rl <- get_heuristic_highway_action();   // NPC luôn heuristic Greedy (đã bỏ random)
        }

        // Thực thi dứt khoát theo Action của RL
        if      (action_rl = 0) { speed <- min(speed_max, speed + acceleration); }
        else if (action_rl = 1) { /* Không làm gì, giữ nguyên vận tốc và hướng */ }
        else if (action_rl = 2) { speed <- max(0.0, speed - deceleration); }
        else if (action_rl = 3) { do execute_lane_change(-1); } // Chuyển làn trái (lên trên)
        else if (action_rl = 4) { do execute_lane_change(1);  } // Chuyển làn phải (xuống dưới)

        // ĐÃ GỠ BỎ: handle_blocking_cars() và "Bức tường vật lý". 
        // Từ bây giờ xe phải tự chịu trách nhiệm với speed và làn đường của nó.
        // 3. THỰC THI DI CHUYỂN
        float target_y <- offset_y + (current_lane_index * lane_width) + (lane_width / 2.0);
        float next_y   <- location.y;

        if (abs(location.y - target_y) > 0.1) {
            if (speed > 0) {
                if (location != nil) {
                    lane_blend_hw <- 0.3;
                    if (is_merging_transition) {
                        lane_blend_hw <- 0.6;
                    }
                    step_y_hw <- speed * lane_blend_hw;
                    if (location.y < target_y) { next_y <- min(target_y, location.y + step_y_hw); }
                    else                       { next_y <- max(target_y, location.y - step_y_hw); }
                    steer_pt_hw <- {location.x + speed, next_y};
                    if (steer_pt_hw != nil) {
                        heading <- atan2(steer_pt_hw.y - location.y, steer_pt_hw.x - location.x);
                    }
                }
            } else {
                heading <- 0.0;  // dừng lại trong lúc chuyển làn → ép thẳng đầu xe
            }
        } else {
            next_y                <- target_y;
            heading               <- 0.0;
            is_merging_transition <- false;
        }

        location <- {location.x + speed, next_y};
        
        // Cập nhật State, Reward
        reward_val <- calculate_reward();
        next_state <- get_current_state();
        
        // Kiểm tra xem có va chạm hay về đích không
        do check_terminal_state();

    }
    action check_terminal_state {
        // 1. Va chạm — dùng collision_distance (3.5m) thay vì car_length*0.9 (3.15m) cho nhất quán.
        list nearby_cars <- filter_collision_neighbors();
        if (length(nearby_cars) > 0) {
            car closest_car <- nil;
            float best_dist_ct <- 1.0E9;
            loop nc over: nearby_cars {
                if (nc != nil) {
                    car nc_car <- nc as car;
                    if (nc_car.location != nil) {
                        if (euclid_point_dist(self.location, nc_car.location) < best_dist_ct) {
                            best_dist_ct <- euclid_point_dist(self.location, nc_car.location);
                            closest_car <- nc_car;
                        }
                    }
                }
            }
            if (closest_car != nil) {
                if (closest_car.location != nil) {
                    if (euclid_point_dist(self.location, closest_car.location) < collision_distance) {
                        reward_val <- -100.0;
                        terminal_reason <- "collision";
                        collision_event <- true;
                        is_done <- true;
                        return;
                    }
                }
            }
        }
        
        // 2. Về đích an toàn — xe highway RL hoàn thành episode khi thoát khỏi đường.
        if (location != nil) {
            if (location.x >= road_length - 5.0) {
                reward_val <- 50.0;
                terminal_reason <- "exited";
                is_done <- true;
                return;
            }
        }
    }

    action check_merging_terminal_state {
        // Va chạm — dùng collision_distance nhất quán với check_terminal_state.
        list nearby_cars <- filter_collision_neighbors();
        if (length(nearby_cars) > 0) {
            car closest_car <- nil;
            float best_dist_cm <- 1.0E9;
            loop nc over: nearby_cars {
                if (nc != nil) {
                    car nc_car <- nc as car;
                    if (nc_car.location != nil) {
                        if (euclid_point_dist(self.location, nc_car.location) < best_dist_cm) {
                            best_dist_cm <- euclid_point_dist(self.location, nc_car.location);
                            closest_car <- nc_car;
                        }
                    }
                }
            }
            if (closest_car != nil) {
                if (closest_car.location != nil) {
                    if (euclid_point_dist(self.location, closest_car.location) < collision_distance) {
                        reward_val <- -100.0;
                        terminal_reason <- "collision";
                        collision_event <- true;
                        is_done <- true;
                        return;
                    }
                }
            }
        }

        // Nếu xe đi quá cuối vùng nhập làn mà chưa merge thì coi là thất bại của episode.
        if (is_merging) {
            if (location != nil) {
                if (location.x >= merge_x - 1.0) {
                    reward_val <- -50.0;
                    terminal_reason <- "failed_merge";
                    failed_merge <- true;
                    is_done <- true;
                    return;
                }
            }
        }

        // Het ramp ma chua merge (con tren polyline merge_mode=1).
        if (merge_mode = 1) {
            if (location != nil) {
                if (location.x >= merge_x + 2.0) {
                    reward_val <- -50.0;
                    terminal_reason <- "failed_merge";
                    failed_merge <- true;
                    is_done <- true;
                    return;
                }
            }
        }
    }

    action execute_lane_change(int dir) {
        if (is_merging) { return; }
        
        // Nếu đang trượt mượt (merge transition), bỏ qua lệnh rẽ — không phạt.
        if (is_merging_transition) { return; } 
        
        int new_lane <- current_lane_index + dir;

        if (new_lane >= 0) {
            if (new_lane < number_of_lanes) {
                target_lane_index <- new_lane;
                current_lane_index <- new_lane;
                patience <- 100;
                action_penalty <- action_penalty - 0.3;   // đồng bộ behave: phạt đổi làn cơ bản nhẹ (cho phép né 1 lần); flip-flop bị -2.1 ở behave
            } else {
                action_penalty <- action_penalty - 2.0;
            }
        } else {
            action_penalty <- action_penalty - 2.0;
        }
    }

    action get_car_ahead(int lane) type: car {
        list ahead_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            // Dung physical_lane (theo Y vat ly) — tranh "ma xe" khi xe khac vua action 3/4
                            // (current_lane_index update tuc thi nhung Y vat ly van o lane cu).
                            if (c_car.physical_lane = lane) {
                                if (c_car.location.x >= self.location.x) {
                                    if (c_car.location.x - self.location.x <= observation_max) {
                                        ahead_cars << c_car;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (length(ahead_cars) = 0) {
            return nil;
        }
        car closest_ahead <- nil;
        float best_dx_ahead <- 1.0E9;
        loop ac over: ahead_cars {
            if (ac != nil) {
                car ac_car <- ac as car;
                if (ac_car.location != nil) {
                    if (ac_car.location.x - self.location.x < best_dx_ahead) {
                        best_dx_ahead <- ac_car.location.x - self.location.x;
                        closest_ahead <- ac_car;
                    }
                }
            }
        }
        return closest_ahead;
    }

    action get_car_behind(int lane) type: car {
        list behind_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            // Dung physical_lane — same rationale nhu get_car_ahead.
                            if (c_car.physical_lane = lane) {
                                if (c_car.location.x <= self.location.x) {
                                    if (self.location.x - c_car.location.x <= observation_max) {
                                        behind_cars << c_car;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (length(behind_cars) = 0) {
            return nil;
        }
        car closest_behind <- nil;
        float best_dx_behind <- 1.0E9;
        loop bc over: behind_cars {
            if (bc != nil) {
                car bc_car <- bc as car;
                if (bc_car.location != nil) {
                    if (self.location.x - bc_car.location.x < best_dx_behind) {
                        best_dx_behind <- self.location.x - bc_car.location.x;
                        closest_behind <- bc_car;
                    }
                }
            }
        }
        return closest_behind;
    }

    action get_target_lane_ahead type: car {
        int merge_lane <- number_of_lanes - 1;
        list ahead_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            if (not c_car.is_merging) {
                                // Dung physical_lane (Y thuc te) — tranh "ma xe" khi xe khac vua action 3/4.
                                if (c_car.physical_lane = merge_lane) {
                                    if (c_car.location.x >= self.location.x) {
                                        if (c_car.location.x - self.location.x <= observation_max) {
                                            ahead_cars << c_car;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (length(ahead_cars) = 0) {
            return nil;
        }
        car closest_target_ahead <- nil;
        float best_dx_tahead <- 1.0E9;
        loop ac over: ahead_cars {
            if (ac != nil) {
                car ac_car <- ac as car;
                if (ac_car.location != nil) {
                    if (ac_car.location.x - self.location.x < best_dx_tahead) {
                        best_dx_tahead <- ac_car.location.x - self.location.x;
                        closest_target_ahead <- ac_car;
                    }
                }
            }
        }
        return closest_target_ahead;
    }

    action get_target_lane_behind type: car {
        int merge_lane <- number_of_lanes - 1;
        list behind_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            if (not c_car.is_merging) {
                                // Dung physical_lane — same rationale.
                                if (c_car.physical_lane = merge_lane) {
                                    if (c_car.location.x <= self.location.x) {
                                        if (self.location.x - c_car.location.x <= observation_max) {
                                            behind_cars << c_car;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (length(behind_cars) = 0) {
            return nil;
        }
        car closest_target_behind <- nil;
        float best_dx_tbehind <- 1.0E9;
        loop bc over: behind_cars {
            if (bc != nil) {
                car bc_car <- bc as car;
                if (bc_car.location != nil) {
                    if (self.location.x - bc_car.location.x < best_dx_tbehind) {
                        best_dx_tbehind <- self.location.x - bc_car.location.x;
                        closest_target_behind <- bc_car;
                    }
                }
            }
        }
        return closest_target_behind;
    }

    // CỔNG THỰC THI merge — TÁCH khỏi is_merge_gap_safe() để hàm đó luôn báo gap THẬT cho
    // observation / reward / dashboard. Cổng mở khi: gap thật an toàn, HOẶC gate đã tắt
    // (Python mode, shield off) → cho merge bất chấp (baseline công bằng, RL tự học an toàn).
    // Heuristic GUI (dashboard != Python) LUÔN giữ cổng theo gap thật → demo không đâm.
    action merge_gate_open type: bool {
        if (is_merge_gap_safe()) { return true; }
        // Cổng merge CHỈ phụ thuộc commit_gate_real (KHÔNG còn dính enable_safety_shield — đã tách
        // để bật khiên car-following cho NPC mà không bật lại cổng merge). Heuristic GUI giữ cổng.
        if (not commit_gate_real and dashboard_policy = "Python/SB3 model") { return true; }
        return false;
    }

    action is_merge_gap_safe type: bool {
        // LUÔN tính gap THẬT (không short-circuit). Việc "merge bất chấp khi gate off" do
        // merge_gate_open() xử lý ở chỗ thực thi — nhờ vậy obs_gap_safe/reward/dashboard trung thực.
        // Lấy xe trước và xe sau ở làn mục tiêu để đánh giá khoảng trống nhập làn.
        car lead_car <- get_target_lane_ahead();
        car lag_car  <- get_target_lane_behind();

        // Nếu không có xe trong tầm quan sát thì coi khoảng cách là rất lớn.
        float gap_front <- 999.0;
        float gap_rear  <- 999.0;
        if (lead_car != nil) {
            if (lead_car.location != nil) {
                gap_front <- lead_car.location.x - self.location.x;
            }
        }
        if (lag_car != nil) {
            if (lag_car.location != nil) {
                gap_rear <- self.location.x - lag_car.location.x;
            }
        }

        // Hệ số gap: xe RL merging_0 nới ro de hoc merge duoc; NPC / rule-merge giữ ngưỡng chặt hơn.
        // Sua bug: required_front=8m (floor gap_front_min) lam gap hiem khi du o medium/low density,
        // ket hop voi 3 highway RL deu o lane 2 -> auto-merge fail -> failed_merge / timeout.
        float k_front <- 4.0;
        float k_rear  <- 3.0;
        float floor_front <- gap_front_min;
        float floor_rear  <- gap_rear_min;
        if (rl_agent_id = "merging_0") {
            k_front <- 1.5;
            k_rear  <- 1.0;
            floor_front <- 5.0;
            floor_rear  <- 4.0;
        }
        // Xe chạy nhanh cần khoảng cách phía trước lớn hơn vì quãng đường phanh dài hơn.
        float required_front <- max(floor_front, car_length * 1.15 + speed * k_front);
        // Xe phía sau chạy nhanh cần gap sau lớn hơn để tránh bị tông khi nhập làn.
        float required_rear  <- max(floor_rear,  car_length * 1.15 + speed * k_rear);
        // CURRICULUM: nới gap cho merging_0 (relax<1 = dễ hơn ở stage 1; =1 = ngặt thật ở stage 2).
        if (rl_agent_id = "merging_0") {
            required_front <- required_front * merge_gap_relax;
            required_rear  <- required_rear  * merge_gap_relax;
        }

        if (gap_front >= required_front) {
            if (gap_rear >= required_rear) {
                return true;
            }
        }
        return false;
    }

    action get_ramp_front_gap type: float {
        list ramp_front_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            if (c_car.is_on_ramp_merge_path()) {
                                if (c_car.location.x >= self.location.x) {
                                    if (c_car.location.x - self.location.x <= observation_max) {
                                        ramp_front_cars << c_car;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        car ramp_front <- nil;
        if (length(ramp_front_cars) > 0) {
            float best_dx_rf <- 1.0E9;
            loop rc over: ramp_front_cars {
                if (rc != nil) {
                    car rc_car <- rc as car;
                    if (rc_car.location != nil) {
                        if (rc_car.location.x - self.location.x < best_dx_rf) {
                            best_dx_rf <- rc_car.location.x - self.location.x;
                            ramp_front <- rc_car;
                        }
                    }
                }
            }
        }
        if (ramp_front = nil) { return 999.0; }
        if (ramp_front.location = nil) { return 999.0; }
        return ramp_front.location.x - self.location.x;
    }

    action get_heuristic_highway_action type: int {
        // Ngữ nghĩa action trong behave (dòng ~2046): action_rl=0 → TĂNG tốc, action_rl=2 → GIẢM tốc.
        // Heuristic này phải dùng đúng quy ước đó.
        // COOLDOWN GUARD: vua chuyen lan -> KHONG de xuat chuyen tiep (chong spam 3/4).
        // Chi tra ve action 0/1/2 (toc do) trong khoang cooldown — xe co thoi gian on dinh tren lane moi.
        bool block_lane_change <- (lane_change_cooldown > 0);
        car ahead_same <- get_car_ahead(current_lane_index);
        float gap_same <- 999.0;
        if (ahead_same != nil) {
            if (ahead_same.location != nil) {
                gap_same <- ahead_same.location.x - self.location.x;
            }
        }
        list merging_ahead <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            if (c_car.is_on_ramp_merge_path()) {
                                if (c_car.location.x >= self.location.x) {
                                    if (c_car.location.x - self.location.x <= 50.0) {
                                        merging_ahead << c_car;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        car merger <- nil;
        if (length(merging_ahead) > 0) {
            float best_dx_mh <- 1.0E9;
            loop mc over: merging_ahead {
                if (mc != nil) {
                    car mc_car <- mc as car;
                    if (mc_car.location != nil) {
                        if (mc_car.location.x - self.location.x < best_dx_mh) {
                            best_dx_mh <- mc_car.location.x - self.location.x;
                            merger <- mc_car;
                        }
                    }
                }
            }
        }

        float merge_zone_lo <- accel_start_x - 30.0;
        float merge_zone_hi <- merge_x + 25.0;

        // Ưu tiên 1: Giữ khoảng cách an toàn với xe phía trước cùng làn.
        // action_rl=2 → GIẢM tốc; action_rl=0 → TĂNG tốc (theo behave).
        if (ahead_same != nil) {
            float safe_distance <- 12.0;
            if (ahead_same.speed < 0.1) {
                safe_distance <- 20.0;
            } else {
                if (speed - ahead_same.speed > 0.5) {
                    safe_distance <- 16.0;
                }
            }

            // NetLogo intent: khi xe truoc DAMAGED/STOP (speed < 0.05), UU TIEN chuyen lan ngay
            // (NetLogo: handle-blocking-cars khi blocking damaged -> choose-new-lane lien tiep).
            // Nguong gap an toan lane ben giam tu 15 -> 8m (gan voi NetLogo observation-distance=2.5 patches).
            // GUARDS:
            //   - skip neu cooldown active (vua chuyen lan, can on dinh)
            //   - SUA BUG: them gate khoang cach. Truoc day khong gate -> get_car_ahead quet
            //     trong observation_max=100m -> 1 xe hong cach 80m van trigger chuyen lan vu vo
            //     du duong truoc thuc su thoang. Cascading lane changes (RL + NPC dung chung
            //     heuristic) gay tac cao toc khi co 1 xe hong.
            bool ahead_stopped <- (ahead_same.speed < 0.05) and (gap_same < 28.0) and (not block_lane_change);
            if (ahead_stopped) {
                // Thu trai truoc
                if (current_lane_index > 0) {
                    car lead_l_stop <- get_car_ahead(current_lane_index - 1);
                    car lag_l_stop  <- get_car_behind(current_lane_index - 1);
                    bool safe_l_stop <- true;
                    if (lead_l_stop != nil) {
                        if (lead_l_stop.location != nil) {
                            if (lead_l_stop.location.x - self.location.x < 8.0) { safe_l_stop <- false; }
                        }
                    }
                    if (lag_l_stop != nil) {
                        if (lag_l_stop.location != nil) {
                            if (self.location.x - lag_l_stop.location.x < 6.0) { safe_l_stop <- false; }
                        }
                    }
                    if (safe_l_stop) { return 3; }
                }
                // Thu phai
                if (current_lane_index < number_of_lanes - 1) {
                    car lead_r_stop <- get_car_ahead(current_lane_index + 1);
                    car lag_r_stop  <- get_car_behind(current_lane_index + 1);
                    bool safe_r_stop <- true;
                    if (lead_r_stop != nil) {
                        if (lead_r_stop.location != nil) {
                            if (lead_r_stop.location.x - self.location.x < 8.0) { safe_r_stop <- false; }
                        }
                    }
                    if (lag_r_stop != nil) {
                        if (lag_r_stop.location != nil) {
                            if (self.location.x - lag_r_stop.location.x < 6.0) { safe_r_stop <- false; }
                        }
                    }
                    if (safe_r_stop) { return 4; }
                }
                // Khong co lane thoat -> phanh hau (return 2 phia duoi)
            }

            // LANE-CHANGE chi khi THUC SU CAN — tranh chuyen lan vu vo khi van di thang OK.
            // 3 dieu kien dong thoi:
            //   (a) Gap thuc su gan (< safe_distance, KHONG +8) — vao zone phai phan ung
            //   (b) Self chua dat toc cao (< 80% speed_max) — co dong luc passing
            //   (c) Xe truoc CHAM hon dang ke (< self.speed - 0.15) — khong overtake xe cung speed
            // Bonus: chi chuyen sang lane co LOI (lead_left/right vang hoac nhanh hon ahead_same).
            // Truoc day: nguong gap < safe+8=20m + khong check speed diff -> chuyen lan vu vo.
            // BỎ overtake tuỳ ý cho xe heuristic (NPC): không đổi làn để vượt xe chậm — chỉ bám đuôi/giảm tốc.
            // Vẫn giữ escape khi xe trước DỪNG hẳn (ahead_stopped ở trên). Tránh "đổi làn ngu" trong demo.
            bool need_passing <- false;
            if (need_passing and (not block_lane_change)) {
                // Thử sang trái (làn nhanh hơn)
                if (current_lane_index > 0) {
                    car lead_left <- get_car_ahead(current_lane_index - 1);
                    car lag_left  <- get_car_behind(current_lane_index - 1);
                    bool safe_left_h <- true;
                    if (lead_left != nil) {
                        if (lead_left.location != nil) {
                            if (lead_left.location.x - self.location.x < 8.0) { safe_left_h <- false; }
                        }
                    }
                    if (lag_left != nil) {
                        if (lag_left.location != nil) {
                            if (self.location.x - lag_left.location.x < 6.0) { safe_left_h <- false; }
                        }
                    }
                    if (safe_left_h) {
                        if (lead_left = nil) {
                            return 3; // Lane trai trong hoan toan -> chuyen
                        } else {
                            if (lead_left.speed > ahead_same.speed + 0.10) {
                                return 3; // Lane trai nhanh hon dang ke -> chuyen
                            }
                            // else: lane trai cung cham -> khong chuyen, giu lan
                        }
                    }
                }
                // Thử sang phải
                if (current_lane_index < number_of_lanes - 1) {
                    car lead_right <- get_car_ahead(current_lane_index + 1);
                    car lag_right  <- get_car_behind(current_lane_index + 1);
                    bool safe_right_h <- true;
                    if (lead_right != nil) {
                        if (lead_right.location != nil) {
                            if (lead_right.location.x - self.location.x < 8.0) { safe_right_h <- false; }
                        }
                    }
                    if (lag_right != nil) {
                        if (lag_right.location != nil) {
                            if (self.location.x - lag_right.location.x < 6.0) { safe_right_h <- false; }
                        }
                    }
                    if (safe_right_h) {
                        if (lead_right = nil) {
                            return 4;
                        } else {
                            if (lead_right.speed > ahead_same.speed + 0.10) {
                                return 4;
                            }
                        }
                    }
                }
            }

            if (gap_same < safe_distance) {
                // Quá gần → GIẢM tốc (action_rl=2 trong behave = giảm)
                return 2;
            } else {
                if (gap_same < safe_distance + 6.0) {
                    // Vùng bám đuôi: đồng bộ tốc độ
                    if (speed > ahead_same.speed + 0.05) {
                        return 2; // Hơi nhanh → GIẢM
                    } else {
                        if (speed < ahead_same.speed - 0.05) {
                            return 0; // Hơi chậm → TĂNG
                        } else {
                            return 1; // Đồng bộ → GIỮ
                        }
                    }
                }
            }
        }

        // Ưu tiên 2: Nhường đường cho xe xin nhập làn (chỉ ở làn phải ngoài cùng).
        // Truoc: dx_merger trong (-5, 20) -> yield range 25m, qua dai. Voi merge_zone ~187m
        // va ramp cars lien tuc xuat hien, lane 2 thanh "vung cham" -> NPC dam vao xe RL trong
        // lane 2 phai luon lane change len lane 1 de qua mat -> traffic pattern trong ky.
        // Sua: thu hep ve (0, 8m) — chi yield khi merger THUC SU truoc minh + RAT GAN.
        // Safety shield M3c (line ~2029, ahead_gap_dx < 12) van lo nhung truong hop con lai.
        if (merger != nil) {
            if (location != nil) {
                if (location.x >= merge_zone_lo) {
                    if (location.x <= merge_zone_hi) {
                        if (current_lane_index = number_of_lanes - 1) {
                            if (merger.location != nil) {
                                float dx_merger <- merger.location.x - self.location.x;
                                if (dx_merger > 0.0) {
                                    if (dx_merger < 8.0) {
                                        return 2; // Giảm tốc nhường đường (action_rl=2 = GIẢM)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Ưu tiên 3: Tăng tốc nếu đường thoáng (action_rl=0 = TĂNG trong behave).
        if (ahead_same != nil) {
            if (gap_same >= 15.0) {
                if (speed < speed_max * 0.9) {
                    return 0; // Tăng tốc
                }
            }
        } else {
            if (speed < speed_max * 0.9) {
                return 0; // Không có xe trước → Tăng tốc
            }
        }

        return 1; // Mặc định giữ tốc độ
    }

    action get_heuristic_merging_action type: int {
        // Rule-based baseline (Greedy) dùng cho dashboard GAMA và NPC ramp.
        float ramp_gap <- get_ramp_front_gap();
        float accel_len <- max(1.0, accel_end_x - accel_start_x);
        float urgency <- 0.0;
        if (location != nil) {
            urgency <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len));
        }

        // Ưu tiên 1: Giữ khoảng cách với xe phía trước trên ramp
        if (ramp_gap < 12.0) { return 0; }

        // Ưu tiên 2: Nhập làn nếu an toàn — kích hoạt từ khi vào accel zone trở đi,
        // KHÔNG dừng thử merge sau khi vượt accel_end_x (execute_merge vẫn cho phép tới merge_x-14).
        // Trước đây gate `in_accel_zone` khiến heuristic bỏ rơi xe khi out-of-zone -> deadlock.
        if (urgency > 0.0) {
            if (is_merge_gap_safe()) {
                return 3; // Nhập làn
            }
        }

        // Ưu tiên 3: Cực urgent (~hết đường chéo) mà gap vẫn chật -> nới ngưỡng để phá deadlock.
        // Khi rear car đang đứng im (gridlock / xe hỏng / xe RL khác stuck), gap 1.5m vẫn an toàn
        // vì không có chuyển động tương đối -> nới thật sự xuống mức vật lý tối thiểu.
        // Khi rear còn tốc độ -> chấp nhận rủi ro nhỏ nhưng giữ buffer động.
        if (urgency > 0.95) {
            car lead_em <- get_target_lane_ahead();
            car lag_em  <- get_target_lane_behind();
            float gap_front_em <- 999.0;
            float gap_rear_em  <- 999.0;
            float lag_speed <- 0.0;
            if (lead_em != nil) {
                if (lead_em.location != nil) {
                    gap_front_em <- lead_em.location.x - self.location.x;
                }
            }
            if (lag_em != nil) {
                if (lag_em.location != nil) {
                    gap_rear_em <- self.location.x - lag_em.location.x;
                    lag_speed <- lag_em.speed;
                }
            }
            float req_front_em <- max(5.0, car_length * 0.8 + speed * 0.8);
            float req_rear_em;
            if (lag_speed < 0.1) {
                // Rear đứng im -> chỉ cần đủ chỗ vật lý (~ nửa chiều dài xe)
                req_rear_em <- max(1.5, car_length * 0.5);
            } else {
                // Rear còn chạy -> giữ buffer theo tốc độ rear (không phải self)
                req_rear_em <- max(2.5, car_length * 0.6 + lag_speed * 0.8);
            }
            if (gap_front_em >= req_front_em) {
                if (gap_rear_em >= req_rear_em) {
                    return 3; // Force-merge khẩn cấp
                }
            }
        }

        // Ưu tiên 4: Sắp hết đường chéo mà không merge được.
        // Đứng im rồi (speed ≈ 0) -> trả Wait (4) để phân biệt "deadlock chờ" vs "phanh chủ động";
        // còn tốc độ -> Dec (0) để phanh chờ cơ hội.
        if (urgency > 0.8) {
            if (speed < 0.1) {
                return 4; // Wait — đã dừng hẳn, không cần phanh thêm
            }
            return 0; // Phanh hẳn chờ cơ hội
        }

        // Ưu tiên 5: Tăng tốc để bắt kịp luồng xe trên cao tốc
        if (speed < speed_max * 0.85) { return 2; }

        return 1; // Giữ tốc độ
    }

    action update_merging_metrics {
        // Gom metric episode ngay trong GAMA để Python có thể log success/collision/min-gap ổn định.
        car lead_car <- get_target_lane_ahead();
        car lag_car  <- get_target_lane_behind();

        float gap_front <- 999.0;
        float gap_rear  <- 999.0;
        if (lead_car != nil) {
            if (lead_car.location != nil) {
                gap_front <- max(0.0, lead_car.location.x - self.location.x);
            }
        }
        if (lag_car != nil) {
            if (lag_car.location != nil) {
                gap_rear <- max(0.0, self.location.x - lag_car.location.x);
            }
        }

        min_front_gap <- min(min_front_gap, gap_front);
        min_rear_gap  <- min(min_rear_gap, gap_rear);
        speed_sum <- speed_sum + speed;
        cumulative_reward <- cumulative_reward + reward_val;
    }

    action get_episode_info type: map {
        info_mean_speed <- 0.0;
        if (episode_step > 0) {
            info_mean_speed <- speed_sum / max(1, episode_step);
        }
        // Xe cao tốc RL (highway_i) dùng terminal_reason khác: "exited" khi về đích, không có merge_success.
        if (rl_agent_id != "") {
            if (rl_agent_id != "merging_0") {
            return [
                "outcome"::terminal_reason,
                "success"::(terminal_reason = "exited"),
                "collision"::collision_event,
                "failed_merge"::false,
                "episode_step"::episode_step,
                "mean_speed"::info_mean_speed,
                "min_front_gap"::min_front_gap,
                "min_rear_gap"::min_rear_gap,
                "cumulative_reward"::cumulative_reward,
                "speed"::speed,
                "agent_role"::"highway",
                "dbg_decel_total"::dbg_decel_total,
                "dbg_decel_mrg"::dbg_decel_mrg,
                "dbg_decel_ahead"::dbg_decel_ahead
            ];
            }
        }
        // Throughput = tỷ lệ xe merge thành công / tổng xe đã thử merge trong simulation.
        info_throughput <- 0.0;
        if (total_ramp_attempts > 0) {
            info_throughput <- total_merge_success / max(1, total_ramp_attempts);
        }

        // Shockwave index = std_dev / mean (Coefficient of Variation tốc độ mainline ở vùng merge).
        info_sw_variance <- 0.0;
        if (sw_n > 1) {
            info_sw_variance <- sw_M2 / max(1, sw_n - 1);
        }
        info_sw_std <- 0.0;
        if (info_sw_variance > 0) {
            info_sw_std <- sqrt(info_sw_variance);
        }
        info_shockwave_index <- 0.0;
        if (sw_mean > 0.001) {
            info_shockwave_index <- info_sw_std / sw_mean;
        }
        // braking_event_rate = số lần phanh gấp / tổng (xe×mẫu) — sóng lùi đúng chuẩn (literature).
        float info_braking_rate <- 0.0;
        if (sw_car_ticks > 0) {
            info_braking_rate <- braking_events / sw_car_ticks;
        }

        return [
            "outcome"::terminal_reason,
            "success"::merge_success,
            "collision"::collision_event,
            "failed_merge"::failed_merge,
            "episode_step"::episode_step,
            "mean_speed"::info_mean_speed,
            "min_front_gap"::min_front_gap,
            "min_rear_gap"::min_rear_gap,
            "cumulative_reward"::cumulative_reward,
            "speed"::speed,
            "gap_safe"::is_merge_gap_safe(),
            "in_accel_zone"::in_accel_zone,
            "agent_role"::"merging",
            "merge_step"::merge_step,
            "throughput"::info_throughput,
            "total_ramp_attempts"::total_ramp_attempts,
            "total_merge_success"::total_merge_success,
            "shockwave_index"::info_shockwave_index,
            "mainline_mean_speed"::sw_mean,
            "braking_event_rate"::info_braking_rate,
            "merge_x_pos"::dbg_merge_x_pos
        ];
    }

    action get_current_state type: list<float> {
        // M3b RADAR 15D: dung 14 scratch fields precomputed trong reflex behave scan loop.
        // 15 features (tat ca normalized [0,1]):
        //   [0]  own speed
        //   [1]  own lane (0=bottom, 0.5=mid, 1=top)
        //   [2]  same-lane ahead dist  | [3]  same-lane ahead speed
        //   [4]  left-lane ahead dist  | [5]  left-lane ahead speed
        //   [6]  left-lane behind dist | [7]  left-lane behind speed
        //   [8]  right-lane ahead dist | [9]  right-lane ahead speed
        //   [10] right-lane behind dist| [11] right-lane behind speed
        //   [12] closest merger dist   | [13] closest merger speed
        //   [14] own patience
        // KHONG return bien tam moi (TempVariable). Tat ca o scratch fields chinh thuc.
        obs_speed <- speed / max(0.001, speed_max);
        if (obs_speed < 0.0) { obs_speed <- 0.0; }
        if (obs_speed > 1.0) { obs_speed <- 1.0; }
        obs_lane <- current_lane_index / max(1.0, number_of_lanes - 1.0);
        obs_patience <- patience / 100.0;
        if (obs_patience < 0.0) { obs_patience <- 0.0; }
        if (obs_patience > 1.0) { obs_patience <- 1.0; }
        // Normalize gap (chia 100m max) va speed (chia speed_max) -> [0,1].
        // Default 1.0 khi khong co xe (gap >= 1.0E9 -> sau min = 1.0).
        return [
            obs_speed,
            obs_lane,
            min(1.0, ahead_gap_dx / 100.0),     min(1.0, ahead_speed_other / max(0.001, speed_max)),
            min(1.0, ahead_gap_left / 100.0),   min(1.0, ahead_speed_left / max(0.001, speed_max)),
            min(1.0, behind_gap_left / 100.0),  min(1.0, behind_speed_left / max(0.001, speed_max)),
            min(1.0, ahead_gap_right / 100.0),  min(1.0, ahead_speed_right / max(0.001, speed_max)),
            min(1.0, behind_gap_right / 100.0), min(1.0, behind_speed_right / max(0.001, speed_max)),
            min(1.0, merger_gap_x / 100.0),     min(1.0, merger_speed / max(0.001, speed_max)),
            obs_patience
        ];
    }

    action get_merging_state type: list<float> {
        // Vector 15D chuan (khop rl/baselines.py + README): speed, progress, dist_merge, lateral,
        // in_accel, gap_front, speed_front, gap_rear, speed_rear, gap_safe, ramp_gap, ramp_spd,
        // urgency, last_action, patience.
        float speed_denom <- max(0.001, speed_max);
        float obs_denom <- max(1.0, observation_max);
        float norm_speed <- min(1.0, max(0.0, speed / speed_denom));
        float accel_len <- max(1.0, accel_end_x - accel_start_x);
        float norm_progress <- 0.0;
        float norm_dist_to_merge <- 1.0;
        float norm_lateral <- 1.0;
        float norm_in_accel <- 0.0;
        float norm_urgency <- 0.0;
        if (location != nil) {
            norm_progress <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len));
            norm_dist_to_merge <- min(1.0, max(0.0, (merge_x - location.x) / accel_len));
            norm_lateral <- min(1.0, max(0.0, abs(location.y - bottom_lane_y) / max(1.0, lane_width * 4.0)));
            if (location.x >= accel_start_x) {
                norm_urgency <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len));
            }
        }
        if (in_accel_zone) {
            norm_in_accel <- 1.0;
        }
        obs_front_gap_raw <- 999.0;
        obs_rear_gap_raw <- 999.0;
        obs_front_speed_raw <- speed_max;
        float obs_rear_speed_raw <- 0.0;
        float ramp_front_gap_raw <- 999.0;
        float ramp_front_speed_raw <- speed_max;
        if (location != nil) {
            if (safe_cars != nil) {
                loop c over: safe_cars {
                    if (c != nil) {
                        car c_car <- c as car;
                        if (c_car != nil) {
                            if (not dead(c_car)) {
                                if (c_car != self) {
                                    if (c_car.location != nil) {
                                        if (not c_car.is_merging) {
                                            // physical_lane: tranh "ma xe" trong obs cua merging_0.
                                            if (c_car.physical_lane = number_of_lanes - 1) {
                                                if (c_car.location.x >= location.x) {
                                                    obs_scan_dx <- c_car.location.x - location.x;
                                                    if (obs_scan_dx < obs_front_gap_raw) {
                                                        obs_front_gap_raw <- obs_scan_dx;
                                                        obs_front_speed_raw <- c_car.speed;
                                                    }
                                                } else {
                                                    obs_scan_dx <- location.x - c_car.location.x;
                                                    if (obs_scan_dx < obs_rear_gap_raw) {
                                                        obs_rear_gap_raw <- obs_scan_dx;
                                                        obs_rear_speed_raw <- c_car.speed;
                                                    }
                                                }
                                            }
                                        } else {
                                            if (c_car.is_on_ramp_merge_path()) {
                                                if (c_car.location.x > location.x) {
                                                    obs_scan_dx <- c_car.location.x - location.x;
                                                    if (obs_scan_dx < ramp_front_gap_raw) {
                                                        ramp_front_gap_raw <- obs_scan_dx;
                                                        ramp_front_speed_raw <- c_car.speed;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        float norm_gap_front <- min(1.0, max(0.0, obs_front_gap_raw / obs_denom));
        float norm_speed_front <- min(1.0, max(0.0, obs_front_speed_raw / speed_denom));
        float norm_gap_rear <- min(1.0, max(0.0, obs_rear_gap_raw / obs_denom));
        float norm_speed_rear <- min(1.0, max(0.0, obs_rear_speed_raw / speed_denom));
        float norm_gap_safe <- 0.0;
        if (is_merge_gap_safe()) {
            norm_gap_safe <- 1.0;
        }
        // Sua bug: truoc day chi gan local `norm_gap_safe`, KHONG ghi field `obs_gap_safe`.
        // -> calculate_merging_reward doc obs_gap_safe = default 0.0 -> bonus "+0.05 khi gap safe" KHONG BAO GIO an.
        obs_gap_safe <- norm_gap_safe;
        float norm_ramp_front_gap <- min(1.0, max(0.0, ramp_front_gap_raw / obs_denom));
        float norm_ramp_front_speed <- min(1.0, max(0.0, ramp_front_speed_raw / speed_denom));
        float norm_last_action <- 0.0;
        if (action_rl >= 0) {
            norm_last_action <- min(1.0, (action_rl + 1) / 5.0);
        }
        float norm_patience <- min(1.0, max(0.0, patience / 100.0));
        return [
            norm_speed,
            norm_progress, norm_dist_to_merge,
            norm_lateral, norm_in_accel,
            norm_gap_front, norm_speed_front,
            norm_gap_rear, norm_speed_rear,
            norm_gap_safe,
            norm_ramp_front_gap, norm_ramp_front_speed,
            norm_urgency, norm_last_action,
            norm_patience
        ];
    }

    // ─── HDV (NPC) car-following + lane-change: IDM + MOBIL ───
    // IDM (Treiber 2000): gia toc doc theo leader cung lan. Tham so scaled don vi sim
    // (speed in [0,1]/cycle, x theo met). gap_center = khoang cach tam-den-tam toi leader.
    action idm_acc(float v, float gap_center, float lead_v) type: float {
        float v0 <- max(0.001, speed_max);
        float a0 <- max(0.001, acceleration);
        float b0 <- max(0.001, deceleration);
        float vr <- v / v0;
        float free_term <- 1.0 - vr * vr * vr * vr;
        float interaction <- 0.0;
        if (gap_center < observation_max) {
            float s <- gap_center - car_length;          // net bumper gap
            if (s < 0.1) { s <- 0.1; }
            float dv <- v - lead_v;                       // approach rate
            float s_star <- idm_min_gap + v * idm_time_headway + (v * dv) / (2.0 * sqrt(a0 * b0));
            if (s_star < idm_min_gap) { s_star <- idm_min_gap; }
            float ratio <- s_star / s;
            interaction <- ratio * ratio;
        }
        float acc <- a0 * (free_term - interaction);
        if (acc > a0) { acc <- a0; }
        if (acc < -0.6) { acc <- -0.6; }                  // floor phanh khan cap
        return acc;
    }

    // MOBIL (Kesting/Treiber 2007): doi lan khi AN TOAN (xe sau lan dich khong bi ep phanh > b_safe)
    // VA LOI ICH (gia toc tang vuot nguong, tru phi politeness). Tra 1=giu, 3=trai, 4=phai.
    // Dung scratch radar ahead/behind_gap_left/right precompute trong reflex behave (1-tick stale, OK).
    action mobil_decision type: int {
        if (lane_change_cooldown > 0) { return 1; }
        float a_cur <- idm_acc(speed, ahead_gap_dx, ahead_speed_other);
        int best <- 1;
        float best_gain <- idm_lc_threshold;
        if (current_lane_index > 0) {
            float a_me_l <- idm_acc(speed, ahead_gap_left, ahead_speed_left);
            float a_nf_l <- idm_acc(behind_speed_left, behind_gap_left, speed);
            if (a_nf_l >= (0.0 - idm_b_safe)) {
                float dec_l <- 0.0;
                if (a_nf_l < 0.0) { dec_l <- 0.0 - a_nf_l; }
                float gain_l <- (a_me_l - a_cur) - idm_politeness * dec_l;
                if (gain_l > best_gain) { best_gain <- gain_l; best <- 3; }
            }
        }
        if (current_lane_index < number_of_lanes - 1) {
            float a_me_r <- idm_acc(speed, ahead_gap_right, ahead_speed_right);
            float a_nf_r <- idm_acc(behind_speed_right, behind_gap_right, speed);
            if (a_nf_r >= (0.0 - idm_b_safe)) {
                float dec_r <- 0.0;
                if (a_nf_r < 0.0) { dec_r <- 0.0 - a_nf_r; }
                float gain_r <- (a_me_r - a_cur) - idm_politeness * dec_r;
                if (gain_r > best_gain) { best_gain <- gain_r; best <- 4; }
            }
        }
        return best;
    }

    action get_safe_distance type: float {
        // Tốc độ cao cần khoảng cách phanh dài hơn để mô hình RL không bị phạt oan
        if (speed <= 0.4) { return 8.0; }        // ~ Tốc độ chậm
        else if (speed <= 0.7) { return 15.0; }  // ~ Tốc độ trung bình
        else { return 25.0; }                    // ~ Tốc độ cao
    }


    action calculate_reward type: float {
        // M2: dung field scalar `ahead_gap_dx`, `recent_merge_*` (precomputed trong `reflex behave`)
        // de tranh point-loop trong hot path. KHONG goi helper co `return` bien cuc bo.
        if (enable_reward_collision_check) {
            if (nearest_collision_dist < collision_distance) {
                return -100.0;
            }
        }
        if (rl_agent_id != "") {
            if (rl_agent_id = "merging_0") {
                reward_cache <- calculate_merging_reward();
                return reward_cache;
            }
        }
        // Reward base cho highway / mainline agents.
        //   - speed * 0.04: khuyến khích duy trì tốc độ; hệ số nhỏ để không đè safety signals.
        //   - low-speed penalty (-0.05): tránh policy "đứng yên" trên cao tốc.
        reward_cache <- action_penalty + speed * 0.04;
        // INTERVENTION COST (đối xứng merger): khiên IDM-cap ép phanh > RL muốn = highway lái ẩu/sắp đâm
        // → phạt -1 → highway học tự phanh-từ-xa (không tăng tốc đâm chết) + không ỷ lại khiên.
        if (shield_intervened) {
            reward_cache <- reward_cache - 1.0;
        }
        if (speed < speed_max * 0.2) {
            reward_cache <- reward_cache - 0.05;
        }
        // ANTI-CRAWL: đường THOÁNG (gap>15) + KHÔNG có merger gần phía trước + đi chậm (<50% max)
        // -> PHẠT MẠNH để phá "bò chậm vô cớ". Khoanh vùng: KHÔNG phạt khi đang nhường merger gần.
        if (ahead_gap_dx > 22.0) {   // chỉ "đường thoáng THẬT" (>22) mới ép nhanh; vùng <20 nhường car-following phanh
            bool merger_near_ac <- false;
            if (active_merger_x >= 0.0) {
                if (move_next_x < active_merger_x) {
                    if ((active_merger_x - move_next_x) < 25.0) { merger_near_ac <- true; }
                }
            }
            if (not merger_near_ac) {
                if (speed < speed_max * 0.5) {
                    reward_cache <- reward_cache - 0.5;
                }
            }
        }
        // Flow-aware (Plan V1 §10): thưởng giữ tốc cao + PHẠT GIẢM TỐC THỪA -> chống highway "phanh thừa".
        if (enable_highway_flow_reward) {
            reward_cache <- reward_cache + speed * 0.06;
            if (speed >= speed_max * 0.7) {
                reward_cache <- reward_cache + 0.05;
            }
            // ③ TARGETED YIELD: phạt GIẢM TỐC (action 2) khi đường thoáng VÀ KHÔNG có merger gần →
            // highway chỉ nhường ĐÚNG LÚC merger tới, giữ flow khi vắng merger (mainline không chậm
            // constant → goal 3 sạch). Khi merger gần thì coop +0.4 thắng (không vào nhánh phạt này).
            if (action_rl = 2) {
                if (ahead_gap_dx > 15.0) {
                    bool merger_near_yld <- false;
                    if (active_merger_x >= 0.0) {
                        if (move_next_x < active_merger_x) {
                            if ((active_merger_x - move_next_x) < 35.0) { merger_near_yld <- true; }
                        }
                    }
                    if (not merger_near_yld) {
                        reward_cache <- reward_cache - 0.6;   // phanh vô cớ (không merger) → phạt mạnh
                    }
                }
            }
        }
        // TTC penalty: phạt liên tục theo (gap_thiếu × speed) khi gần xe trước. Trigger sớm để
        // agent học "phanh khi gần". Flow mode: nhẹ hơn (gap<12, coef 0.8) để bớt phanh thừa.
        if (enable_highway_ttc_reward) {
            if (enable_highway_flow_reward) {
                // Siết: trigger SỚM hơn (gap<20, was 12) + coef mạnh hơn (1.3, was 0.8) → fast agent
                // bắt đầu hãm từ xa, không chờ tới sát mới phanh (chống rear-end khi chạy nhanh 0.7+).
                if (ahead_gap_dx < 20.0) {
                    if (speed > 0.15) {
                        reward_cache <- reward_cache - ((20.0 - ahead_gap_dx) / 20.0) * speed * 1.3;
                    }
                }
            } else {
                if (ahead_gap_dx < 20.0) {
                    if (speed > 0.15) {
                        reward_cache <- reward_cache - ((20.0 - ahead_gap_dx) / 20.0) * speed * 1.2;
                    }
                }
            }
        }
        // CHÚ Ý: action mapping của highway NGƯỢC với merging — xem reflex behave:
        //   action 0 = accelerate, action 1 = keep, action 2 = decelerate, 3/4 = lane change.
        // Phạt trực tiếp quyết định "accel khi gap quá gần" (TTC chung không đủ specific).
        if (action_rl = 0) {
            if (ahead_gap_dx < 18.0) {
                reward_cache <- reward_cache - 0.8;
            }
        }
        // Không thưởng "phanh khi gần" (đã thử dạng bonus per-tick) — tạo degenerate trap
        // "đứng yên": speed=0 → gap không đổi → +bonus mãi. Dùng TTC penalty (< 0) thay thế.
        // Cooperative reward MARL: thuong khi gan diem merge thanh cong trong TTL `recent_merge_coop_ticks`.
        if (enable_marl_coop_reward) {
            if (rl_agent_id != "") {
                // (a) NHƯỜNG TRƯỚC MERGE: có merger trong accel-lane gần ngay phía trước (highway đứng sau,
                //     cách < 25u). Thưởng GIẢM TỐC (action 2 = mở khe cho merger), phạt TĂNG TỐC vượt qua.
                // GOAL 2 (tăng mạnh để thắng slow-highway equilibrium): merger gần (<35u) → thưởng
                // LỚN khi giảm tốc nhường (action 2), phạt LỚN khi tăng tốc vượt (action 0).
                // GOAL 2 coop reward = ±0.4 (yt9 stable). ĐÃ THỬ 5 biến thể mạnh hơn (±0.7/±1.5/outcome-based/
                // lane-change-yield) — TẤT CẢ rơi vào gridlock attractor (highway all-brake → mainline 0.000 →
                // merger 100% timeout). Bất kỳ coop reward nào ≠ ±0.4 đều diverge shared-policy. Goal 2 (highway
                // nhường chủ động) KHÔNG trị được bằng reward-tuning ở setup này → cần DESIGN DECISION (user).
                // coop ±0.4 (±0.7 hại merger 60%/40% qua shared policy — yt21). Highway passive vì coop hiếm
                // kích (merger nhập giữa IDM, ko gần RL highway) → goal 2 cần POSITIONING, không phải magnitude.
                if (active_merger_x >= 0.0) {
                    if (move_next_x < active_merger_x) {
                        if ((active_merger_x - move_next_x) < 25.0) {
                            if (action_rl = 2) {
                                reward_cache <- reward_cache + 0.4;
                            } else if (action_rl = 0) {
                                reward_cache <- reward_cache - 0.4;
                            }
                        }
                    }
                }
                if (recent_merge_coop_ticks > 0) {
                    if (recent_merge_x >= 0.0) {
                        if (abs(move_next_x - recent_merge_x) < 20.0) {
                            reward_cache <- reward_cache + 0.3;
                        }
                    }
                }
            }
        }
        // GLOBAL/REGIONAL REWARD (MARL4AV local+global — research-backed chống gridlock): thưởng theo
        // FLOW dòng chính (sw_mean = space-mean-speed mainline). Khác local-decel-reward (bị farm →
        // gridlock yt12-15): gridlock ⇒ sw_mean→0 ⇒ MẤT thưởng này ⇒ KHÔNG còn là attractor. Khuyến
        // khích highway giữ flow + hợp tác cho merge (merge thành công ⇒ thêm xe chảy ⇒ flow tăng).
        // CHỈ áp cho highway (không đụng merger đang ổn) — shared net vẫn cân bằng qua reward riêng.
        reward_cache <- reward_cache + global_flow_coef * sw_mean;
        return reward_cache;
    }

    action calculate_merging_reward type: float {
        // M2: dung field `obs_front_gap_raw`, `obs_rear_gap_raw`, `obs_gap_safe` precomputed
        // trong `get_merging_state` (Python query moi tick) + field `ahead_gap_dx` tu loop INLINE
        // collision trong `reflex behave`. KHONG dung target-lane point-loop hay helper co `return` bien cuc bo.
        if (enable_reward_collision_check) {
            if (nearest_collision_dist < collision_distance) {
                return -100.0;
            }
        }
        // Base reward: -0.01/tick (time pressure) + action_penalty + speed * 0.10.
        // Hệ số speed nhỏ để không lấn át safety signals (collision -100, gap penalties).
        reward_cache <- -0.01 + action_penalty + speed * 0.10;
        // INTERVENTION COST (post-merge): KHIÊN IDM-cap ép phanh > RL muốn = RL lái ẩu/ỷ lại → phạt
        // nặng (−5/tick) → RL học TỰ phanh-từ-xa/điều-tốc-mượt trước khi khiên kích hoạt (goal-3 của RL,
        // không phải IDM). Phối hợp: RL = chủ động (tầm xa); khiên = phản xạ cứu phút chót (hiếm khi đạt).
        if (shield_intervened) {
            reward_cache <- reward_cache - 1.0;   // -5 quá nặng → merger freeze (yt41). -1 đủ biết-sai, không sợ đến mức đứng im.
        }
        if (merge_mode = 1) {
            // Trên ramp: shaping signal hướng agent vào accel zone (vùng cuối có thể merge).
            //   - Penalty nhẹ khi speed quá thấp (đứng yên trên ramp).
            //   - Bonus liên tục khi đã trong zone (incentive duy trì vị trí merge).
            //   - One-shot bonus +15 khi VÀO zone (signal dense cho exploration ban đầu).
            if (speed < 0.10) {
                reward_cache <- reward_cache - 0.06;
            }
            if (in_accel_zone) {
                reward_cache <- reward_cache + 0.05;
            }
            if (in_accel_zone) {
                if (not was_in_accel_prev) {
                    reward_cache <- reward_cache + 15.0;
                }
            }
            was_in_accel_prev <- in_accel_zone;
        }
        // Precompute scalar dau action (tranh declare local trong nested if -> TempVariable trong hot path).
        m2_merge_gap_front <- obs_front_gap_raw;
        if (m2_merge_gap_front >= 999.0) {
            m2_merge_gap_front <- ahead_gap_dx;
        }
        m2_accel_len <- max(1.0, accel_end_x - accel_start_x);
        m2_urgency <- min(1.0, max(0.0, (move_next_x - accel_start_x) / m2_accel_len));
        if (enable_merging_gap_reward) {
            // Chỉ thưởng nhẹ khi gap an toàn (không phạt per-tick gap nhỏ — sẽ tạo reward
            // âm tích lũy ngay khi agent đứng yên trên ramp). Safety vẫn được giữ qua:
            //   (1) Observation: gap_front_raw / gap_rear_raw / gap_safe đưa vào policy obs.
            //   (2) Action 3 gate: ``is_merge_gap_safe()`` chặn execute_merge khi unsafe.
            //   (3) Collision penalty -100 (terminal).
            //   (4) Penalty -2 cho action 3 với gap sai (đặt ở dưới).
            if (obs_gap_safe >= 1.0) {
                reward_cache <- reward_cache + 0.05;
            }
        }
        // Urgency penalty: trừ tuyến tính theo vị trí trong accel zone — khuyến khích merge
        // sớm thay vì chờ đến cuối zone (vùng "merge muộn" có ít cơ hội tìm gap an toàn).
        if (in_accel_zone) {
            reward_cache <- reward_cache - m2_urgency * 0.05;
        }
        // OPPORTUNITY-COST penalty (đòn bẩy chính trị merge-muộn) — đang trong zone + gap AN TOÀN
        // nhưng VẪN trên ramp (merge_mode=1, chưa nhập) ⇒ agent BỎ LỠ một gap có thể merge ngay.
        // Phạt tăng dần theo urgency: -1.0 đầu zone → -3.0 cuối zone. Lý do: bonus success +200
        // (behave reflex ~2473) được trao BẤT KỂ merge sớm/muộn hay qua action 3 / auto-fallback
        // cuối ramp (rl_merging_behavior ~3052), nên reward action-3 đơn lẻ KHÔNG có đòn bẩy —
        // policy học "creep tới cuối rồi tự merge" (eval: action 3 = 0%, merge_step cụm ~239).
        // Khi mỗi step ngồi-trên-gap-an-toàn bị tính phí, lấy gap ĐẦU TIÊN (action 3) trở thành
        // advantage dương rõ rệt. Nếu execute_merge chạy trong step này thì in_accel_zone đã <-
        // false (execute_merge ~3079) ⇒ điều kiện dưới tự loại step merge, chỉ phạt step BỎ LỠ.
        if (merge_mode = 1) {
            // ĐÓNG CỬA THOÁT (chỉ khi force_action3_only): phạt MỖI step còn trên ramp chưa merge,
            // KHÔNG điều kiện → "bò chờ timeout" thành rất đắt → ép policy phải bấm action-3 để nhập.
            if (force_action3_only) {
                reward_cache <- reward_cache - 0.5;
            }
            if (in_accel_zone) {
                if (obs_gap_safe >= 1.0) {
                    reward_cache <- reward_cache - (1.0 + m2_urgency * 2.0);
                    // MERGER SPEED-MATCH (PlanB-2, giảm merger collision 27%) — chỉ kích KHI GAP AN
                    // TOÀN (sắp nhập): phạt NHẸ phần vượt tốc so với xe dẫn làn đích. Merger floor
                    // accel (~0.9) lách vào gap mà highway giờ chạy ~0.54 → đâm đuôi xe dẫn lúc lách.
                    // K=0.7 + gate gap_safe. K=1.5 quá mạnh → merger rụt rè → timeout 47%. Hạ xuống
                    // 0.7 để vừa giảm va chạm vừa không làm merger chậm tới mức không nhập kịp.
                    if (speed > obs_front_speed_raw) {
                        reward_cache <- reward_cache - (speed - obs_front_speed_raw) * 0.7;
                    }
                }
            }
        }
        // Action 3 (merge attempt) shaping — tạo sharp gradient hướng policy về quyết định merge:
        //   - Trong zone + gap an toàn: bonus +50 → +25 (scale NGHỊCH theo urgency — merge SỚM
        //     thưởng cao hơn). Đúng thực tế làn tăng tốc: lấy gap an toàn ĐẦU TIÊN, không đi đến
        //     cuối zone (cuối = deadline/gore, gap muộn ít cơ hội hơn). Trước đây dùng
        //     `+ m2_urgency*25` -> thưởng merge MUỘN gấp đôi -> policy học "đi đến cuối mới merge".
        //   - Trong zone + gap không an toàn: -2 (phạt nhẹ, đủ để học "chọn đúng gap" mà
        //     không khiến PPO né tránh hoàn toàn action 3 → policy collapse).
        //   - Ngoài zone: -0.5 (signal "merge chỉ trong zone").
        if (action_rl = 3) {
            if (in_accel_zone) {
                if (obs_gap_safe >= 1.0) {
                    reward_cache <- reward_cache + 25.0 + (1.0 - m2_urgency) * 25.0;
                } else {
                    reward_cache <- reward_cache - 2.0;
                }
            } else {
                reward_cache <- reward_cache - 0.5;
            }
        }
        // Hierarchy phạt non-progress actions trong zone (BRAKE/WAIT nặng nhất do "không tiến").
        // Action 2 (accel) trong zone KHÔNG phạt — giữ tốc độ để bắt kịp traffic mainline.
        if (in_accel_zone) {
            if (action_rl = 0) {  // Brake/Dec
                reward_cache <- reward_cache - 2.0;
            }
            if (action_rl = 1) {  // Keep
                reward_cache <- reward_cache - 0.5;
            }
            if (action_rl = 4) {  // Wait
                reward_cache <- reward_cache - 3.0;
            }
        }
        // Cap [-60, 60] để bonus action 3 (+50) không bị clip mà vẫn ràng buộc magnitude
        // cho value function ổn định (giảm variance gradient).
        if (reward_cache < -60.0) {
            reward_cache <- -60.0;
        }
        if (reward_cache > 60.0) {
            reward_cache <- 60.0;
        }
        return reward_cache;
    }

    aspect default {
        if (location != nil) {
            rgb draw_color <- color;
            point draw_loc <- {location.x, location.y, 0.25};
            if (self = selected_car) {
                draw_color <- #magenta;
            }
            draw rectangle(car_length, car_width) at: draw_loc rotate: heading color: draw_color border: #black;

            float ch <- cos(heading); float sh <- sin(heading);
            point wp <- {location.x + (car_length / 4.0) * ch, location.y + (car_length / 4.0) * sh, 0.26};
            draw rectangle(car_length * 0.25, car_width * 0.8) at: wp rotate: heading color: rgb(30, 30, 30);

            point l1 <- {location.x + (car_length/2)*ch - (car_width/3)*sh, location.y + (car_length/2)*sh + (car_width/3)*ch, 0.27};
            point l2 <- {location.x + (car_length/2)*ch + (car_width/3)*sh, location.y + (car_length/2)*sh - (car_width/3)*ch, 0.27};
            draw circle(0.22) at: l1 color: #yellow;
            draw circle(0.22) at: l2 color: #yellow;
            if (rl_agent_id != "") {
                draw circle(1.25) at: {location.x, location.y, 0.28} color: rgb(255, 0, 255, 70) border: #magenta;
                draw "RL" at: {location.x - 1.3, location.y - 2.2, 0.29} color: #magenta font: font("Arial", 8, #bold);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// TrafficMARLHeadless: MARL 4 agents — pipeline chinh (train_marl, run_experiments).
//   Khởi động: gama-headless.bat -socket 1001
//   Sau đó: python rl/smoke_test_env.py | python rl/run_experiments.py --preset ...
// Baseline Greedy/Random chạy qua Python (rl/baselines.py) trên chính experiment này.
experiment TrafficMARLHeadless type: gui {
    init {
        // Headless socket + GAMA 2025.06.4: parallel species stepping can lose agent scope in `do` statements.
        gama.pref_parallel_species <- false;
        gama.pref_parallel_simulations <- false;
        gama.pref_parallel_grids <- false;
    }

    output {}
}

// ─────────────────────────────────────────────────────────────
experiment TrafficSimulation type: gui {
    parameter "Dashboard policy" var: dashboard_policy category: "RL Demo"
        among: ["Python/SB3 model", "Heuristic"];
    parameter "Show dashboard" var: show_dashboard category: "RL Demo";
    parameter "Max cars (spawn cap)" var: nb_cars_max category: "RL Demo" min: 20 max: 80;

    action select_car {
        point     click_loc    <- #user_location;
        list clicked_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car.location != nil) {
                        if (sqrt((c_car.location.x - click_loc.x) * (c_car.location.x - click_loc.x) + (c_car.location.y - click_loc.y) * (c_car.location.y - click_loc.y)) < 4.0) {
                            clicked_cars << c_car;
                        }
                    }
                }
            }
        }
        selected_car <- nil;
        if (length(clicked_cars) > 0) {
            selected_car <- clicked_cars[0];
        }
    }

    output {
        display MainView type: opengl axes: false background: rgb(46, 139, 87) {
            camera 'default' location: {100.0, 50.0, 155.0} target: {100.0, 50.0, 0.0};

            graphics "Infrastructure" {
                // Nen lane tang toc / gore truoc mesh ramp (chi UI).
                loop g over: ui_road_fill_geoms { if (g != nil) { draw g color: rgb(80, 80, 80); } }
                // Nen ramp (ramp_surface_geoms) + bien/joint da tinh trong init.
                // Khong ve polyline waypoint xam (chi UI tham chieu); xe van bam ramp_waypoints trong behave.
                loop g over: ramp_surface_geoms { if (g != nil) { draw g color: rgb(80, 80, 80); } }
                loop g over: ramp_joint_geoms { if (g != nil) { draw g color: rgb(80, 80, 80); } }
                loop g over: ramp_edge_geoms { if (g != nil) { draw g color: #white width: 1.6; } }
                loop g over: ramp_joint_edge_geoms { if (g != nil) { draw g color: #white width: 1.6; } }

                float top_edge <- offset_y;
                float gore_y   <- offset_y + number_of_lanes * lane_width;
                float outer_y  <- gore_y + lane_width;
                float rescue_lane_y <- offset_y - lane_width;

                draw line([{0.0, top_edge, 0.06}, {road_length, top_edge, 0.06}]) color: #white width: 2.0;
                draw line([{0.0, gore_y, 0.06}, {accel_start_x, gore_y, 0.06}]) color: #white width: 2.0;
                float gore_dash_len <- 2.5;
                float gore_period <- 6.0;
                int gore_k_max <- int(road_length / gore_period) + 2;
                loop k from: 0 to: gore_k_max {
                    float ds <- k * gore_period;
                    float de <- ds + gore_dash_len;
                    if (de > accel_start_x) {
                        if (ds < merge_x) {
                            draw line([
                                {max(ds, accel_start_x), gore_y, 0.06},
                                {min(de, merge_x), gore_y, 0.06}
                            ]) color: #white width: 1.5;
                        }
                    }
                }
                draw line([{merge_x, gore_y, 0.06}, {road_length, gore_y, 0.06}]) color: #white width: 2.0;
                loop g over: accel_outer_dash_geoms { if (g != nil) { draw g color: #white width: 2.0; } }
                // Mép duoi lane tang toc: ngang + cheo gore (khong ve day taper theo tam — gay net X).
                draw line([{accel_start_x, outer_y, 0.07}, {accel_end_x, outer_y, 0.07}]) color: #white width: 2.0;
                draw line([{accel_end_x, outer_y, 0.12}, {merge_x, gore_y, 0.12}]) color: #white width: 2.2;
                loop g over: ui_accel_bottom_edge_geoms { if (g != nil) { draw g color: #white width: 2.0; } }
                draw line([{0.0, rescue_lane_y, 0.06}, {road_length, rescue_lane_y, 0.06}]) color: #yellow width: 2.0;
            }

            species road_segment aspect: default;
            species car          aspect: default;

            event #mouse_down action: select_car;

            overlay position: {20 #px, 20 #px} size: {760 #px, 650 #px}
                    background: rgb(30, 30, 30, 220) border: #white rounded: true {

                if (show_dashboard) {
                    car ego <- nil;
                    loop c over: safe_cars {
                        if (c != nil) {
                            car c_car <- c as car;
                            if (not dead(c_car)) {
                                if (c_car.rl_agent_id = "merging_0") { ego <- c_car; }
                            }
                        }
                    }
                    car tracked <- ego;
                    if (selected_car != nil) {
                        if (not dead(selected_car)) {
                            tracked <- selected_car;
                        }
                    }

                    draw "HIGHWAY MERGING RL DASHBOARD" at: {15 #px, 38 #px}
                        color: #cyan font: font("Arial", 24, #bold);
                    draw ("Policy: " + dashboard_policy + " (model .zip do Python --model)")
                        at: {15 #px, 78 #px} color: #white font: font("Arial", 21, #plain);
                    draw "Action map: 0 Dec | 1 Keep | 2 Acc | 3 Merge | 4 Wait"
                        at: {15 #px, 113 #px} color: #yellow font: font("Arial", 17, #plain);

                    if (tracked = nil) {
                        draw "No RL agent found. Reset the experiment to spawn merging_0."
                            at: {15 #px, 158 #px} color: #red font: font("Arial", 21, #bold);
                    } else {
                        if (dead(tracked)) {
                            draw "No RL agent found. Reset the experiment to spawn merging_0."
                                at: {15 #px, 158 #px} color: #red font: font("Arial", 21, #bold);
                        } else {
                        car lead <- tracked.get_target_lane_ahead();
                        car lag <- tracked.get_target_lane_behind();
                        float front_gap <- 999.0;
                        float rear_gap <- 999.0;
                        if (lead != nil) {
                            if (lead.location != nil) {
                                if (tracked.location != nil) {
                                    front_gap <- max(0.0, lead.location.x - tracked.location.x);
                                }
                            }
                        }
                        if (lag != nil) {
                            if (lag.location != nil) {
                                if (tracked.location != nil) {
                                    rear_gap <- max(0.0, tracked.location.x - lag.location.x);
                                }
                            }
                        }
                        float progress <- 0.0;
                        if (ramp_waypoints != nil) {
                            if (length(ramp_waypoints) > 0) {
                                if (ramp_waypoints[0] != nil) {
                                    float ramp_start_x <- ramp_waypoints[0].x;
                                    float merge_length <- max(1.0, merge_x - ramp_start_x);
                                    if (tracked.location != nil) {
                                        progress <- min(1.0, max(0.0, (tracked.location.x - ramp_start_x) / merge_length));
                                    }
                                }
                            }
                        }
                        float mean_speed <- 0.0;
                        if (tracked.episode_step > 0) {
                            mean_speed <- tracked.speed_sum / tracked.episode_step;
                        }

                        string tracked_title_prefix <- "Tracked: selected car ";
                        rgb tracked_title_rgb <- #white;
                        if (tracked.is_rl_agent) {
                            tracked_title_prefix <- "Tracked: RL agent ";
                            tracked_title_rgb <- #magenta;
                        }

                        rgb outcome_rgb <- #white;
                        if (tracked.merge_success) {
                            outcome_rgb <- #lime;
                        } else {
                            if (tracked.collision_event) {
                                outcome_rgb <- #red;
                            }
                        }

                        rgb reward_line_rgb <- #lime;
                        if (tracked.reward_val < 0) {
                            reward_line_rgb <- #orange;
                        }

                        rgb gap_safe_rgb <- #red;
                        if (tracked.is_merge_gap_safe()) {
                            gap_safe_rgb <- #lime;
                        }

                        draw (tracked_title_prefix + tracked.rl_agent_id)
                            at: {15 #px, 143 #px} color: tracked_title_rgb font: font("Arial", 21, #bold);
                        draw ("Outcome: " + tracked.terminal_reason + " | Step: " + tracked.episode_step)
                            at: {15 #px, 178 #px} color: outcome_rgb font: font("Arial", 21, #plain);
                        draw ("Action: " + tracked.action_rl + " | Reward: " + (tracked.reward_val with_precision 2) + " | CumReward: " + (tracked.cumulative_reward with_precision 2))
                            at: {15 #px, 213 #px} color: reward_line_rgb font: font("Arial", 21, #plain);
                        draw ("Speed: " + (tracked.speed with_precision 2) + " | Mean speed: " + (mean_speed with_precision 2) + " | Progress: " + ((progress * 100.0) with_precision 1) + "%")
                            at: {15 #px, 248 #px} color: #white font: font("Arial", 21, #plain);
                        draw ("Gap front/rear: " + (front_gap with_precision 2) + " / " + (rear_gap with_precision 2) + " | Safe: " + tracked.is_merge_gap_safe())
                            at: {15 #px, 283 #px} color: gap_safe_rgb font: font("Arial", 21, #plain);
                        draw ("Min gap front/rear: " + (tracked.min_front_gap with_precision 2) + " / " + (tracked.min_rear_gap with_precision 2))
                            at: {15 #px, 318 #px} color: #white font: font("Arial", 21, #plain);
                        draw ("Ramp front gap: " + (tracked.get_ramp_front_gap() with_precision 2) + " | In accel zone: " + tracked.in_accel_zone)
                            at: {15 #px, 353 #px} color: #white font: font("Arial", 21, #plain);
                        draw ("Tốc độ dòng chính TB: " + (sw_mean with_precision 2) + " (" + (sw_mean < 0.25 ? "KẸT" : "thông") + ") | Sóng lùi (CV): " + (tracked.info_shockwave_index with_precision 3))
                            at: {15 #px, 388 #px} color: (sw_mean < 0.4 ? rgb(255,140,0) : #white) font: font("Arial", 21, #plain);
                        }
                    }

                    int mainline_count <- 0;
                    int ramp_count <- 0;
                    int rl_count <- 0;
                    int rule_based_merging_count <- 0;
                    int damaged_count <- 0;
                    loop c over: safe_cars {
                        if (c != nil) {
                            car c_car <- c as car;
                            if (c_car != nil) {
                                if (not dead(c_car)) {
                                    if (c_car.is_on_ramp_merge_path()) {
                                        ramp_count <- ramp_count + 1;
                                        if (not c_car.is_rl_agent) { rule_based_merging_count <- rule_based_merging_count + 1; }
                                    } else {
                                        mainline_count <- mainline_count + 1;
                                    }
                                    if (c_car.rl_agent_id != "") { rl_count <- rl_count + 1; }
                                    if (c_car.is_damaged) { damaged_count <- damaged_count + 1; }
                                }
                            }
                        }
                    }
                    draw ("Agents: RL=" + rl_count + " ramp NPC=" + rule_based_merging_count + " mainline env=" + mainline_count)
                        at: {15 #px, 438 #px} color: #white font: font("Arial", 21, #plain);
                    draw ("Traffic: total=" + length(safe_cars) + " ramp=" + ramp_count + " damaged=" + damaged_count)
                        at: {15 #px, 473 #px} color: #white font: font("Arial", 21, #plain);
                    draw ("Cycle: " + cycle + " | Slow demo: GAMA speed slider or Python --step-delay")
                        at: {15 #px, 508 #px} color: #white font: font("Arial", 21, #plain);
                    draw "SB3 .zip models are loaded by Python; this GUI labels/observes the selected run." at: {15 #px, 550 #px} color: #cyan font: font("Arial", 17, #italic);
                    draw "Legend: RL=learning agent | unlabelled=traffic | red=breakdown (tow to yellow lane)" at: {15 #px, 585 #px} color: #cyan font: font("Arial", 17, #italic);
                }
            }
        }
    }
}
