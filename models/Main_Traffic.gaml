// MARL-only (repo chinh): 4 RL agents merging_0 + highway_0/1/2, experiment TrafficMARLHeadless.
// Single-agent archive (baseline/legacy): models/single_agent/Main_Traffic_SingleAgent.gaml + TrafficSingleHeadless.
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
    int   nb_cars_max          <- 36;   // GUI mac dinh nhe hon 45 de giam gridlock; headless co the tang qua tham so / Python
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

    // Chi so waypoint dang huong toi diem ket thuc doan lane tang toc ngang (tru diem merge cuoi).
    // Tranh hard-code `4` vi length ramp_waypoints co the doi; invariant: 2 diem cuoi = {accel_end} -> {merge}.
    int ramp_accel_waypoint_ix <- 4;

    float bottom_lane_y;
    float accel_lane_y;
    float accel_start_x <- 48.0;
    float accel_end_x   <- 162.0;
    float merge_x       <- 180.0;

    float gap_front_max <- 7.0;
    float gap_front_min <- 2.5;
    float gap_rear_max  <- 6.0;
    float gap_rear_min  <- 2.0;

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
    // dashboard_algorithm chỉ là label hiển thị trong GUI — đổi thành algo đang demo.
    string dashboard_algorithm <- "PPO";
    string dashboard_model_version <- "none";
    int    manual_action <- 1;
    bool   show_dashboard <- true;
    int    demo_step_delay_ms <- 0;

    // Legacy restore flags: bat tung nhom logic cu co kiem soat, tranh bat dong loat gay NPE/timeout headless.
    bool enable_rl_respawn <- true;
    bool enable_highway_rl_respawn <- true;
    bool enable_balance_traffic <- true;
    bool enable_collision_neighbor_scan <- true;
    // Mesh ramp (polygon + joint + vach giua): luon tinh mot lan trong global init — khong dung co bat/tat.
    bool enable_demo_slowdown <- false;
    bool enable_reward_collision_check <- true;
    bool enable_highway_ttc_reward <- true;
    bool enable_merging_gap_reward <- true;
    bool enable_marl_coop_reward <- false;

    // Throughput counters: đếm tổng số xe ramp đã cố merge và số lần merge thành công.
    int total_ramp_attempts <- 0;   // Tăng khi xe ramp vào acceleration zone
    int total_merge_success <- 0;   // Tăng khi bất kỳ xe nào merge thành công

    // Cooperative reward: số cycle còn lại để highway RL nhận bonus (TTL), tránh phụ thuộc thứ tự reflex bool + reset.
    // Mỗi đầu cycle global reflex `apply_pz_python_each_cycle` giảm TTL — merge set TTL=3 → bao phủ ~2–3 cycle sau khi merge.
    int    recent_merge_coop_ticks <- 0;
    float  recent_merge_x          <- -1.0;   // Vị trí x merge gần nhất

    // Shockwave index: đo độ bất ổn tốc độ xe mainline ở vùng merge.
    // Dùng Welford's online algorithm để tính mean và variance không cần lưu toàn bộ mẫu.
    // shockwave_index = std_dev / mean_speed (Coefficient of Variation) — thấp = êm, cao = sóng lùi.
    int    sw_n      <- 0;      // Số mẫu đã thu thập
    float  sw_mean   <- 0.0;   // Running mean (Welford)
    float  sw_M2     <- 0.0;   // Running sum of squared deviations (Welford)

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
    
    reflex update_safe_cars {
        // GAMA 2025.06.4 co the NPE AddStatement khi prune bang list cuc bo + `<<`
        // trong global reflex. Giu cache non-nil; cac diem dung safe_cars da guard nil/dead.
        if (safe_cars = nil) {
            safe_cars <- [];
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
    }

    // PettingZoo: nap `pz_actions` -> `action_rl`. Chi dung reflex + field global `pz_action_target` — KHONG `do action(...)`
    // (GAMA 2025.6.4: DoStatement.getContext NPE khi getAgent() null trong reflex architecture).
    reflex apply_pz_python_each_cycle when: every(1 #cycles) {
        if (recent_merge_coop_ticks > 0) {
            recent_merge_coop_ticks <- recent_merge_coop_ticks - 1;
        }
        if (pz_actions != nil) {
            loop c over: safe_cars {
                pz_action_target <- nil;
                pz_apply_aid <- "";
                if (c != nil) {
                    pz_action_target <- c as car;
                    if (pz_action_target != nil) {
                        if (not dead(pz_action_target)) {
                            pz_apply_aid <- pz_action_target.rl_agent_id;
                            if (pz_apply_aid != "") {
                                if (pz_actions contains_key pz_apply_aid) {
                                    pz_action_target.action_rl <- int(pz_actions at pz_apply_aid);
                                }
                            }
                        }
                    }
                }
                pz_action_target <- nil;
                pz_apply_aid <- "";
            }
        }
    }

    action pz_default_info(string outcome) type: map {
        return ["outcome"::outcome, "success"::false, "collision"::false, "failed_merge"::false];
    }

    action pz_find_rl_car(string aid) type: car {
        if (aid = nil) { return nil; }
        if (safe_cars = nil) { return nil; }
        loop c over: safe_cars {
            if (c != nil) {
                if ((c as car) != nil) {
                    if (not dead(c as agent)) {
                        if ((c as car).rl_agent_id = aid) {
                            return c as car;
                        }
                    }
                }
            }
        }
        return nil;
    }

    action pz_put_agent_payload(string aid) {
        if (aid = nil) { return; }
        if (pz_agents = nil) { pz_agents <- []; }
        if (pz_observations = nil) { pz_observations <- []; }
        if (pz_data = nil) { pz_data <- []; }
        car rc <- pz_find_rl_car(aid);
        if (rc = nil) {
            pz_data <- [
                "Observations"::pz_observations,
                "Rewards"::[],
                "Terminations"::[aid::true],
                "Truncations"::[aid::false],
                "Infos"::[aid::pz_default_info("missing_agent")]
            ];
            return;
        }
    }

    action pz_sync_from_world {
        map pz_rewards <- [
            "merging_0"::0.0,
            "highway_0"::0.0,
            "highway_1"::0.0,
            "highway_2"::0.0
        ];
        map pz_terminations <- [
            "merging_0"::false,
            "highway_0"::false,
            "highway_1"::false,
            "highway_2"::false
        ];
        map pz_truncations <- [
            "merging_0"::false,
            "highway_0"::false,
            "highway_1"::false,
            "highway_2"::false
        ];
        pz_infos <- [
            "merging_0"::pz_default_info("running"),
            "highway_0"::pz_default_info("running"),
            "highway_1"::pz_default_info("running"),
            "highway_2"::pz_default_info("running")
        ];
        pz_observations <- [
            "merging_0"::list_with(15, 0.0),
            "highway_0"::list_with(15, 0.0),
            "highway_1"::list_with(15, 0.0),
            "highway_2"::list_with(15, 0.0)
        ];
        pz_agents <- [];

        car rc0 <- pz_find_rl_car("merging_0");
        if (rc0 != nil) {
            pz_rewards << "merging_0"::rc0.reward_val;
            pz_terminations << "merging_0"::rc0.is_done;
            pz_observations << "merging_0"::rc0.get_merging_state();
            pz_infos << "merging_0"::rc0.get_episode_info();
            if (not rc0.is_done) { pz_agents << "merging_0"; }
        }

        car rc1 <- pz_find_rl_car("highway_0");
        if (rc1 != nil) {
            pz_rewards << "highway_0"::rc1.reward_val;
            pz_terminations << "highway_0"::rc1.is_done;
            pz_observations << "highway_0"::rc1.get_current_state();
            pz_infos << "highway_0"::rc1.get_episode_info();
            if (not rc1.is_done) { pz_agents << "highway_0"; }
        }

        car rc2 <- pz_find_rl_car("highway_1");
        if (rc2 != nil) {
            pz_rewards << "highway_1"::rc2.reward_val;
            pz_terminations << "highway_1"::rc2.is_done;
            pz_observations << "highway_1"::rc2.get_current_state();
            pz_infos << "highway_1"::rc2.get_episode_info();
            if (not rc2.is_done) { pz_agents << "highway_1"; }
        }

        car rc3 <- pz_find_rl_car("highway_2");
        if (rc3 != nil) {
            pz_rewards << "highway_2"::rc3.reward_val;
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

        // Quy dao ramp: bat dau sat phia duoi lane tang toc (cung he toa do voi mainline),
        // khong dung diem "xa" y=78 (gay UI nhu xe chay tren co tach khoi cao toc).
        float ramp_dy_drop <- accel_lane_y - bottom_lane_y;
        // Them 2 diem noi suon giua accel_end va merge de giam goc dot ngot (it hon "vuong goc" nhap lan).
        ramp_waypoints <- [
            {12.0, accel_lane_y + 12.0},
            {22.0, accel_lane_y + 7.0},
            {34.0, accel_lane_y + 3.5},
            {accel_start_x, accel_lane_y},
            {accel_end_x,   accel_lane_y},
            {accel_end_x + 8.0, accel_lane_y - ramp_dy_drop * 0.30},
            {merge_x - 12.0, bottom_lane_y + ramp_dy_drop * 0.18},
            {merge_x, bottom_lane_y}
        ];

        ramp_accel_waypoint_ix <- 0;
        loop wi from: 0 to: length(ramp_waypoints) - 1 {
            if (ramp_waypoints[wi] != nil) {
                if (abs(ramp_waypoints[wi].x - accel_end_x) < 0.02) {
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
        ramp_wp0_y <- accel_lane_y + 12.0;
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
                                float half_w <- lane_width / 2.0;
                                float nx <- -dy / seg_len;
                                float ny <- dx / seg_len;
                                point a <- {p0.x + nx * half_w, p0.y + ny * half_w, 0.01};
                                point b <- {p1.x + nx * half_w, p1.y + ny * half_w, 0.01};
                                point c <- {p1.x - nx * half_w, p1.y - ny * half_w, 0.01};
                                point d <- {p0.x - nx * half_w, p0.y - ny * half_w, 0.01};
                                if (polygon([a, b, c, d]) != nil) {
                                    ramp_surface_geoms << polygon([a, b, c, d]);
                                }
                                float e1y_mid <- (a.y + b.y) / 2.0;
                                float e2y_mid <- (d.y + c.y) / 2.0;
                                float e1x_mid <- (a.x + b.x) / 2.0;
                                float e2x_mid <- (d.x + c.x) / 2.0;
                                // Chan cac edge "loi" len lane cao toc o doan taper (gay net cheo khong hop ly).
                                // Tranh bien bool tam trong loop phuc tap (co the gay TempVariableExpression NPE o GAMA 2025.06.4).
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
                                            // Dung 2 tam giac de tranh tu-cat quad tai khuc ngoat manh (gay den/nhap nhay).
                                            if (polygon([L0, L1, pj3]) != nil) {
                                                ramp_joint_geoms << polygon([L0, L1, pj3]);
                                            }
                                            if (polygon([R0, R1, pj3]) != nil) {
                                                ramp_joint_geoms << polygon([R0, R1, pj3]);
                                            }
                                            float lenL <- sqrt((L1.x - L0.x) * (L1.x - L0.x) + (L1.y - L0.y) * (L1.y - L0.y));
                                            float lenR <- sqrt((R1.x - R0.x) * (R1.x - R0.x) + (R1.y - R0.y) * (R1.y - R0.y));
                                            // Bien tren: uu tien noi lien de khong bi dut.
                                            if (lenL < lane_width * 1.25) {
                                                if (line([{L0.x, L0.y, 0.04}, {L1.x, L1.y, 0.04}]) != nil) {
                                                    ramp_joint_edge_geoms << line([{L0.x, L0.y, 0.04}, {L1.x, L1.y, 0.04}]);
                                                }
                                            }
                                            // Bien duoi: loc chat hon o vung taper de tranh net thua.
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
        // Khong sync bridge trong init de tranh context local chua on dinh (NPE this.local).
    }

    // Trì hoãn create car sang reflex đầu tiên để chắc chắn simulation scope đã sẵn population.
    reflex bootstrap_initial_cars when: ((not initial_cars_created) and (cycle > 0)) {
        // --- 1. KHỞI TẠO XE TRÊN CAO TỐC CHÍNH ---
        int cars_per_lane_boot <- int(nb_cars_max / number_of_lanes);
        float spacing_boot <- road_length / (cars_per_lane_boot + 1);
        loop i from: 0 to: number_of_lanes - 1 {
            loop j from: 0 to: cars_per_lane_boot - 1 {
                create car number: 1 {
                    speed              <- rnd(0.4, speed_max);
                    color              <- one_of([#crimson, #dodgerblue, #silver, #white, #gold, #darkviolet]);
                    if (flip(0.05)) {
                        speed <- 0.0;
                        color <- #red;
                    }
                    is_merging         <- false;
                    in_accel_zone      <- false;
                    current_lane_index <- i;
                    target_lane_index  <- i;
                    location           <- {10.0 + j * spacing_boot + rnd(-3.0, 3.0), offset_y + (i * lane_width) + (lane_width / 2.0)};
                    move_next_x        <- location.x;
                    move_next_y        <- location.y;
                    direction          <- 1;
                    heading            <- 0.0;
                    is_rl_agent        <- false;
                    rl_agent_id        <- "";
                }
            }
        }

        // --- 2. KHỞI TẠO XE TRÊN NHÁNH NHẬP LÀN ---
        int nb_ramp_cars_boot <- 3;
        if (length(ramp_waypoints) >= 2) {
            point wp0_boot <- ramp_waypoints[0];
            point wp1_boot <- ramp_waypoints[1];
            if (wp0_boot != nil) {
                if (wp1_boot != nil) {
                    loop k from: 0 to: nb_ramp_cars_boot - 1 {
                        create car number: 1 {
                            is_merging         <- true;
                            merge_mode         <- 1;
                            in_accel_zone      <- false;
                            waypoint_index     <- 1;
                            float ratio <- (k * 1.0) / nb_ramp_cars_boot;
                            float lx <- wp0_boot.x + ratio * (wp1_boot.x - wp0_boot.x);
                            float ly <- wp0_boot.y + ratio * (wp1_boot.y - wp0_boot.y);
                            location <- {lx, ly};
                            move_next_x <- lx;
                            move_next_y <- ly;
                            heading            <- atan2(wp1_boot.y - ly, wp1_boot.x - lx);
                            speed              <- rnd(0.2, 0.4);
                            color              <- #orange;
                            direction          <- 1;
                            current_lane_index <- number_of_lanes - 1;
                            target_lane_index  <- number_of_lanes - 1;
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

        // --- 3. KHỞI TẠO 3 XE RL TRÊN LÀN DƯỚI CÙNG ---
        list<string> hw_ids_boot <- ["highway_0", "highway_1", "highway_2"];
        float bottom_y_boot <- offset_y + (number_of_lanes - 1) * lane_width + lane_width / 2.0;
        loop hw_k from: 0 to: 2 {
            float hw_x_boot <- road_length * (0.25 + hw_k * 0.25);
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
                speed              <- rnd(0.5, speed_max);
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
    int   spawn_ramp_cap   <- 5;        // Cap toi da xe ramp active cung luc

    // Reflex 1: counter tick + set flag due moi 20 tick. Khong dung loop, khong dung binary phuc tap.
    // Phase 6.2: BAT LAI (bo `when: false`) sau khi them cleanup (`die_if_out_of_road` +
    // `purge_safe_cars`). N tu day duoc bo: bootstrap 16 + ramp_cap 5 = max 21 xe.
    reflex spawn_ramp_tickup {
        spawn_ramp_tick <- spawn_ramp_tick + 1;
        spawn_ramp_due <- false;
        if (spawn_ramp_tick >= 20) {
            spawn_ramp_tick <- 0;
            if (ramp_wp0_ready) {
                spawn_ramp_due <- true;
            }
        }
    }

    // Reflex 2: spawn khi `spawn_ramp_due`, co cho an toan quanh wp0, va chua het cap.
    // Phase 6.2: them count `spawn_ramp_count` trong cung loop -> khong cost O(N) them.
    reflex spawn_ramp_cars when: spawn_ramp_due {
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
    }

    // Tránh empty(car where ...) và empty(nil): một số bản GAMA trả nil / không rút gọn or → UnaryOperator.empty NPE.
    reflex respawn_rl_merging_agent when: enable_rl_respawn {
        if (not initial_cars_created) { return; }
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
                            recent_merge_coop_ticks  <- 0;
                            recent_merge_x           <- -1.0;
                            create car number: 1 {
                                is_merging         <- true;
                                merge_mode         <- 1;
                                in_accel_zone      <- false;
                                waypoint_index     <- 1;
                                location           <- {wp0.x, wp0.y};
                                move_next_x        <- wp0.x;
                                move_next_y        <- wp0.y;
                                heading            <- 0.0;
                                if (length(ramp_waypoints) > 1) {
                                    if (ramp_waypoints[1] != nil) {
                                        point w1r <- ramp_waypoints[1];
                                        heading   <- atan2(w1r.y - wp0.y, w1r.x - wp0.x);
                                    }
                                }
                                speed              <- rnd(0.2, 0.4);
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
    }

    // MARL: Tái tạo xe highway RL. every(3) thay vì every(1) giảm overhead mỗi cycle.
    // 3 cycles trễ tái tạo là không đáng kể so với episode dài 300 steps.
    reflex respawn_highway_rl_agents when: enable_highway_rl_respawn {
        if (not initial_cars_created) { return; }
        list<string> hw_ids <- ["highway_0", "highway_1", "highway_2"];
        float bottom_y <- offset_y + (number_of_lanes - 1) * lane_width + lane_width / 2.0;
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
                                        if (c_car.current_lane_index = number_of_lanes - 1) {
                                            if (not c_car.is_merging) {
                                                if (c_car.location.x < 8.0) {
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
                if (is_safe) {
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
                        speed              <- 0.58;
                        color              <- #cyan;
                        is_rl_agent        <- true;
                        rl_agent_id        <- hid;
                    }
                    // Khong goi sync bridge o day de tranh local context null.
                }
            }
        }
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
        if (safe_cars != nil) {
            loop c over: safe_cars {
                if (c != nil) {
                    car c_car <- c as car;
                    if (c_car != nil) {
                        if (not dead(c_car)) {
                            if (c_car.merge_mode != 1) {
                                if (not c_car.is_merging_transition) {
                                    if (c_car.current_lane_index = number_of_lanes - 1) {
                                        if (c_car.move_next_x >= accel_start_x - 20.0) {
                                            if (c_car.move_next_x <= merge_x + 30.0) {
                                                sw_n <- sw_n + 1;
                                                sw_delta_cache <- c_car.speed - sw_mean;
                                                sw_mean <- sw_mean + sw_delta_cache / sw_n;
                                                sw_delta2_cache <- c_car.speed - sw_mean;
                                                sw_M2 <- sw_M2 + sw_delta_cache * sw_delta2_cache;
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

    // slow_down_demo: GAMA không có API pause(ms). Để làm chậm demo GUI,
    // giảm tốc độ simulation thủ công qua menu Simulation > Speed hoặc tắt continuous.
    // reflex này bị vô hiệu hóa để tránh lỗi runtime khi demo_step_delay_ms > 0.
    reflex slow_down_demo when: enable_demo_slowdown {}

    int  balance_tick <- 0;
    bool balance_due <- false;

    reflex balance_traffic_tickup {
        balance_due <- false;
        if (enable_balance_traffic) {
            balance_tick <- balance_tick + 1;
            if (balance_tick >= 20) {
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
                    color  <- one_of([#crimson, #dodgerblue, #silver, #white, #gold, #darkviolet]);
                    if (flip(0.05)) {
                        speed <- rnd(0.1, 0.25);
                        color <- #red;
                    }
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
                    // 5% xe chướng ngại vật khi khởi tạo — nhất quán với balance_traffic (cũng flip(0.05)).
                    speed              <- rnd(0.4, speed_max);
                    color              <- one_of([#crimson, #dodgerblue, #silver, #white, #gold, #darkviolet]);
                    if (flip(0.05)) {
                        speed <- 0.0;
                        color <- #red;
                    }
                    
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

    action update_data {
        if (observations = nil) { observations <- []; }
        if (rewards = nil) { rewards <- []; }
        if (terminations = nil) { terminations <- []; }
        if (truncations = nil) { truncations <- []; }
        if (infos = nil) { infos <- []; }
        data <- [
            "Observations"::observations,
            "Rewards"::rewards,
            "Terminations"::terminations,
            "Truncations"::truncations,
            "Infos"::infos
        ];
        pz_agents <- agents;
        pz_possible_agents <- possible_agents;
        pz_observation_spaces <- observation_spaces;
        pz_action_spaces <- action_spaces;
        pz_observations <- observations;
        pz_infos <- infos;
        pz_data <- data;
    }

    action default_info(string outcome) type: map {
        return ["outcome"::outcome, "success"::false, "collision"::false, "failed_merge"::false];
    }

    action find_rl_car(string aid) type: car {
        if (aid = nil) { return nil; }
        if (safe_cars = nil) { return nil; }
        loop c over: safe_cars {
            if (c != nil) {
                if ((c as car) != nil) {
                    if (not dead(c as agent)) {
                        if ((c as car).rl_agent_id = aid) {
                            return c as car;
                        }
                    }
                }
            }
        }
        return nil;
    }

    action put_agent_payload(string aid) {
        if (aid = nil) { return; }
        if (agents = nil) { agents <- []; }
        if (observations = nil) { observations <- []; }
        if (rewards = nil) { rewards <- []; }
        if (terminations = nil) { terminations <- []; }
        if (truncations = nil) { truncations <- []; }
        if (infos = nil) { infos <- []; }
        if (data = nil) { data <- []; }
        car rc <- find_rl_car(aid);
        if (rc = nil) {
            terminations << aid::true;
            infos << aid::default_info("missing_agent");
            return;
        }
        rewards << aid::rc.reward_val;
        terminations << aid::rc.is_done;
        if (not rc.is_done) {
            agents << aid;
        }
    }

    action sync_from_world {
        if (agents = nil) { agents <- []; }
        if (possible_agents = nil) { possible_agents <- []; }
        if (observations = nil) { observations <- []; }
        if (rewards = nil) { rewards <- []; }
        if (terminations = nil) { terminations <- []; }
        if (truncations = nil) { truncations <- []; }
        if (infos = nil) { infos <- []; }
        observations <- [];
        rewards <- [];
        terminations <- [];
        truncations <- [];
        infos <- [];
        agents <- [];
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
            "merging_0"::default_info("running"),
            "highway_0"::default_info("running"),
            "highway_1"::default_info("running"),
            "highway_2"::default_info("running")
        ];
        do put_agent_payload("merging_0");
        do put_agent_payload("highway_0");
        do put_agent_payload("highway_1");
        do put_agent_payload("highway_2");
        do update_data;
    }

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
                if ((c as car) != nil) {
                    if (not dead(c as agent)) {
                        if ((c as car).rl_agent_id = aid) {
                            return c as car;
                        }
                    }
                }
            }
        }
        return nil;
    }

    action tick_sync_from_world {
        map pz_rewards <- [
            "merging_0"::0.0,
            "highway_0"::0.0,
            "highway_1"::0.0,
            "highway_2"::0.0
        ];
        map pz_terminations <- [
            "merging_0"::false,
            "highway_0"::false,
            "highway_1"::false,
            "highway_2"::false
        ];
        map pz_truncations <- [
            "merging_0"::false,
            "highway_0"::false,
            "highway_1"::false,
            "highway_2"::false
        ];
        pz_infos <- [
            "merging_0"::tick_default_info("running"),
            "highway_0"::tick_default_info("running"),
            "highway_1"::tick_default_info("running"),
            "highway_2"::tick_default_info("running")
        ];
        pz_observations <- [
            "merging_0"::list_with(15, 0.0),
            "highway_0"::list_with(15, 0.0),
            "highway_1"::list_with(15, 0.0),
            "highway_2"::list_with(15, 0.0)
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

        car rc1 <- tick_find_rl_car("highway_0");
        if (rc1 != nil) {
            pz_rewards << "highway_0"::rc1.reward_val;
            pz_terminations << "highway_0"::rc1.is_done;
            pz_observations << "highway_0"::rc1.get_current_state();
            pz_infos << "highway_0"::rc1.get_episode_info();
            if (not rc1.is_done) { pz_agents << "highway_0"; }
        }

        car rc2 <- tick_find_rl_car("highway_1");
        if (rc2 != nil) {
            pz_rewards << "highway_1"::rc2.reward_val;
            pz_terminations << "highway_1"::rc2.is_done;
            pz_observations << "highway_1"::rc2.get_current_state();
            pz_infos << "highway_1"::rc2.get_episode_info();
            if (not rc2.is_done) { pz_agents << "highway_1"; }
        }

        car rc3 <- tick_find_rl_car("highway_2");
        if (rc3 != nil) {
            pz_rewards << "highway_2"::rc3.reward_val;
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

    reflex push_outputs when: every(1 #cycles) {
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
    int   waypoint_index  <- 0;
    bool  is_rl_agent     <- false;
    string rl_agent_id     <- "";

    float speed   <- 0.0;   // thủ công thay MovingSkill
    float heading <- 0.0;   // thủ công thay MovingSkill

    float car_length    <- 3.5;
    float car_width     <- 1.7;
    int   patience      <- 100;
    int   stuck_counter <- 0;

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
    bool        collision_event <- false;
    bool        failed_merge <- false;
	
	bool is_damaged <- false;
    int  damaged_timer <- 0;
    int  max_damaged_duration <- 150; // Thời gian xe đứng im sửa chữa
    // Xác suất xe “hỏng” ngẫu nhiên mỗi cycle (chỉ xe mainline trong highway_rl_move). 0.001 ≈ điều kiện động thực tế;
    // làm tăng phương sai shockwave/episode — báo cáo nên nhấn mạnh nhiều seed. Đặt 0.0 nếu muốn baseline tĩnh.
    float prob_damaged_car <- 0.001;

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

    reflex behave {
        // Headless batch ổn định: không gọi ramp_behavior/highway_rl_move vì các action đó
        // chứa nhiều point-loop dễ NPE trong GAMA 2025.06.4 khi chạy socket dài.
        if (terminal_reason != "running") {
            speed <- 0.0;
            dead_timer <- dead_timer + 1;
            // Sau episode terminal, xe van ton tai trong population -> respawn_rl_merging_agent
            // khong bao gio tao merging_0 moi (slot luon "bi chiem"). Huy agent sau vai tick de
            // Python/doc GUI doc reward/termination roi respawn (purge chi loai dead).
            if (dead_timer > 48) {
                do die;
            }
            return;
        }

        dead_timer <- 0;

        merge_rl_accel_gate <- false;

        prev_move_tick_x <- move_next_x;
        prev_move_tick_y <- move_next_y;

        // Penalty theo action chi ap dung cho tick hien tai, khong tich luy qua episode.
        action_penalty <- 0.0;

        // Dashboard / socket: `rl_merging_behavior` khong con goi trong behave — gan action_rl o day.
        // Python/SB3: global reflex `apply_pz_python_each_cycle` da nap tu `pz_actions`; neu thieu key thi fallback keep.
        if (dashboard_policy != "Python/SB3 model") {
            if (rl_agent_id != "") {
                if (rl_agent_id = "merging_0") {
                    if (dashboard_policy = "Manual") {
                        action_rl <- manual_action;
                    } else if (dashboard_policy = "Random") {
                        // Uniform rnd(0,4) -> ~20% phanh/tick + shield -> shockwave / gridlock khi demo.
                        int rw_m <- rnd(0, 99);
                        if (rw_m < 7) {
                            action_rl <- 0;
                        } else if (rw_m < 62) {
                            action_rl <- 1;
                        } else if (rw_m < 82) {
                            action_rl <- 2;
                        } else if (rw_m < 91) {
                            action_rl <- 3;
                        } else {
                            action_rl <- 4;
                        }
                    } else if (dashboard_policy = "Heuristic") {
                        action_rl <- get_heuristic_merging_action();
                    }
                } else {
                    if (dashboard_policy = "Manual") {
                        action_rl <- manual_action;
                    } else if (dashboard_policy = "Random") {
                        int rw_h <- rnd(0, 99);
                        if (rw_h < 8) {
                            action_rl <- 0;
                        } else if (rw_h < 70) {
                            action_rl <- 1;
                        } else if (rw_h < 85) {
                            action_rl <- 2;
                        } else if (rw_h < 93) {
                            action_rl <- 3;
                        } else {
                            action_rl <- 4;
                        }
                    } else if (dashboard_policy = "Heuristic") {
                        action_rl <- get_heuristic_highway_action();
                    }
                }
            }
        } else {
            if (rl_agent_id != "") {
                if (action_rl < 0) {
                    action_rl <- 1;
                }
            }
        }

        // merging_0: 0 giam, 2 tang. highway_* (PettingZoo / heuristic / highway_rl_move): 0 tang, 2 giam.
        if (rl_agent_id = "merging_0") {
            if (action_rl = 0) {
                speed <- max(speed_min, speed - deceleration);
            } else if (action_rl = 2) {
                speed <- min(speed_max, speed + acceleration);
            } else {
                speed <- min(speed_max, max(speed_min, speed));
            }
        } else if (rl_agent_id != "") {
            if (action_rl = 0) {
                speed <- min(speed_max, speed + acceleration);
            } else if (action_rl = 2) {
                speed <- max(speed_min, speed - deceleration);
            } else {
                speed <- min(speed_max, max(speed_min, speed));
            }
        } else {
            speed <- min(speed_max, max(speed_min, speed));
        }

        // merging_0 tren ramp: action 4 = cho (giam toc nhe) — truoc day behave khong xu ly 4 giong rl_merging_behavior.
        if (rl_agent_id = "merging_0") {
            if (merge_mode = 1) {
                if (action_rl = 4) {
                    speed <- max(speed_min, speed - deceleration * 0.28);
                }
                // Random/keep co the ha speed ve 0 nhieu tick -> ket tren ramp; san nho khi van episode running.
                if (terminal_reason = "running") {
                    speed <- max(0.085, min(speed_max, speed));
                }
            }
        }

        // M3: Lane change cho highway agents (rl_agent_id != "merging_0" va != "").
        // action 3 = lane left (giam lane_index), 4 = lane right (tang). Hot path INLINE,
        // khong goi `do execute_lane_change()` de tranh DoStatement.getContext NPE.
        if (rl_agent_id != "") {
            if (rl_agent_id != "merging_0") {
                if (action_rl = 3) {
                    if (current_lane_index > 0) {
                        target_lane_index <- current_lane_index - 1;
                        current_lane_index <- target_lane_index;
                        action_penalty <- action_penalty - 0.05;
                    } else {
                        action_penalty <- action_penalty - 2.0;
                    }
                } else if (action_rl = 4) {
                    if (current_lane_index < number_of_lanes - 1) {
                        target_lane_index <- current_lane_index + 1;
                        current_lane_index <- target_lane_index;
                        action_penalty <- action_penalty - 0.05;
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
        if (merge_mode != 1) {
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

        // Sau shield: neu van gan dung im nhung van con khoang dx an toan, ep lun de tranh deadlock giua duong (dac biet GUI Random).
        if (merge_mode != 1) {
            if (terminal_reason = "running") {
                if (speed < 0.035) {
                    // +0.2 qua lon so voi khe OVM thuc te (~car_length) -> hang xe khong toi nguong die o cuoi mainline.
                    if (ahead_gap_dx > collision_distance + 0.02) {
                        if (ahead_gap_dx < 25.0) {
                            speed <- 0.12;
                        }
                    }
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

        // Movement: xe ramp (merge_mode=1) KHONG duoc chi +move_next_x — can polyline / ramp_behavior
        // (tranh RB/RL chi chay ngang tren nen, ramp_behavior khong con duoc goi o day).
        if (merge_mode = 1) {
            if (rl_agent_id = "") {
                location <- {move_next_x, move_next_y};
                do ramp_behavior();
                move_next_x <- min(road_move_clip_x, max(0.0, location.x));
                move_next_y <- location.y;
            } else {
                if (waypoint_index = ramp_accel_waypoint_ix) {
                    merge_rl_accel_gate <- true;
                }
                list<float> merging_obs_tmp <- get_merging_state();
                if (ramp_waypoints != nil) {
                    if (waypoint_index < length(ramp_waypoints)) {
                        point tp_rl <- ramp_waypoints[waypoint_index];
                        if (tp_rl != nil) {
                            float rh <- atan2(tp_rl.y - move_next_y, tp_rl.x - move_next_x);
                            float nmx <- move_next_x + speed * cos(rh);
                            float nmy <- move_next_y + speed * sin(rh);
                            move_next_x <- min(road_move_clip_x, max(0.0, nmx));
                            move_next_y <- nmy;
                            heading <- rh;
                            point here_rl <- {move_next_x, move_next_y};
                            if (euclid_point_dist(here_rl, tp_rl) < max(speed * 2.0, 1.0)) {
                                waypoint_index <- waypoint_index + 1;
                            }
                        }
                    }
                }
                in_accel_zone <- (waypoint_index = ramp_accel_waypoint_ix);
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
                    m3_lane_blend <- 0.2;
                    if (is_merging_transition) {
                        m3_lane_blend <- 0.38;
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
        }
        location <- {move_next_x, move_next_y};

        if (merge_mode != 1) {
            float ddx_h <- move_next_x - prev_move_tick_x;
            float ddy_h <- move_next_y - prev_move_tick_y;
            if (abs(ddx_h) > 0.001) {
                heading <- atan2(ddy_h, ddx_h);
            } else {
                heading <- 0.0;
            }
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

        // Terminal merge toi thieu: dung field scalar da on dinh, khong doc location.x/y.
        // RL: action 3 chi xet khi merge_rl_accel_gate (waypoint truoc buoc di chuyen) + obs_gap_safe tu get_merging_state.
        // RB: ramp_behavior da xu ly merge theo luat; khong dung nhanh action_rl o day.
        if (merge_mode = 1) {
            if (rl_agent_id != "") {
                if (merge_rl_accel_gate) {
                    if (action_rl = 3) {
                        if (obs_gap_safe >= 1.0) {
                            is_merging <- false;
                            merge_mode <- 0;
                            in_accel_zone <- false;
                            is_merging_transition <- true;
                            current_lane_index <- number_of_lanes - 1;
                            target_lane_index <- number_of_lanes - 1;
                            move_next_y <- bottom_lane_y;
                            location <- {move_next_x, move_next_y};
                            reward_val <- 100.0;
                            terminal_reason <- "success";
                            merge_success <- true;
                            merge_step <- episode_step;
                            is_done <- true;
                            total_merge_success <- total_merge_success + 1;
                            recent_merge_coop_ticks <- 3;
                            recent_merge_x <- move_next_x;
                            cumulative_reward <- cumulative_reward + reward_val;
                            speed_sum <- speed_sum + speed;
                            return;
                        } else {
                            action_penalty <- action_penalty - 0.25;
                        }
                    }
                }
            }
            if (move_next_x >= merge_x - 1.0) {
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

        reward_val <- calculate_reward();
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
        
        // 1. VÁ LỖI RADAR: Phải nhìn cả xe đang chờ nhập làn VÀ xe đang trượt vào cao tốc
        // Snapshot car trước loop để tránh CME; kiểm tra nil trước truy cập .x.
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
                                    blocking_ramp_cars << c_car;
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
        
        // 2. BỨC TƯỜNG VẬT LÝ CHO ĐƯỜNG NHÁNH
        if (ramp_ahead != nil) {
            if (ramp_ahead.location != nil) {
                float dist <- euclid_point_dist(location, ramp_ahead.location);
                if (dist <= car_length + 0.5) {
                    max_allowed_speed <- 0.0;
                } else {
                    max_allowed_speed <- max(0.0, dist - (car_length + 1.5));
                }
            }
        }

        int accel_flag_prev <- 0;
        if (in_accel_zone) { accel_flag_prev <- 1; }
        in_accel_zone <- (waypoint_index = ramp_accel_waypoint_ix);
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

            list bottom_lane_cars <- [];
            loop c over: car_snap_rb {
                if (c != nil) {
                    car c_car <- c as car;
                    if (not dead(c_car)) {
                        if (c_car != self) {
                            if (c_car.location != nil) {
                                if (not c_car.is_merging) {
                                    if (c_car.current_lane_index = number_of_lanes - 1) {
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

            if (lead_car != nil) {
                if (gap_front < req_front) {
                speed <- max(speed_min, speed - deceleration * 0.6);
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

            if (gap_front >= req_front) {
                if (gap_rear >= req_rear) {
                    do execute_merge();
                    return;
                }
            }
            if (stuck_counter > 95) {
                if (gap_front > car_length * 0.95) {
                    if (gap_rear > car_length * 0.95) {
                        do execute_merge();
                        return;
                    }
                }
            }
            stuck_counter <- stuck_counter + 1;

        } else {
            speed <- min(speed_max * 0.7, speed + acceleration);
            stuck_counter <- 0;
        }

        // 3. ÉP GIỚI HẠN TỐC ĐỘ CUỐI CÙNG (Không cho phép vượt quá max_allowed_speed)
        speed <- min(speed, max_allowed_speed);
        speed <- max(0.0, speed); // Đảm bảo speed không bị âm lùi lại

        // Thực thi di chuyển trên nhánh
        if (ramp_waypoints != nil) {
            if (waypoint_index < length(ramp_waypoints)) {
                point tp <- ramp_waypoints[waypoint_index];
                if (tp != nil) {
                    heading  <- atan2(tp.y - location.y, tp.x - location.x);
                    location <- {location.x + speed * cos(heading), location.y + speed * sin(heading)};

                    if (euclid_point_dist(location, tp) < max(speed * 2.0, 1.0)) {
                        waypoint_index <- waypoint_index + 1;
                        if (ramp_waypoints != nil) {
                            if (waypoint_index >= length(ramp_waypoints)) {
                                do execute_merge();
                            }
                        }
                    }
                }
            }
        }
    }

    action rl_merging_behavior {
        if (location = nil) { return; }
        bool was_in_accel <- in_accel_zone;
        // Lưu observation trước khi thực thi action để Python có trạng thái hiện tại của xe nhập làn.
        state <- get_current_state();
        episode_step <- episode_step + 1;

        // Reset phạt tạm thời ở mỗi tick để reward chỉ phản ánh action vừa thực hiện.
        action_penalty <- 0.0;

        // Cập nhật cờ vùng tăng tốc; chỉ trong vùng này agent mới được phép ra lệnh nhập làn.
        // Throughput: đếm RL agent khi lần đầu vào accel zone (tương đương NPC trong ramp_behavior).
        in_accel_zone <- (waypoint_index = ramp_accel_waypoint_ix);
        if (in_accel_zone) {
            if (not was_in_accel) {
                total_ramp_attempts <- total_ramp_attempts + 1;
            }
        }

        // Mặc định là giữ tốc nếu Python chưa kịp gửi action trong tick đầu tiên.
        if (action_rl < 0) {
            action_rl <- 1;
        }

        // Dashboard GUI có thể chạy policy demo nội bộ; SB3 model thật vẫn do Python gửi action.
        if (dashboard_policy = "Manual") {
            action_rl <- manual_action;
        } else if (dashboard_policy = "Random") {
            int rw_ml <- rnd(0, 99);
            if (rw_ml < 7) {
                action_rl <- 0;
            } else if (rw_ml < 62) {
                action_rl <- 1;
            } else if (rw_ml < 82) {
                action_rl <- 2;
            } else if (rw_ml < 91) {
                action_rl <- 3;
            } else {
                action_rl <- 4;
            }
        } else if (dashboard_policy = "Heuristic") {
            action_rl <- get_heuristic_merging_action();
        }

        // Action 0: giảm tốc để tạo khoảng trống với xe phía trước hoặc chờ gap an toàn.
        if (action_rl = 0) {
            speed <- max(speed_min, speed - deceleration);
        }
        // Action 1: giữ tốc, dùng khi agent muốn duy trì quỹ đạo hiện tại trên ramp.
        else if (action_rl = 1) {
            speed <- speed;
        }
        // Action 2: tăng tốc để bắt kịp tốc độ dòng chính trước khi nhập làn.
        else if (action_rl = 2) {
            speed <- min(speed_max, speed + acceleration);
        }
        // Action 3: thử nhập làn; chỉ thành công khi đang ở vùng tăng tốc và gap đủ an toàn.
        else if (action_rl = 3) {
            if (in_accel_zone) {
                if (is_merge_gap_safe()) {
                    do execute_merge();
                    reward_val <- 100.0;
                    terminal_reason <- "success";
                    merge_success <- true;
                    merge_step    <- episode_step;
                    do update_merging_metrics();
                    next_state <- get_current_state();
                    is_done <- true;
                    return;
                }
            }
            // Phat nho theo tick neu thu merge sai thoi diem; terminal fail/success moi la tin hieu lon.
            action_penalty <- action_penalty - 0.25;
        }
        // Action 4: chờ nhập làn, giảm tốc nhẹ để tránh lao tới cuối làn tăng tốc quá sớm.
        else if (action_rl = 4) {
            speed <- max(speed_min, speed - deceleration * 0.3);
            action_penalty <- action_penalty - 0.02;
        }

        // Không cho xe RL đè lên xe nhập làn phía trước trên cùng nhánh.
        float max_allowed_speed <- speed_max;
        list blocking_ramp_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            if (c_car.is_on_ramp_merge_path()) {
                                if (c_car.location.x > self.location.x) {
                                    blocking_ramp_cars << c_car;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Xe gần nhất phía trước trên ramp quyết định tốc độ tối đa an toàn của agent (không dùng ? : — GAMA có thể đánh giá cả hai nhánh).
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

        // Nếu bám quá sát xe trước trên ramp thì ép giảm tốc để tránh va chạm giả do hình học.
        if (ramp_ahead != nil) {
            if (ramp_ahead.location != nil) {
                float dist <- euclid_point_dist(location, ramp_ahead.location);
                if (dist <= car_length + 0.5) {
                    max_allowed_speed <- 0.0;
                } else {
                    max_allowed_speed <- max(0.0, dist - (car_length + 1.5));
                }
            }
        }

        // Ép tốc độ cuối cùng nằm trong giới hạn vật lý và giới hạn an toàn theo xe trước.
        speed <- min(speed, max_allowed_speed);
        speed <- max(speed_min, min(speed_max, speed));

        // Di chuyển dọc theo polyline ramp; RL chỉ điều khiển tốc độ và quyết định nhập làn.
        if (ramp_waypoints != nil) {
            if (waypoint_index < length(ramp_waypoints)) {
                point tp <- ramp_waypoints[waypoint_index];
                if (tp != nil) {
                    heading  <- atan2(tp.y - location.y, tp.x - location.x);
                    location <- {location.x + speed * cos(heading), location.y + speed * sin(heading)};

                    if (euclid_point_dist(location, tp) < max(speed * 2.0, 1.0)) {
                        waypoint_index <- waypoint_index + 1;
                    }
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
        next_state <- get_current_state();
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
        // Throughput: đếm mỗi lần có xe merge thành công vào cao tốc.
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
        // --- CƠ CHẾ XE HỎNG ĐỘT XUẤT ---
        if (not is_damaged) {
            if (flip(prob_damaged_car)) {
                is_damaged <- true;
                damaged_timer <- max_damaged_duration;
                color <- #red;
                speed <- 0.0;
            }
        }

        if (is_damaged) {
            // Xe hỏng không dừng khựng lại ngay, mà giảm tốc từ từ theo quán tính
            speed <- max(0.0, speed - deceleration * 0.5); 
            
            // Nếu đã chậm lại, xe bắt đầu đánh lái tấp vào làn cứu hộ (phía trên cùng)
            if (speed < 0.2) {
                if (location.y > rescue_y_hw) {
                loop c over: safe_cars {
                    if (c != nil) {
                        car c_car <- c as car;
                        if (not dead(c_car)) {
                            if (c_car != self) {
                                if (c_car.location != nil) {
                                    if (c_car.location.y < self.location.y + 0.5) {
                                        if (c_car.location.y > self.location.y - lane_width * 1.5) {
                                            if (abs(c_car.location.x - self.location.x) < car_length * 1.2) {
                                                heading <- 0.0;
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                    if (location != nil) {
                        ox_hw <- location.x;
                        oy_hw <- location.y;
                        nx_hw <- ox_hw + speed;
                        ny_hw <- oy_hw - 0.08;
                        location <- {nx_hw, ny_hw};
                        heading <- atan2(ny_hw - oy_hw, nx_hw - ox_hw);
                    }
                }
            } else {
                if (location.y <= rescue_y_hw) {
                    heading <- 0.0;
                }
            }

            damaged_timer <- damaged_timer - 1;
            
            // Phục hồi xe hỏng
            if (damaged_timer <= 0) {
                is_damaged <- false;
                color <- one_of([#crimson, #dodgerblue, #silver, #white, #gold]);
                current_lane_index <- 0;
                target_lane_index <- 0;
                speed <- 0.1; 
            }
            return;
        }

        state <- get_current_state();
        
        action_penalty <- 0.0;
        
        // 1. TÁC TỬ CHỌN HÀNH ĐỘNG (0: Tăng tốc, 1: Giữ nguyên, 2: Giảm tốc, 3: Rẽ Trái, 4: Rẽ Phải)
        if (rl_agent_id != "") {
            if (action_rl < 0) {
                action_rl <- 1;
            }
        } else if (driving_policy = "Greedy") {
            action_rl <- get_heuristic_highway_action();
        } else if (flip(exp_rate / (1 + exp_decay * cycle))) {
            action_rl <- rnd(0, 4); // Random thám hiểm đủ 5 action
        } else {
            action_rl <- 1; // Gắn mô hình predict vào đây sau
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

        // Nếu xe trôi ra ngoài chiều dài đường mô phỏng thì cũng kết thúc để tránh episode treo.
        if (location != nil) {
            if (location.x >= road_length - 5.0) {
                reward_val <- -50.0;
                terminal_reason <- "failed_merge";
                failed_merge <- true;
                is_done <- true;
                return;
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
                action_penalty <- action_penalty - 0.05;
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
                            if (c_car.current_lane_index = lane) {
                                if (c_car.location.x > self.location.x) {
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
                            if (c_car.current_lane_index = lane) {
                                if (c_car.location.x < self.location.x) {
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
                                if (c_car.current_lane_index = merge_lane) {
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
                                if (c_car.current_lane_index = merge_lane) {
                                    if (c_car.location.x < self.location.x) {
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

    action is_merge_gap_safe type: bool {
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

        // Xe chạy nhanh cần khoảng cách phía trước lớn hơn vì quãng đường phanh dài hơn.
        float required_front <- max(gap_front_min, car_length * 1.2 + speed * 4.0);
        // Xe phía sau chạy nhanh cần gap sau lớn hơn để tránh bị tông khi nhập làn.
        float required_rear  <- max(gap_rear_min,  car_length * 1.2 + speed * 3.0);

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
                                if (c_car.location.x > self.location.x) {
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
                                if (c_car.location.x > self.location.x) {
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
        if (ahead_same != nil) {
            if (gap_same < 9.0) {
                if (speed > 0.12) {
                    return 2;
                }
            }
        }

        if (merger != nil) {
            if (location != nil) {
                if (location.x >= merge_zone_lo) {
                    if (location.x <= merge_zone_hi) {
                        if (current_lane_index = number_of_lanes - 1) {
                            if (current_lane_index > 0) {
                                car al <- get_car_ahead(current_lane_index - 1);
                                car bl <- get_car_behind(current_lane_index - 1);
                                float g_af <- 999.0;
                                float g_ab <- 999.0;
                                if (al != nil) {
                                    if (al.location != nil) {
                                        g_af <- al.location.x - self.location.x;
                                    }
                                }
                                if (bl != nil) {
                                    if (bl.location != nil) {
                                        g_ab <- self.location.x - bl.location.x;
                                    }
                                }
                                if (g_af > 11.0) {
                                    if (g_ab > 7.0) {
                                        return 3;
                                    }
                                }
                            }
                            return 2;
                        }
                    }
                }
            }
        }

        if (speed < speed_max * 0.72) {
            if (gap_same > 12.0) {
                return 0;
            }
        }
        return 1;
    }

    action get_heuristic_merging_action type: int {
        // Rule-based baseline dùng cho dashboard GAMA và làm mốc so sánh với DQN/PPO/A2C.
        float ramp_gap <- get_ramp_front_gap();
        float accel_len <- max(1.0, accel_end_x - accel_start_x);
        float urgency <- 0.0;
        if (location != nil) {
            urgency <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len));
        }

        if (in_accel_zone) {
            if (is_merge_gap_safe()) {
                return 3;
            }
        }
        if (ramp_gap < car_length * 2.0) { return 0; }
        if (urgency > 0.75) {
            if (not is_merge_gap_safe()) {
                return 4;
            }
        }
        if (speed < speed_max * 0.7) { return 2; }
        return 1;
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
                "agent_role"::"highway"
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
            "mainline_mean_speed"::sw_mean
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

    /* Disabled old radar-heavy get_current_state body for headless batch stability.

        float speed_denom_state <- max(0.001, speed_max);
        float obs_denom_state <- max(1.0, observation_max);
        float norm_speed <- speed / speed_denom_state;
        float norm_current_lane <- current_lane_index / max(1.0, (number_of_lanes - 1.0));

        // Radar Làn Hiện Tại
        car ahead_same <- get_car_ahead(current_lane_index);
        float norm_dist_ahead <- safe_obs_gap_norm(ahead_same, false);
        float norm_speed_ahead <- safe_obs_speed_norm(ahead_same);

        // Radar Làn Bên Trái
        float norm_dist_ahead_left <- 1.0; float norm_dist_behind_left <- 1.0;
        float norm_speed_ahead_left <- 1.0; float norm_speed_behind_left <- 1.0;
        if (current_lane_index > 0) {
            car ahead_left  <- get_car_ahead(current_lane_index - 1);
            car behind_left <- get_car_behind(current_lane_index - 1);

            norm_dist_ahead_left  <- safe_obs_gap_norm(ahead_left, false);
            norm_speed_ahead_left <- safe_obs_speed_norm(ahead_left);

            norm_dist_behind_left <- safe_obs_gap_norm(behind_left, true);
            norm_speed_behind_left <- safe_obs_speed_norm(behind_left);
        } else {
            norm_dist_ahead_left <- 0.0; norm_dist_behind_left <- 0.0; 
            norm_speed_ahead_left <- 0.0; norm_speed_behind_left <- 0.0; // Coi như tường đứng im
        }

        // Radar Làn Bên Phải
        float norm_dist_ahead_right <- 1.0; float norm_dist_behind_right <- 1.0;
        float norm_speed_ahead_right <- 1.0; float norm_speed_behind_right <- 1.0;
        if (current_lane_index < number_of_lanes - 1) {
            car ahead_right  <- get_car_ahead(current_lane_index + 1);
            car behind_right <- get_car_behind(current_lane_index + 1);

            norm_dist_ahead_right  <- safe_obs_gap_norm(ahead_right, false);
            norm_speed_ahead_right <- safe_obs_speed_norm(ahead_right);

            norm_dist_behind_right <- safe_obs_gap_norm(behind_right, true);
            norm_speed_behind_right <- safe_obs_speed_norm(behind_right);
        } else {
            norm_dist_ahead_right <- 0.0; norm_dist_behind_right <- 0.0; 
            norm_speed_ahead_right <- 0.0; norm_speed_behind_right <- 0.0; 
        }

        // Radar Xe Nhập Làn — loop thay where chuỗi (tránh nil / không short-circuit).
        list merging_cars <- [];
        if (safe_cars != nil) {
            loop c over: safe_cars {
                if (c != nil) {
                    car c_car <- c as car;
                    if (not dead(c_car)) {
                        if (c_car.location != nil) {
                            if (c_car.is_merging) {
                                if (c_car.location.x > self.location.x) {
                                    if ((c_car.location.x - self.location.x) <= observation_max) {
                                        merging_cars << c_car;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        car closest_merger <- nil;
        if (length(merging_cars) > 0) {
            float best_dx_ms <- 1.0E9;
            loop mc over: merging_cars {
                if (mc != nil) {
                    car mc_car <- mc as car;
                    if (mc_car.location != nil) {
                        if (mc_car.location.x - self.location.x < best_dx_ms) {
                            best_dx_ms <- mc_car.location.x - self.location.x;
                            closest_merger <- mc_car;
                        }
                    }
                }
            }
        }

        float norm_dist_merger <- 1.0;
        float norm_speed_merger <- 1.0;
        if (closest_merger != nil) {
            norm_speed_merger <- safe_obs_speed_norm(closest_merger);
            if (closest_merger.location != nil) {
                if (location != nil) {
                    norm_dist_merger <- min(1.0, (closest_merger.location.x - self.location.x) / obs_denom_state);
                }
            }
        }

        // Vector trả về bây giờ có 15 thông số
        return [
            norm_speed, norm_current_lane, 
            norm_dist_ahead, norm_speed_ahead, 
            norm_dist_ahead_left, norm_speed_ahead_left, norm_dist_behind_left, norm_speed_behind_left, 
            norm_dist_ahead_right, norm_speed_ahead_right, norm_dist_behind_right, norm_speed_behind_right, 
            norm_dist_merger, norm_speed_merger, 
            patience / 100.0
        ];
    }
    */

    action get_merging_state type: list<float> {
        // Observation hot path khong return bien tam: GAMA headless de mat TempVariable context.
        obs_speed <- min(1.0, max(0.0, speed));
        obs_merge_len <- max(1.0, merge_x - accel_start_x);
        obs_progress <- 0.0;
        obs_dist_to_merge <- 1.0;
        obs_lateral <- 1.0;
        if (location != nil) {
            obs_progress <- min(1.0, max(0.0, (location.x - accel_start_x) / obs_merge_len));
            obs_dist_to_merge <- min(1.0, max(0.0, (merge_x - location.x) / obs_merge_len));
            obs_lateral <- min(1.0, max(0.0, abs(location.y - bottom_lane_y) / max(1.0, lane_width * 4.0)));
        }
        obs_in_accel <- 0.0;
        if (in_accel_zone) {
            obs_in_accel <- 1.0;
        }
        obs_action <- 0.5;
        if (action_rl <= 0) {
            obs_action <- 0.0;
        } else if (action_rl = 1) {
            obs_action <- 0.25;
        } else if (action_rl = 2) {
            obs_action <- 0.5;
        } else if (action_rl = 3) {
            obs_action <- 0.75;
        } else if (action_rl >= 4) {
            obs_action <- 1.0;
        }
        obs_terminal <- 0.0;
        if (terminal_reason != "running") {
            obs_terminal <- 1.0;
        }
        obs_front_gap_raw <- 999.0;
        obs_rear_gap_raw <- 999.0;
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
                                            if (c_car.current_lane_index = number_of_lanes - 1) {
                                                if (c_car.location.x >= location.x) {
                                                    obs_scan_dx <- c_car.location.x - location.x;
                                                    if (obs_scan_dx < obs_front_gap_raw) {
                                                        obs_front_gap_raw <- obs_scan_dx;
                                                    }
                                                } else {
                                                    obs_scan_dx <- location.x - c_car.location.x;
                                                    if (obs_scan_dx < obs_rear_gap_raw) {
                                                        obs_rear_gap_raw <- obs_scan_dx;
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
        obs_target_front_gap <- min(1.0, max(0.0, obs_front_gap_raw / max(1.0, observation_max)));
        obs_target_rear_gap <- min(1.0, max(0.0, obs_rear_gap_raw / max(1.0, observation_max)));
        obs_gap_safe <- 0.0;
        obs_required_front <- max(gap_front_min, car_length * 1.2 + speed * 4.0);
        obs_required_rear <- max(gap_rear_min, car_length * 1.2 + speed * 3.0);
        if (obs_front_gap_raw >= obs_required_front) {
            if (obs_rear_gap_raw >= obs_required_rear) {
                obs_gap_safe <- 1.0;
            }
        }
        return [
            obs_speed,
            obs_progress, obs_dist_to_merge,
            obs_lateral, obs_in_accel,
            obs_gap_safe, obs_target_front_gap, obs_target_rear_gap, 1.0,
            1.0, 1.0,
            obs_action, obs_terminal,
            0.0,
            1.0
        ];
    }

    /* Disabled old radar-heavy get_merging_state body for headless batch stability.

        list<float> merging_state_nil <- [
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0
        ];
        // GAMA: không gọi empty(ramp_waypoints) khi list có thể nil (or không rút gọn → NPE IContainer.isEmpty).
        if (location = nil) {
            return merging_state_nil;
        }
        if (ramp_waypoints = nil) {
            return merging_state_nil;
        }
        if (length(ramp_waypoints) = 0) {
            return merging_state_nil;
        }
        if (ramp_waypoints[0] = nil) {
            return merging_state_nil;
        }
        // Chuẩn hóa tốc độ hiện tại vào [0, 1] để phù hợp Box observation của PettingZoo.
        float speed_denom_merge <- max(0.001, speed_max);
        float obs_denom_merge <- max(1.0, observation_max);
        float norm_speed <- min(1.0, max(0.0, speed / speed_denom_merge));

        // Tính tiến độ theo trục x từ đầu ramp tới điểm nhập làn.
        float ramp_start_x <- ramp_waypoints[0].x;
        float merge_length <- max(1.0, merge_x - ramp_start_x);
        float norm_progress <- min(1.0, max(0.0, (location.x - ramp_start_x) / merge_length));

        // Khoảng cách còn lại tới điểm merge càng nhỏ thì bài toán càng khẩn cấp.
        float norm_dist_to_merge <- min(1.0, max(0.0, (merge_x - location.x) / merge_length));

        // Độ lệch ngang so với làn mục tiêu giúp agent biết mình còn ở ramp hay đã sát cao tốc.
        float norm_lateral_error <- min(1.0, abs(location.y - bottom_lane_y) / (lane_width * 4.0));

        float norm_in_accel_zone <- 0.0;
        if (in_accel_zone) {
            norm_in_accel_zone <- 1.0;
        }

        // Xe trước và sau ở làn mục tiêu là hai đối tượng quan trọng nhất khi quyết định nhập làn.
        car lead_car <- get_target_lane_ahead();
        car lag_car  <- get_target_lane_behind();

        float norm_gap_front <- 1.0;
        float norm_speed_front <- 1.0;
        if (lead_car != nil) {
            norm_speed_front <- min(1.0, max(0.0, lead_car.speed / speed_denom_merge));
            if (lead_car.location != nil) {
                norm_gap_front <- min(1.0, max(0.0, (lead_car.location.x - self.location.x) / obs_denom_merge));
            }
        }

        float norm_gap_rear <- 1.0;
        float norm_speed_rear <- 1.0;
        if (lag_car != nil) {
            norm_speed_rear <- min(1.0, max(0.0, lag_car.speed / speed_denom_merge));
            if (lag_car.location != nil) {
                norm_gap_rear <- min(1.0, max(0.0, (self.location.x - lag_car.location.x) / obs_denom_merge));
            }
        }

        float norm_gap_safe <- 0.0;
        if (is_merge_gap_safe()) {
            norm_gap_safe <- 1.0;
        }

        list ramp_front_cars <- [];
        loop c over: safe_cars {
            if (c != nil) {
                car c_car <- c as car;
                if (not dead(c_car)) {
                    if (c_car != self) {
                        if (c_car.location != nil) {
                            if (c_car.is_on_ramp_merge_path()) {
                                if (c_car.location.x > self.location.x) {
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
            float best_dx_rfm <- 1.0E9;
            loop rc over: ramp_front_cars {
                if (rc != nil) {
                    car rc_car <- rc as car;
                    if (rc_car.location != nil) {
                        if (rc_car.location.x - self.location.x < best_dx_rfm) {
                            best_dx_rfm <- rc_car.location.x - self.location.x;
                            ramp_front <- rc_car;
                        }
                    }
                }
            }
        }

        float norm_ramp_front_gap <- 1.0;
        float norm_ramp_front_speed <- 1.0;
        if (ramp_front != nil) {
            norm_ramp_front_speed <- min(1.0, max(0.0, ramp_front.speed / speed_denom_merge));
            if (ramp_front.location != nil) {
                norm_ramp_front_gap <- min(1.0, max(0.0, (ramp_front.location.x - self.location.x) / obs_denom_merge));
            }
        }

        // Urgency tăng dần khi xe đi gần hết làn tăng tốc mà vẫn chưa nhập làn.
        float accel_len <- max(1.0, accel_end_x - accel_start_x);
        float norm_urgency <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len));

        float norm_last_action <- 0.0;
        if (action_rl >= 0) {
            norm_last_action <- min(1.0, (action_rl + 1) / 5.0);
        }

        // patience giữ chiều thứ 15 tương thích với observation space cũ.
        float norm_patience <- min(1.0, max(0.0, patience / 100.0));

        // Vector 15 chiều: động học xe, tiến độ ramp, gap làn mục tiêu, xe trước trên ramp và mức khẩn cấp.
        return [
            norm_speed, norm_progress, norm_dist_to_merge,
            norm_lateral_error, norm_in_accel_zone,
            norm_gap_front, norm_speed_front,
            norm_gap_rear, norm_speed_rear,
            norm_gap_safe,
            norm_ramp_front_gap, norm_ramp_front_speed,
            norm_urgency, norm_last_action,
            norm_patience
        ];
    }
    */

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
        // Speed reward + action_penalty cho highway / mainline.
        reward_cache <- action_penalty + speed * 0.10;
        // Phat ru bo: giu speed > 20% speed_max.
        if (speed < speed_max * 0.2) {
            reward_cache <- reward_cache - 0.05;
        }
        // TTC penalty: neu xe phia truoc cung lane gan (< 10m) va minh dang chay nhanh.
        if (enable_highway_ttc_reward) {
            if (ahead_gap_dx < 10.0) {
                if (speed > 0.3) {
                    reward_cache <- reward_cache - ((10.0 - ahead_gap_dx) / 10.0) * speed * 0.5;
                }
            }
        }
        // Phase 7 ROLLBACK: proximity penalty cong don lam MARL collapse -> bo.
        // Goc re la OBSERVATION (M3b da fix: 15D radar thuc).
        // Phase 8 ROLLBACK: yield-to-merger per-tick reward lam highway dung yen -> MARL collapse.
        // Cooperative reward MARL: thuong khi gan diem merge thanh cong trong TTL `recent_merge_coop_ticks`.
        if (enable_marl_coop_reward) {
            if (rl_agent_id != "") {
                if (recent_merge_coop_ticks > 0) {
                    if (recent_merge_x >= 0.0) {
                        if (abs(move_next_x - recent_merge_x) < 20.0) {
                            reward_cache <- reward_cache + 0.3;
                        }
                    }
                }
            }
        }
        return reward_cache;
    }

    /* Disabled old radar-heavy calculate_reward body for headless batch stability.

        // 1. Phạt cực nặng nếu đâm xe — dùng collision_distance nhất quán.
        list nearby_cars <- filter_collision_neighbors();
        if (length(nearby_cars) > 0) {
            car closest_car <- nil;
            float best_dist_cr <- 1.0E9;
            loop nc over: nearby_cars {
                if (nc != nil) {
                    car nc_car <- nc as car;
                    if (nc_car.location != nil) {
                        if (euclid_point_dist(self.location, nc_car.location) < best_dist_cr) {
                            best_dist_cr <- euclid_point_dist(self.location, nc_car.location);
                            closest_car <- nc_car;
                        }
                    }
                }
            }
            if (closest_car != nil) {
                if (closest_car.location != nil) {
                    if (euclid_point_dist(self.location, closest_car.location) < collision_distance) {
                        return -100.0;
                    }
                }
            }
        }
        
        // 2. Thưởng khi duy trì tốc độ cao
        float reward <- (speed / speed_denom_reward) * 0.1; 
        
        reward <- reward + action_penalty;

        // Phạt rùa bò
        if (speed < speed_max * 0.2) {
            reward <- reward - 0.05; 
        }

        // Phạt bám đuôi nguy hiểm (Time-To-Collision penalty)
        car ahead_car <- get_car_ahead(current_lane_index);
        if (ahead_car != nil) {
            if (ahead_car.location != nil) {
                float dist_to_ahead <- ahead_car.location.x - self.location.x;
                if (dist_to_ahead < 10.0) {
                    if (speed > 0.3) {
                        ttc_penalty_hw <- ((10.0 - dist_to_ahead) / 10.0) * (speed / speed_denom_reward) * 0.5;
                        reward <- reward - ttc_penalty_hw;
                    }
                }
            }
        }

        // 5. MARL Cooperative reward: thưởng khi merge thành công gần highway agent (TTL recent_merge_coop_ticks).
        if (rl_agent_id != "") {
            if (recent_merge_coop_ticks > 0) {
                if (recent_merge_x >= 0.0) {
                    if (location != nil) {
                        float dist_to_merge <- abs(location.x - recent_merge_x);
                        if (dist_to_merge < 20.0) {
                            reward <- reward + 0.3;
                        }
                    }
                }
            }
        }

        return reward;
    }
    */

    action calculate_merging_reward type: float {
        // M2: dung field `obs_front_gap_raw`, `obs_rear_gap_raw`, `obs_gap_safe` precomputed
        // trong `get_merging_state` (Python query moi tick) + field `ahead_gap_dx` tu loop INLINE
        // collision trong `reflex behave`. KHONG dung target-lane point-loop hay helper co `return` bien cuc bo.
        if (enable_reward_collision_check) {
            if (nearest_collision_dist < collision_distance) {
                return -100.0;
            }
        }
        reward_cache <- -0.02 + action_penalty + speed * 0.05;
        // Precompute scalar dau action (tranh declare local trong nested if -> TempVariable trong hot path).
        m2_merge_gap_front <- obs_front_gap_raw;
        if (m2_merge_gap_front >= 999.0) {
            m2_merge_gap_front <- ahead_gap_dx;
        }
        m2_accel_len <- max(1.0, accel_end_x - accel_start_x);
        m2_urgency <- min(1.0, max(0.0, (move_next_x - accel_start_x) / m2_accel_len));
        if (enable_merging_gap_reward) {
            // Gap front penalty.
            if (m2_merge_gap_front < gap_front_min) {
                reward_cache <- reward_cache - (gap_front_min - m2_merge_gap_front) * 0.5;
            }
            // Gap rear penalty (chi tu observation).
            if (obs_rear_gap_raw < gap_rear_min) {
                reward_cache <- reward_cache - (gap_rear_min - obs_rear_gap_raw) * 0.5;
            }
            // Thuong nho khi gap an toan.
            if (obs_gap_safe >= 1.0) {
                reward_cache <- reward_cache + 0.05;
            }
        }
        // Urgency penalty trong accel zone.
        if (in_accel_zone) {
            reward_cache <- reward_cache - m2_urgency * 0.05;
        }
        // Bonus khi chon action merge dung luc gap an toan.
        if (action_rl = 3) {
            if (in_accel_zone) {
                if (obs_gap_safe >= 1.0) {
                    reward_cache <- reward_cache + 1.0;
                }
            }
        }
        if (reward_cache < -40.0) {
            reward_cache <- -40.0;
        }
        if (reward_cache > 40.0) {
            reward_cache <- 40.0;
        }
        return reward_cache;
    }

    /* Disabled old radar-heavy calculate_merging_reward body for headless batch stability.

        // Va chạm → phạt nặng nhất. Dùng collision_distance nhất quán với toàn bộ codebase.
        list nearby_cars <- filter_collision_neighbors();
        if (length(nearby_cars) > 0) {
            car closest_car <- nil;
            float best_dist_cmr <- 1.0E9;
            loop nc over: nearby_cars {
                if (nc != nil) {
                    car nc_car <- nc as car;
                    if (nc_car.location != nil) {
                        if (euclid_point_dist(self.location, nc_car.location) < best_dist_cmr) {
                            best_dist_cmr <- euclid_point_dist(self.location, nc_car.location);
                            closest_car <- nc_car;
                        }
                    }
                }
            }
            if (closest_car != nil) {
                if (closest_car.location != nil) {
                    if (euclid_point_dist(self.location, closest_car.location) < collision_distance) {
                        return -100.0;
                    }
                }
            }
        }

        // Phạt nhỏ mỗi timestep để khuyến khích nhập làn gọn, không kéo dài episode.
        float reward <- -0.02;

        // Cộng điểm phạt/thưởng tạm thời sinh ra từ action vừa chọn.
        reward <- reward + action_penalty;

        // Thưởng nhẹ khi xe duy trì tốc độ đủ cao, tránh học chính sách đứng yên trên ramp.
        reward <- reward + (speed / speed_denom_merging_reward) * 0.05;

        // Lấy xe trước/sau trên làn mục tiêu để đánh giá an toàn gap.
        car lead_car <- get_target_lane_ahead();
        car lag_car  <- get_target_lane_behind();

        float gap_front <- 999.0;
        if (lead_car != nil) {
            if (lead_car.location != nil) {
                gap_front <- lead_car.location.x - self.location.x;
            }
        }
        if (gap_front < gap_front_min) {
            reward <- reward - (gap_front_min - gap_front) * 0.5;
        }

        float gap_rear <- 999.0;
        if (lag_car != nil) {
            if (lag_car.location != nil) {
                gap_rear <- self.location.x - lag_car.location.x;
            }
        }
        if (gap_rear < gap_rear_min) {
            reward <- reward - (gap_rear_min - gap_rear) * 0.5;
        }

        // Khi gap an toàn, thưởng nhỏ để agent học nhận diện thời điểm nhập làn tốt.
        if (is_merge_gap_safe()) {
            reward <- reward + 0.05;
        }

        // Càng gần cuối làn tăng tốc mà chưa nhập làn thì càng bị phạt vì nguy cơ hết đường.
        float accel_len <- max(1.0, accel_end_x - accel_start_x);
        float urgency <- 0.0;
        if (location != nil) {
            urgency <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len));
        }
        if (in_accel_zone) {
            reward <- reward - urgency * 0.05;
        }

        // Nếu action nhập làn được chọn đúng lúc gap an toàn, thưởng phụ để tăng tín hiệu học.
        if (action_rl = 3) {
            if (in_accel_zone) {
                if (is_merge_gap_safe()) {
                    reward <- reward + 1.0;
                }
            }
        }

        return reward;
    }
    */

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
            } else {
                if (is_on_ramp_merge_path()) {
                    draw "RB" at: {location.x - 1.3, location.y - 2.2, 0.29} color: #orange font: font("Arial", 8, #bold);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// TrafficMARLHeadless: MARL 4 agents — pipeline chinh (train_marl, run_experiments).
//   Khởi động: gama-headless.bat -socket 1001
//   Sau đó: python rl/smoke_test_env.py | python rl/run_experiments.py --preset ...
// Baseline Greedy dùng archive GAML (TrafficSingleHeadless), không experiment này.
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
        among: ["Python/SB3 model", "Heuristic", "Random", "Manual"];
    parameter "Algorithm label" var: dashboard_algorithm category: "RL Demo"
        among: ["DQN", "PPO", "A2C", "Heuristic", "Random"];
    parameter "Model version label" var: dashboard_model_version category: "RL Demo";
    parameter "Manual action" var: manual_action category: "RL Demo" min: 0 max: 4;
    parameter "Show dashboard" var: show_dashboard category: "RL Demo";
    parameter "Demo step delay ms" var: demo_step_delay_ms category: "RL Demo" min: 0 max: 1000;
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
                // Nen ramp (ramp_surface_geoms) + bien/joint da tinh trong init.
                if (ramp_waypoints != nil) {
                    if (length(ramp_waypoints) > 1) {
                        loop ri from: 0 to: length(ramp_waypoints) - 2 {
                            if (ramp_waypoints[ri] != nil) {
                                if (ramp_waypoints[ri + 1] != nil) {
                                    point rpa <- ramp_waypoints[ri];
                                    point rpb <- ramp_waypoints[ri + 1];
                                    draw line([{rpa.x, rpa.y, 0.035}, {rpb.x, rpb.y, 0.035}]) color: rgb(70, 70, 70) width: 2.8;
                                }
                            }
                        }
                    }
                }
                loop g over: ramp_surface_geoms { if (g != nil) { draw g color: rgb(80, 80, 80); } }
                loop g over: ramp_joint_geoms { if (g != nil) { draw g color: rgb(80, 80, 80); } }
                loop g over: ramp_edge_geoms { if (g != nil) { draw g color: #white width: 1.6; } }
                loop g over: ramp_joint_edge_geoms { if (g != nil) { draw g color: #white width: 1.6; } }
                // Bo line dut trong long duong nhap lan/duong gia toc theo yeu cau.

                float top_edge <- offset_y;
                float gore_y   <- offset_y + number_of_lanes * lane_width;
                float outer_y  <- gore_y + lane_width;
                float rescue_lane_y <- offset_y - lane_width;
                
                draw line([{0.0, top_edge, 0.06}, {road_length, top_edge, 0.06}]) color: #white width: 2.0;
                // Giu nguyen: ranh gioi merge co net dut trong vung merge, net lien hai dau.
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
                // Giu nguyen: bien tren lane gia toc la net dut.
                loop g over: accel_outer_dash_geoms { if (g != nil) { draw g color: #white width: 2.0; } }
                // Canh cheo duoi cung cua lane gia toc noi vao mep duoi lane cao toc duoi cung.
                draw line([{accel_end_x, outer_y, 0.06}, {merge_x, gore_y, 0.06}]) color: #white width: 2.0;
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
                    draw ("Policy: " + dashboard_policy + " | Algo: " + dashboard_algorithm)
                        at: {15 #px, 78 #px} color: #white font: font("Arial", 21, #plain);
                    draw ("Model version: " + dashboard_model_version)
                        at: {15 #px, 113 #px} color: #white font: font("Arial", 21, #plain);
                    draw "Action map: 0 Dec | 1 Keep | 2 Acc | 3 Merge | 4 Wait"
                        at: {15 #px, 148 #px} color: #yellow font: font("Arial", 17, #plain);

                    if (tracked = nil) {
                        draw "No RL agent found. Reset the experiment to spawn merging_0."
                            at: {15 #px, 193 #px} color: #red font: font("Arial", 21, #bold);
                    } else {
                        if (dead(tracked)) {
                            draw "No RL agent found. Reset the experiment to spawn merging_0."
                                at: {15 #px, 193 #px} color: #red font: font("Arial", 21, #bold);
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
                            at: {15 #px, 178 #px} color: tracked_title_rgb font: font("Arial", 21, #bold);
                        draw ("Outcome: " + tracked.terminal_reason + " | Step: " + tracked.episode_step)
                            at: {15 #px, 213 #px} color: outcome_rgb font: font("Arial", 21, #plain);
                        draw ("Action: " + tracked.action_rl + " | Reward: " + (tracked.reward_val with_precision 2) + " | CumReward: " + (tracked.cumulative_reward with_precision 2))
                            at: {15 #px, 248 #px} color: reward_line_rgb font: font("Arial", 21, #plain);
                        draw ("Speed: " + (tracked.speed with_precision 2) + " | Mean speed: " + (mean_speed with_precision 2) + " | Progress: " + ((progress * 100.0) with_precision 1) + "%")
                            at: {15 #px, 283 #px} color: #white font: font("Arial", 21, #plain);
                        draw ("Gap front/rear: " + (front_gap with_precision 2) + " / " + (rear_gap with_precision 2) + " | Safe: " + tracked.is_merge_gap_safe())
                            at: {15 #px, 318 #px} color: gap_safe_rgb font: font("Arial", 21, #plain);
                        draw ("Min gap front/rear: " + (tracked.min_front_gap with_precision 2) + " / " + (tracked.min_rear_gap with_precision 2))
                            at: {15 #px, 353 #px} color: #white font: font("Arial", 21, #plain);
                        draw ("Ramp front gap: " + (tracked.get_ramp_front_gap() with_precision 2) + " | In accel zone: " + tracked.in_accel_zone)
                            at: {15 #px, 388 #px} color: #white font: font("Arial", 21, #plain);
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
                    draw ("Agents: RL=" + rl_count + " rule-based merge=" + rule_based_merging_count + " mainline env=" + mainline_count)
                        at: {15 #px, 438 #px} color: #white font: font("Arial", 21, #plain);
                    draw ("Traffic: total=" + length(safe_cars) + " ramp=" + ramp_count + " damaged=" + damaged_count)
                        at: {15 #px, 473 #px} color: #white font: font("Arial", 21, #plain);
                    draw ("Cycle: " + cycle + " | Delay: " + demo_step_delay_ms + " ms | Manual action only when policy=Manual")
                        at: {15 #px, 508 #px} color: #white font: font("Arial", 21, #plain);
                    draw "SB3 .zip models are loaded by Python; this GUI labels/observes the selected run." at: {15 #px, 550 #px} color: #cyan font: font("Arial", 17, #italic);
                    draw "Legend: RL=learning agent | RB=rule-based merging car | unlabelled=traffic environment" at: {15 #px, 585 #px} color: #cyan font: font("Arial", 17, #italic);
                }
            }
        }
    }
}
