model HorizontalTraffic

global {
    int   number_of_lanes <- 3;
    float lane_width      <- 3.5;
    float road_length     <- 200.0;
    float offset_y        <- 18.0;

    geometry shape <- rectangle(road_length, 100.0);

    float speed_max            <- 1.0;
    float speed_min            <- 0.0;
    float acceleration         <- 0.05;
    float deceleration         <- 0.1;
    int   nb_cars_max          <- 20;
    float observation_distance <- 10.0;

    list<point> ramp_waypoints;
    geometry    ramp_shape;

    float bottom_lane_y;
    float accel_lane_y;
    float accel_start_x <- 48.0;
    float accel_end_x   <- 162.0;
    float merge_x       <- 180.0;

    // ── Gap-acceptance thresholds ──
    float gap_front_max <- 7.0;   
    float gap_front_min <- 2.5;   
    float gap_rear_max  <- 6.0;   
    float gap_rear_min  <- 2.0;   

    // ── Tham số RL & Thuật toán (từ NetLogo) ──
    string driving_policy <- "Greedy"; // Tạm để Greedy để xe chạy tránh va chạm, sau này đổi lại "Q-Learning"
    float  discount_factor <- 0.99;
    float  learning_rate   <- 0.001;
    float  exp_rate        <- 1.0;
    float  exp_decay       <- 0.005;
    
    float  collision_distance <- 1.0;
    int    max_patience       <- 100;
    float  observation_max    <- 100.0;

    init {
        bottom_lane_y <- offset_y + (number_of_lanes - 1) * lane_width + lane_width / 2.0;
        accel_lane_y  <- offset_y + number_of_lanes * lane_width + lane_width / 2.0;

        ramp_waypoints <- [
            {18.0, 78.0},
            {28.0, 60.0},
            {40.0, accel_lane_y + 4.0},
            {accel_start_x, accel_lane_y},   
            {accel_end_x,   accel_lane_y},   
            {merge_x, bottom_lane_y}         
        ];

        ramp_shape <- polyline(ramp_waypoints) + (lane_width / 2.0);

        do build_road();
        do create_initial_cars();
    }

    reflex spawn_ramp_cars when: every(25 #cycles) {
        if (flip(0.55)) {
            create car number: 1 {
                is_merging     <- true;
                waypoint_index <- 1;
                location       <- ramp_waypoints[0];
                heading        <- towards(location, ramp_waypoints[1]);
                speed          <- rnd(0.30, 0.55);
                color          <- #orange;
                direction      <- 1;
                current_lane_index <- number_of_lanes - 1;
                target_lane_index  <- number_of_lanes - 1;
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
        create car number: nb_cars_max {
            speed              <- rnd(0.4, speed_max);
            color              <- one_of([#crimson, #dodgerblue, #silver, #white, #gold, #darkviolet]);
            is_merging         <- false;
            in_accel_zone      <- false;
            current_lane_index <- rnd(0, number_of_lanes - 1);
            target_lane_index  <- current_lane_index;
            float sx <- rnd(10.0, road_length - 10.0);
            float sy <- offset_y + (current_lane_index * lane_width) + (lane_width / 2.0);
            location  <- {sx, sy};
            direction <- 1;
            heading   <- 0.0;
        }
    }
}

// ────────────────────────────────────────────────────────────
species road_segment {
    int lane_index;
    rgb color;

    aspect default {
        draw shape color: color;
        if (lane_index < number_of_lanes - 1) {
            float dy <- location.y + lane_width / 2.0;
            float dx <- 0.0;
            loop while: dx < road_length {
                draw line([{dx, dy}, {dx + 2.5, dy}]) color: #white width: 1.5;
                dx <- dx + 6.0;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
species car skills: [moving] {
    // ══════════════════════════════════════════════════════════
    // KHAI BÁO BIẾN (BẮT BUỘC Ở ĐẦU SPECIES)
    // ══════════════════════════════════════════════════════════
    int   current_lane_index;
    int   target_lane_index;
    int   direction;
    rgb   color;

    bool  is_merging;      
    bool  in_accel_zone;   
    int   waypoint_index;

    float car_length <- 3.5;
    float car_width  <- 1.7;
    int   patience   <- 100;

    // Các biến phục vụ thuật toán RL
    float speed_top;
    float reward_val;
    list<float> state;
    list<float> next_state;
    int action_rl <- -1;

    // ══════════════════════════════════════════════════════════
    // REFLEX CHÍNH
    // ══════════════════════════════════════════════════════════
    reflex behave {
        if (is_merging) {
            do ramp_behavior();
        } else {
            do highway_rl_move();
        }
    }

    // ══════════════════════════════════════════════════════════
    // HÀNH VI NHẬP LÀN (GIỮ NGUYÊN)
    // ══════════════════════════════════════════════════════════
    action ramp_behavior {
        in_accel_zone <- (waypoint_index = 4);

        if (in_accel_zone) {
            speed <- min(speed_max, speed + acceleration);

            float urgency <- 0.0;
            float accel_len <- accel_end_x - accel_start_x;
            if (accel_len > 0.0) {
                urgency <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len));
            }

            float req_front <- gap_front_max - urgency * (gap_front_max - gap_front_min);
            float req_rear  <- gap_rear_max  - urgency * (gap_rear_max  - gap_rear_min);

            list<car> bottom_lane_cars <- (car - self) where (
                not each.is_merging and
                each.current_lane_index = number_of_lanes - 1
            );

            car lead_car <- (bottom_lane_cars where (each.location.x > self.location.x))
                            with_min_of (each.location.x - self.location.x);
            car lag_car  <- (bottom_lane_cars where (each.location.x <= self.location.x))
                            with_min_of (self.location.x - each.location.x);

            float gap_front <- (lead_car = nil) ? 999.0 : (lead_car.location.x - self.location.x);
            float gap_rear  <- (lag_car  = nil) ? 999.0 : (self.location.x - lag_car.location.x);

            if (lead_car != nil and gap_front < req_front) {
                speed <- max(speed_min, speed - deceleration * 0.6);
            } else if (lag_car != nil and gap_rear < req_rear) {
                speed <- min(speed_max, speed + acceleration * 0.8);
            }

            bool gap_ok <- (gap_front >= req_front) and (gap_rear >= req_rear);

            if (gap_ok) {
                do execute_merge();
                return;
            }

        } else {
            speed <- min(speed_max * 0.7, speed + acceleration);
        }

        if (waypoint_index < length(ramp_waypoints)) {
            point tp <- ramp_waypoints[waypoint_index];
            heading  <- towards(location, tp);
            location <- {location.x + speed * cos(heading), location.y + speed * sin(heading)};

            if (location distance_to tp < max(speed * 2.0, 1.0)) {
                waypoint_index <- waypoint_index + 1;
                if (waypoint_index >= length(ramp_waypoints)) {
                    do execute_merge();
                }
            }
        }
    }

    // Thực hiện quyết định nhập làn
    action execute_merge {
        is_merging         <- false;
        in_accel_zone      <- false;
        
        // ── ĐÃ XÓA 2 DÒNG ÉP TỌA ĐỘ VÀ ÉP GÓC QUAY TỨC THỜI ──
        // heading <- 0.0; 
        // location <- {location.x, bottom_lane_y}; 
        
        // Chỉ cần báo cho hệ thống biết xe đã thuộc về làn cao tốc dưới cùng
        current_lane_index <- number_of_lanes - 1;
        target_lane_index  <- number_of_lanes - 1;
        
        // Giữ tốc độ hiện tại (đã tăng trong accel lane)
        speed              <- min(speed_max, speed);
        patience           <- 100;
    }

    // ══════════════════════════════════════════════════════════
    // HÀNH VI CHẠY CAO TỐC & HỌC TĂNG CƯỜNG
    // ══════════════════════════════════════════════════════════
    action highway_rl_move {
        state <- get_current_state();
        
        float epsilon <- exp_rate / (1 + exp_decay * cycle);
        if (driving_policy = "Greedy") {
		    // Đi thẳng và giữ tốc độ, chỉ giảm khi cần
		    action_rl <- 1; // maintain speed mặc định
		} else if (flip(epsilon)) {
		    action_rl <- rnd(0, 3); // explore
		} else {
		    action_rl <- 1; // exploit (Q-table sau này)
		}

        if (action_rl = 0) { speed <- min(speed_max, speed + acceleration); }
        else if (action_rl = 1) {} 
        else if (action_rl = 2) { speed <- max(speed_min, speed - deceleration); }
        else if (action_rl = 3) { do try_change_lane(); }

        do handle_blocking_cars();
        
        // --- FIX: DI CHUYỂN CHÉO MƯỢT MÀ VÀ GIỮ THẲNG LÁI ---
        float target_y <- offset_y + (current_lane_index * lane_width) + (lane_width / 2.0);
        float next_y <- location.y;
        
        if (abs(location.y - target_y) > 0.1) {
            // Lái xe thực tế: Chỉ có thể lạng sang ngang khi xe đang lăn bánh
            if (speed > 0) {
                // Tốc độ lách ngang tỷ lệ với tốc độ tiến (tránh bị xoay 90 độ)
                float step_y <- speed * 0.3; 
                if (location.y < target_y) { next_y <- min(target_y, location.y + step_y); }
                else { next_y <- max(target_y, location.y - step_y); }
                
                // Chủ động cập nhật góc quay mượt mà theo vector di chuyển
                heading <- towards(location, {location.x + speed, next_y});
            }
        } else {
            next_y <- target_y;
            heading <- 0.0; // Trả thẳng lái khi đã vào chính giữa làn
        }

        location <- {location.x + speed, next_y};
        if (location.x > road_length) { location <- {0.0, next_y}; }

        reward_val <- calculate_reward();
        next_state <- get_current_state();
    }
    
    action handle_blocking_cars {
        car ahead <- get_car_ahead();
        if (ahead != nil and (ahead.location.x - self.location.x) <= get_safe_distance()) {
            if (ahead.speed > 0) {
                speed <- min(speed, ahead.speed); // Khớp tốc độ với xe trước
                speed <- max(speed_min, speed - deceleration);
            } else {
                speed <- 0.0; // Xe trước dừng hẳn thì mình cũng dừng
                patience <- patience - 1;
                
                // Chỉ đánh lái chuyển làn khi đã chờ hết kiên nhẫn
                if (patience <= 0) {
                    do try_change_lane();
                }
            }
        } else {
            // Đường thoáng thì phục hồi sự kiên nhẫn
            patience <- min(100, patience + 1); 
        }
    }

    action try_change_lane {
        if (is_merging) { return; }
        
        list<int> possible_lanes <- [];
        if (current_lane_index > 0) { possible_lanes << current_lane_index - 1; }
        if (current_lane_index < number_of_lanes - 1) { possible_lanes << current_lane_index + 1; }
        
        int best_lane <- current_lane_index;
        float min_required_gap <- car_length * 2.5; // ~8.75 units
		float max_gap <- min_required_gap;
        
        loop l over: possible_lanes {
	    list<car> cars_in_l <- car where (each.current_lane_index = l);
	    
	    car ahead_l <- (cars_in_l where (each.location.x > self.location.x)) 
	                   with_min_of(each.location.x - self.location.x);
	    car lag_l   <- (cars_in_l where (each.location.x <= self.location.x)) 
	                   with_min_of(self.location.x - each.location.x);
	    
	    // ← Khai báo gap ở đây, trước khi dùng
	    float gap      <- (ahead_l = nil) ? 999.0 : (ahead_l.location.x - self.location.x);
	    float rear_gap <- (lag_l   = nil) ? 999.0 : (self.location.x - lag_l.location.x);
	
	    if (gap > min_required_gap and rear_gap > min_required_gap and gap > max_gap) {
	        max_gap   <- gap;
	        best_lane <- l;
	    }
	}
        
        if (best_lane != current_lane_index) {
            target_lane_index  <- best_lane;
            current_lane_index <- best_lane;
            patience  <- 100; // Đã tìm được đường, reset kiên nhẫn
        } else {
            patience <- 100; // Không có đường nào thoát, đành chấp nhận chờ tiếp
        }
    }

    // ══════════════════════════════════════════════════════════
    // HÀM CẢM BIẾN DỮ LIỆU RL
    // ══════════════════════════════════════════════════════════
    action get_car_ahead type: car {
        list<car> ahead_cars <- (car - self) where (
            each.current_lane_index = self.current_lane_index and 
            each.location.x > self.location.x
        );
        if (empty(ahead_cars)) { return nil; }
        return ahead_cars with_min_of (each.location.x - self.location.x);
    }

    action get_car_behind type: car {
        list<car> behind_cars <- (car - self) where (
            each.current_lane_index = self.current_lane_index and 
            each.location.x < self.location.x
        );
        if (empty(behind_cars)) { return nil; }
        return behind_cars with_min_of (self.location.x - each.location.x);
    }

    action get_safe_distance type: float {
    if (speed <= 0.6) { return 5.0; }
    else if (speed <= 0.8) { return 8.0; }
    else { return 12.0; }
}

    action get_current_state type: list<float> {
        car ahead <- get_car_ahead();
        car behind <- get_car_behind();
        
        float dist_ahead  <- (ahead != nil) ? (ahead.location.x - self.location.x) : observation_max;
        float speed_ahead <- (ahead != nil) ? ahead.speed : speed_max;
        float dist_behind <- (behind != nil) ? (self.location.x - behind.location.x) : observation_max;
        float speed_behind<- (behind != nil) ? behind.speed : speed_max;
        
        return [speed, dist_ahead, speed_ahead, dist_behind, speed_behind];
    }

    action calculate_reward type: float {
        car ahead <- get_car_ahead();
        if (ahead != nil and (ahead.location.x - self.location.x) < collision_distance) {
            return -100.0; 
        }
        
        if (action_rl = 0) { return 3.0; }  
        if (action_rl = 1) { return 1.0; }  
        if (action_rl = 2) { return -1.0; } 
        if (action_rl = 3) { return -2.0; } 
        
        return 0.0;
    }

    // ── Hiển thị ─────────────────────────────────────────────
    aspect default {
        draw rectangle(car_length, car_width) rotate: heading color: color border: #black;
        float ch <- cos(heading);
        float sh <- sin(heading);
        point wp <- location + {(car_length / 4.0) * ch, (car_length / 4.0) * sh};
        draw rectangle(car_length * 0.25, car_width * 0.8)
            at: wp rotate: heading color: rgb(30, 30, 30);
        point l1 <- location + {(car_length/2)*ch - (car_width/3)*sh,
                                 (car_length/2)*sh + (car_width/3)*ch};
        point l2 <- location + {(car_length/2)*ch + (car_width/3)*sh,
                                 (car_length/2)*sh - (car_width/3)*ch};
        draw circle(0.22) at: l1 color: #yellow;
        draw circle(0.22) at: l2 color: #yellow;
    }
}

// ────────────────────────────────────────────────────────────
experiment TrafficSimulation type: gui {
    output {
        display MainView type: 2d axes: false background: rgb(46, 139, 87) {
            camera 'default' location: {100.0, 50.0, 155.0} target: {100.0, 50.0, 0.0};

            graphics "Infrastructure" {
                draw ramp_shape color: rgb(80, 80, 80);

                float top_edge  <- offset_y;
                float gore_y    <- offset_y + number_of_lanes * lane_width;
                float outer_y   <- gore_y + lane_width;

                draw line([{0.0, top_edge}, {road_length, top_edge}])
                    color: #white width: 2.0;
                draw line([{0.0, gore_y}, {road_length, gore_y}])
                    color: #white width: 2.0;
                draw line([{accel_start_x, outer_y}, {accel_end_x, outer_y}])
                    color: #white width: 2.0;
            }

            species road_segment aspect: default;
            species car          aspect: default;
        }
    }
}