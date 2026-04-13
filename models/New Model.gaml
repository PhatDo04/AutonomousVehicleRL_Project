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

    float gap_front_max <- 7.0;   
    float gap_front_min <- 2.5;   
    float gap_rear_max  <- 6.0;   
    float gap_rear_min  <- 2.0;   

    string driving_policy <- "Greedy"; 
    float  discount_factor <- 0.99;
    float  learning_rate   <- 0.001;
    float  exp_rate        <- 1.0;
    float  exp_decay       <- 0.005;
    float  collision_distance <- 1.0;
    int    max_patience       <- 100;
    float  observation_max    <- 100.0;

    car selected_car <- nil;

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
            bool is_safe <- empty(car where (each.is_merging and each.location distance_to ramp_waypoints[0] < 6.0));
            if (is_safe) {
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
    }

    reflex balance_traffic {
        int active_cars <- length(car where (not each.is_merging));
        if (active_cars < nb_cars_max) {
            int spawn_lane <- rnd(0, number_of_lanes - 1);
            float spawn_y <- offset_y + (spawn_lane * lane_width) + (lane_width / 2.0);
            bool is_safe <- empty(car where (each.current_lane_index = spawn_lane and each.location.x < 20.0));
            if (is_safe) {
                create car number: 1 {
                    speed <- rnd(0.4, speed_max);
                    color <- one_of([#crimson, #dodgerblue, #silver, #white, #gold, #darkviolet]);
                    is_merging <- false;
                    in_accel_zone <- false;
                    current_lane_index <- spawn_lane;
                    target_lane_index <- spawn_lane;
                    location <- {0.0, spawn_y};
                    direction <- 1;
                    heading <- 0.0;
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
        int cars_per_lane <- int(nb_cars_max / number_of_lanes);
        loop i from: 0 to: number_of_lanes - 1 {
            loop j from: 0 to: cars_per_lane - 1 {
                create car number: 1 {
                    speed <- rnd(0.4, speed_max);
                    color <- one_of([#crimson, #dodgerblue, #silver, #white, #gold, #darkviolet]);
                    is_merging <- false;
                    in_accel_zone <- false;
                    current_lane_index <- i;
                    target_lane_index <- i;
                    float sx <- 20.0 + j * 20.0; 
                    float sy <- offset_y + (i * lane_width) + (lane_width / 2.0);
                    location <- {sx, sy};
                    direction <- 1;
                    heading <- 0.0;
                }
            }
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

    float speed_top;
    float reward_val;
    list<float> state;
    list<float> next_state;
    int action_rl <- -1;

    reflex behave {
        if (is_merging) {
            do ramp_behavior();
        } else {
            do highway_rl_move();
        }
    }

    action ramp_behavior {
        in_accel_zone <- (waypoint_index = 4);

        if (in_accel_zone) {
            speed <- min(speed_max, speed + acceleration);

            float urgency <- 0.0;
            float accel_len <- accel_end_x - accel_start_x;
            if (accel_len > 0.0) { urgency <- min(1.0, max(0.0, (location.x - accel_start_x) / accel_len)); }

            float req_front <- gap_front_max - urgency * (gap_front_max - gap_front_min);
            float req_rear  <- gap_rear_max  - urgency * (gap_rear_max  - gap_rear_min);

            list<car> bottom_lane_cars <- (car - self) where (not each.is_merging and each.current_lane_index = number_of_lanes - 1);
            car lead_car <- (bottom_lane_cars where (each.location.x >= self.location.x)) with_min_of (each.location.x - self.location.x);
            car lag_car  <- (bottom_lane_cars where (each.location.x <= self.location.x)) with_min_of (self.location.x - each.location.x);

            float gap_front <- (lead_car = nil) ? 999.0 : (lead_car.location.x - self.location.x);
            float gap_rear  <- (lag_car  = nil) ? 999.0 : (self.location.x - lag_car.location.x);

            if (lead_car != nil and gap_front < req_front) { speed <- max(speed_min, speed - deceleration * 0.6); } 
            else if (lag_car != nil and gap_rear < req_rear) { speed <- min(speed_max, speed + acceleration * 0.8); }

            bool gap_ok <- (gap_front >= req_front) and (gap_rear >= req_rear);

            if (gap_ok) {
                do execute_merge();
                return;
            }

        } else {
            speed <- min(speed_max * 0.7, speed + acceleration);
        }

        // ── FIX: TÍNH KHOẢNG CÁCH ĐƯỜNG CHÉO VÀ PHANH GẤP CHỐNG XUYÊN THẤU ──
        car ramp_ahead <- (car - self) where (each.is_merging and each.location.x > self.location.x) with_min_of(each.location.x - self.location.x);
        if (ramp_ahead != nil) {
            float real_dist <- location distance_to ramp_ahead.location;
            if (real_dist < car_length + 0.5) {
                speed <- 0.0; // Phanh gấp
            } else if (real_dist < 8.0) {
                speed <- min(speed, ramp_ahead.speed);
                speed <- max(0.0, speed - deceleration);
            }
        }

        if (waypoint_index < length(ramp_waypoints)) {
            point tp <- ramp_waypoints[waypoint_index];
            heading  <- towards(location, tp);
            location <- {location.x + speed * cos(heading), location.y + speed * sin(heading)};

            if (location distance_to tp < max(speed * 2.0, 1.0)) {
                waypoint_index <- waypoint_index + 1;
                if (waypoint_index >= length(ramp_waypoints)) { do execute_merge(); }
            }
        }
    }

    action execute_merge {
        is_merging         <- false;
        in_accel_zone      <- false;
        current_lane_index <- number_of_lanes - 1;
        target_lane_index  <- number_of_lanes - 1;
        speed              <- min(speed_max, speed);
        patience           <- 100;
    }

    action highway_rl_move {
        state <- get_current_state();
        
        float epsilon <- exp_rate / (1 + exp_decay * cycle);
        if (driving_policy = "Greedy") {
		    action_rl <- 1; 
		} else if (flip(epsilon)) {
		    action_rl <- rnd(0, 3); 
		} else {
		    action_rl <- 1; 
		}

        if (action_rl = 0) { speed <- min(speed_max, speed + acceleration); }
        else if (action_rl = 1) {} 
        else if (action_rl = 2) { speed <- max(speed_min, speed - deceleration); }
        else if (action_rl = 3) { do try_change_lane(); }

        do handle_blocking_cars();
        
        float target_y <- offset_y + (current_lane_index * lane_width) + (lane_width / 2.0);
        float next_y <- location.y;
        
        if (abs(location.y - target_y) > 0.1) {
            if (speed > 0) {
                float step_y <- speed * 0.3; 
                if (location.y < target_y) { next_y <- min(target_y, location.y + step_y); }
                else { next_y <- max(target_y, location.y - step_y); }
                heading <- towards(location, {location.x + speed, next_y});
            }
        } else {
            next_y <- target_y;
            heading <- 0.0; 
        }

        location <- {location.x + speed, next_y};
        
        if (location.x > road_length) { 
            if (self = selected_car) { selected_car <- nil; }
            do die; 
        }

        reward_val <- calculate_reward();
        next_state <- get_current_state();
    }
    
    // ── FIX: THÊM LỆNH PHANH GẤP TRÊN CAO TỐC ĐỂ KHÔNG ĐÈ NHAU ──
    action handle_blocking_cars {
        car ahead <- get_car_ahead();
        if (ahead != nil) {
            float dist <- ahead.location.x - self.location.x;
            
            if (dist < car_length + 0.5) { // Quá sát đít -> Đạp lút phanh
                speed <- 0.0;
            } 
            else if (dist <= get_safe_distance()) { // Trong vùng cần giảm tốc
                if (ahead.speed > 0) {
                    speed <- min(speed, ahead.speed); 
                    speed <- max(speed_min, speed - deceleration);
                } else {
                    speed <- max(0.0, speed - deceleration * 2.0); 
                    patience <- patience - 1;
                    if (patience <= 0) { do try_change_lane(); }
                }
            } else {
                patience <- min(100, patience + 1); 
            }
        } else {
            patience <- min(100, patience + 1); 
        }
    }

    action try_change_lane {
        if (is_merging) { return; }
        
        list<int> possible_lanes <- [];
        if (current_lane_index > 0) { possible_lanes << current_lane_index - 1; }
        if (current_lane_index < number_of_lanes - 1) { possible_lanes << current_lane_index + 1; }
        
        int best_lane <- current_lane_index;
        float min_required_gap <- car_length * 2.5; 
		float max_gap <- min_required_gap;
        
        loop l over: possible_lanes {
            list<car> cars_in_l <- car where (each.current_lane_index = l);
            car ahead_l <- (cars_in_l where (each.location.x >= self.location.x)) with_min_of(each.location.x - self.location.x);
            car lag_l   <- (cars_in_l where (each.location.x <= self.location.x)) with_min_of(self.location.x - each.location.x);
            
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
            patience  <- 100; 
        } else {
            patience <- 100; 
        }
    }

    action get_car_ahead type: car {
        list<car> ahead_cars <- (car - self) where (each.current_lane_index = self.current_lane_index and each.location.x > self.location.x);
        if (empty(ahead_cars)) { return nil; }
        return ahead_cars with_min_of (each.location.x - self.location.x);
    }

    action get_car_behind type: car {
        list<car> behind_cars <- (car - self) where (each.current_lane_index = self.current_lane_index and each.location.x < self.location.x);
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
        if (ahead != nil and (ahead.location.x - self.location.x) < collision_distance) { return -100.0; }
        if (action_rl = 0) { return 3.0; }  
        if (action_rl = 1) { return 1.0; }  
        if (action_rl = 2) { return -1.0; } 
        if (action_rl = 3) { return -2.0; } 
        return 0.0;
    }

    aspect default {
        rgb draw_color <- (self = selected_car) ? #magenta : color;
        
        draw rectangle(car_length, car_width) rotate: heading color: draw_color border: #black;
        float ch <- cos(heading); float sh <- sin(heading);
        point wp <- location + {(car_length / 4.0) * ch, (car_length / 4.0) * sh};
        draw rectangle(car_length * 0.25, car_width * 0.8) at: wp rotate: heading color: rgb(30, 30, 30);
        point l1 <- location + {(car_length/2)*ch - (car_width/3)*sh, (car_length/2)*sh + (car_width/3)*ch};
        point l2 <- location + {(car_length/2)*ch + (car_width/3)*sh, (car_length/2)*sh - (car_width/3)*ch};
        draw circle(0.22) at: l1 color: #yellow;
        draw circle(0.22) at: l2 color: #yellow;
    }
}

// ────────────────────────────────────────────────────────────
experiment TrafficSimulation type: gui {
    
    action select_car {
        point click_loc <- #user_location;
        list<car> clicked_cars <- car where (each.location distance_to click_loc < 4.0);
        
        if (!empty(clicked_cars)) {
            selected_car <- clicked_cars[0]; 
        } else {
            selected_car <- nil; 
        }
    }

    output {
        display MainView type: 2d axes: false background: rgb(46, 139, 87) {
            camera 'default' location: {100.0, 50.0, 155.0} target: {100.0, 50.0, 0.0};

            graphics "Infrastructure" {
                draw ramp_shape color: rgb(80, 80, 80);
                float top_edge  <- offset_y;
                float gore_y    <- offset_y + number_of_lanes * lane_width;
                float outer_y   <- gore_y + lane_width;
                draw line([{0.0, top_edge}, {road_length, top_edge}]) color: #white width: 2.0;
                draw line([{0.0, gore_y}, {road_length, gore_y}]) color: #white width: 2.0;
                draw line([{accel_start_x, outer_y}, {accel_end_x, outer_y}]) color: #white width: 2.0;
            }

            species road_segment aspect: default;
            species car          aspect: default;
            
            event mouse_down action: select_car;
            
            overlay position: { 20 #px, 20 #px } size: { 400 #px, 160 #px } background: rgb(30, 30, 30, 220) border: #white rounded: true {
                if (selected_car != nil and not dead(selected_car)) {
                    draw "🎯 SELECTED CAR (RL STATS)" at: {15 #px, 30 #px} color: #magenta font: font("Arial", 14, #bold);
                    
                    string s_speed <- "Speed: " + (selected_car.speed with_precision 2);
                    string s_action <- "Action (0:Acc, 1:Stay, 2:Dec, 3:Lane): " + selected_car.action_rl;
                    string s_reward <- "Reward: " + (selected_car.reward_val with_precision 2);
                    
                    string st <- selected_car.state != nil ? 
                                 "[" + (selected_car.state[0] with_precision 2) + ", " + 
                                       (selected_car.state[1] with_precision 1) + ", " + 
                                       (selected_car.state[2] with_precision 2) + ", " + 
                                       (selected_car.state[3] with_precision 1) + ", " + 
                                       (selected_car.state[4] with_precision 2) + "]" 
                                 : "[Waiting for data...]";
                    string s_state <- "State: " + st;

                    draw s_speed at: {15 #px, 60 #px} color: #white font: font("Arial", 12, #plain);
                    draw s_action at: {15 #px, 85 #px} color: #yellow font: font("Arial", 12, #plain);
                    draw s_reward at: {15 #px, 110 #px} color: (selected_car.reward_val < 0 ? #red : #green) font: font("Arial", 12, #bold);
                    draw s_state at: {15 #px, 135 #px} color: #cyan font: font("Arial", 12, #plain);
                } else {
                    draw "🖱️ Click on any car to track its RL data" at: {20 #px, 80 #px} color: #white font: font("Arial", 14, #italic);
                }
            }
        }
    }
}