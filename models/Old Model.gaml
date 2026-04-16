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

    int   nb_cars_max          <- 80;   // ← tăng từ 20 lên 45

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



    string driving_policy  <- "Greedy";

    float  discount_factor <- 0.99;

    float  learning_rate   <- 0.001;

    float  exp_rate        <- 1.0;

    float  exp_decay       <- 0.005;



    float  collision_distance <- 3.5;

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



    // ← interval 40, flip 0.5

    reflex spawn_ramp_cars when: every(20 #cycles) {

        if (flip(0.5)) {

            bool is_safe <- empty(car where (each.is_merging and each.location distance_to ramp_waypoints[0] < 15.0));

            if (is_safe) {

                create car number: 1 {

                    is_merging         <- true;

                    waypoint_index     <- 1;

                    location           <- ramp_waypoints[0];

                    heading            <- towards(location, ramp_waypoints[1]);

                    speed              <- rnd(0.30, 0.55);

                    color              <- #orange;

                    direction          <- 1;

                    current_lane_index <- number_of_lanes - 1;

                    target_lane_index  <- number_of_lanes - 1;

                }

            }

        }

    }



    // ← interval 15, spawn cả 3 làn

    reflex balance_traffic when: every(5 #cycles) {

        int total_cars <- length(car);

        if (total_cars < nb_cars_max) {

            int spawn_lane <- rnd(0, number_of_lanes - 1);

            float spawn_y  <- offset_y + (spawn_lane * lane_width) + (lane_width / 2.0);

            // SỬA Ở ĐÂY: Giảm vùng check an toàn xuống 8.0 để xe sinh ra sát nhau hơn

            bool is_safe   <- empty(car where (each.current_lane_index = spawn_lane and each.location.x < 8.0));

            if (is_safe) {

                create car number: 1 {

                bool is_obstacle <- flip(0.1);

                    // 1. Dùng trực tiếp flip(0.2) (20% xác suất) để gán tốc độ chậm

                    speed <- flip(0.05) ? rnd(0.1, 0.25) : rnd(0.5, speed_max);

                    

                    // 2. Dựa vào tốc độ vừa được gán để quyết định màu sắc (tốc độ <= 0.25 thì là xe tải màu đỏ)

                    color <- (speed <= 0.25) ? #red : one_of([#crimson, #dodgerblue, #silver, #white, #gold, #darkviolet]);

                    is_merging         <- false;

                    in_accel_zone      <- false;

                    current_lane_index <- spawn_lane;

                    target_lane_index  <- spawn_lane;

                    location           <- {0.0, spawn_y};

                    direction          <- 1;

                    heading            <- 0.0;

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

                    bool is_obstacle <- flip(0.02);

                    speed              <- is_obstacle ? 0.0 : rnd(0.4, speed_max);

                    color              <- is_obstacle ? #red : one_of([#crimson, #dodgerblue, #silver, #white, #gold, #darkviolet]);

                    

                    is_merging         <- false;

                    in_accel_zone      <- false;

                    current_lane_index <- i;

                    target_lane_index  <- i;

                    location           <- {10.0 + j * spacing + rnd(-3.0, 3.0), offset_y + (i * lane_width) + (lane_width / 2.0)};

                    direction          <- 1;

                    heading            <- 0.0;

                }

            }

        }



        // --- 2. THÊM MỚI: KHỞI TẠO XE TRÊN NHÁNH NHẬP LÀN (ON-RAMP) ---

        int nb_ramp_cars <- 3; // Số lượng xe ban đầu trên nhánh (bạn có thể tùy chỉnh)

        point wp0 <- ramp_waypoints[0];

        point wp1 <- ramp_waypoints[1];



        loop k from: 0 to: nb_ramp_cars - 1 {

            create car number: 1 {

                is_merging         <- true;

                in_accel_zone      <- false;

                waypoint_index     <- 1; // Hướng mục tiêu là điểm thứ 2 của ramp

                

                // Trải đều vị trí xe dọc theo đoạn thẳng từ wp0 đến wp1

                // Dùng (k / nb_ramp_cars) để chia tỉ lệ đoạn đường

                float ratio <- k / nb_ramp_cars;

                location <- {wp0.x + ratio * (wp1.x - wp0.x), wp0.y + ratio * (wp1.y - wp0.y)};

                

                heading            <- towards(location, wp1); // Quay đầu xe hướng về wp1

                speed              <- rnd(0.2, 0.4);          // Đi chậm hơn cao tốc một chút

                color              <- #orange;                // Màu cam để dễ phân biệt xe nhập làn

                direction          <- 1;

                

                // Về mặt logic, xe nhập làn sẽ hướng tới làn dưới cùng của cao tốc

                current_lane_index <- number_of_lanes - 1;

                target_lane_index  <- number_of_lanes - 1;

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

                draw line([{dx, dy}, {dx + 2.5, dy}]) color: #white width: 1.5;

                dx <- dx + 6.0;

            }

        }

    }

}



// ─────────────────────────────────────────────────────────────

species car skills: [moving] {

    int   current_lane_index;

    int   target_lane_index;

    int   direction;

    rgb   color;

bool is_merging_transition <- false;

    bool  is_merging;

    bool  in_accel_zone;

    int   waypoint_index;



    float car_length    <- 3.5;

    float car_width     <- 1.7;

    int   patience      <- 100;

    int   stuck_counter <- 0;



    float       reward_val;

    list<float> state;

    list<float> next_state;

    int         action_rl <- -1;


bool is_damaged <- false;

    int  damaged_timer <- 0;

    int  max_damaged_duration <- 150; // Thời gian xe đứng im sửa chữa

    float prob_damaged_car <- 0.001;  // Xác suất hỏng xe mỗi cycle 

    

    bool is_done <- false;

    

    reflex behave {

        if (is_merging) { do ramp_behavior(); }

        else            { do highway_rl_move(); }

    }



action ramp_behavior {

        float max_allowed_speed <- speed_max;

        

        // 1. VÁ LỖI RADAR: Phải nhìn cả xe đang chờ nhập làn VÀ xe đang trượt vào cao tốc

        list<car> blocking_ramp_cars <- (car - self) where (

            (each.is_merging or each.is_merging_transition) and 

            each.location.x > self.location.x

        );

        

        car ramp_ahead <- empty(blocking_ramp_cars) ? nil : blocking_ramp_cars with_min_of (each.location distance_to self.location);

        

        // 2. BỨC TƯỜNG VẬT LÝ CHO ĐƯỜNG NHÁNH

        if (ramp_ahead != nil) {

            float dist <- location distance_to ramp_ahead.location;

            if (dist <= car_length + 0.5) {

                max_allowed_speed <- 0.0; // Khoảng cách < 4.0m -> Phanh cháy đường, tuyệt đối không đè hình

            } else {

                max_allowed_speed <- max(0.0, dist - (car_length + 1.5));

            }

        }



        in_accel_zone <- (waypoint_index = 4);



        if (in_accel_zone) {

            speed <- min(speed_max, speed + acceleration);



            float accel_len <- accel_end_x - accel_start_x;

            float urgency   <- (accel_len > 0) ? min(1.0, max(0.0, (location.x - accel_start_x) / accel_len)) : 0.0;



            float req_front <- gap_front_max - urgency * (gap_front_max - gap_front_min);

            float req_rear  <- gap_rear_max  - urgency * (gap_rear_max  - gap_rear_min);



            list<car> bottom_lane_cars <- (car - self) where (not each.is_merging and each.current_lane_index = number_of_lanes - 1);

            car lead_car <- (bottom_lane_cars where (each.location.x >= self.location.x)) with_min_of (each.location.x - self.location.x);

            car lag_car  <- (bottom_lane_cars where (each.location.x <  self.location.x)) with_min_of (self.location.x - each.location.x);



            float gap_front <- (lead_car = nil) ? 999.0 : (lead_car.location.x - self.location.x);

            float gap_rear  <- (lag_car  = nil) ? 999.0 : (self.location.x - lag_car.location.x);



            if (lead_car != nil and gap_front < req_front) {

                speed <- max(speed_min, speed - deceleration * 0.6);

            } else if (lag_car != nil and gap_rear < req_rear) {

                speed <- min(speed_max, speed + acceleration * 0.8);

            }



            bool gap_ok <- (gap_front >= req_front) and (gap_rear >= req_rear);


bool absolute_safe <- (gap_front > car_length * 1.1) and (gap_rear > car_length * 1.1);


            if (gap_ok or (stuck_counter > 150 and absolute_safe)) {

                do execute_merge();

                return; // Đã nhập làn thì thoát nhánh này

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

    is_merging            <- false;

    in_accel_zone         <- false;

    stuck_counter         <- 0;

    is_merging_transition <- true;   // ← bật flag trượt mượt

    // KHÔNG snap location.y, KHÔNG reset heading ngay

    // Chỉ khai báo xe đã thuộc làn dưới cùng về mặt logic

    current_lane_index <- number_of_lanes - 1;

    target_lane_index  <- number_of_lanes - 1;

    speed              <- min(speed_max, speed);

    patience           <- 100;

}



    action highway_rl_move {

        // --- CƠ CHẾ XE HỎNG ĐỘT XUẤT ---

        if (not is_damaged and flip(prob_damaged_car)) {

            is_damaged <- true;

            damaged_timer <- max_damaged_duration;

            color <- #red;

            speed <- 0.0;

        }



        if (is_damaged) {

            // Xe hỏng không dừng khựng lại ngay, mà giảm tốc từ từ theo quán tính

            speed <- max(0.0, speed - deceleration * 0.5); 

            

            // Nếu đã chậm lại, xe bắt đầu đánh lái tấp vào làn cứu hộ (phía trên cùng)

            float rescue_y <- offset_y - (lane_width / 2.0);

            if (speed < 0.2 and location.y > rescue_y) {

                

                // --- SỬA Ở ĐÂY: CHECK "GƯƠNG CHIẾU HẬU" TRƯỚC KHI TẤP LỀ ---

                // Quét xem có xe nào đang chiếm không gian bên hông (hướng tấp lề) không

                bool clear_to_drift <- empty((car - self) where (

                    each.location.y < self.location.y + 0.5 and              // Gồm cả xe đang cọ sát

                    each.location.y > self.location.y - lane_width * 1.5 and // Quét lố sang làn bên cạnh một chút

                    abs(each.location.x - self.location.x) < car_length * 1.2 // Giới hạn chiều dọc (ngay sát sườn xe)

                ));



                if (clear_to_drift) {

                    // Đường trống -> Trôi ngang sang làn cứu hộ

                    location <- {location.x + speed, location.y - 0.08};

                    heading <- towards(location, {location.x + speed, location.y - 0.08});

                } else {

                    // Có xe bên hông cản đường -> Ngừng trôi ngang, giữ thẳng lái chờ thời cơ

                    heading <- 0.0; 

                }

                

            } else if (location.y <= rescue_y) {

                heading <- 0.0; // Đã vào bãi đỗ an toàn, thẳng lái

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



        // 1. TÁC TỬ CHỌN HÀNH ĐỘNG

        if (driving_policy = "Greedy") {

            action_rl <- 1;

        } else if (flip(exp_rate / (1 + exp_decay * cycle))) {

            action_rl <- rnd(0, 3);

        } else {

            action_rl <- 1;

        }



        if      (action_rl = 0) { speed <- min(speed_max, speed + acceleration); }

        else if (action_rl = 1) { speed <- min(speed_max, speed + acceleration * 0.3); }

        else if (action_rl = 2) { 

            speed <- max(speed_min, speed - deceleration); 

            // VÁ Ở ĐÂY: Trừ điểm kiên nhẫn khi xe bị ép phanh

            patience <- max(0, patience - 5); 

        }

        else if (action_rl = 3) { do try_change_lane(); }



        // 2. MÔI TRƯỜNG CAN THIỆP (GIỐNG NETLOGO)

        do handle_blocking_cars();

// --- BỨC TƯỜNG VẬT LÝ TUYỆT ĐỐI V2 (Quét mọi xe, không phụ thuộc làn) ---

        list<car> physical_blockers <- (car - self) where (

            each.location.x > self.location.x and 

            (each.location.x - self.location.x) < 10.0 and 

            abs(each.location.y - self.location.y) <= 2.5 // Bắt được cả xe đang lấp lửng nhập làn

        );

        

        if (!empty(physical_blockers)) {

            car ahead_final <- physical_blockers with_min_of (each.location.x - self.location.x);

            float dist_final <- ahead_final.location.x - self.location.x;

            if (dist_final <= car_length + 0.1) {

                speed <- 0.0; // Phanh cháy đường, không cho đè hình

            } else {

                float max_physical_speed <- max(0.0, dist_final - car_length - 0.1);

                speed <- min(speed, max_physical_speed);

            }

        }



        // 3. THỰC THI DI CHUYỂN

        float target_y <- offset_y + (current_lane_index * lane_width) + (lane_width / 2.0);

        float next_y   <- location.y;



        if (abs(location.y - target_y) > 0.1) {

            if (speed > 0) {

                float step_y <- speed * (is_merging_transition ? 0.6 : 0.3);

                if (location.y < target_y) { next_y <- min(target_y, location.y + step_y); }

                else                       { next_y <- max(target_y, location.y - step_y); }

                heading <- towards(location, {location.x + speed, next_y});

            } else {

                // SỬA Ở ĐÂY: Nếu đang chuyển làn mà phải dừng khựng lại, ép thẳng đầu xe để không dạt

                heading <- 0.0;

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

        

        // NẾU DONE -> Xóa xe khỏi bản đồ để không làm cản trở các xe khác đang học

        if (is_done) {

            if (self = selected_car) { selected_car <- nil; } // Bỏ focus trên UI

            do die; // Tiêu hủy tác tử

        }



    }

    action check_terminal_state {

        // 1. Va chạm (Game Over)

        list<car> nearby_cars <- (car - self) at_distance 4.0;

        if (!empty(nearby_cars)) {

            car closest_car <- nearby_cars with_min_of (self.location distance_to each.location);

            if (self.location distance_to closest_car.location < car_length * 0.9) {

                reward_val <- -100.0; // Phạt nặng

                is_done <- true;      // Chốt hạ ván học

                return;

            }

        }

        

        // 2. Về đích an toàn (Win Game)

        if (location.x >= road_length - 5.0) { 

            reward_val <- 50.0;   

            is_done <- true;      // Chốt hạ ván học

            return;

        }

    }

action handle_blocking_cars {

        car ahead <- get_car_ahead();

        if (ahead != nil) {

            float dist <- ahead.location.x - self.location.x;

            if (dist <= get_safe_distance()) {

                if (ahead.speed > 0.0) { // Sửa > 0.1 thành > 0.0

                    speed <- ahead.speed; 

                    speed <- max(speed_min, speed - deceleration);

                    patience <- max(0, patience - 1); 

                } else {

                    // Xe trước đứng im (hỏng) -> Phanh gấp và tự động môi trường ép đổi làn

                    speed <- max(0.0, speed - deceleration * 2.0);

                    do try_change_lane(); 

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

float current_target_y <- offset_y + (current_lane_index * lane_width) + (lane_width / 2.0);

        if (abs(location.y - current_target_y) > 0.1) { return; }

        list<int> possible_lanes <- [];

        if (current_lane_index > 0)                   { possible_lanes << current_lane_index - 1; }

        if (current_lane_index < number_of_lanes - 1) { possible_lanes << current_lane_index + 1; }



        int   best_lane        <- current_lane_index;

        float min_required_gap <- car_length * 1.2;

        float max_gap          <- min_required_gap;



        loop l over: possible_lanes {

            list<car> cars_in_l <- car where (each.current_lane_index = l);

            car ahead_l <- (cars_in_l where (each.location.x >= self.location.x)) with_min_of (each.location.x - self.location.x);

            car lag_l   <- (cars_in_l where (each.location.x <  self.location.x)) with_min_of (self.location.x - each.location.x);



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

            patience <- 100; // Đưa vào đây: Chỉ khi đổi làn THÀNH CÔNG mới hết bực mình!

        }

    }



    action get_car_ahead type: car {

        // Quét tất cả xe phía trước (x > self.x), nằm trong tầm nhìn observation_max

        // và dao động trục Y không quá 1 làn đường (để bắt được cả xe đang lấp lửng đè vạch)

        list<car> ahead_cars <- (car - self) where (

            each.location.x > self.location.x + 0.1 and

            each.location.x - self.location.x <= observation_max and

            abs(each.location.y - self.location.y) <= 2.5

        );

        if (empty(ahead_cars)) { return nil; }

        return ahead_cars with_min_of (each.location.x - self.location.x);

    }



    action get_car_behind type: car {

        list<car> behind_cars <- (car - self) where (

            each.location.x < self.location.x - 0.1 and

            self.location.x - each.location.x <= observation_max and

            abs(each.location.y - self.location.y) <= 2.5

        );

        if (empty(behind_cars)) { return nil; }

        return behind_cars with_min_of (self.location.x - each.location.x);

    }



    action get_safe_distance type: float {

        // Tốc độ cao cần khoảng cách phanh dài hơn để mô hình RL không bị phạt oan

        if (speed <= 0.4) { return 8.0; }        // ~ Tốc độ chậm

        else if (speed <= 0.7) { return 15.0; }  // ~ Tốc độ trung bình

        else { return 25.0; }                    // ~ Tốc độ cao

    }



    action get_current_state type: list<float> {

        car ahead  <- get_car_ahead();

        car behind <- get_car_behind();



        float raw_dist_ahead <- (ahead != nil) ? (ahead.location.x - self.location.x) : observation_max;

        float norm_dist_ahead <- min(1.0, raw_dist_ahead / observation_max);



        float norm_speed <- speed / speed_max;

        float norm_speed_ahead <- (ahead != nil) ? (ahead.speed / speed_max) : 1.0;



        float raw_dist_behind <- (behind != nil) ? (self.location.x - behind.location.x) : observation_max;

        float norm_dist_behind <- min(1.0, raw_dist_behind / observation_max);

        float norm_speed_behind <- (behind != nil) ? (behind.speed / speed_max) : 0.0;

        

        float norm_patience <- patience / 100.0;



        // --- BỔ SUNG MỚI: CẢM BIẾN QUÉT XE NHẬP LÀN (ON-RAMP SENSOR) ---

        // Tìm xe đang nhập làn (is_merging = true) ở phía trước mặt trong tầm nhìn

        list<car> merging_cars <- car where (each.is_merging and each.location.x > self.location.x and (each.location.x - self.location.x) <= observation_max);

        

        // Tính khoảng cách tới xe nhập làn gần nhất (dùng trục X để đồng bộ logic)

        float raw_dist_merger <- empty(merging_cars) ? observation_max : (merging_cars with_min_of (each.location.x - self.location.x)).location.x - self.location.x;

        float norm_dist_merger <- min(1.0, raw_dist_merger / observation_max);



        // Trả về 7 thông số

        return [norm_speed, norm_dist_ahead, norm_speed_ahead, norm_dist_behind, norm_speed_behind, norm_patience, norm_dist_merger];

    }



    action calculate_reward type: float {

        car ahead <- get_car_ahead();

        if (ahead != nil and (ahead.location.x - self.location.x) < collision_distance) { return -100.0; }

        if (action_rl = 0) { return  3.0; }

        if (action_rl = 1) { return  1.0; }

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

        if (self = selected_car) {

    car ahead <- get_car_ahead();

    if (ahead != nil) {

        draw line([location, ahead.location]) color: #cyan width: 2.0;

    }

}

    }

}



// ─────────────────────────────────────────────────────────────

experiment TrafficSimulation type: gui {



    action select_car {

        point     click_loc    <- #user_location;

        list<car> clicked_cars <- car where (each.location distance_to click_loc < 4.0);

        selected_car <- empty(clicked_cars) ? nil : clicked_cars[0];

    }



    output {

        display MainView type: 2d axes: false background: rgb(46, 139, 87) {

            camera 'default' location: {100.0, 50.0, 155.0} target: {100.0, 50.0, 0.0};



            graphics "Infrastructure" {

                draw ramp_shape color: rgb(80, 80, 80);



                float top_edge <- offset_y;

                float gore_y   <- offset_y + number_of_lanes * lane_width;

                float outer_y  <- gore_y + lane_width;

                float rescue_lane_y <- offset_y - lane_width;

                

                draw line([{0.0, top_edge}, {road_length, top_edge}]) color: #white width: 2.0;

                draw line([{0.0, gore_y},   {road_length, gore_y}])   color: #white width: 2.0;

                draw line([{accel_start_x, outer_y}, {accel_end_x, outer_y}]) color: #white width: 2.0;

                draw line([{0.0, rescue_lane_y}, {road_length, rescue_lane_y}]) color: #yellow width: 2.0;

            }



            species road_segment aspect: default;

            species car          aspect: default;



            event mouse_down action: select_car;



            overlay position: {20 #px, 20 #px} size: {420 #px, 185 #px}

                    background: rgb(30, 30, 30, 220) border: #white rounded: true {



                if (selected_car != nil and not dead(selected_car)) {

                    draw "🎯 SELECTED CAR (RL STATS)" at: {15 #px, 30 #px}

                        color: #magenta font: font("Arial", 14, #bold);

                    draw ("Speed: " + (selected_car.speed with_precision 2))

                        at: {15 #px, 60 #px} color: #white font: font("Arial", 12, #plain);

                    draw ("Action (0:Acc 1:Stay 2:Dec 3:Lane): " + selected_car.action_rl)

                        at: {15 #px, 85 #px} color: #yellow font: font("Arial", 12, #plain);

                    draw ("Reward: " + (selected_car.reward_val with_precision 2))

                        at: {15 #px, 110 #px}

                        color: (selected_car.reward_val < 0 ? #red : #lime)

                        font: font("Arial", 12, #bold);

                    draw ("Cars on road: " + length(car))

                        at: {15 #px, 135 #px} color: #white font: font("Arial", 12, #plain);

                    string st <- (selected_car.state != nil and length(selected_car.state) >= 7) ?

                        ("[" + (selected_car.state[0] with_precision 2) + ", " +

                               (selected_car.state[1] with_precision 1) + ", " +

                               (selected_car.state[2] with_precision 2) + ", " +

                               (selected_car.state[3] with_precision 1) + ", " +

                               (selected_car.state[4] with_precision 2) + ", " +

                               (selected_car.state[5] with_precision 2) + ", " +

                               (selected_car.state[6] with_precision 2) + "]")

                        : "[Waiting...]";

                    draw ("State: " + st) at: {15 #px, 158 #px} color: #cyan font: font("Arial", 12, #plain);

                } else {

                    draw "🖱️ Click on any car to track its RL data" at: {20 #px, 70 #px}

                        color: #white font: font("Arial", 14, #italic);

                    draw ("Cars on road: " + length(car)) at: {20 #px, 110 #px}

                        color: #lime font: font("Arial", 13, #bold);

                }

            }

        }

    }

}
