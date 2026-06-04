# GIẢI THÍCH MÃ NGUỒN CHO NGƯỜI MỚI HOÀN TOÀN

> Tài liệu này viết cho người **chưa biết gì** về lập trình, AI hay xe tự hành. Đọc từ trên xuống, mỗi phần xây trên phần trước. Mục tiêu: đọc xong bạn hiểu **dự án này làm gì, hoạt động ra sao, và mỗi file để làm gì**. Bản kỹ thuật chi tiết: [BAO_CAO_MA_NGUON_CHI_TIET.md](BAO_CAO_MA_NGUON_CHI_TIET.md).

---

## 1. Dự án này giải quyết chuyện gì?

Bạn hình dung một **đường cao tốc** có 3 làn. Bên hông có một **nhánh nhập làn** (giống lối lên cao tốc ngoài đời — gọi là *on-ramp*). Một chiếc xe đi từ nhánh phụ này phải **hòa vào dòng xe đang chạy nhanh** trên cao tốc mà **không đâm ai**, **không phải dừng khựng**, và **không làm kẹt xe phía sau**.

Đây là tình huống khó vì:
- Xe nhập làn phải canh **đúng khoảng trống** giữa 2 xe trên cao tốc.
- Các xe trên cao tốc nên **nhường đường** một chút.
- Mọi người phải phối hợp, nếu không sẽ va chạm hoặc gây "sóng phanh" lan ngược (ùn tắc).

**Dự án dạy cho các xe tự "học" cách làm việc này một cách khéo léo** — bằng một kỹ thuật trí tuệ nhân tạo tên là **Học Tăng Cường (Reinforcement Learning)**.

---

## 2. "Học Tăng Cường" là gì? (ví dụ huấn luyện chó)

Hãy nghĩ tới cách dạy một chú chó:
- Chó **làm hành động** (ngồi, nằm, sủa…).
- Bạn **thưởng** (cho bánh) khi nó làm đúng, **không thưởng/phạt nhẹ** khi sai.
- Lặp lại nhiều lần, chó dần **học** được hành động nào mang lại nhiều bánh nhất.

Máy tính học **y hệt vậy**:
- Chiếc xe (gọi là **agent** — "tác tử") **chọn hành động** (tăng tốc, giảm tốc, nhập làn…).
- Môi trường **chấm điểm** (gọi là **reward** — "phần thưởng"): nhập làn an toàn được **+200 điểm**, đâm xe bị **−100 điểm**.
- Lặp lại **hàng trăm nghìn lần**, xe dần học được chiến lược ghi nhiều điểm nhất = lái khéo.

> Toàn bộ "trí khôn" của xe nằm trong một **mạng nơ-ron** (network) — bạn cứ hiểu nó là một "bộ não số" gồm nhiều con số, được tinh chỉnh dần qua mỗi lần thử.

**"Đa tác tử" (Multi-Agent):** ở đây không chỉ 1 xe mà **4 xe cùng học một lúc** (1 xe nhập làn + 3 xe cao tốc). Chúng ảnh hưởng lẫn nhau, nên phải học cách **phối hợp**.

---

## 3. Các "nhân vật" trong dự án

| Nhân vật | Là gì | Nhiệm vụ |
|---|---|---|
| 🟣 **merging_0** | Xe màu tím, đi từ nhánh phụ | Nhập vào cao tốc an toàn |
| 🔵 **highway_0, highway_1, highway_2** | 3 xe màu xanh trên cao tốc | Chạy đều + nhường chỗ cho xe tím |
| ⚪ **Xe nền (NPC)** | Các xe khác chạy theo luật cứng | Tạo "giao thông" thật, không tự học |
| 🖥️ **GAMA** | Phần mềm mô phỏng | Đóng vai **"thế giới giả lập"** — vẽ đường, di chuyển xe, tính va chạm |
| 🐍 **Python** | Bộ code AI | Đóng vai **"bộ não học"** — quyết định mỗi xe nên làm gì |

**Cách 2 bên nói chuyện:** GAMA (thế giới) và Python (bộ não) trao đổi qua một "đường dây" (socket). Mỗi nhịp:
1. GAMA gửi cho Python biết: *"xe đang ở đâu, tốc độ bao nhiêu, xe khác cách bao xa"* (gọi là **observation** — quan sát).
2. Python suy nghĩ rồi trả lời: *"hãy tăng tốc"* / *"nhập làn đi"* (gọi là **action** — hành động).
3. GAMA thực hiện, tính xem được/mất bao nhiêu điểm (**reward**), rồi lặp lại.

---

## 4. Một "ván" (episode) diễn ra thế nào?

Một **episode** là một lượt chơi, giống một ván game:

```
BẮT ĐẦU: xe tím xuất hiện ở đầu nhánh phụ, 3 xe xanh chạy trên cao tốc
   │
   ├─ Mỗi nhịp: 4 xe cùng quan sát → chọn hành động → di chuyển → nhận điểm
   │
KẾT THÚC khi một trong các điều sau xảy ra với xe tím:
   • Nhập làn xong + chạy hết đường   → "success" (+200 điểm) 🎉
   • Đâm vào xe khác                   → "collision" (−100 điểm) 💥
   • Đi hết nhánh mà chưa nhập được    → "failed_merge" (−50 điểm) 😞
   • Quá lâu chưa xong                 → "timeout" ⏰
```

Máy chơi đi chơi lại episode **hàng trăm nghìn lần**. Lúc đầu xe lái rất ngu (đâm liên tục), nhưng nhờ cơ chế thưởng/phạt, nó **giỏi dần lên**.

---

## 5. Xe "nhìn thấy" gì và "làm được" gì?

### Xe nhìn thấy (Observation) — 15 con số
Mỗi nhịp, xe nhận một danh sách 15 con số mô tả tình hình, ví dụ với xe tím:
- Tốc độ của chính nó.
- Đã đi được bao xa trên nhánh.
- Còn cách điểm phải nhập làn bao xa.
- Khoảng trống phía trước/sau trên làn cần nhập có đủ rộng không.
- Khoảng trống đó có **đủ an toàn** để chen vào không (1 = an toàn, 0 = không).
- Mức độ "khẩn cấp" (càng gần cuối nhánh càng gấp).

### Xe làm được (Action) — chọn 1 trong 5
Với **xe tím (nhập làn):** 0=giảm tốc, 1=giữ tốc, 2=tăng tốc, 3=**nhập làn**, 4=chờ.
Với **xe xanh (cao tốc):** 0=tăng tốc, 1=giữ tốc, 2=giảm tốc, 3=rẽ trái, 4=rẽ phải.

> Lưu ý vui: cùng số 0 nhưng xe tím là "giảm tốc" còn xe xanh là "tăng tốc" — vì 2 loại xe có vai trò khác nhau. Bộ não phân biệt được nhờ một "thẻ tên" gắn kèm quan sát.

### Điểm thưởng/phạt (Reward) — kim chỉ nam
- Nhập làn thành công: **+200** (phần thưởng lớn nhất → đích nhắm).
- Va chạm: **−100** (phạt nặng → tránh xa).
- Chọn "nhập làn" đúng lúc (trong vùng tăng tốc + có khoảng trống an toàn): **+25 đến +50**.
- Cố nhập khi không an toàn: **−2** (phạt nhẹ để học chọn đúng thời điểm).
- Đứng ì, do dự: bị trừ điểm để khuyến khích tiến lên.

Chính bộ điểm này "nắn" hành vi của xe theo hướng ta muốn: **nhanh nhưng an toàn**.

---

## 6. Hai thuật toán được so sánh

Có nhiều "kiểu học" khác nhau. Dự án thử **2 kiểu** và so xem cái nào tốt hơn:

- **MAPPO** (dựa trên thuật toán PPO): kiểu học cẩn thận, ổn định.
- **MAA2C** (dựa trên A2C): kiểu học đơn giản hơn.

Và so cả hai với **2 chuẩn không học** (không tập, chỉ làm theo luật cố định):
- **Greedy** ("tham lam"): xe cứ tăng tốc + cố nhập làn sớm + lách làn, không tính trước. Là "vạch mốc" xem AI có giỏi hơn luật tay đơn giản không.
- **Random** ("ngẫu nhiên"): bấm bừa — đóng vai "sàn" tệ nhất để biết bài toán không phải tự dễ.

> Có một kiểu thứ ba là **DQN** đã bị **loại bỏ**, vì nó không hợp với bài toán nhiều xe cùng học (kết quả không ổn định).

**"Khiên an toàn" của thế giới (rất quan trọng):** bản thân thế giới GAMA có sẵn vài "lưới an toàn" — ví dụ tự ép xe phanh khi quá gần xe trước, hoặc chỉ cho nhập làn khi đủ khoảng trống. Dự án làm một thí nghiệm **A/B**: chạy y hệt nhưng **bật khiên** vs **tắt khiên** (áp đều cho mọi xe) để xem ai thật sự *biết* lái an toàn, ai chỉ dựa vào khiên.

**Mẹo hay trong dự án:** trong lúc học, máy lưu lại "ảnh chụp" bộ não ở các mốc (sau 40k, 80k, … lần thử). Cuối cùng nó **chọn ảnh chụp tốt nhất** chứ không nhất thiết lấy bản cuối — vì đôi khi học thêm lại dở đi. (Gọi là **checkpoint** và **select-best-checkpoint**.)

---

## 7. Cách đo "giỏi hay không"

| Chỉ số | Nghĩa đời thường | Tốt khi |
|---|---|---|
| **Success rate** | % số ván nhập làn thành công | Cao |
| **Collision rate** | % số ván bị đâm | Thấp |
| **Shockwave (sóng lùi)** | Mức độ "giật cục" của dòng xe (đo bằng độ dao động tốc độ) | Thấp = dòng mượt |
| **Throughput** | Tỷ lệ xe nhập làn trót lọt | Cao |
| **Reward** | Tổng điểm trung bình mỗi ván | Cao |

**Kết quả chính của dự án** (thử ở 3 mức đông xe: thưa/vừa/đông × bật/tắt khiên):

- **MAA2C giỏi & ổn định nhất** — luôn ~82–89% thành công ở mọi điều kiện, ít thất thường. Nhỉnh hơn MAPPO (hay dao động hơn).
- **Phát hiện thú vị nhất:** khi **bật khiên**, Greedy đạt tới 96% — nhìn còn hơn AI! Nhưng khi **tắt khiên** ở đường đông, Greedy **sụp còn 0%** (đâm liên tục), trong khi AI vẫn giữ ~85%.
  → Nghĩa là: **Greedy không thật sự biết lái an toàn — nó "ăn theo" lưới an toàn của thế giới.** Còn AI thì *đã học được* cách tự tránh va chạm nên vẫn ổn khi gỡ lưới. **Đường càng đông, khiên càng quan trọng, và lợi thế của AI càng rõ.**
- Random: 0% (luôn đâm) — đúng vai "sàn".

> Một câu: *bỏ lưới an toàn ra, chỉ có xe-đã-học mới còn lái được; xe lái-theo-luật-tham-lam thì bó tay.*

---

## 8. Các file trong dự án để làm gì? (giải thích dân dã)

### Phần "thế giới" (GAMA)
- **`models/Main_Traffic.gaml`** — Bản thiết kế cả thế giới: vẽ đường, nhánh nhập làn, sinh xe, di chuyển xe, tính va chạm, chấm điểm. To nhất, quan trọng nhất.

### Phần "bộ não" (Python, thư mục `rl/`)
| File | Ví như… |
|---|---|
| `config.py` | Bảng cấu hình chung (các con số, đường dẫn) |
| `marl_env.py` | "Phiên dịch viên" nối thế giới GAMA với bộ não Python |
| `centralized_policy.py` | Kiến trúc bộ não (phần ra quyết định + phần đánh giá) |
| `train_marl.py` | "Lớp huấn luyện" — nơi xe tập đi tập lại để giỏi lên |
| `evaluate_marl.py` | "Phòng thi" — kiểm tra xe đã giỏi tới đâu |
| `baselines.py` | "Thí sinh đối chứng" không học: Greedy (tham lam) + Random (bừa) để so sánh |
| `metrics.py` | "Sổ điểm" — ghi lại kết quả mỗi ván ra file |
| `run_experiments.py` | "Nhạc trưởng" — bấm 1 nút là chạy toàn bộ: tập → thi → vẽ biểu đồ |
| `analysis.py` | Lập bảng tổng kết so sánh |
| `plots.py` | Vẽ 12 biểu đồ minh họa |
| `scenario_utils.py` | Chỉnh mật độ giao thông (thưa/vừa/đông) + bật/tắt "khiên an toàn" trước khi chạy |
| `gama_compat.py` | "Miếng vá" giúp GAMA và Python tương thích trơn tru |
| `gama_episode_reset.py` | Lo việc "bắt đầu ván mới" cho sạch sẽ |

### Công cụ phụ (chạy tay khi cần)
- `smoke_test_env.py` — kiểm tra nhanh "mọi thứ có kết nối ổn không" trước khi tập lâu.
- `diagnose_marl.py` — soi kỹ 1 ván để tìm lỗi.
- `model_registry.py` — liệt kê các bộ não đã lưu.

---

## 9. Quy trình chạy (từ đầu đến cuối)

```
1) Bật "thế giới" GAMA ở chế độ không màn hình (headless), nghe ở cổng 1001.
2) Chạy "nhạc trưởng":  python rl/run_experiments.py --preset thesis
3) Máy tự động:
      a. Cho xe Greedy chạy thử (lấy mốc so sánh)
      b. Huấn luyện MAPPO và MAA2C, mỗi cái 5 lần (5 "seed" khác nhau cho khách quan)
      c. Chọn bộ não tốt nhất của mỗi lần
      d. Thi 50 ván để đo điểm
      e. Vẽ 12 biểu đồ + lập bảng so sánh
4) Xem kết quả trong thư mục outputs/
```

**Kết quả lưu gọn theo từng lần chạy:** mỗi lần chạy tạo một thư mục riêng có dán nhãn thời gian (vd `thesis_20260530_133739`), nên **chạy lại không đè kết quả cũ**.

---

## 10. Những điểm "dễ hiểu nhầm" — làm rõ luôn

- **"200k" trong tên file model KHÔNG phải tốc độ hay điểm** — nó là "đã tập 200.000 lần". Còn nội dung bên trong là bộ não **tốt nhất đã chọn** (có thể là ảnh chụp ở mốc 120k).
- **Mỗi lần "tập lại" là tập từ đầu**, không phải học tiếp bộ não cũ. Làm vậy để mỗi thí nghiệm khách quan, lặp lại được.
- **Chế độ đánh giá có 2 kiểu:** *deterministic* (luôn chọn nước tốt nhất) cho số cao nhưng "dễ"; báo cáo dùng *stochastic* (có ngẫu nhiên) — khắt khe, thực tế hơn (vì vậy số ~82–89% chứ không phải 100%).
- **Greedy nhìn cao (96%) khi CÓ khiên KHÔNG có nghĩa AI thua** — đó là vì Greedy ăn theo lưới an toàn. Bỏ lưới ra (nhất là đường đông), Greedy sụp về 0% còn AI vẫn ~85%. Phải so ở **cùng điều kiện** mới công bằng.

---

## 11. Tóm tắt một câu

> Dự án dựng một **đường cao tốc giả lập** trong GAMA, rồi dùng **Học Tăng Cường đa tác tử** (Python + Stable-Baselines3) để dạy **4 chiếc xe tự học cách nhập làn an toàn và phối hợp nhường đường**, sau đó **so sánh** 2 thuật toán học (MAPPO vs MAA2C) với 2 chuẩn không học (Greedy, Random) trên nhiều mức đông xe và có/không "lưới an toàn" — qua đó cho thấy **xe-đã-học bền vững hơn luật tham lam, nhất là khi gỡ lưới an toàn ở đường đông**.

---

*Muốn đi sâu vào kỹ thuật (MDP, siêu tham số, từng wrapper…): đọc [BAO_CAO_MA_NGUON_CHI_TIET.md](BAO_CAO_MA_NGUON_CHI_TIET.md). Hướng dẫn cài đặt & chạy: [README.md](README.md).*
