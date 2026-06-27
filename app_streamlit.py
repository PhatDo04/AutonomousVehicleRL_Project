"""Bảng điều khiển Streamlit cho dự án AutonomousVehicleRL.

⚠️ VENV RIÊNG: streamlit cần websockets>=12, nhưng gama-client cần websockets~=10.3
(xung đột!). Vì vậy CHẠY UI BẰNG VENV RIÊNG, còn training chạy bằng .venv chính
(qua cfg["venv_python"]). Setup 1 lần:

    python -m venv .venv-ui
    .venv-ui\\Scripts\\pip install streamlit
    .venv-ui\\Scripts\\streamlit run app_streamlit.py

UI chỉ dùng socket để kiểm GAMA + subprocess gọi .venv chính để train/eval → KHÔNG
import gama-client, nên không vướng websockets. Lõi logic ở rl/controller.py.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl import controller as C  # noqa: E402

st.set_page_config(page_title="AV-RL Control Panel", page_icon="🚗", layout="wide")

# --- state ---
if "cfg" not in st.session_state:
    st.session_state.cfg = C.load_config()
if "job" not in st.session_state:
    st.session_state.job = None  # type: C.Job | None

cfg = st.session_state.cfg


def _job_running() -> bool:
    j = st.session_state.job
    return j is not None and C.proc_alive(j.pid)


def pick_zip(label: str, zips: list[str], key: str):
    """Selectbox model .zip kèm ô LỌC theo tên (đỡ cuộn list dài). Hiện <thư mục>/<file>."""
    q = st.text_input(f"🔎 Lọc {label}", key=f"{key}_q",
                      placeholder="gõ lọc theo tên: vd a2c, planb2_k07, shield_on, seed0…")
    filt = [z for z in zips if q.lower() in z.lower()] if q else zips
    st.caption(f"{len(filt)}/{len(zips)} model khớp")
    if not filt:
        st.warning("Không có zip khớp bộ lọc.")
        return None
    return st.selectbox(label, filt, key=key,
                        format_func=lambda p: "/".join(Path(p).parts[-2:]))


@st.dialog(" ", width="large")
def _png_dialog(path: str, name: str):
    """Dialog phóng to 1 ảnh — co vừa TRỌN khung (giữ tỉ lệ, không tràn)."""
    st.markdown(f"**{name}**")
    data = base64.b64encode(Path(path).read_bytes()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{data}" '
        f'style="display:block;margin:auto;max-width:100%;max-height:78vh;'
        f'object-fit:contain;border-radius:6px;" />',
        unsafe_allow_html=True,
    )


def png_gallery(pngs, key_prefix: str, cols_per_row: int = 3):
    """Đổ ảnh dạng thumbnail nhỏ trong lưới; bấm nút 🔍 dưới ảnh → mở dialog xem to."""
    st.caption("Bấm 🔍 dưới mỗi ảnh để xem to.")
    pngs = [str(p) for p in pngs]
    for i in range(0, len(pngs), cols_per_row):
        row = st.columns(cols_per_row)
        for col, png in zip(row, pngs[i:i + cols_per_row]):
            name = Path(png).name
            with col:
                st.image(png, use_container_width=True)
                if st.button(f"🔍 {name}", key=f"{key_prefix}_{i}_{name}", use_container_width=True):
                    _png_dialog(png, name)


# --- header + GAMA status badge ---
st.title("🚗 AutonomousVehicleRL — Control Panel")
gs = C.gama_status(cfg) if C.config_ok(cfg) else {"listening": False, "pid": None, "port": cfg.get("gama_port")}
c1, c2, c3 = st.columns(3)
c1.metric("GAMA port", gs["port"], "🟢 LISTENING" if gs["listening"] else "🔴 OFF")
c2.metric("Config", "✓ OK" if C.config_ok(cfg) else "✗ thiếu path")
c3.metric("Job", st.session_state.job.kind if _job_running() else "—",
          "đang chạy" if _job_running() else "rảnh")

tabs = st.tabs(["⚙️ Config", "🟢 GAMA", "🚀 Train", "📡 Monitor",
                "📊 Evaluate", "🧪 Experiments", "📈 Results", "▶️ Play GUI", "🎓 Thesis"])

# ===================== CONFIG =====================
with tabs[0]:
    st.subheader("Cấu hình đường dẫn & port")
    v = C.validate_config(cfg)
    cfg["gama_bat_path"] = st.text_input(
        f"{'✓' if v['gama_bat_path'] else '✗'} gama-headless.bat", cfg.get("gama_bat_path", ""))
    cfg["gama_headless_dir"] = st.text_input(
        f"{'✓' if v['gama_headless_dir'] else '✗'} thư mục GAMA headless (auto từ .bat nếu trống)",
        cfg.get("gama_headless_dir", ""))
    cfg["venv_python"] = st.text_input(
        f"{'✓' if v['venv_python'] else '✗'} venv python.exe", cfg.get("venv_python", ""))
    cfg["project_dir"] = st.text_input(
        f"{'✓' if v['project_dir'] else '✗'} thư mục dự án", cfg.get("project_dir", ""))
    cfg["gama_port"] = int(st.number_input("GAMA socket port", value=int(cfg.get("gama_port", 1001)), step=1))
    if st.button("💾 Lưu cấu hình", type="primary"):
        if not cfg.get("gama_headless_dir") and cfg.get("gama_bat_path"):
            cfg["gama_headless_dir"] = str(Path(cfg["gama_bat_path"]).parent)
        C.save_config(cfg)
        st.success(f"Đã lưu vào {C.CONFIG_PATH}")
        st.rerun()

# ===================== GAMA =====================
with tabs[1]:
    st.subheader("GAMA headless server")
    if not C.config_ok(cfg):
        st.warning("Chưa đủ cấu hình — vào tab Config.")
    cc1, cc2, cc3, cc4 = st.columns(4)
    if cc1.button("▶️ Bật GAMA"):
        st.info(C.gama_start(cfg)["msg"])
    if cc2.button("⏹️ Tắt GAMA"):
        st.info(C.gama_stop(int(cfg["gama_port"]))["msg"])
    if cc3.button("🔁 Restart"):
        C.gama_stop(int(cfg["gama_port"]))
        st.info(C.gama_start(cfg)["msg"])
    if cc4.button("🔄 Trạng thái"):
        st.rerun()
    st.write("**Trạng thái:**", "🟢 LISTENING" if gs["listening"] else "🔴 OFF",
             f"(PID {gs['pid']})" if gs["pid"] else "")
    st.divider()
    st.write("**Soi lỗi GAMA (bắt nhiễu):**")
    err = C.gama_check_errors()
    e1, e2, e3 = st.columns(3)
    e1.metric("NullPointerException", err["npe"])
    e2.metric("Exception", err["exception"])
    e3.metric("stderr bytes", err["stderr_bytes"])
    for s in err["samples"]:
        st.code(s)

# ===================== TRAIN =====================
with tabs[2]:
    st.subheader("Huấn luyện")
    if _job_running():
        st.warning(f"⏳ Đang có job '{st.session_state.job.kind}' chạy (PID {st.session_state.job.pid}). "
                   "1 GAMA port = 1 train tại một thời điểm. Xem tab Monitor.")
    mode = st.radio("Loại", ["Train mới", "Train tiếp từ zip"], horizontal=True)
    cA, cB = st.columns(2)
    algo = cA.selectbox("Thuật toán", C.ALGOS, index=1, help="PPO khuyến nghị (sau fix episode-boundary, PPO 90-100%; A2C kém hơn ở code hiện tại).")
    timesteps = cB.number_input("Timesteps", value=200000, step=10000)
    seeds_str = cA.text_input("Seed(s)", "0",
                              help="1 seed: '0'. Nhiều seed (train tuần tự trong 1 job): '0,1,2,3,4'.")
    scenario = cB.selectbox("Scenario / density", C.SCENARIOS, index=1)
    shield = cA.radio("Shield (khiên an toàn)", ["on", "off"], horizontal=True)
    resume_from, resume_mode = None, "continue"
    if mode == "Train tiếp từ zip":
        zips = C.list_model_zips()
        resume_from = pick_zip("Model .zip", zips, "train_zip") if zips else None
        resume_mode = st.radio("Chế độ", ["continue", "warmstart"], horizontal=True,
                               help="continue = resume optimizer (train tiếp cùng setup). "
                                    "warmstart = chỉ weights (curriculum/đổi cấu hình).")
        if not zips:
            st.info("Chưa có .zip trong outputs/models.")
    # Patch GAML thật để density/shield đúng (mặc định BẬT khi train-tiếp-zip để khớp môi trường zip).
    patch_real = st.checkbox(
        "🔧 Patch GAML THẬT (density + shield) trước khi train",
        value=(mode == "Train tiếp từ zip"),
        help="BẬT: vá GAML đúng density+shield đã chọn (GAMA recompile khi connect) — KHỚP môi "
             "trường zip. TẮT: scenario chỉ là nhãn (density giữ nguyên GAML hiện tại).")
    if patch_real:
        st.caption("⚠️ Sau train, GAML giữ ở density/shield này — dùng ♻️ Restore GAML (tab Monitor) để hoàn về mặc định.")
    ckpt = st.number_input(
        "💾 Checkpoint mỗi N steps (0 = tắt)", value=40000, step=10000,
        help="Lưu zip định kỳ vào outputs/models/checkpoints/<run>/ → KILL giữa chừng không mất "
             "bản dở; checkpoint là model SB3 đầy đủ, dùng luôn 'Train tiếp từ zip' (continue) để chạy tiếp.")
    ent_anneal = st.checkbox(
        "🎯 Entropy annealing (deterministic sạch — recipe a3_anneal2)",
        value=True,
        help="BẬT (mặc định): entropy 0.05→0.002, giữ cao tới 65% rồi nhọn → policy chạy argmax sạch như "
             "a3_anneal2 (1 run từ đầu, không cần BC). TẮT: entropy cố định, so sánh thuần — cần stochastic eval.")
    seeds = [int(x) for x in seeds_str.replace(" ", "").split(",") if x.lstrip("-").isdigit()]
    if len(seeds) > 1:
        st.success(f"🔁 MULTI-SEED: {len(seeds)} seed {seeds} sẽ train TUẦN TỰ (1 job nền) → run-name/seedN. "
                   + ("Resume nên dùng **warmstart** (mỗi seed init từ cùng zip)." if resume_from else ""))
    else:
        st.info("💡 Muốn train **NHIỀU seed**: gõ vào ô **Seed(s)** ở trên dạng `0,1,2,3,4` (cách nhau dấu phẩy) → chạy tuần tự.")
    run_name = st.text_input("run-name", f"ui_{algo}")
    if st.button("🚀 Launch train", type="primary", disabled=_job_running() or not C.config_ok(cfg)):
        if not C.port_listening(int(cfg["gama_port"])):
            st.error("GAMA chưa chạy — bật ở tab GAMA trước.")
        elif mode == "Train tiếp từ zip" and not resume_from:
            st.error("Chưa chọn zip.")
        elif not seeds:
            st.error("Seed(s) không hợp lệ (vd: 0 hoặc 0,1,2).")
        elif len(seeds) == 1:
            job, pmsg = C.launch_train(cfg, algo=algo, timesteps=int(timesteps), seed=seeds[0],
                                       run_name=run_name, scenario=scenario, shield=shield,
                                       patch_gaml_real=patch_real, checkpoint_interval=int(ckpt),
                                       resume_from=resume_from, resume_mode=resume_mode, ent_anneal=ent_anneal)
            st.session_state.job = job
            if pmsg:
                st.info(pmsg)
            st.success(f"Đã launch (PID {job.pid}). Sang tab Monitor để theo dõi.")
            st.rerun()
        else:
            job, pmsg = C.launch_train_multiseed(cfg, algo=algo, timesteps=int(timesteps), seeds=seeds,
                                                 run_name=run_name, scenario=scenario, shield=shield,
                                                 patch_gaml_real=patch_real, checkpoint_interval=int(ckpt),
                                                 resume_from=resume_from, resume_mode=resume_mode, ent_anneal=ent_anneal)
            st.session_state.job = job
            if pmsg:
                st.info(pmsg)
            st.success(f"Đã launch {len(seeds)} seed tuần tự (PID {job.pid}). Theo dõi ở Monitor.")
            st.rerun()

# ===================== MONITOR =====================
with tabs[3]:
    st.subheader("Theo dõi job hiện tại")
    j = st.session_state.job
    if j is None:
        st.info("Chưa có job nào được launch từ UI.")
        if st.button("🔄 Cập nhật"):
            st.rerun()
    else:
        alive = C.proc_alive(j.pid)
        st.write(f"**{j.kind}** | PID {j.pid} | {'🟢 đang chạy' if alive else '⚪ đã kết thúc'}")
        st.caption(j.cmd)
        stt = C.train_status(j.log_path)
        line = f"**Pha:** {stt['phase']}  ·  **số model đã train:** {stt['trainings']} (= seed × thuật toán)"
        if stt.get("algo"):
            line += f"  ·  **đang:** {stt['algo']} seed={stt['seed']}"
        st.markdown(line)
        ts = stt.get("timesteps")
        if ts is not None:
            target = getattr(j, "target_timesteps", None) or 200000
            st.progress(min(1.0, ts / target), text=f"seed hiện tại ~{ts:,}/{target:,} timesteps")
        if stt.get("last_ep"):
            st.caption("📋 Episode gần nhất: " + stt["last_ep"])
        st.write("**GAMA NPE:**", C.gama_check_errors()["npe"])
        kc1, kc2, kc3 = st.columns(3)
        if kc1.button("⏹️ KILL job", type="primary", disabled=not alive):
            st.warning(C.kill_job(j)["msg"])
            st.session_state.job = None
            st.rerun()
        if kc2.button("♻️ Restore GAML",
                      help="Bấm sau khi kill Experiment giữa chừng — GAML có thể còn bị patch density/shield."):
            st.info(C.restore_gaml())
        if kc3.button("🔄 Cập nhật"):
            st.rerun()
        st.code(C.tail_log(j.log_path, 40), language="log")
    st.caption("Streamlit không live như terminal — bấm 🔄 Cập nhật để refresh tiến độ.")

# ===================== EVALUATE =====================
with tabs[4]:
    st.subheader("Đánh giá model")
    zips = C.list_model_zips()
    if not zips:
        st.info("Chưa có model .zip.")
    else:
        ez = pick_zip("Model", zips, "eval_zip")
        ealgo = st.selectbox("Thuật toán", C.ALGOS, index=1, key="eval_algo")
        ep = st.number_input("Episodes", value=30, step=5)
        stoch = st.checkbox("Stochastic eval (cho A/B shield)", value=False)
        la = st.checkbox("Log action histogram", value=True)
        ee2e = st.checkbox("🏁 E2E (chạy tới cuối đường — đo sóng lùi + tai nạn post-merge)", value=True, key="eval_e2e",
                           help="--eval-continue --rl-postmerge --postmerge-window 400, max-steps 800. Chuẩn eval thesis.")
        ens = st.checkbox("🛡️ BỎ KHIÊN (ablation nội tâm hóa)", value=False, key="eval_noshield",
                          help="--no-shield: tắt IDM-cap + gate ngang — policy lái trần. So với có-khiên để đo mức tự lái thật.")
        if st.button("📊 Eval", disabled=_job_running() or not ez):
            if not C.port_listening(int(cfg["gama_port"])):
                st.error("GAMA chưa chạy.")
            else:
                job = C.launch_eval(cfg, algo=ealgo, model_zip=ez, episodes=int(ep),
                                    stochastic=stoch, log_actions=la, e2e=ee2e, no_shield=ens)
                st.session_state.job = job
                st.success(f"Eval launched (PID {job.pid}) — xem Monitor.")

# ===================== EXPERIMENTS =====================
with tabs[5]:
    st.subheader("Experiment preset (run_experiments — tự patch density+shield+restore)")
    p = st.selectbox("Preset", C.PRESETS, index=2)
    sc = st.selectbox("Scenario", C.SCENARIOS, index=1, key="exp_sc")
    sh = st.radio("Shield", ["on", "off"], horizontal=True)
    al = st.multiselect("Thuật toán", list(C.ALGOS), default=["a2c"])
    emode = st.radio("Mode", ["both", "eval"], horizontal=True,
                     help="both = chạy MỚI đầy đủ (baseline+train+eval+plots). "
                          "eval = chỉ EVAL LẠI 1 run đã train (chọn ở dropdown) — khỏi train lại. "
                          "(Muốn CHỈ train thì dùng tab 🚀 Train.)")
    if emode == "eval":
        tags = C.list_run_tags()
        run_tag_val = st.selectbox("📁 Chọn run đã train để EVAL LẠI", tags) if tags else None
        if not tags:
            st.info("Chưa có run nào ở outputs/models/<tag>/. Chạy mode 'both' trước (hoặc train ở tab Train).")
    else:
        run_tag_val = (st.text_input("📁 Tên thư mục (run-tag)", "",
                                     placeholder="để trống = <preset>_<timestamp>; vd: thesis_final_a2c").strip() or None)
    ec1, ec2 = st.columns(2)
    sb = ec1.checkbox("Skip baselines (nhanh hơn)", value=True)
    estoch = ec2.checkbox("Eval stochastic (cho A/B shield)", value=False)
    eanneal = st.checkbox(
        "🎯 Entropy annealing (deterministic sạch — recipe a3_anneal2)", value=True,
        help="BẬT (mặc định): mỗi seed train entropy 0.05→0.002 giữ cao 65% rồi nhọn → policy argmax sạch "
             "như a3_anneal2. TẮT: entropy cố định (so sánh thuần, cần Eval stochastic).")
    edry = st.checkbox("Dry-run (chỉ in lệnh, không chạy)", value=False)
    st.caption("thesis = 5 seed × 200k + 50 eval (~vài giờ). Bộ số chính cho báo cáo.")
    if st.button("🧪 Chạy experiment", disabled=_job_running() or not al):
        if not C.port_listening(int(cfg["gama_port"])):
            st.error("GAMA chưa chạy.")
        elif emode == "eval" and not run_tag_val:
            st.error("Mode eval: chưa chọn run để eval (hoặc chưa có run nào đã train).")
        else:
            job = C.launch_experiment(cfg, preset=p, scenario=sc, shield=sh, algos=al, skip_baselines=sb,
                                      run_tag=run_tag_val, eval_stochastic=estoch, mode=emode, dry_run=edry,
                                      ent_anneal=eanneal)
            st.session_state.job = job
            st.success(f"Experiment ({emode}) launched (PID {job.pid}) — xem Monitor."
                       + (f" Folder: {run_tag_val}" if run_tag_val else " (folder: <preset>_<timestamp>)"))

# ===================== RESULTS =====================
with tabs[6]:
    st.subheader("Kết quả các lần chạy")
    runs = C.list_run_dirs()
    if not runs:
        st.info("Chưa có run nào trong outputs/logs.")
    else:
        rd = st.selectbox("Run", runs, format_func=lambda p: Path(p).name)
        summ = C.summarize_run(rd)
        if summ["per_algo"]:
            st.table([
                {"algo": a.upper(), "seeds": d["seeds"],
                 "success%": f"{d['success_mean']}±{d['success_std']}",
                 "collision%": f"{d['collision_mean']}±{d['collision_std']}"}
                for a, d in summ["per_algo"].items()
            ])
        else:
            st.warning("Run này chưa có CSV eval.")
        # biểu đồ PNG của run đang chọn (folder cùng tên trong outputs/plots/**)
        run_name = Path(rd).name
        matches = [m for m in (ROOT / "outputs" / "plots").rglob(run_name) if m.is_dir()]
        plot_dir = matches[0] if matches else None
        pngs = sorted(plot_dir.glob("*.png")) if plot_dir else []
        if pngs:
            with st.expander(f"📊 Biểu đồ của run '{run_name}' ({len(pngs)} PNG)", expanded=True):
                png_gallery(pngs, key_prefix=f"run_{run_name}")
        else:
            st.caption(f"Run '{run_name}' chưa có biểu đồ (chỉ Experiments mode=both mới sinh plots).")

    st.divider()
    st.markdown("### 📊 Vẽ biểu đồ + bảng từ CSV (cho model lẻ / resume — như bước cuối Experiments)")
    csvs = C.list_eval_csvs()
    if not csvs:
        st.info("Chưa có CSV eval — eval model ở tab Evaluate trước.")
    else:
        sel = st.multiselect("Chọn CSV (chọn nhiều để SO SÁNH các model cùng lúc)", csvs,
                             format_func=lambda p: "/".join(Path(p).parts[-2:]))
        rep_name = st.text_input("Tên thư mục báo cáo (trong outputs/plots/)", "ui_report")
        if st.button("📊 Vẽ biểu đồ + bảng", type="primary", disabled=not sel):
            out_dir = str(ROOT / "outputs" / "plots" / rep_name)
            with st.spinner("Chạy plots.py + analysis.py…"):
                res = C.generate_report(cfg, sel, out_dir)
            for m in res["msgs"]:
                st.write("•", m)
            if res["table"]:
                import csv as _csv
                rows = list(_csv.DictReader(open(res["table"], encoding="utf-8")))
                if rows:
                    st.markdown("**comparison_table.csv:**")
                    st.dataframe(rows, use_container_width=True)
            if res["pngs"]:
                st.markdown(f"**Biểu đồ mới ({len(res['pngs'])}):**")
                png_gallery(res["pngs"], key_prefix="report")
            st.success(f"Xong → {out_dir}")

# ===================== PLAY GUI =====================
with tabs[7]:
    st.subheader("▶️ Xem model chạy trên GAMA Desktop (test trực quan sau train)")
    st.caption("CÁCH DÙNG: mở **Gama.exe (Desktop)** → chạy experiment **TrafficSimulation** "
               "(Desktop sẽ serve socket ở **port 1000** + hiển thị) → bấm Chạy bên dưới với **port 1000**. "
               "Python load zip → điều khiển cửa sổ Desktop → THẤY animation. (Port 1001 là headless cho train, KHÔNG hiển thị.)")
    zips_p = C.list_model_zips()
    if not zips_p:
        st.info("Chưa có model .zip — train trước đã.")
    else:
        pz = pick_zip("Model .zip", zips_p, "play_zip")
        pcol1, pcol2 = st.columns(2)
        pa = pcol1.selectbox("Thuật toán", C.ALGOS, index=1, key="play_algo")
        pep = pcol2.number_input("Số episode", value=3, step=1, key="play_ep")
        pexp = pcol1.text_input("Experiment GAML", "TrafficSimulation")
        pport = pcol2.number_input("Port GAMA Desktop (GUI server)", value=1000, step=1, key="play_port",
                                   help="1000 = GAMA Desktop (Gama.exe) đang chạy experiment + hiển thị. KHÔNG dùng 1001 (headless).")
        pscen = pcol1.selectbox("Mật độ xe (patch GAML)", C.SCENARIOS, index=2, key="play_scen",
                                help="low/medium/high = nb_cars_max 24/54/84 (road 240). Patch GAML trước khi play; "
                                     "server Desktop nạp lại file khi play load_experiment → mật độ áp dụng. "
                                     "⚠️ Patch này từng lẫn vào commit khi play bị kill — restore GAML sau khi xem.")
        pdelay = st.slider("Độ trễ mỗi bước (giây) — chậm để dễ nhìn", 0.0, 1.0, 0.35, 0.05)
        pstoch = st.checkbox("Stochastic (lấy mẫu thay vì argmax)", value=False, key="play_stoch")
        pe2e = st.checkbox("🏁 Đi tới cuối (e2e: RL lái tiếp sau merge, không end ở merge)", value=True, key="play_e2e",
                           help="Bật cho model train --rl-postmerge (vd thesis_e2e_yt50_150k): merger nhập làn xong CHẠY TIẾP tới cuối đường bằng RL, ván kết tại cuối đường. TẮT = end ngay khi merge (terminate).")
        pns = st.checkbox("🛡️ BỎ KHIÊN (xem policy lái trần — demo ablation)", value=False, key="play_noshield",
                          help="Tắt IDM-cap + gate ngang trên GUI. Kỳ vọng: model thường sẽ đâm nhiều hơn — minh họa vai trò khiên.")
        if st.button("▶️ Chạy Play GUI", type="primary", disabled=_job_running() or not pz):
            if not C.port_listening(int(pport)):
                st.error(f"❌ Không có GAMA server ở port {int(pport)}. Mở **Gama.exe** → chạy experiment "
                         f"**{pexp}** (Desktop tự serve port đó) rồi mới bấm Chạy.")
            else:
                job, pmsg = C.launch_play(cfg, algo=pa, model_zip=pz, episodes=int(pep),
                                          experiment=pexp, port=int(pport), step_delay=float(pdelay),
                                          stochastic=pstoch, scenario=pscen, e2e=pe2e, no_shield=pns)
                st.session_state.job = job
                if pmsg:
                    st.info(pmsg)
                st.success(f"Đã chạy Play (PID {job.pid}). Nhìn cửa sổ GAMA Desktop để xem animation; log ở Monitor.")
                st.caption("⚠️ Nếu Desktop KHÔNG động: model có thể không khớp kiến trúc — xem lỗi ở tab 📡 Monitor.")
    # Nút dừng Play (kill job đang chạy)
    if _job_running() and st.session_state.job and st.session_state.job.kind == "play":
        if st.button("⏹️ DỪNG Play", type="secondary"):
            st.warning(C.kill_job(st.session_state.job)["msg"])
            st.session_state.job = None
            st.rerun()

# ===================== THESIS =====================
with tabs[8]:
    st.subheader("🎓 Biểu đồ & model thesis")
    st.caption("Bộ hình PNG 300dpi cho báo cáo — sinh bởi rl/make_thesis_plots.py từ CSV train/eval. "
               "Số liệu bar chart lấy từ bảng SUMMARY đã kiểm chứng trong script (kèm nguồn log).")

    pl_dir = Path(cfg["project_dir"]) / "outputs" / "plots" if cfg.get("project_dir") else Path("outputs/plots")
    if st.button("🖼️ Tạo / làm mới biểu đồ", disabled=_job_running()):
        import subprocess as _sp
        r = _sp.run([cfg["venv_python"], "rl/make_thesis_plots.py"],
                    cwd=cfg.get("project_dir") or ".", capture_output=True, text=True,
                    encoding="utf-8", errors="replace")
        if r.returncode == 0:
            st.success("Đã xuất biểu đồ vào outputs/plots/")
        else:
            st.error(f"Lỗi: {r.stderr[-800:]}")

    figs = sorted(pl_dir.glob("fig*.png")) if pl_dir.exists() else []
    if figs:
        png_gallery(figs, "thesis", cols_per_row=2)
    else:
        st.info("Chưa có biểu đồ — bấm nút trên để tạo.")

    st.divider()
    st.markdown("**Model thesis (eval @ mật độ cao 84, road 240, e2e):**")
    st.table([
        {"Model": "thesis_curriculum_final.zip", "Nguồn": "curriculum 2-stage from-scratch (600k)",
         "Success": "90%", "Nhường": "41%", "No-shield": "30%"},
        {"Model": "thesis_e2e_yt50_150k.zip", "Nguồn": "warmstart lineage (~2-3M tích lũy)",
         "Success": "90%", "Nhường": "94%", "No-shield": "10%"},
        {"Model": "yt52_propshield5_150000_steps.zip", "Nguồn": "yt50-150k + phạt khiên tỷ lệ k=5",
         "Success": "100%", "Nhường": "thấp", "No-shield": "50%"},
    ])
    st.caption("Train lại từ đầu: `bash train_curriculum.sh` (2 stage, ~3h) — learning curve ghi vào "
               "outputs/logs/curr_stage*_episodes.csv, tự dùng cho biểu đồ ở trên.")
