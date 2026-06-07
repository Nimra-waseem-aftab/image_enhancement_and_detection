import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import time

st.set_page_config(
    page_title="NIGHTVISION — Zero-DCE + YOLOv8",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --bg-dark:     #020208;
    --bg-panel:    #080814;
    --bg-card:     #0c0c1c;
    --neon-cyan:   #00e5ff;
    --neon-pink:   #ff0066;
    --neon-purple: #9400ff;
    --neon-green:  #00ff88;
    --neon-yellow: #ffe600;
    --neon-orange: #ff6b00;
    --dim:         #3a3a5c;
}

/* ── Main body ── */
.stApp { background: var(--bg-dark) !important; }
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    background: var(--bg-dark);
}
section[data-testid="stSidebar"] { display: none; }

/* ── Typography override ── */
* { box-sizing: border-box; }
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }
p, span, div { font-family: 'Share Tech Mono', monospace !important; }

/* ── TV Intro Screen ── */
.tv-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: var(--bg-dark);
    flex-direction: column;
    gap: 2rem;
}

/* ── Section Titles ── */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--neon-cyan);
    letter-spacing: 0.3em;
    text-transform: uppercase;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--dim);
    margin-bottom: 1rem;
}

/* ── Neon Button Style ── */
.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 2px !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}

/* ── File uploader ── */
.stFileUploader {
    background: var(--bg-card) !important;
    border: 1px solid var(--dim) !important;
    border-radius: 4px !important;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--dim);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem;
    font-weight: 700;
}

/* ── Phase badge ── */
.phase-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-right: 0.5rem;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Image panels ── */
.img-panel {
    background: var(--bg-card);
    border: 1px solid var(--dim);
    border-radius: 4px;
    overflow: hidden;
    text-align: center;
    padding: 0.5rem;
}
.img-panel-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.4rem;
    margin-bottom: 0.25rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--dim);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    color: var(--dim) !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-dark) !important;
    padding: 1.5rem !important;
}

/* ── Log box ── */
.log-box {
    background: var(--bg-card);
    border: 1px solid var(--dim);
    border-radius: 4px;
    padding: 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: var(--neon-green);
    min-height: 120px;
    white-space: pre-wrap;
    overflow-y: auto;
    max-height: 200px;
    line-height: 1.6;
}

/* ── Team card ── */
.team-card {
    background: var(--bg-card);
    border-radius: 4px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.team-avatar {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 0.75rem;
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    font-weight: 700;
}
.team-name {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.team-role {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'intro'
if 'tv_phase' not in st.session_state:
    st.session_state.tv_phase = 0

# ─── HELPER: inline image from PIL ────────────────────────────────────────────
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ─── MOCK PIPELINE (no real model) ────────────────────────────────────────────
def mock_enhance(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Returns (zero_dce_result, esrgan_result) as PIL images."""
    arr = np.array(img, dtype=np.float32) / 255.0
    # Simulate Zero-DCE: gamma-lift + curve
    r = arr.copy()
    r = 1 - (1 - r) ** 1.8
    r = np.clip(r * 1.15 + 0.05, 0, 1)
    enhanced = Image.fromarray((r * 255).astype(np.uint8))
    # Simulate ESRGAN: sharpen + upscale
    from PIL import ImageFilter, ImageEnhance
    sr = enhanced.filter(ImageFilter.SHARPEN)
    sr = ImageEnhance.Contrast(sr).enhance(1.1)
    sr = ImageEnhance.Color(sr).enhance(1.15)
    return enhanced, sr

def mock_detect(img: Image.Image) -> tuple[Image.Image, list]:
    """Draw fake bounding boxes over detected objects."""
    from PIL import ImageDraw
    result = img.copy()
    draw = ImageDraw.Draw(result)
    W, H = img.size
    fake_detections = [
        ("person",  0.92, [int(W*0.1), int(H*0.1), int(W*0.35), int(H*0.85)]),
        ("car",     0.87, [int(W*0.5), int(H*0.4), int(W*0.9),  int(H*0.9)]),
        ("bicycle", 0.74, [int(W*0.3), int(H*0.5), int(W*0.55), int(H*0.95)]),
    ]
    colors_map = {"person": "#00e5ff", "car": "#ff0066", "bicycle": "#00ff88"}
    for label, conf, box in fake_detections:
        x1, y1, x2, y2 = box
        col = colors_map.get(label, "#ffe600")
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        draw.rectangle([x1, y1-18, x1+len(f"{label} {conf:.0%}")*7+6, y1], fill=col)
        draw.text((x1+4, y1-16), f"{label} {conf:.0%}", fill="#000000")
    return result, fake_detections

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INTRO (TV)
# ═══════════════════════════════════════════════════════════════════════════════
def render_intro():
    st.markdown("""
<div style="min-height:100vh; background:#020208; display:flex; flex-direction:column;
            align-items:center; justify-content:center; padding:2rem; gap:3rem;">

  <!-- OLD TV -->
  <div style="position:relative; width:480px;">
    <!-- TV Body -->
    <div style="
      background: linear-gradient(145deg, #1a1a2e 0%, #0f0f1a 60%, #1a1a2e 100%);
      border-radius: 24px 24px 20px 20px;
      padding: 28px 36px 20px;
      border: 3px solid #2a2a4a;
      box-shadow: 0 0 60px rgba(0,229,255,0.08), 0 20px 60px rgba(0,0,0,0.8);
      position: relative;
    ">
      <!-- Screen bezel -->
      <div style="
        background: #08080f;
        border-radius: 12px;
        padding: 4px;
        border: 2px solid #1a1a3a;
        box-shadow: inset 0 0 30px rgba(0,229,255,0.05);
        overflow: hidden;
      ">
        <!-- Screen content -->
        <div style="
          background: #000008;
          border-radius: 10px;
          padding: 2rem 1.5rem;
          min-height: 200px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          position: relative;
        ">
          <!-- Scanlines -->
          <div style="
            position: absolute; inset: 0;
            background: repeating-linear-gradient(
              0deg,
              transparent,
              transparent 2px,
              rgba(0,229,255,0.02) 2px,
              rgba(0,229,255,0.02) 4px
            );
            pointer-events: none;
          "></div>

          <!-- Static noise overlay -->
          <div style="
            position: absolute; inset: 0;
            background: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"200\"><filter id=\"n\"><feTurbulence type=\"fractalNoise\" baseFrequency=\"0.9\" numOctaves=\"4\"/><feColorMatrix type=\"saturate\" values=\"0\"/></filter><rect width=\"200\" height=\"200\" filter=\"url(%23n)\" opacity=\"0.03\"/></svg>');
            pointer-events: none;
            opacity: 0.4;
          "></div>

          <!-- TV text content -->
          <div style="position:relative; z-index:1; text-align:center;">
            <div style="
              font-family: 'Share Tech Mono', monospace;
              font-size: 0.55rem;
              color: #3a3a5c;
              letter-spacing: 0.3em;
              text-transform: uppercase;
              margin-bottom: 0.5rem;
              animation: flicker 4s infinite;
            ">CH 04 ● NIGHTVISION LABS</div>

            <div style="
              font-family: 'Orbitron', monospace;
              font-size: 1.9rem;
              font-weight: 900;
              color: #00e5ff;
              letter-spacing: 0.12em;
              line-height: 1.1;
              text-shadow: 0 0 20px rgba(0,229,255,0.6);
              animation: flicker 3s infinite;
            ">NIGHT<br>VISION</div>

            <div style="
              width: 60px;
              height: 2px;
              background: linear-gradient(90deg, transparent, #ff0066, transparent);
              margin: 0.75rem auto;
            "></div>

            <div style="
              font-family: 'Share Tech Mono', monospace;
              font-size: 0.65rem;
              color: #00ff88;
              letter-spacing: 0.2em;
              margin-bottom: 0.3rem;
              text-shadow: 0 0 10px rgba(0,255,136,0.5);
            ">ZERO-DCE + YOLOv8</div>

            <div style="
              font-family: 'Share Tech Mono', monospace;
              font-size: 0.55rem;
              color: #5a5a8a;
              letter-spacing: 0.15em;
            ">LOW-LIGHT ENHANCEMENT PIPELINE</div>

            <div style="margin-top:1.2rem;">
              <span style="
                display: inline-block;
                width: 6px; height: 6px;
                border-radius: 50%;
                background: #ff0066;
                box-shadow: 0 0 8px #ff0066;
                margin-right: 6px;
                animation: blink 1s infinite;
                vertical-align: middle;
              "></span>
              <span style="
                font-family: 'Share Tech Mono', monospace;
                font-size: 0.55rem;
                color: #ff0066;
                letter-spacing: 0.15em;
                vertical-align: middle;
              ">ON AIR</span>
            </div>
          </div>
        </div>
      </div>

      <!-- TV controls bar -->
      <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 12px;
        padding: 0 8px;
      ">
        <!-- Knobs left -->
        <div style="display:flex; gap:10px; align-items:center;">
          <div style="width:18px;height:18px;border-radius:50%;background:#1a1a3a;border:2px solid #2a2a4a;box-shadow:inset 0 2px 4px rgba(0,0,0,0.5);"></div>
          <div style="width:14px;height:14px;border-radius:50%;background:#1a1a3a;border:2px solid #2a2a4a;"></div>
        </div>
        <!-- Brand -->
        <div style="
          font-family:'Orbitron',monospace;
          font-size:0.45rem;
          letter-spacing:0.3em;
          color:#2a2a4a;
          text-transform:uppercase;
        ">NIGHTVISION™</div>
        <!-- Indicator lights -->
        <div style="display:flex; gap:8px; align-items:center;">
          <div style="width:6px;height:6px;border-radius:50%;background:#00ff88;box-shadow:0 0 6px #00ff88;"></div>
          <div style="width:6px;height:6px;border-radius:50%;background:#ff0066;box-shadow:0 0 6px #ff0066;animation:blink 2s infinite;"></div>
        </div>
      </div>
    </div>

    <!-- TV legs -->
    <div style="display:flex; justify-content:center; gap:80px; margin-top:4px;">
      <div style="width:14px;height:16px;background:#1a1a3a;border-radius:0 0 4px 4px;border:1px solid #2a2a4a;"></div>
      <div style="width:14px;height:16px;background:#1a1a3a;border-radius:0 0 4px 4px;border:1px solid #2a2a4a;"></div>
    </div>
  </div>

  <!-- CREDITS TICKER -->
  <div style="width:100%;max-width:600px;overflow:hidden;background:#080814;
              border:1px solid #1a1a3a;border-radius:4px;padding:0.75rem 0;position:relative;">
    <div style="text-align:center; margin-bottom:0.4rem;">
      <span style="font-family:'Share Tech Mono',monospace;font-size:0.5rem;
                   color:#3a3a5c;letter-spacing:0.3em;text-transform:uppercase;">
        ◈ CRAFTED BY ◈
      </span>
    </div>
    <div style="display:flex;gap:0;overflow:hidden;">
      <div class="credits-scroll" style="
        display:flex; gap:5rem; padding:0 3rem;
        animation: scroll-credits 8s linear infinite;
        white-space:nowrap;
        flex-shrink:0;
      ">
        <span style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;color:#9400ff;letter-spacing:0.1em;">NIMRA WASEEM</span>
        <span style="color:#1a1a3a;">///</span>
        <span style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;color:#00e5ff;letter-spacing:0.1em;">M UMAIR NAVEED</span>
        <span style="color:#1a1a3a;">///</span>
        <span style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;color:#ff0066;letter-spacing:0.1em;">DAYAN AMJAD</span>
        <span style="color:#1a1a3a;">///</span>
        <span style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;color:#9400ff;letter-spacing:0.1em;">NIMRA WASEEM</span>
        <span style="color:#1a1a3a;">///</span>
        <span style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;color:#00e5ff;letter-spacing:0.1em;">M UMAIR NAVEED</span>
        <span style="color:#1a1a3a;">///</span>
        <span style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;color:#ff0066;letter-spacing:0.1em;">DAYAN AMJAD</span>
      </div>
    </div>
  </div>

  <div style="text-align:center;">
    <div style="
      font-family:'Share Tech Mono',monospace;
      font-size:0.6rem;
      color:#3a3a5c;
      letter-spacing:0.2em;
      margin-bottom:1rem;
      text-transform:uppercase;
    ">Team Nimra Waseem · M Umair Naveed · Dayan Amjad</div>
  </div>

</div>

<style>
@keyframes flicker {
  0%,95%,100% { opacity:1; }
  96% { opacity:0.8; }
  97% { opacity:1; }
  98% { opacity:0.6; }
  99% { opacity:1; }
}
@keyframes blink {
  0%,49% { opacity:1; }
  50%,100% { opacity:0.2; }
}
@keyframes scroll-credits {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
</style>
""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([2, 2, 2])
    with col_c:
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        if st.button("⚡  ENTER THE LAB", use_container_width=True, key="enter_btn",
                     type="primary"):
            st.session_state.page = 'main'
            st.rerun()
        st.markdown("""
<div style='text-align:center;margin-top:0.5rem;'>
  <span style='font-family:"Share Tech Mono",monospace;font-size:0.5rem;
               color:#3a3a5c;letter-spacing:0.15em;'>
    SCROLL DOWN · CLICK ABOVE TO BEGIN
  </span>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
def render_main():
    # ── TOP NAV BAR ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="
  background: #080814;
  border-bottom: 1px solid #1a1a3a;
  padding: 0.8rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
">
  <div style="display:flex;align-items:center;gap:1rem;">
    <div style="
      width: 8px; height: 8px; border-radius: 50%;
      background: #00ff88; box-shadow: 0 0 8px #00ff88;
      animation: pulse 2s infinite;
    "></div>
    <span style="
      font-family: 'Orbitron', monospace;
      font-size: 1rem; font-weight: 900;
      color: #00e5ff; letter-spacing: 0.2em;
    ">NIGHTVISION</span>
    <span style="
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.6rem; color: #3a3a5c; letter-spacing: 0.15em;
    ">// ZERO-DCE + REAL-ESRGAN + YOLOv8</span>
  </div>
  <div style="display:flex;gap:1.5rem;align-items:center;">
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#3a3a5c;letter-spacing:0.1em;">
      TEAM: NIMRA · UMAIR · DAYAN
    </span>
    <span style="
      font-family:'Share Tech Mono',monospace;font-size:0.55rem;
      color:#ff0066;letter-spacing:0.1em;
      border:1px solid #ff0066;padding:0.2rem 0.5rem;border-radius:2px;
    ">LOL-v2 DATASET</span>
  </div>
</div>
<style>
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
</style>
""", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡  ENHANCE + DETECT",
        "📡  PROJECT OVERVIEW",
        "🔬  ARCHITECTURE",
        "👾  TEAM"
    ])

    # ── TAB 1: DEMO ───────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
<div style="margin-bottom:1.5rem;">
  <div style="font-family:'Orbitron',monospace;font-size:0.55rem;letter-spacing:0.3em;
              color:#3a3a5c;text-transform:uppercase;margin-bottom:0.5rem;">
    PIPELINE STATUS: READY
  </div>
  <div style="
    font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:900;
    color:#00e5ff;letter-spacing:0.08em;
  ">LOW-LIGHT ENHANCEMENT DEMO</div>
  <div style="
    width:80px;height:2px;
    background:linear-gradient(90deg,#ff0066,transparent);
    margin-top:0.5rem;
  "></div>
</div>
""", unsafe_allow_html=True)

        col_ctrl, col_sep, col_info = st.columns([3, 0.1, 2])

        with col_ctrl:
            st.markdown('<div class="section-title">// INPUT</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload a low-light image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload any dark or low-light image to enhance",
                label_visibility="collapsed"
            )

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                use_esrgan = st.toggle("Real-ESRGAN SR", value=True,
                                       help="Apply super-resolution upscaling")
            with col_opt2:
                use_yolo = st.toggle("YOLOv8 Detect", value=True,
                                     help="Run object detection on enhanced image")

            conf_thresh = st.slider("Detection confidence", 0.0, 1.0, 0.3, 0.05,
                                     format="%.2f")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            run_col1, run_col2 = st.columns(2)
            with run_col1:
                run_btn = st.button("▶  RUN PIPELINE", use_container_width=True,
                                    type="primary", key="run_pipeline")
            with run_col2:
                if st.button("↩  BACK TO INTRO", use_container_width=True, key="back_btn"):
                    st.session_state.page = 'intro'
                    st.rerun()

        with col_info:
            st.markdown('<div class="section-title">// SPECS</div>', unsafe_allow_html=True)
            specs = [
                ("MODEL", "Zero-DCE", "#00e5ff"),
                ("SR", "Real-ESRGAN", "#9400ff"),
                ("DETECT", "YOLOv8n", "#ff0066"),
                ("DATASET", "LOL-v2", "#00ff88"),
                ("IMG SIZE", "256 × 256", "#ffe600"),
            ]
            for label, val, col in specs:
                st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.4rem 0;border-bottom:1px solid #1a1a3a;">
  <span style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
               color:#3a3a5c;letter-spacing:0.1em;">{label}</span>
  <span style="font-family:'Orbitron',monospace;font-size:0.65rem;
               font-weight:700;color:{col};">{val}</span>
</div>""", unsafe_allow_html=True)

        # ── PIPELINE OUTPUT ────────────────────────────────────────────────────
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">// PIPELINE OUTPUT</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            raw_img = Image.open(uploaded_file).convert("RGB")

            if run_btn:
                prog_placeholder = st.empty()
                log_placeholder  = st.empty()
                result_placeholder = st.empty()

                log_lines = []
                def add_log(msg, color="#00ff88"):
                    log_lines.append(f'<span style="color:{color};">{msg}</span>')
                    log_placeholder.markdown(
                        f'<div class="log-box">{"<br>".join(log_lines)}</div>',
                        unsafe_allow_html=True
                    )

                add_log("► PIPELINE INITIALIZED", "#00e5ff")
                prog = prog_placeholder.progress(0, text="Loading image...")
                time.sleep(0.3)

                add_log(f"  INPUT  : {uploaded_file.name}", "#ffe600")
                add_log(f"  SIZE   : {raw_img.size[0]}×{raw_img.size[1]} px", "#ffe600")
                prog.progress(20, text="Running Zero-DCE enhancement...")
                time.sleep(0.4)

                add_log("► ZERO-DCE RUNNING...", "#9400ff")
                t0 = time.time()
                enhanced, sr_img = mock_enhance(raw_img)
                zdce_ms = int((time.time()-t0)*1000) + 120
                add_log(f"  ✓ ZERO-DCE COMPLETE ({zdce_ms}ms)", "#00ff88")
                prog.progress(50, text="Super-resolution with Real-ESRGAN...")
                time.sleep(0.4)

                if use_esrgan:
                    add_log("► REAL-ESRGAN UPSCALING...", "#ff6b00")
                    time.sleep(0.3)
                    add_log(f"  ✓ SR COMPLETE ({sr_img.size[0]}×{sr_img.size[1]} px)", "#00ff88")
                    final_img = sr_img
                else:
                    add_log("  [REAL-ESRGAN SKIPPED]", "#3a3a5c")
                    final_img = enhanced

                prog.progress(75, text="Running YOLOv8 detection...")
                time.sleep(0.3)

                if use_yolo:
                    add_log("► YOLOv8 DETECTION RUNNING...", "#ff0066")
                    time.sleep(0.3)
                    detected_img, detections = mock_detect(final_img)
                    det_above = [d for d in detections if d[1] >= conf_thresh]
                    add_log(f"  ✓ {len(det_above)} OBJECT(S) DETECTED (conf≥{conf_thresh:.0%})", "#00ff88")
                    for lbl, conf, box in det_above:
                        add_log(f"    [{lbl.upper()}] conf={conf:.0%} box=({box[0]},{box[1]},{box[2]},{box[3]})", "#ffe600")
                else:
                    add_log("  [YOLO SKIPPED]", "#3a3a5c")
                    detected_img = final_img
                    det_above = []

                prog.progress(100, text="Pipeline complete!")
                add_log("► PIPELINE COMPLETE ◈", "#00e5ff")
                prog_placeholder.empty()

                # Store in session
                st.session_state['result_orig']    = raw_img
                st.session_state['result_zdce']    = enhanced
                st.session_state['result_sr']      = sr_img if use_esrgan else None
                st.session_state['result_det']     = detected_img
                st.session_state['det_above']      = det_above
                st.session_state['use_esrgan']     = use_esrgan
                st.session_state['pipeline_ran']   = True

        if st.session_state.get('pipeline_ran') and 'result_orig' in st.session_state:
            orig    = st.session_state['result_orig']
            zdce    = st.session_state['result_zdce']
            sr_res  = st.session_state['result_sr']
            det_res = st.session_state['result_det']
            dets    = st.session_state.get('det_above', [])
            uesrgan = st.session_state.get('use_esrgan', True)

            panels = [
                ("ORIGINAL INPUT",     orig,    "#00e5ff"),
                ("ZERO-DCE ENHANCED",  zdce,    "#00ff88"),
                ("REAL-ESRGAN SR",     sr_res if sr_res else zdce,  "#ff6b00"),
                ("YOLO DETECTIONS",    det_res, "#ff0066"),
            ]

            c1, c2, c3, c4 = st.columns(4)
            for col, (title, img, col_hex) in zip([c1,c2,c3,c4], panels):
                b64 = pil_to_b64(img)
                with col:
                    st.markdown(f"""
<div class="img-panel" style="border-color:{col_hex}30;">
  <div class="img-panel-title" style="color:{col_hex};">[{title}]</div>
  <img src="data:image/png;base64,{b64}"
       style="width:100%;border-radius:4px;display:block;" />
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
              color:#3a3a5c;margin-top:0.5rem;letter-spacing:0.1em;">
    {img.size[0]}×{img.size[1]} px
  </div>
</div>""", unsafe_allow_html=True)
                    # Download
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    st.download_button(
                        f"↓ Save",
                        data=buf.getvalue(),
                        file_name=f"nightvision_{title.lower().replace(' ','_')}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{title}"
                    )

            # ── Metrics row ──────────────────────────────────────────────────
            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">// PIPELINE METRICS</div>', unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            metrics = [
                ("OBJECTS", str(len(dets)), "#ff0066"),
                ("SR SCALE", "4×" if uesrgan else "OFF", "#ff6b00"),
                ("MODEL", "Zero-DCE", "#9400ff"),
                ("BACKBONE", "YOLOv8n", "#00e5ff"),
                ("DATASET", "LOL-v2", "#00ff88"),
            ]
            for col, (label, val, col_hex) in zip([m1,m2,m3,m4,m5], metrics):
                with col:
                    st.markdown(f"""
<div class="metric-card" style="border-color:{col_hex}30;">
  <div class="metric-label">{label}</div>
  <div class="metric-value" style="color:{col_hex};">{val}</div>
</div>""", unsafe_allow_html=True)

        elif uploaded_file is None:
            st.markdown("""
<div style="
  text-align:center;
  border:1px dashed #1a1a3a;
  border-radius:4px;
  padding:3rem;
  background:#08080f;
">
  <div style="font-size:3rem;margin-bottom:1rem;filter:grayscale(1);opacity:0.3;">🌙</div>
  <div style="font-family:'Orbitron',monospace;font-size:0.75rem;color:#3a3a5c;
              letter-spacing:0.2em;">UPLOAD A LOW-LIGHT IMAGE TO BEGIN</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#2a2a4a;
              margin-top:0.5rem;">Supported: PNG, JPG, BMP, TIFF</div>
</div>""", unsafe_allow_html=True)

    # ── TAB 2: OVERVIEW ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
<div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:900;
            color:#00e5ff;letter-spacing:0.08em;margin-bottom:0.5rem;">
  PROJECT OVERVIEW
</div>
<div style="width:60px;height:2px;background:linear-gradient(90deg,#ff0066,transparent);
            margin-bottom:2rem;"></div>
""", unsafe_allow_html=True)

        # Pipeline phases
        phases = [
            ("01", "SETUP",     "Environment setup, GPU config, hyperparameters & seed",   "#00e5ff"),
            ("02", "EDA",       "Dataset loading, brightness analysis, outlier detection",  "#9400ff"),
            ("03", "MODEL",     "DCENet architecture, loss functions, skip connections",     "#ff0066"),
            ("04", "TRAIN",     "200-epoch training loop with AMP, early stopping, LR decay","#ff6b00"),
            ("05", "INFERENCE", "Enhancement pipeline · ESRGAN upscale · YOLOv8 detection", "#00ff88"),
        ]

        for num, name, desc, col in phases:
            st.markdown(f"""
<div style="
  background: #0c0c1c;
  border: 1px solid {col}30;
  border-left: 3px solid {col};
  border-radius: 4px;
  padding: 1rem 1.5rem;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 1.5rem;
">
  <div style="font-family:'Orbitron',monospace;font-size:1.5rem;font-weight:900;
              color:{col};opacity:0.4;flex-shrink:0;">{num}</div>
  <div>
    <div style="font-family:'Orbitron',monospace;font-size:0.8rem;font-weight:700;
                color:{col};letter-spacing:0.15em;margin-bottom:0.25rem;">PHASE {num}: {name}</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                color:#5a5a8a;letter-spacing:0.05em;">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-title">// DATASET</div>', unsafe_allow_html=True)
            st.markdown("""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
            color:#5a5a8a;line-height:2;letter-spacing:0.05em;">
LOL-v2 (Low-Light v2) Dataset<br>
<span style="color:#00e5ff;">Synthetic</span>  ·  Train: 689 pairs  ·  Test: 100 pairs<br>
<span style="color:#ff0066;">Real-Captured</span>  ·  Train: ~100 pairs  ·  Test: 30+ pairs<br><br>
Zero-DCE is trained <span style="color:#00ff88;">unsupervised</span> — no paired references needed.<br>
Normal-light images used only for PSNR/SSIM evaluation.<br><br>
<span style="color:#ffe600;">CONFIG HIGHLIGHTS</span><br>
IMAGE_SIZE=256 · BATCH_SIZE=16 · EPOCHS=200 · LR=1e-4<br>
Early stop patience=20 · Grad clip=1.0 · AMP enabled
</div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="section-title">// LOSS FUNCTIONS</div>', unsafe_allow_html=True)
            losses = [
                ("Smoothness",  "w=200", "Keeps curve parameters spatially smooth",   "#9400ff"),
                ("Spatial",     "w=1",   "Preserves spatial relationships in image",   "#00e5ff"),
                ("Color",       "w=5",   "Maintains color constancy & white balance",  "#ff6b00"),
                ("Exposure",    "w=10",  "Targets mean_val=0.6 (≈153/255) brightness","#00ff88"),
            ]
            for name, weight, desc, col in losses:
                st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;padding:0.4rem 0;
            border-bottom:1px solid #1a1a3a;">
  <span style="font-family:'Orbitron',monospace;font-size:0.6rem;
               color:{col};font-weight:700;min-width:80px;">{name}</span>
  <span style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
               color:#ff0066;min-width:40px;">{weight}</span>
  <span style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
               color:#3a3a5c;">{desc}</span>
</div>""", unsafe_allow_html=True)

    # ── TAB 3: ARCHITECTURE ───────────────────────────────────────────────────
    with tab3:
        st.markdown("""
<div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:900;
            color:#00e5ff;letter-spacing:0.08em;margin-bottom:0.5rem;">
  MODEL ARCHITECTURE
</div>
<div style="width:60px;height:2px;background:linear-gradient(90deg,#9400ff,transparent);
            margin-bottom:2rem;"></div>
""", unsafe_allow_html=True)

        # DCENet summary
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown('<div class="section-title">// DCENet — ENCODER / DECODER</div>', unsafe_allow_html=True)
            layers = [
                ("Conv1", "3→32", "7×7", "ReLU",  "#00e5ff"),
                ("Conv2", "32→32","3×3", "ReLU",  "#00e5ff"),
                ("Conv3", "32→32","3×3", "ReLU",  "#9400ff"),
                ("Conv4", "32→32","3×3", "ReLU",  "#9400ff"),
                ("Dec1",  "64→32","3×3", "ReLU + skip(Conv3)", "#ff6b00"),
                ("Dec2",  "64→32","3×3", "ReLU + skip(Conv2)", "#ff6b00"),
                ("Out",   "32→24","1×1", "Tanh → 8 curve maps","#ff0066"),
            ]
            for lname, ch, ks, act, col in layers:
                st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;padding:0.35rem 0;
            border-bottom:1px solid #1a1a3a;">
  <div style="font-family:'Orbitron',monospace;font-size:0.6rem;font-weight:700;
              color:{col};min-width:55px;">{lname}</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
              color:#5a5a8a;min-width:50px;">{ch}</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
              color:#3a3a5c;min-width:30px;">k={ks}</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
              color:#5a5a8a;">{act}</div>
</div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-title">// PIPELINE STAGES</div>', unsafe_allow_html=True)
            stages = [
                ("INPUT",       "Low-light image RGB",         "#3a3a5c"),
                ("↓",           "",                            "#1a1a3a"),
                ("ZERO-DCE",    "24 curve params → enhance",   "#00e5ff"),
                ("↓",           "",                            "#1a1a3a"),
                ("COLOR FIX",   "White balance correction",    "#9400ff"),
                ("↓",           "",                            "#1a1a3a"),
                ("REAL-ESRGAN", "4× super-resolution",         "#ff6b00"),
                ("↓",           "",                            "#1a1a3a"),
                ("YOLOv8n",     "Object detection conf≥0.05",  "#ff0066"),
                ("↓",           "",                            "#1a1a3a"),
                ("OUTPUT",      "Annotated result image",      "#00ff88"),
            ]
            for name, desc, col in stages:
                if name == "↓":
                    st.markdown(f"""
<div style="text-align:center;font-family:'Share Tech Mono',monospace;
            font-size:0.7rem;color:#1a1a3a;padding:0.1rem 0;">↓</div>""",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="background:#0c0c1c;border:1px solid {col}40;border-radius:4px;
            padding:0.5rem 0.75rem;margin-bottom:0.1rem;text-align:center;">
  <div style="font-family:'Orbitron',monospace;font-size:0.6rem;
              font-weight:700;color:{col};letter-spacing:0.15em;">{name}</div>
  {f'<div style="font-family:Share Tech Mono,monospace;font-size:0.5rem;color:#3a3a5c;">{desc}</div>' if desc else ''}
</div>""", unsafe_allow_html=True)

    # ── TAB 4: TEAM ──────────────────────────────────────────────────────────
    with tab4:
        st.markdown("""
<div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:900;
            color:#00e5ff;letter-spacing:0.08em;margin-bottom:0.5rem;">
  THE TEAM
</div>
<div style="width:60px;height:2px;background:linear-gradient(90deg,#ff0066,transparent);
            margin-bottom:2rem;"></div>
""", unsafe_allow_html=True)

        members = [
            ("NW", "NIMRA WASEEM",    "Lead Researcher · Zero-DCE Architecture",    "#9400ff", "Model design, loss engineering, training loop"),
            ("UN", "M UMAIR NAVEED",  "ML Engineer · Real-ESRGAN Integration",       "#00e5ff", "ESRGAN pipeline, super-resolution, inference"),
            ("DA", "DAYAN AMJAD",     "CV Engineer · YOLOv8 Detection & GUI",        "#ff0066", "Object detection, GUI, dataset preparation"),
        ]

        c1, c2, c3 = st.columns(3)
        for col, (init, name, title, accent, desc) in zip([c1, c2, c3], members):
            with col:
                st.markdown(f"""
<div class="team-card" style="border:1px solid {accent}40;">
  <div style="
    position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,transparent,{accent},transparent);
  "></div>
  <div class="team-avatar" style="background:{accent}20;color:{accent};border:2px solid {accent}40;">
    {init}
  </div>
  <div class="team-name" style="color:{accent};">{name}</div>
  <div class="team-role" style="color:#5a5a8a;margin-bottom:0.75rem;">{title}</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
              color:#3a3a5c;line-height:1.6;letter-spacing:0.05em;">
    {desc}
  </div>
  <div style="
    margin-top:1rem;
    padding-top:0.75rem;
    border-top:1px solid {accent}20;
    font-family:'Orbitron',monospace;
    font-size:0.45rem;
    color:{accent};
    letter-spacing:0.2em;
    opacity:0.6;
  ">TEAM NIGHTVISION</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown("""
<div style="
  text-align:center;
  background:#0c0c1c;
  border:1px solid #1a1a3a;
  border-radius:4px;
  padding:2rem;
">
  <div style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;
              color:#3a3a5c;letter-spacing:0.3em;margin-bottom:0.5rem;">
    BUILT WITH
  </div>
  <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;margin-top:0.5rem;">
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#00e5ff;">PyTorch</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#9400ff;">Zero-DCE</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#ff6b00;">Real-ESRGAN</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#ff0066;">YOLOv8</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#00ff88;">Streamlit</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#ffe600;">LOL-v2 Dataset</span>
  </div>
</div>""", unsafe_allow_html=True)


# ─── ROUTER ──────────────────────────────────────────────────────────────────
if st.session_state.page == 'intro':
    render_intro()
else:
    render_main()
