"""
NIGHTVISION — Streamlit GUI
Zero-DCE Low-Light Image Enhancement + YOLOv8 Object Detection
--------------------------------------------------------------------
Run with:
    streamlit run streamlit_gui.py
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import io
import os
import time
import sys

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIGHTVISION — Low-Light Enhancer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DARK NEON THEME  (injected CSS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── global background ──────────────────────────────────────────────────── */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      background-color: #050510 !important;
      color: #e0e0ff !important;
  }
  [data-testid="stSidebar"] {
      background-color: #0a0a1a !important;
      border-right: 1px solid #1a1a3e;
  }
  [data-testid="stSidebar"] * { color: #e0e0ff !important; }

  /* ── headline ────────────────────────────────────────────────────────────── */
  .hero-title {
      font-family: 'Courier New', monospace;
      font-size: 2.6rem;
      font-weight: 900;
      color: #00fff5;
      text-shadow: 0 0 18px #00fff5, 0 0 40px #00fff580;
      letter-spacing: 0.12em;
      margin-bottom: 0;
  }
  .hero-sub {
      font-family: 'Courier New', monospace;
      font-size: 0.85rem;
      color: #5a5a8a;
      letter-spacing: 0.06em;
      margin-top: 2px;
  }

  /* ── metric cards ───────────────────────────────────────────────────────── */
  .metric-card {
      background: #0d0d20;
      border: 1px solid #1a1a3e;
      border-radius: 10px;
      padding: 14px 18px;
      text-align: center;
      font-family: 'Courier New', monospace;
  }
  .metric-label { font-size: 0.7rem; color: #5a5a8a; letter-spacing: 0.1em; }
  .metric-value { font-size: 1.4rem; font-weight: 700; }

  /* ── image panel wrapper ─────────────────────────────────────────────────── */
  .img-panel {
      border: 1px solid #1a1a3e;
      border-radius: 10px;
      padding: 6px;
      background: #0d0d20;
  }
  .panel-label {
      font-family: 'Courier New', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      font-weight: 700;
      padding: 4px 0 6px 4px;
  }

  /* ── detection badge ────────────────────────────────────────────────────── */
  .det-badge {
      display:inline-block;
      background:#0d0d20;
      border:1px solid #ff006e;
      border-radius:6px;
      padding:3px 10px;
      font-family:'Courier New',monospace;
      font-size:0.75rem;
      color:#ff006e;
      margin:3px 4px;
  }

  /* ── status bar ─────────────────────────────────────────────────────────── */
  .status-bar {
      background:#0a0a1a;
      border:1px solid #1a1a3e;
      border-radius:8px;
      padding:8px 16px;
      font-family:'Courier New',monospace;
      font-size:0.78rem;
      color:#00fff5;
      margin-bottom:10px;
  }

  /* ── file uploader custom ───────────────────────────────────────────────── */
  [data-testid="stFileUploadDropzone"] {
      background: #0d0d20 !important;
      border: 2px dashed #00fff540 !important;
      border-radius: 10px !important;
  }

  /* ── buttons ────────────────────────────────────────────────────────────── */
  .stButton>button {
      background: #050510 !important;
      border: 1px solid #00fff5 !important;
      color: #00fff5 !important;
      font-family: 'Courier New', monospace !important;
      font-weight: 700 !important;
      letter-spacing: 0.08em !important;
      border-radius: 8px !important;
      transition: all 0.2s;
  }
  .stButton>button:hover {
      background: #00fff510 !important;
      box-shadow: 0 0 14px #00fff550 !important;
  }

  /* ── slider / checkbox ──────────────────────────────────────────────────── */
  .stSlider [data-testid="stSlider"] > div { color: #00fff5; }
  div[data-baseweb="checkbox"] label { font-family: 'Courier New', monospace; }

  /* ── spinner override ────────────────────────────────────────────────────── */
  .stSpinner > div { border-top-color: #ff006e !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (copied verbatim from the notebook)
# ─────────────────────────────────────────────────────────────────────────────

class DCENet(nn.Module):
    """Original Zero-DCE backbone (7-layer CNN with skip connections)."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 32, 3, padding=1)
        self.final = nn.Conv2d(32, 24, 3, padding=1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x4, x3], 1)))
        x6 = self.relu(self.conv6(torch.cat([x5, x2], 1)))
        x7 = self.relu(self.conv7(torch.cat([x6, x1], 1)))
        return self.tanh(self.final(x7))


class DCENetPro(nn.Module):
    """Enhanced Zero-DCE with BatchNorm, Dropout, and MaxPool."""
    def __init__(self, dropout_p=0.1):
        super().__init__()
        def conv_bn_relu(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1  = conv_bn_relu(3,  32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2  = conv_bn_relu(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_bn_relu(64, 64)
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = conv_bn_relu(64 + 64, 32)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = conv_bn_relu(32 + 32, 32)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.final = nn.Sequential(nn.Conv2d(32, 24, 3, padding=1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return self.final(self.dropout(d2))


def enhance(x, r):
    """Iterative curve enhancement — 8 rounds of x + r*(x²-x)."""
    for i in range(0, 24, 3):
        r_i = r[:, i:i+3, :, :]
        x = x + r_i * (x * x - x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_zero_dce(model_path: str, model_variant: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_variant == "DCENet-Pro (BN+Dropout)":
        model = DCENetPro(dropout_p=0.1).to(device)
    else:
        model = DCENet().to(device)

    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            loaded = True
        except Exception as e:
            st.warning(f"Could not load weights: {e}. Using random weights for demo.")
            loaded = False
    else:
        loaded = False

    model.eval()
    return model, device, loaded


@st.cache_resource(show_spinner=False)
def load_yolo(model_name: str):
    try:
        from ultralytics import YOLO
        yolo = YOLO(model_name)
        return yolo, True
    except Exception as e:
        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_already_bright(img_pil, threshold=0.60):
    arr = np.array(img_pil).astype(np.float32) / 255.0
    return arr.mean() >= threshold, float(arr.mean())


def run_zero_dce(img_pil: Image.Image, model, device, image_size: int = 256):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    orig_w, orig_h = img_pil.size
    x = tfm(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        r   = model(x)
        out = enhance(x, r)
    enh_small = out.squeeze().cpu().permute(1, 2, 0).numpy()
    enh_small = np.clip(enh_small, 0, 1)
    enh_pil   = Image.fromarray((enh_small * 255).astype(np.uint8))
    enh_pil   = enh_pil.resize((orig_w, orig_h), Image.LANCZOS)
    return enh_pil, enh_small


def run_yolo(enh_np: np.ndarray, yolo, conf: float = 0.05, iou: float = 0.30):
    import cv2
    img_uint8 = np.ascontiguousarray((np.clip(enh_np, 0, 1) * 255).astype(np.uint8))
    results   = yolo(img_uint8, conf=conf, iou=iou, verbose=False)
    annotated = cv2.cvtColor(results[0].plot(line_width=2), cv2.COLOR_BGR2RGB)
    boxes     = results[0].boxes
    detections = []
    if boxes is not None and len(boxes):
        for cls, cf in zip(
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy()
        ):
            detections.append({"class": yolo.names[cls], "conf": float(cf)})
    return Image.fromarray(annotated), detections


def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    model_variant = st.selectbox(
        "Zero-DCE Variant",
        ["DCENet (Original)", "DCENet-Pro (BN+Dropout)"],
        help="DCENet-Pro has BatchNorm and Dropout for better generalisation."
    )

    model_path = st.text_input(
        "Model weights path (.pth)",
        value="best_zero_dce_lolv2.pth",
        help="Path to your trained Zero-DCE checkpoint. Leave default if you have trained the notebook."
    )

    st.markdown("---")
    st.markdown("### 🎯 Detection Settings")

    yolo_model_name = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt",
         "yolo11x-oi.pt", "yolov8x-oiv7.pt"],
        index=0,
        help="Larger models are more accurate but slower."
    )

    conf_thresh = st.slider(
        "Detection Confidence Threshold",
        min_value=0.01, max_value=0.95, value=0.10, step=0.01,
        help="Lower = more detections (but more false positives)."
    )

    iou_thresh = st.slider(
        "NMS IoU Threshold",
        min_value=0.10, max_value=0.90, value=0.30, step=0.05,
        help="Controls overlap suppression between boxes."
    )

    st.markdown("---")
    st.markdown("### 🖼️ Enhancement Settings")

    image_size = st.select_slider(
        "Processing Resolution",
        options=[128, 256, 384, 512],
        value=256,
        help="Higher = more detail preserved, but slower. Zero-DCE is fully-convolutional."
    )

    brightness_thresh = st.slider(
        "Already-Bright Threshold",
        min_value=0.2, max_value=0.9, value=0.60, step=0.05,
        help="Images brighter than this won't be enhanced (they're already well-lit)."
    )

    st.markdown("---")
    st.markdown("### 🚀 Pipeline Steps")
    run_yolo_flag = st.checkbox("Run YOLO Detection", value=True)
    show_diff     = st.checkbox("Show Brightness Comparison", value=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-family:Courier New;font-size:0.7rem;color:#5a5a8a;'>"
        "Zero-DCE + YOLOv8 Pipeline<br>"
        "LOL-v2 trained model<br>"
        "Streamlit GUI"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<p class="hero-title">🌙 NIGHTVISION</p>'
    '<p class="hero-sub">ZERO-DCE LOW-LIGHT ENHANCEMENT  //  YOLOV8 OBJECT DETECTION  //  LOL-v2 MODEL</p>',
    unsafe_allow_html=True
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING STATUS
# ─────────────────────────────────────────────────────────────────────────────

col_status1, col_status2 = st.columns(2)

with col_status1:
    with st.spinner("Loading Zero-DCE model..."):
        zdce_model, device, zdce_loaded = load_zero_dce(model_path, model_variant)
    device_name = str(device).upper()
    if zdce_loaded:
        st.success(f"✅ Zero-DCE loaded from `{model_path}` · Device: **{device_name}**")
    else:
        st.warning(f"⚠️ No saved weights found at `{model_path}`. Running in demo mode (untrained). Train the notebook first!")

with col_status2:
    if run_yolo_flag:
        with st.spinner(f"Loading {yolo_model_name}..."):
            yolo_model, yolo_loaded = load_yolo(yolo_model_name)
        if yolo_loaded:
            st.success(f"✅ YOLO loaded: `{yolo_model_name}`")
        else:
            st.error("❌ YOLO failed to load. Install `ultralytics`: `pip install ultralytics`")
    else:
        st.info("ℹ️ YOLO detection is disabled in settings.")


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD AREA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📤 Upload Your Image")
st.markdown(
    "<div style='font-family:Courier New;font-size:0.78rem;color:#5a5a8a;'>"
    "Upload a low-light image (JPG, PNG, WEBP). The model will enhance brightness and run object detection."
    "</div>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed"
)

if uploaded is None:
    # Demo placeholder
    st.markdown("""
    <div style='
        background:#0d0d20;
        border:2px dashed #1a1a3e;
        border-radius:12px;
        padding:40px;
        text-align:center;
        font-family:"Courier New",monospace;
        color:#5a5a8a;
        margin-top:10px;
    '>
        <div style='font-size:2.5rem;margin-bottom:12px;'>🌑</div>
        <div style='font-size:1rem;color:#00fff560;'>DROP A LOW-LIGHT IMAGE HERE</div>
        <div style='font-size:0.72rem;margin-top:8px;'>
            Supported: JPG · PNG · WEBP<br>
            The model will automatically detect if the image is already bright.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS IMAGE
# ─────────────────────────────────────────────────────────────────────────────

original_pil = Image.open(uploaded).convert("RGB")
orig_w, orig_h = original_pil.size
orig_brightness = np.array(original_pil).astype(np.float32).mean() / 255.0

st.markdown("---")
st.markdown("### 🔬 Processing Pipeline")

progress_bar = st.progress(0, text="Starting pipeline…")
log_container = st.empty()
log_lines = []

def log(msg):
    ts = time.strftime("%H:%M:%S")
    log_lines.append(f"[{ts}] {msg}")
    log_container.code("\n".join(log_lines[-8:]), language=None)

log(f"Image loaded: {orig_w}×{orig_h} px · Mean brightness: {orig_brightness:.3f}")
progress_bar.progress(10, text="Image loaded…")

# ── Step 1: Brightness check ─────────────────────────────────────────────────
already_bright, mean_bri = is_already_bright(original_pil, brightness_thresh)
if already_bright:
    log(f"Image is already bright ({mean_bri:.3f} ≥ threshold {brightness_thresh:.2f}) — skipping Zero-DCE")
    enhanced_pil = original_pil.copy()
    enhanced_arr = np.array(enhanced_pil).astype(np.float32) / 255.0
    skip_enhance = True
else:
    skip_enhance = False

progress_bar.progress(20, text="Brightness check done…")

# ── Step 2: Zero-DCE ─────────────────────────────────────────────────────────
t0 = time.time()
if not skip_enhance:
    log("Running Zero-DCE enhancement…")
    enhanced_pil, enhanced_small = run_zero_dce(original_pil, zdce_model, device, image_size)
    enhanced_arr = np.array(enhanced_pil).astype(np.float32) / 255.0
    dce_time = time.time() - t0
    enh_brightness = enhanced_arr.mean()
    log(f"Zero-DCE done in {dce_time:.2f}s · Output brightness: {enh_brightness:.3f}")
else:
    enhanced_arr = np.array(enhanced_pil).astype(np.float32) / 255.0
    enh_brightness = enhanced_arr.mean()
    dce_time = 0.0

progress_bar.progress(55, text="Enhancement done…")

# ── Step 3: YOLO Detection ───────────────────────────────────────────────────
detections     = []
detected_pil   = None
yolo_time      = 0.0

if run_yolo_flag and yolo_loaded:
    log(f"Running YOLO detection (conf={conf_thresh:.2f}, iou={iou_thresh:.2f})…")
    t0 = time.time()
    detected_pil, detections = run_yolo(
        np.array(enhanced_pil).astype(np.float32) / 255.0,
        yolo_model,
        conf=conf_thresh,
        iou=iou_thresh
    )
    yolo_time = time.time() - t0
    log(f"YOLO done in {yolo_time:.2f}s · {len(detections)} object(s) found")
elif run_yolo_flag and not yolo_loaded:
    log("YOLO not available — skipping detection step")

progress_bar.progress(90, text="Detection done…")

# ── Done ─────────────────────────────────────────────────────────────────────
progress_bar.progress(100, text="✅ Pipeline complete!")
log("Pipeline complete.")
time.sleep(0.3)
progress_bar.empty()


# ─────────────────────────────────────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📊 Results Summary")

m1, m2, m3, m4, m5 = st.columns(5)

brightness_gain = enh_brightness - orig_brightness
brightness_gain_str = f"+{brightness_gain:.3f}" if brightness_gain >= 0 else f"{brightness_gain:.3f}"
gain_color = "#00ff88" if brightness_gain >= 0 else "#ff006e"

def metric_card(label, value, color="#00fff5"):
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'</div>'
    )

m1.markdown(metric_card("INPUT SIZE", f"{orig_w}×{orig_h}"), unsafe_allow_html=True)
m2.markdown(metric_card("ORIG BRIGHTNESS", f"{orig_brightness:.3f}"), unsafe_allow_html=True)
m3.markdown(metric_card("ENH BRIGHTNESS",  f"{enh_brightness:.3f}", gain_color), unsafe_allow_html=True)
m4.markdown(metric_card("BRIGHTNESS GAIN", brightness_gain_str, gain_color), unsafe_allow_html=True)
m5.markdown(metric_card("OBJECTS FOUND",   str(len(detections)), "#ff006e"), unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PANELS
# ─────────────────────────────────────────────────────────────────────────────

n_panels = 3 if (detected_pil is not None) else 2
cols = st.columns(n_panels)

# Panel 1: Original
with cols[0]:
    st.markdown(
        '<div class="panel-label" style="color:#00fff5;">[ ORIGINAL INPUT ]</div>',
        unsafe_allow_html=True
    )
    st.image(original_pil, use_container_width=True)
    st.caption(f"Size: {orig_w}×{orig_h}  |  Brightness: {orig_brightness:.3f}")

# Panel 2: Enhanced
with cols[1]:
    label2_text = "ALREADY BRIGHT — UNCHANGED" if skip_enhance else "[ ZERO-DCE ENHANCED ]"
    label2_color = "#ffe600" if skip_enhance else "#00ff88"
    st.markdown(
        f'<div class="panel-label" style="color:{label2_color};">[ {label2_text} ]</div>',
        unsafe_allow_html=True
    )
    st.image(enhanced_pil, use_container_width=True)
    note = "Enhancement skipped (image already bright)" if skip_enhance else f"Time: {dce_time:.2f}s  |  Brightness: {enh_brightness:.3f}"
    st.caption(note)

# Panel 3: Detection
if detected_pil is not None:
    with cols[2]:
        st.markdown(
            '<div class="panel-label" style="color:#ff006e;">[ YOLO DETECTIONS ]</div>',
            unsafe_allow_html=True
        )
        st.image(detected_pil, use_container_width=True)
        st.caption(f"Time: {yolo_time:.2f}s  |  Objects: {len(detections)}  |  conf≥{conf_thresh:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# BRIGHTNESS BAR CHART (optional)
# ─────────────────────────────────────────────────────────────────────────────

if show_diff:
    st.markdown("---")
    st.markdown("### 📈 Brightness Comparison")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        orig_arr = np.array(original_pil).astype(np.float32) / 255.0
        enh_arr  = np.array(enhanced_pil).astype(np.float32) / 255.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 3))
        fig.patch.set_facecolor("#050510")

        def channel_hist(ax, arr, channel_idx, color, title):
            data = arr[:, :, channel_idx].flatten()
            ax.hist(data, bins=64, color=color, alpha=0.8, edgecolor="none")
            ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1.2,
                       label=f"mean={data.mean():.3f}")
            ax.set_facecolor("#0d0d20")
            ax.set_title(title, color="#e0e0ff", fontsize=8, fontfamily="monospace")
            ax.tick_params(colors="#5a5a8a", labelsize=7)
            for spine in ax.spines.values(): spine.set_color("#1a1a3e")
            ax.legend(fontsize=7, labelcolor="white", facecolor="#0d0d20", edgecolor="#1a1a3e")

        channel_hist(axes[0], orig_arr, 0, "#e74c3c", "ORIGINAL — Red channel")
        channel_hist(axes[0], enh_arr,  0, "#ff8888", "")
        channel_hist(axes[1], orig_arr, 1, "#2ecc71", "ORIGINAL — Green channel")
        channel_hist(axes[1], enh_arr,  1, "#88ffbb", "")
        channel_hist(axes[2], orig_arr, 2, "#3498db", "ORIGINAL — Blue channel")
        channel_hist(axes[2], enh_arr,  2, "#88ccff", "")

        for i, (label, color) in enumerate([
            ("Original", "#5a5a8a"),
            ("Enhanced", "#00fff5"),
        ]):
            for ax in axes:
                pass  # already plotted with colour distinction

        plt.suptitle(
            "Pixel Intensity Distributions: Original (dark) vs Enhanced (bright)",
            color="#e0e0ff", fontsize=9, fontfamily="monospace"
        )
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="#050510", dpi=120, bbox_inches="tight")
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close(fig)
    except ImportError:
        st.info("Install `matplotlib` for the brightness chart: `pip install matplotlib`")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION TABLE
# ─────────────────────────────────────────────────────────────────────────────

if detections:
    st.markdown("---")
    st.markdown("### 🎯 Detection Results")

    # Badges
    badge_html = ""
    from collections import Counter
    cls_counts = Counter(d["class"] for d in detections)
    for cls, cnt in cls_counts.most_common():
        badge_html += f'<span class="det-badge">{cls}  ×{cnt}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # Table
    import pandas as pd
    df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Class": d["class"],
            "Confidence": f"{d['conf']:.1%}",
            "Confidence (raw)": round(d["conf"], 4),
        }
        for i, d in enumerate(sorted(detections, key=lambda x: -x["conf"]))
    ])
    st.dataframe(
        df[["Rank", "Class", "Confidence"]],
        use_container_width=True,
        hide_index=True,
    )

elif run_yolo_flag and yolo_loaded:
    st.info(f"No objects detected above confidence threshold ({conf_thresh:.0%}). "
            "Try lowering the threshold in the sidebar, or the image may not contain recognisable objects.")


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD BUTTONS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 💾 Download Results")

dl_cols = st.columns(3)

with dl_cols[0]:
    st.download_button(
        label="⬇ Download Enhanced Image",
        data=pil_to_bytes(enhanced_pil),
        file_name=f"enhanced_{uploaded.name}",
        mime="image/png",
        use_container_width=True,
    )

with dl_cols[1]:
    if detected_pil is not None:
        st.download_button(
            label="⬇ Download Detection Result",
            data=pil_to_bytes(detected_pil),
            file_name=f"detected_{uploaded.name}",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("⬇ Detection (disabled)", disabled=True, use_container_width=True)

with dl_cols[2]:
    # Side-by-side comparison image
    comp_w = orig_w * 2 if detected_pil is None else orig_w * 3
    comp   = Image.new("RGB", (comp_w, orig_h + 36), (5, 5, 16))
    comp.paste(original_pil, (0, 36))
    comp.paste(enhanced_pil.resize((orig_w, orig_h)), (orig_w, 36))
    if detected_pil is not None:
        comp.paste(detected_pil.resize((orig_w, orig_h)), (orig_w * 2, 36))
    draw = ImageDraw.Draw(comp)
    draw.rectangle([0, 0, comp_w, 35], fill=(10, 10, 26))
    draw.text((10, 10), "Original", fill=(0, 255, 245))
    draw.text((orig_w + 10, 10), "Enhanced", fill=(0, 255, 136))
    if detected_pil:
        draw.text((orig_w * 2 + 10, 10), "Detected", fill=(255, 0, 110))

    st.download_button(
        label="⬇ Download Side-by-Side",
        data=pil_to_bytes(comp),
        file_name=f"comparison_{uploaded.name}",
        mime="image/png",
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE — collapsible
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️  How to use this app / FAQ"):
    st.markdown("""
**How do I use this?**
1. Make sure you have trained the Zero-DCE model (run your Jupyter notebook through Phase 4).
2. Set the **Model weights path** in the sidebar to your `.pth` file (default: `best_zero_dce_lolv2.pth`).
3. Upload any dark or low-light image using the uploader above.
4. The app will automatically enhance it and run object detection.

**What models does this use?**
- **Zero-DCE (DCENet / DCENet-Pro)** — Trained on LOL-v2 dataset, performs unsupervised low-light enhancement using curve estimation.
- **YOLOv8** — Pre-trained on COCO (or Open Images), runs object detection on the enhanced image.

**The weights file isn't found — what do I do?**
- Run your notebook to completion. The best model is saved as `best_zero_dce_lolv2.pth` in the same folder.
- Or change the path in the sidebar to wherever you saved your model.

**Why is the image not being enhanced?**
- If the input image is already bright (above the threshold), Zero-DCE is skipped.
- You can lower the **Already-Bright Threshold** slider in the sidebar.

**How do I get more/fewer detections?**
- Lower the **Confidence Threshold** to detect more objects (with potentially more false positives).
- Raise it to only see high-confidence detections.

**Can I use my own custom YOLO model?**
- Yes! Enter the path to your `.pt` file in the **YOLO Model** dropdown or type directly.

**Running locally:**
```bash
pip install streamlit torch torchvision ultralytics pillow numpy matplotlib pandas
streamlit run streamlit_gui.py
```
""")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='font-family:Courier New;font-size:0.68rem;color:#2a2a4a;text-align:center;'>"
    "NIGHTVISION  //  Zero-DCE + YOLOv8 Pipeline  //  LOL-v2 Trained Model"
    "</div>",
    unsafe_allow_html=True
)
