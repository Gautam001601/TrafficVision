from pathlib import Path
from collections import Counter
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ─── Model Download from Google Drive ───────────────────────────────────────
MODEL_PATH = Path("best.pt")

def download_model():
    if not MODEL_PATH.exists():
        try:
            import gdown
            st.info("⬇️ Downloading model from Google Drive...")
            file_id = "1XWlnqkcUtW86K5qVz5nXNKg3OuZvfL4C"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(MODEL_PATH), quiet=False)
            st.success("✅ Model downloaded!")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            st.stop()

download_model()

from ultralytics import YOLO

# ─── Config (inline, no local config.py needed) ──────────────────────────────
RUN_NAME = "vehicle_detection_nano-3"
DEVICE = "cpu"  # Streamlit Cloud has no GPU

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VehicleDetect · YOLOv8",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800;900&display=swap');

:root {
    --bg: #0b0c08;
    --gold: #E3A545;
    --sage: #91A268;
    --muted: rgba(244,245,239,0.65);
    --text: #F4F5EF;
    --card: rgba(27,28,15,0.75);
    --border: rgba(227,165,69,0.18);
    --shadow: 0 20px 60px rgba(0,0,0,0.5);
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}
.stApp {
    background: linear-gradient(160deg, #0f1009 0%, #0b0c08 60%, #0e0d07 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1200px; }

.hero-banner {
    background:
        linear-gradient(135deg, rgba(227,165,69,0.08) 0%, rgba(145,162,104,0.05) 100%),
        linear-gradient(180deg, rgba(27,28,15,0.9) 0%, rgba(11,12,8,0.95) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 52px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(227,165,69,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(145,162,104,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-label {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 12px;
}
.hero-title {
    font-size: clamp(36px, 5vw, 64px);
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -1.5px;
    margin: 0 0 16px;
    background: linear-gradient(135deg, #F4F5EF 0%, #E3A545 70%, #91A268 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: var(--muted) !important;
    font-size: 16px;
    line-height: 1.6;
    max-width: 55ch;
    margin-bottom: 24px;
}
.hero-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 14px;
    border-radius: 999px;
    border: 1px solid rgba(227,165,69,0.25);
    background: rgba(227,165,69,0.08);
    font-size: 13px;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card .label {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted) !important;
    margin-bottom: 6px;
}
.metric-card .value { font-size: 26px; font-weight: 900; color: var(--gold) !important; line-height: 1; }
.metric-card .value.sage { color: var(--sage) !important; }
.metric-card .value.white { color: var(--text) !important; }
.section-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: hidden;
    backdrop-filter: blur(14px);
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}
.section-header {
    padding: 22px 24px 10px;
    border-bottom: 1px solid rgba(227,165,69,0.1);
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(227,165,69,0.2), rgba(145,162,104,0.15));
    border: 1px solid rgba(227,165,69,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.section-title { font-size: 18px; font-weight: 800; margin: 0; color: var(--text) !important; }
.section-sub { font-size: 13px; color: var(--muted) !important; margin: 0; }
.upload-hint {
    text-align: center;
    padding: 18px;
    font-size: 13px;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
}
.stFileUploader > div {
    background: rgba(18,20,12,0.5) !important;
    border: 1.5px dashed rgba(227,165,69,0.4) !important;
    border-radius: 16px !important;
}
.stFileUploader > div:hover {
    border-color: rgba(227,165,69,0.7) !important;
    background: rgba(227,165,69,0.05) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(18,20,12,0.4) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(227,165,69,0.15) !important;
    color: var(--gold) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #E3A545, #c98f38) !important;
    color: #1B1C0F !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
}
div[data-testid="stSidebar"] {
    background: rgba(11,12,8,0.95) !important;
    border-right: 1px solid var(--border) !important;
}
div[data-testid="stSidebar"] * { color: var(--text) !important; }
.det-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 8px; }
.det-table th {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid rgba(227,165,69,0.15);
}
.det-table td { padding: 10px 14px; border-bottom: 1px solid rgba(142,145,140,0.1); color: var(--text); }
.det-table tr:last-child td { border-bottom: none; }
.conf-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid rgba(227,165,69,0.35);
    background: rgba(227,165,69,0.1);
    color: #E3A545;
}
.conf-badge.high {
    border-color: rgba(145,162,104,0.4);
    background: rgba(145,162,104,0.1);
    color: #91A268;
}
.class-tag {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 8px;
    background: rgba(27,28,15,0.6);
    border: 1px solid rgba(142,145,140,0.2);
    font-weight: 600;
}
.info-box {
    background: rgba(145,162,104,0.08);
    border: 1px solid rgba(145,162,104,0.25);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 13px;
    color: rgba(244,245,239,0.8) !important;
    margin: 8px 0;
}
.warn-box {
    background: rgba(227,165,69,0.08);
    border: 1px solid rgba(227,165,69,0.25);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 13px;
    color: rgba(244,245,239,0.8) !important;
    margin: 8px 0;
}
.img-panel-label {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted) !important;
    margin-bottom: 8px;
    text-align: center;
}
.sidebar-section {
    background: rgba(27,28,15,0.6);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.sidebar-label {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH)), MODEL_PATH, "trained"
    return None, None, None


def get_detection_table(result):
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    names = result.names
    rows = []
    for cls_id, conf, box in zip(cls_ids, confs, boxes):
        rows.append({
            "Class": names[int(cls_id)],
            "Confidence": round(float(conf), 3),
            "x1": int(box[0]), "y1": int(box[1]),
            "x2": int(box[2]), "y2": int(box[3]),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Confidence", ascending=False).reset_index(drop=True)
    return df


def run_detection(image_np, conf_threshold):
    result = model.predict(
        source=image_np,
        conf=conf_threshold,
        device=DEVICE,
        verbose=False,
    )[0]
    return result


model, model_path, model_type = load_model()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px'>
        <div style='font-size:22px; font-weight:900; letter-spacing:-0.5px'>🚗 VehicleDetect</div>
        <div style='font-family:"Space Mono",monospace; font-size:10px; letter-spacing:3px;
        color:rgba(244,245,239,0.5); margin-top:4px'>YOLOv8 NANO · DETECTION</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Detection Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider(
        "Confidence Threshold", 0.10, 1.0, 0.25, 0.05,
        help="Lower = more detections. Higher = only confident detections."
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Model Info</div>', unsafe_allow_html=True)

    if model is None:
        st.error("No model found.")
    else:
        st.markdown(f"""
        <div class="sidebar-section">
            <div style='font-size:13px; margin-bottom:8px'>
                <span style='background:rgba(27,28,15,0.8); border:1px solid #91A26840;
                color:#91A268; padding:3px 10px; border-radius:999px; font-size:11px;
                font-family:"Space Mono",monospace'>✅ Fine-tuned</span>
            </div>
            <div style='font-family:"Space Mono",monospace; font-size:11px;
            color:rgba(244,245,239,0.6)'>best.pt</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">12 Classes</div>', unsafe_allow_html=True)
    classes = ["Big Bus","Big Truck","Bus-L","Bus-S","Car","Mid Truck",
               "Small Bus","Small Truck","Truck-L","Truck-M","Truck-S","Truck-XL"]
    for cls in classes:
        st.markdown(f"<div style='font-size:12px; color:rgba(244,245,239,0.6); padding:2px 0'>• {cls}</div>",
                    unsafe_allow_html=True)


# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-label">Real-Time AI Detection System</div>
    <h1 class="hero-title">Vehicle Detection<br>with YOLOv8 Nano</h1>
    <p class="hero-sub">Upload an image or use your camera to instantly detect and classify vehicles —
    cars, trucks, buses, and more — using a deep learning model trained on 4,058 real-world traffic images.</p>
    <div class="hero-pills">
        <div class="pill">🎯 12 Classes</div>
        <div class="pill">⚡ ~12ms Inference</div>
        <div class="pill">🧠 3.2M Params</div>
        <div class="pill">🌍 Day · Night · Rain</div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not loaded. Please refresh the page.")
    st.stop()


# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷  Image Upload", "📸  Camera"])


def display_results(image, result, source_label="Original"):
    table = get_detection_table(result)
    annotated = result.plot()

    total = len(table)
    top_conf = f"{table['Confidence'].max() * 100:.1f}%" if total > 0 else "—"
    unique_cls = table["Class"].nunique() if total > 0 else 0
    avg_conf = f"{table['Confidence'].mean() * 100:.1f}%" if total > 0 else "—"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Detections</div>
            <div class="value">{total}</div>
        </div>
        <div class="metric-card">
            <div class="label">Top Confidence</div>
            <div class="value sage">{top_conf}</div>
        </div>
        <div class="metric-card">
            <div class="label">Unique Classes</div>
            <div class="value white">{unique_cls}</div>
        </div>
        <div class="metric-card">
            <div class="label">Avg Confidence</div>
            <div class="value">{avg_conf}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(f'<div class="img-panel-label">🖼 {source_label}</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col2:
        st.markdown('<div class="img-panel-label">🎯 Annotated Output</div>', unsafe_allow_html=True)
        st.image(annotated, channels="BGR", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <div class="section-icon">📋</div>
                <div>
                    <div class="section-title">Detection Results</div>
                    <div class="section-sub">Sorted by confidence (highest first)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if table.empty:
            st.markdown('<div class="warn-box">⚠️ No vehicles detected. Try lowering the confidence threshold.</div>',
                        unsafe_allow_html=True)
        else:
            rows_html = ""
            for _, row in table.iterrows():
                conf_pct = row["Confidence"] * 100
                badge_class = "high" if conf_pct >= 70 else ""
                rows_html += f"""
                <tr>
                    <td><span class="class-tag">{row['Class']}</span></td>
                    <td><span class="conf-badge {badge_class}">{conf_pct:.1f}%</span></td>
                    <td style='font-family:"Space Mono",monospace; font-size:12px;
                    color:rgba(244,245,239,0.6)'>
                        [{row['x1']}, {row['y1']}, {row['x2']}, {row['y2']}]
                    </td>
                </tr>"""
            st.markdown(f"""
            <table class="det-table">
                <thead><tr>
                    <th>Class</th><th>Confidence</th><th>Bounding Box</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <div class="section-icon">📊</div>
                <div>
                    <div class="section-title">Class Distribution</div>
                    <div class="section-sub">Count per vehicle type</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not table.empty:
            counts = table["Class"].value_counts()
            st.bar_chart(counts, color="#E3A545", use_container_width=True)
        else:
            st.markdown('<div class="warn-box" style="margin-top:12px">No data to chart.</div>',
                        unsafe_allow_html=True)


# ─── Tab 1: Image Upload ─────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Drop an image here, or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        with st.spinner("🔍 Running detection…"):
            result = run_detection(image_np, conf_threshold)

        display_results(image, result, source_label="Uploaded Image")
    else:
        st.markdown('<div class="upload-hint">← Adjust confidence in sidebar &nbsp;|&nbsp; Supported: JPG, PNG, WEBP</div>',
                    unsafe_allow_html=True)


# ─── Tab 2: Camera ───────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="info-box">📸 Take a photo using your device camera. Detection runs automatically.</div>',
                unsafe_allow_html=True)
    camera_image = st.camera_input("", label_visibility="collapsed")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)

        with st.spinner("🔍 Running detection…"):
            result = run_detection(image_np, conf_threshold)

        display_results(image, result, source_label="Camera Capture")