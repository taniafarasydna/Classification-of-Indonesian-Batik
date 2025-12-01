import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================
# BACKGROUND MOTIF BATIK
# ============================================
batik_bg = "https://i.imgur.com/7sYzjNw.png"


# ============================================
# GLOBAL CSS
# ============================================
st.markdown(f"""
<style>

.block-container {{
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}}

body {{
    background-image: url('{batik_bg}');
    background-size: 260px;
    background-repeat: repeat;
    background-attachment: fixed;
}}

.header-container {{
    width: 100%;
    padding: 40px 10px 20px 10px;
    text-align: center;
    background: rgba(255,255,255,0.92);
}}

.header-title {{
    font-family: 'Georgia', serif;
    font-size: 40px;
    font-weight: 700;
    color: #2b2b2b;
}}

.upload-title {{
    font-family: 'Georgia', serif;
    font-size: 20px;
    font-weight: 600;
    color: #8A5A44;
    text-align: center;
    margin-top: 25px;
    margin-bottom: 10px;
}}

.footer {{
    width: 100%;
    background: #CBA35C;
    padding: 20px 0;
    text-align: center;
    position: fixed;
    bottom: 0;
    left: 0;
}}

.footer-text {{
    font-family: 'Georgia', serif;
    color: #3b2e1e;
    font-size: 14px;
}}

</style>
""", unsafe_allow_html=True)


# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="header-container">
    <div class="header-title">Klasifikasi Citra Batik Indonesia</div>
</div>
""", unsafe_allow_html=True)


# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_batik.h5")

model = load_model()

labels = [
    'Batik Bali', 'Batik Betawi', 'Batik Cendrawasih', 'Batik Dayak',
    'Batik Geblek Renteng', 'Batik Ikat Celup', 'Batik Insang',
    'Batik Kawung', 'Batik Lasem', 'Batik Megamendung',
    'Batik Pala', 'Batik Parang', 'Batik Poleng',
    'Batik Sekar Jagad', 'Batik Tambal'
]


# ============================================
# SESSION STATE
# ============================================
if "uploaded" not in st.session_state:
    st.session_state.uploaded = None


# ============================================
# BEFORE UPLOAD (SHOW TITLE + UPLOADER)
# ============================================
if st.session_state.uploaded is None:

    # TEXT TITLE tambahannya
    st.markdown("<div class='upload-title'>Unggah Gambar Batik</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.session_state.uploaded = uploaded_file
        st.rerun()

    st.stop()  # early stop agar footer tetap tampil


# ============================================
# AFTER UPLOAD → HIDE UPLOADER ELEMENTS
# ============================================
st.markdown("""
<style>
.stFileUploader {display:none !important;}
.css-1y4p8pa, .css-12ttj6m {display:none !important;}
</style>
""", unsafe_allow_html=True)


# ============================================
# PREDICT SECTION
# ============================================
img = Image.open(st.session_state.uploaded).convert("RGB")
display_img = img.resize((260, 260))

# Preprocess for model
arr = img.resize((224, 224))
arr = np.array(arr) / 255.0
arr = np.expand_dims(arr, 0)

with st.spinner("Sedang memproses..."):
    pred = model.predict(arr)
    idx = np.argmax(pred)
    conf = np.max(pred) * 100
    predicted_label = labels[idx]


# ============================================
# LAYOUT: IMAGE LEFT + PREDICTION RIGHT
# ============================================
col_spacer, col_img, col_pred = st.columns([0.7, 1, 1.2])

# LEFT IMAGE
with col_img:
    st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='
        border: 4px solid #8A5A44;
        border-radius: 12px;
        padding: 5px;
        display: inline-block;
    '>
    """, unsafe_allow_html=True)
    st.image(display_img)
    st.markdown("</div></div>", unsafe_allow_html=True)

# RIGHT PREDICTION BOX
with col_pred:
    st.markdown("""
    <div style='
        background: #ffffff;
        border: 2px solid #d7c2a8;
        border-radius: 14px;
        padding: 20px 26px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        max-width: 450px;
    '>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:22px; font-weight:bold; color:#8A5A44; font-family:Georgia;'>Hasil Prediksi</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:16px; font-family:Georgia;'>Jenis Batik: <b>{predicted_label}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:16px; font-family:Georgia;'>Confidence: <b>{conf:.2f}%</b></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# RESET BUTTON
st.markdown("<div style='text-align:center; margin-top:30px;'>", unsafe_allow_html=True)

reset = st.button("Reset Gambar")

if reset:
    st.session_state.uploaded = None
    st.rerun()

st.markdown("""
<style>
button[kind="secondary"] {
    background-color: #7A4B2A !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    border: none !important;
    font-family: 'Georgia', serif;
    font-size: 15px;
}
button[kind="secondary"]:hover {
    background-color: #5e381f !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <div class="footer-text">
        Aplikasi ini digunakan untuk mengklasifikasikan jenis batik Indonesia menggunakan model MobileNetV2.<br>
        Developed by <strong>Tania</strong>
    </div>
</div>
""", unsafe_allow_html=True)
