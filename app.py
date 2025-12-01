import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================
# MOTIF BATIK BACKGROUND
# ============================================
batik_bg = "https://i.imgur.com/7sYzjNw.png"   # motif batik halus


# ============================================
# CUSTOM CSS
# ============================================
st.markdown(f"""
<style>

/* FULL WIDTH */
.block-container {{
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}}

/* BACKGROUND */
body {{
    background-image: url('{batik_bg}');
    background-size: 260px;
    background-repeat: repeat;
    background-attachment: fixed;
}}

/* HEADER */
.header-container {{
    width: 100%;
    padding: 40px 10px 15px 10px;
    text-align: center;
    background: rgba(255,255,255,0.92);
}}

.header-title {{
    font-family: 'Georgia', serif;
    font-size: 40px;
    font-weight: 700;
    color: #2b2b2b;
}}

.section-title {{
    margin-top: 35px;
    text-align: center;
    font-family: 'Georgia', serif;
    font-size: 18px;
    font-weight: 600;
    color: #3d3d3d;
}}

/* UPLOADER */
div[data-testid="stFileUploader"] {{
    border: 2px dashed #b6b6b6 !important;
    border-radius: 18px;
    padding: 40px 20px;
    background-color: #ffffffee;
    text-align: center;
    max-width: 360px;
    margin: 20px auto 0 auto;
}}
div[data-testid="stFileUploader"] label {{
    display: none !important;
}}

/* CARD HASIL PREDIKSI */
.prediction-card {{
    background: #ffffff;
    border: 2px solid #d7c2a8;
    border-radius: 14px;
    padding: 20px 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    max-width: 450px;
}}

.prediction-title {{
    font-size: 22px;
    font-weight: bold;
    color: #8A5A44;
    margin-bottom: 10px;
    font-family: 'Georgia', serif;
}}

.prediction-text {{
    font-size: 16px;
    color: #3c2f27;
    margin-top: 6px;
    font-family: 'Georgia', serif;
}}

/* Gambar lebih ke tengah */
.image-wrapper {{
    display: flex;
    justify-content: center;
    margin-left: 80px;
}}

/* FOOTER */
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
    font-family: 'Georgia', serif';
    color: #3b2e1e;
    font-size: 14px;
}}

/* TOOLTIP */
.tooltip {{
    position: relative;
    display: inline-block;
    cursor: pointer;
    color: #8A5A44;
    font-weight: 600;
}}

.tooltip .tooltiptext {{
    visibility: hidden;
    width: 240px;
    background-color: #fdf7ee;
    color: #5a4a36;
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #d2b48c;
    position: absolute;
    z-index: 1;
    top: -8px;
    left: 110%;
}}

.tooltip:hover .tooltiptext {{
    visibility: visible;
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
# LOAD MODEL
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
# UPLOAD TITLE
# ============================================
st.markdown("<div class='section-title'>Unggah Gambar Batik</div>", unsafe_allow_html=True)


# ============================================
# UPLOADER
# ============================================
if "uploaded" not in st.session_state:
    st.session_state.uploaded = None

if st.session_state.uploaded is None:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.session_state.uploaded = uploaded_file
        st.rerun()

else:
    # ============================================
    # DISPLAY + PREDICTION
    # ============================================

    img = Image.open(st.session_state.uploaded).convert("RGB")
    display_img = img.resize((250, 250))

    # Preprocess for model
    arr = img.resize((224, 224))
    arr = np.array(arr) / 255.0
    arr = np.expand_dims(arr, 0)

    with st.spinner("Sedang memproses..."):
        pred = model.predict(arr)
        idx = np.argmax(pred)
        conf = np.max(pred) * 100
        predicted_label = labels[idx]

    col1, col2 = st.columns([1, 1], gap="large")

    # Gambar kiri
    with col1:
        st.markdown("<div class='image-wrapper'>", unsafe_allow_html=True)

        st.markdown("""
        <div style='
            border: 4px solid #8A5A44;
            border-radius: 12px;
            padding: 5px;
            display: inline-block;
        '>
        """, unsafe_allow_html=True)

        st.image(display_img, use_column_width=False)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # Hasil prediksi kanan
    with col2:
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown("<div class='prediction-title'>Hasil Prediksi</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='prediction-text'>Jenis Batik: <b>{predicted_label}</b></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='prediction-text'>Confidence: <b>{conf:.2f}%</b></div>",
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Reset button
    st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
    if st.button("Reset Gambar"):
        st.session_state.uploaded = None
        st.rerun()
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
