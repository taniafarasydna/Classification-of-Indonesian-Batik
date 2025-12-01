import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
html, body, .main {
    padding: 0;
    margin: 0;
}

/* HEADER */
.header-container {
    background: #E8EEF5; /* warna pastel lembut */
    padding: 35px 10px;
    text-align: center;
    border-bottom: 3px solid #c7d3e0;
}
.header-title {
    font-size: 34px;
    font-weight: 700;
    color: #3a3a3a;
}
.header-subtitle {
    font-size: 18px;
    color: #555;
}

/* Uploader jadi 1 kotak dashed */
div[data-testid="stFileUploader"] {
    border: 2px dashed #b8b8b8;
    border-radius: 16px;
    padding: 40px 20px;
    background-color: #fafafa;
    text-align: center;
    max-width: 320px;
    margin: 35px auto 10px auto;
}
div[data-testid="stFileUploader"] section {
    text-align: center;
}
div[data-testid="stFileUploader"] label {
    font-weight: 500;
}

/* Text kecil di bawah kotak upload */
.upload-helper {
    text-align: center;
    color: #888;
    font-size: 13px;
}

/* Box hasil prediksi */
.pred-box {
    border-radius: 12px;
    padding: 20px;
    background-color: #ffffff;
    border-left: 6px solid #4F8EF7;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin-top: 20px;
}
.pred-label {
    font-size: 20px;
    font-weight: bold;
    color: #4F8EF7;
}

/* FOOTER */
.main {
    padding-bottom: 90px; /* supaya konten tidak ketutupan footer */
}
.footer {
    width: 100%;
    background: #c7d3e0; /* sedikit lebih gelap dari header */
    padding: 14px 0;
    text-align: center;
    position: fixed;
    bottom: 0;
    left: 0;
}
.footer-text {
    color: #333;
    font-size: 13px;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-container">
    <div class="header-title">🧵 Klasifikasi Citra Batik Indonesia</div>
    <div class="header-subtitle">Menggunakan Transfer Learning MobileNetV2 + Fine-Tuning</div>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
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

# =========================
# UPLOAD (1 KOTAK)
# =========================
uploaded_file = st.file_uploader(
    "Klik atau drag untuk upload gambar batik",
    type=["jpg", "png", "jpeg"]
)

st.markdown('<div class="upload-helper">Hanya 1 gambar batik yang akan diprediksi</div>',
            unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diupload", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Sedang memproses..."):
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        class_name = labels[class_idx]
        confidence = np.max(pred) * 100

    st.markdown('<div class="pred-box">', unsafe_allow_html=True)
    st.markdown('<div class="pred-label">🎉 Hasil Prediksi:</div>', unsafe_allow_html=True)
    st.write(f"**Jenis Batik:** {class_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER (TENTANG APLIKASI)
# =========================
st.markdown("""
<div class="footer">
    <div class="footer-text">
        <b>ℹ️ Tentang Aplikasi</b><br/>
        Aplikasi ini digunakan untuk mengklasifikasikan jenis batik asli Indonesia
        menggunakan model deep learning MobileNetV2.<br/>
        Developed by <b>Tania 🌸</b>
    </div>
</div>
""", unsafe_allow_html=True)
