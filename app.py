import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Custom CSS untuk UI
# =========================
st.markdown("""
<style>
.main {
    background-color: #f9f9f9;
}
.header-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #4a4a4a;
    padding-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #6e6e6e;
    margin-bottom: 30px;
}
.upload-card {
    border-radius: 12px;
    padding: 25px;
    background-color: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.pred-box {
    border-radius: 12px;
    padding: 20px;
    background-color: #ffffff;
    border-left: 6px solid #4F8EF7;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.pred-label {
    font-size: 22px;
    font-weight: bold;
    color: #4F8EF7;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown('<div class="header-title">🧵 Klasifikasi Citra Batik Indonesia</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Menggunakan Transfer Learning MobileNetV2 + Fine-Tuning</div>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.write("""
    Aplikasi ini digunakan untuk mengklasifikasikan jenis batik asli Indonesia
    menggunakan model deep learning MobileNetV2.
    """)
    st.write("Developed by **Tania** 🌸")

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_batik.h5")

model = load_model()

# Label kelas
labels = [
    'Batik Bali', 'Batik Betawi', 'Batik Cendrawasih', 'Batik Dayak',
    'Batik Geblek Renteng', 'Batik Ikat Celup', 'Batik Insang',
    'Batik Kawung', 'Batik Lasem', 'Batik Megamendung',
    'Batik Pala', 'Batik Parang', 'Batik Poleng',
    'Batik Sekar Jagad', 'Batik Tambal'
]

# =========================
# Upload Card
# =========================
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload gambar batik...", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Prediction
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diupload", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Sedang memproses dan memprediksi..."):
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        class_name = labels[class_idx]
        confidence = np.max(pred) * 100

    # Prediction Box
    st.markdown('<div class="pred-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="pred-label">🎉 Hasil Prediksi:</div>', unsafe_allow_html=True)
    st.write(f"**Jenis Batik:** {class_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)
