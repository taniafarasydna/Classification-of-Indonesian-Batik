import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# =========================
# ICON CANTING (base64)
# =========================
canting_icon_base64 = """
iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABKElEQVR4nM3Tv0oDQRDG8d8HcpA2
QAU0r9AR2ggZgomwUjZCwAh2gn0CVpZwbgWJmbm5qTPGWx8o0lxxz5p7/M7Ox1YwAzwBUWiNUM6g
ZyAu8ALuAGjAIhykQyf/UH1GXeV9N8w7vZPUuFz/27ziaqemYiqkM1hBxtRmDYqSTS3MpmN+gqnc
5o41TXU87lN2GJ2nbaZ8gK3gO90u/QBD7sC6Go3d775lPXJLAjbjLCAHQI6v0NO6drt+3qcw3FgB
zBvw7a8j9YRcErjSaILE6nyoj2XfYubKGm+7s7HjTp0NYB/gH32bRa7VrVKwB/wADa6zCzy5l7Qs
T5+7WfmT9Avq0uFnmnQZXFUTioM8O0JtVf5PmnmAAAAAElFTkSuQmCC
"""

def canting_icon():
    return f"<img src='data:image/png;base64,{canting_icon_base64}' width='30' style='margin-right:10px; margin-bottom:-5px;'/>"


# =========================
# CUSTOM CSS
# =========================
st.markdown(f"""
<style>

/* Remove container padding so background can expand */
.block-container {{
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}}

html, body {{
    margin: 0;
    padding: 0;
}}

/* Background motif batik halus */
body {{
    background-image: url("https://i.imgur.com/NF7yHqX.png");
    background-size: 300px;
    background-repeat: repeat;
    opacity: 0.98;
}}

/* Header clean */
.header-container {{
    width: 100%;
    padding: 40px 10px 20px 10px;
    text-align: center;
    background: rgba(255, 255, 255, 0.85);
}}

.header-title {{
    font-family: 'Georgia', serif;
    font-size: 36px;
    font-weight: 700;
    color: #2e2e2e;
}}

.header-subtitle {{
    font-family: 'Georgia', serif;
    font-size: 19px;
    color: #555;
    margin-top: 6px;
}}

.section-title {{
    margin-top: 35px;
    text-align: center;
    font-family: 'Georgia', serif;
    font-size: 20px;
    font-weight: 600;
    color: #333;
}}

/* Upload box */
div[data-testid="stFileUploader"] {{
    border: 2px dashed #b6b6b6 !important;
    border-radius: 16px;
    padding: 40px 20px;
    background-color: #ffffffd9;
    text-align: center;
    max-width: 340px;
    margin: 20px auto 8px auto;
}}
div[data-testid="stFileUploader"] label {{
    display: none !important;
}}

/* Prediction Box */
.pred-box {{
    border-radius: 12px;
    padding: 20px;
    background-color: #ffffffee;
    border-left: 6px solid #8A5A44;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    max-width: 600px;
    margin: 20px auto;
}}

.pred-label {{
    font-size: 20px;
    font-weight: bold;
    color: #8A5A44;
}}

/* Footer soft brown */
.footer {{
    width: 100%;
    background: #D8C4B0;
    padding: 18px 0;
    text-align: center;
    position: fixed;
    bottom: 0;
    left: 0;

    font-family: 'Georgia', serif;
}}

.footer-text {{
    color: #4a3a2a;
    font-size: 14px;
    line-height: 1.5;
}}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">
        {canting_icon()} Klasifikasi Citra Batik Indonesia
    </div>
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
# UPLOAD SECTION TITLE
# =========================
st.markdown("<div class='section-title'>Unggah Gambar Batik</div>", unsafe_allow_html=True)


# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])


# =========================
# PREDICTION
# =========================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Sedang memproses..."):
        pred = model.predict(img_array)
        idx = np.argmax(pred)
        name = labels[idx]
        conf = np.max(pred) * 100

    st.markdown('<div class="pred-box">', unsafe_allow_html=True)
    st.markdown('<div class="pred-label">Hasil Prediksi:</div>', unsafe_allow_html=True)
    st.write(f"Jenis Batik: {name}")
    st.write(f"Confidence: {conf:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    <div class="footer-text">
        Aplikasi ini digunakan untuk mengklasifikasikan jenis batik Indonesia menggunakan model MobileNetV2.<br>
        Developed by <strong>Tania</strong>
    </div>
</div>
""", unsafe_allow_html=True)
