import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# =========================
# LOGO BATIK (base64 inline)
# =========================
# Ini adalah icon batik simple warna gelap (emoji convert PNG → base64)
batik_logo_base64 = """
iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAACXBIWXMAAAsTAAALEwEAmpwYAAABY0lEQVR4nO2ZPUsDQRSGv7QBEijYgYhgMhGBEYqGZo4AEsWCxINqJCnSWkQIT/CTELyE0BBEoJIhYoFkZp6GDpa2grlzznczs7szuzszldG5nZubuefO953vnX+/+XHqJ/HbPU4zr3nAMypxF3gBpsgC14BK2AC70BJbgGnyAProAdewYrJlogWQIPcDPkADPwE+QkCchX+D6UkFFbLcQf6D0I9Ue7lAflWDrzB0EHeAlvRhD9z9hV1xo4vq9r7ECKrq70dFJZiqT+aHS63VdpJ0Vrgl7gVs9kfhFJoJjra9QSwzPSRXN+Hk2R9k9Rc9MWrM3uwt16hSKzop29m2rRpdFhmBpSWiVJt4U263AFTi+jBz7gTtoM+/bHVhJINmcHz6wC89xWbvRZ9TtshglDzuDcvUTQ8TnHDYOgElNBxeHik6HKltmVlgrnnoiN7FAKqLFFz8ErbFnbkH913drQmSMLR2JLpstLb4p8yCiVy5gA4+eHhsc4vsrBWYIQayhywYhjtkQ1o+Ahpwh22knqTO3rLUN9n+M3QmXwDlLXh0u9UiYbPGyqfFXG9uQtVjXzQO0H5bQ+Z9CElgAAAABJRU5ErkJggg==
"""

def get_batik_logo():
    return f"<img src='data:image/png;base64,{batik_logo_base64}' width='38' style='margin-right:10px;'/>"


# =========================
# CUSTOM CSS (FULL UI)
# =========================
st.markdown(f"""
<style>

.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

html, body {{
    margin: 0;
    padding: 0;
}}

.header-container {{
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;

    background: #E8EEF5;
    padding: 45px 10px 35px 10px;
    text-align: center;

    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);   /* shadow lembut */
}}

.header-title {{
    font-size: 34px;
    font-weight: 700;
    color: #2f2f2f;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}}

.header-subtitle {{
    font-size: 18px;
    color: #555;
    margin-top: 5px;
}}

/* UPLOAD BOX */
div[data-testid="stFileUploader"] {{
    border: 2px dashed #b8b8b8 !important;
    border-radius: 16px;
    padding: 40px 20px;
    background-color: #fafafa;
    text-align: center;
    max-width: 340px;
    margin: 45px auto 8px auto;
}}
div[data-testid="stFileUploader"] label {{
    display: none !important;
}}

.upload-info {{
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 6px;
}}

/* PREDICTION BOX */
.pred-box {{
    border-radius: 12px;
    padding: 20px;
    background-color: #ffffff;
    border-left: 6px solid #4F8EF7;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    max-width: 600px;
    margin: 20px auto;
}}
.pred-label {{
    font-size: 20px;
    font-weight: bold;
    color: #4F8EF7;
}}

/* FOOTER */
.main {{
    padding-bottom: 100px;
}}
.footer {{
    width: 100%;
    background: #c7d3e0;
    padding: 14px 0;
    text-align: center;
    position: fixed;
    bottom: 0;
    left: 0;
}}
.footer-text {{
    color: #333;
    font-size: 13px;
    line-height: 1.4;
}}
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">
        {get_batik_logo()}
        Klasifikasi Citra Batik Indonesia
    </div>
    <div class="header-subtitle">
        Menggunakan Transfer Learning MobileNetV2 + Fine-Tuning
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
# UPLOAD
# =========================
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

st.markdown("""
<div class="upload-info">
    Limit 200MB per file • JPG, PNG, JPEG
</div>
""", unsafe_allow_html=True)


# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Sedang memproses..."):
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        class_name = labels[class_idx]
        confidence = np.max(pred) * 100

    st.markdown('<div class="pred-box">', unsafe_allow_html=True)
    st.markdown('<div class="pred-label">Hasil Prediksi:</div>', unsafe_allow_html=True)
    st.write(f"Jenis Batik: {class_name}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    <div class="footer-text">
        <b>Tentang Aplikasi</b><br/>
        Aplikasi ini digunakan untuk mengklasifikasikan jenis batik asli Indonesia
        menggunakan model deep learning MobileNetV2.<br/>
        Developed by <b>Tania</b>
    </div>
</div>
""", unsafe_allow_html=True)
