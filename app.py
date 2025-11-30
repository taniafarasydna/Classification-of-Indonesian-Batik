import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Klasifikasi Batik Menggunakan CNN")

# Load model
model = tf.keras.models.load_model("model_batik.h5")

# Daftar label (urutannya harus sesuai training)
labels = [
    'Batik Bali', 'Batik Betawi', 'Batik Cendrawasih', 'Batik Dayak',
    'Batik Geblek Renteng', 'Batik Ikat Celup', 'Batik Insang',
    'Batik Kawung', 'Batik Lasem', 'Batik Megamendung',
    'Batik Pala', 'Batik Parang', 'Batik Poleng',
    'Batik Sekar Jagad', 'Batik Tambal'
]

uploaded_file = st.file_uploader("Upload gambar batik...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = labels[class_idx]

    st.subheader("Hasil Prediksi:")
    st.write(class_name)
