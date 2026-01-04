import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_v2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_v3

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Klasifikasi Batik | MobileNetV2 vs V3",
    page_icon="🎨",
    layout="wide"
)

# ===============================
# GLOBAL CONFIG
# ===============================
IMG_SIZE = (224, 224)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.big-title {
    font-size: 38px;
    font-weight: 700;
}
.sub-title {
    font-size: 18px;
    color: #555;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: #ffffff;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background: #f8f9fa;
    text-align: center;
}
/* =========================
   SIDEBAR RADIO BUTTON
========================= */
section[data-testid="stSidebar"] .stRadio label {
    display: flex;
    align-items: center;
    padding: 12px 14px;
    margin-bottom: 6px;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.25s ease;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: #f1f5f9;
}

/* aktif */
section[data-testid="stSidebar"] .stRadio input:checked + div {
    background-color: #eef2ff;
    border-left: 4px solid var(--accent);
    font-weight: 600;
}
/* =========================
   BUTTON STYLE
========================= */
button[kind="primary"] {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.25s ease;
}

button[kind="primary"]:hover {
    background: #4338ca !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(79,70,229,0.35);
}

button {
    border-radius: 10px !important;
}
/* =========================
   SIDEBAR RADIO - ACTIVE STATE
========================= */
section[data-testid="stSidebar"] .stRadio label {
    display: flex;
    align-items: center;
    padding: 12px 14px;
    margin-bottom: 6px;
    border-radius: 12px;
    cursor: pointer;
    transition: background-color .25s ease, transform .15s ease;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: #f1f5f9;
}

/* aktif */
section[data-testid="stSidebar"] .stRadio input:checked + div {
    background-color: var(--accent-soft);
    border-left: 4px solid var(--accent);
    padding-left: 10px;
    font-weight: 700;
}
/* =========================
   PAGE TRANSITION
========================= */
.block-container {
    animation: pageFade .35s ease;
}

@keyframes pageFade {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
section[data-testid="stSidebar"] .stRadio label:active {
    transform: scale(0.98);
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD CLASS NAMES
# ===============================
with open("assets/class_names.json", "r") as f:
    class_names = json.load(f)

# ===============================
# LOAD MODELS (CACHE)
# ===============================
@st.cache_resource
def load_models():
    model_v2 = load_model("models/mobilenetv2_best.keras")
    model_v3 = load_model("models/mobilenetv3_best.keras")
    return model_v2, model_v3

model_v2, model_v3 = load_models()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.title("🎨 Batik Classifier")
    st.markdown("---")
    menu = st.radio(
        "Navigasi",
        ["Data Overview", "Klasifikasi", "Evaluasi Model"]
    )

    st.markdown("---")
    st.caption("Dibuat Oleh : Tania Fara Sayyidina (202210715156)")
    st.caption("Klasifikasi Citra Batik Indonesia Berbasis Transfer Learning Menggunakan MobileNetV2 dan MobileNetV3")

# ===============================
# HOME PAGE
# ===============================
if menu == "Data Overview":

    import os
    from collections import Counter

    st.markdown('<div class="big-title">📊 Data Overview Dataset Batik</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Source : https://www.kaggle.com/code/mpwolke/indonesian-batiks-cnn</div>', unsafe_allow_html=True)
    st.markdown("---")

    DATASET_PATH = "data/DATASET/DATASET"
    TRAIN_PATH = os.path.join(DATASET_PATH, "TRAIN")
    TEST_PATH  = os.path.join(DATASET_PATH, "TEST")

    # ===============================
    # HELPER FUNCTION
    # ===============================
    def count_images(base_path):
        count = 0
        for root, _, files in os.walk(base_path):
            count += len([f for f in files if f.lower().endswith(("jpg","jpeg","png"))])
        return count

    def get_image_sizes(sample_paths, max_samples=50):
        sizes = []
        for p in sample_paths[:max_samples]:
            try:
                img = Image.open(p)
                sizes.append(img.size)
            except:
                pass
        return sizes

    # ===============================
    # BASIC STATS
    # ===============================
    classes = sorted(os.listdir(TRAIN_PATH))
    num_classes = len(classes)

    train_count = count_images(TRAIN_PATH)
    test_count  = count_images(TEST_PATH)
    total_images = train_count + test_count

    # ambil contoh gambar
    sample_images = []
    for c in classes:
        class_dir = os.path.join(TRAIN_PATH, c)
        imgs = os.listdir(class_dir)
        if imgs:
            sample_images.append(os.path.join(class_dir, imgs[0]))

    sizes = get_image_sizes(sample_images)

    # ===============================
    # SUMMARY CARDS
    # ===============================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📁 Total Images", total_images)
    col2.metric("🏷️ Total Classes", num_classes)
    col3.metric("🧪 Train Images", train_count)
    col4.metric("🧪 Test Images", test_count)

    st.markdown("---")

    # ===============================
    # DATASET STRUCTURE
    # ===============================
    st.subheader("📂 Struktur Dataset")

    st.code(f"""
DATASET/
├── TRAIN/ ({train_count} images)
│   ├── {classes[0]}
│   ├── ...
│   └── {classes[-1]}
├── TEST/ ({test_count} images)
""")

    # ===============================
    # CONTOH GAMBAR PER KELAS
    # ===============================
    st.subheader("🖼️ Contoh Gambar per Kelas")

    cols = st.columns(5)
    for i, img_path in enumerate(sample_images):
        with cols[i % 5]:
            img = Image.open(img_path)
            st.image(
    img,
    caption=os.path.basename(os.path.dirname(img_path)),
    use_container_width=True
)


    # ===============================
    # DISTRIBUSI GAMBAR PER KELAS
    # ===============================
    st.subheader("📊 Distribusi Gambar per Kelas (TRAIN)")

    class_counts = {
        c: len(os.listdir(os.path.join(TRAIN_PATH, c)))
        for c in classes
    }

    df_dist = pd.DataFrame({
        "Class": list(class_counts.keys()),
        "Images": list(class_counts.values())
    })

    st.bar_chart(df_dist.set_index("Class"))

    # ===============================
    # STATISTIK UKURAN GAMBAR
    # ===============================
    if sizes:
        widths, heights = zip(*sizes)

        st.subheader("📐 Statistik Ukuran Gambar (Sample)")

        col_w, col_h = st.columns(2)

        with col_w:
            st.write("**Lebar Gambar (px)**")
            st.bar_chart(pd.Series(widths))

        with col_h:
            st.write("**Tinggi Gambar (px)**")
            st.bar_chart(pd.Series(heights))




# ===============================
# KLASIFIKASI PAGE
# ===============================
elif menu == "Klasifikasi":

    st.markdown('<div class="big-title">Klasifikasi Batik</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload gambar batik dan bandingkan hasil prediksi</div>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload satu atau beberapa gambar batik (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for file in uploaded_files:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            col_img, col_res = st.columns([1, 2])

            # ===============================
            # LOAD IMAGE
            # ===============================
            img = Image.open(file).convert("RGB")
            img = img.resize(IMG_SIZE)

            with col_img:
                st.image(img, caption=file.name, use_container_width=True)


            # ===============================
            # PREPROCESS
            # ===============================
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            img_v2 = preprocess_v2(img_array.copy())
            img_v3 = preprocess_v3(img_array.copy())

            # ===============================
            # PREDICT
            # ===============================
            with st.spinner("Memprediksi gambar..."):
                pred_v2 = model_v2.predict(img_v2, verbose=0)[0]
                pred_v3 = model_v3.predict(img_v3, verbose=0)[0]

            idx_v2 = int(np.argmax(pred_v2))
            idx_v3 = int(np.argmax(pred_v3))

            conf_v2 = float(pred_v2[idx_v2]) * 100
            conf_v3 = float(pred_v3[idx_v3]) * 100

            with col_res:
                st.subheader("Hasil Prediksi")

                col_m1, col_m2 = st.columns(2)

                with col_m1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric(
                        label="MobileNetV2",
                        value=class_names[idx_v2],
                        delta=f"{conf_v2:.2f}%"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_m2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric(
                        label="MobileNetV3",
                        value=class_names[idx_v3],
                        delta=f"{conf_v3:.2f}%"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                # ===============================
                # CONFIDENCE BAR CHART
                # ===============================
                df = pd.DataFrame({
                    "Model": ["MobileNetV2", "MobileNetV3"],
                    "Confidence (%)": [conf_v2, conf_v3]
                })

                fig, ax = plt.subplots()
                ax.bar(df["Model"], df["Confidence (%)"])
                ax.set_ylim(0, 100)
                ax.set_ylabel("Confidence (%)")
                ax.set_title("Perbandingan Confidence")
                st.pyplot(fig)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("⬆️ Upload gambar batik untuk melihat hasil prediksi")

elif menu == "Evaluasi Model":

    import json
    import numpy as np
    import matplotlib.pyplot as plt

    st.markdown('<div class="big-title">Evaluasi & Perbandingan Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hasil Evaluasi dari Model MobileNetV2 dan MobileNetV3</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ======================================================
    # LOAD FILE EVALUASI (HASIL NOTEBOOK)
    # ======================================================
    try:
        with open("assets/eval/metrics.json", "r") as f:
            metrics = json.load(f)

        cm_v2 = np.load("assets/eval/cm_v2.npy")
        cm_v3 = np.load("assets/eval/cm_v3.npy")

    except FileNotFoundError:
        st.error(
            "File evaluasi tidak ditemukan. "
            "Pastikan folder assets/eval berisi:\n"
            "- metrics.json\n- cm_v2.npy\n- cm_v3.npy"
        )
        st.stop()

    acc_v2 = metrics["mobilenetv2"]["accuracy"]
    loss_v2 = metrics["mobilenetv2"]["loss"]

    acc_v3 = metrics["mobilenetv3"]["accuracy"]
    loss_v3 = metrics["mobilenetv3"]["loss"]

    # ======================================================
    # SUMMARY METRICS
    # ======================================================
    st.subheader("Ringkasan Performa Model")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MobileNetV2 Accuracy", f"{acc_v2*100:.2f}%")
        st.metric("MobileNetV2 Loss", f"{loss_v2:.4f}")

    with col2:
        st.metric("MobileNetV3 Accuracy", f"{acc_v3*100:.2f}%")
        st.metric("MobileNetV3 Loss", f"{loss_v3:.4f}")

    # ======================================================
    # HIGHLIGHT MODEL TERBAIK
    # ======================================================
    st.markdown("### 🏆 Model Terbaik")
    if acc_v3 > acc_v2:
        st.success("MobileNetV3 memiliki performa terbaik berdasarkan akurasi.")
    else:
        st.success("MobileNetV2 memiliki performa terbaik berdasarkan akurasi.")

    st.markdown("---")

    # ======================================================
    # CONFUSION MATRIX (NORMALIZED %)
    # ======================================================
    st.subheader("Perbandingan Confusion Matrix (Normalized %)")

    def normalize_cm(cm):
        return cm / cm.sum(axis=1, keepdims=True) * 100

    cm_v2_norm = normalize_cm(cm_v2)
    cm_v3_norm = normalize_cm(cm_v3)

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.markdown("**MobileNetV2**")
        fig, ax = plt.subplots()
        im = ax.imshow(cm_v2_norm)
        ax.set_title("Normalized CM - V2 (%)")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
        plt.colorbar(im)
        st.pyplot(fig)

    with col_cm2:
        st.markdown("**MobileNetV3**")
        fig, ax = plt.subplots()
        im = ax.imshow(cm_v3_norm)
        ax.set_title("Normalized CM - V3 (%)")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
        plt.colorbar(im)
        st.pyplot(fig)

    st.markdown("---")

    # ======================================================
    # TABEL RINGKASAN
    # ======================================================
    st.subheader("Tabel Ringkasan Evaluasi Model")

    df_summary = pd.DataFrame({
        "Model": ["MobileNetV2", "MobileNetV3"],
        "Accuracy (%)": [acc_v2 * 100, acc_v3 * 100],
        "Loss": [loss_v2, loss_v3]
    })

    st.dataframe(df_summary, use_container_width=True)
    # ======================================================
    # GRAFIK ACCURACY vs LOSS
    # ======================================================
    st.subheader("Grafik Perbandingan Accuracy dan Loss")

    df_metric_plot = pd.DataFrame({
        "Model": ["MobileNetV2", "MobileNetV3"],
        "Accuracy (%)": [acc_v2 * 100, acc_v3 * 100],
        "Loss": [loss_v2, loss_v3]
    })

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Accuracy Comparison**")
        st.bar_chart(
            df_metric_plot.set_index("Model")[["Accuracy (%)"]]
        )

    with col_g2:
        st.markdown("**Loss Comparison**")
        st.bar_chart(
            df_metric_plot.set_index("Model")[["Loss"]]
        )
    # ======================================================
    # PER-CLASS ACCURACY
    # ======================================================
    st.subheader("Per-Class Accuracy")

    def per_class_accuracy(cm):
        return np.diag(cm) / np.sum(cm, axis=1)

    pca_v2 = per_class_accuracy(cm_v2)
    pca_v3 = per_class_accuracy(cm_v3)

    df_pca = pd.DataFrame({
        "Class": class_names,
        "MobileNetV2 (%)": pca_v2 * 100,
        "MobileNetV3 (%)": pca_v3 * 100
    })

    st.dataframe(df_pca, use_container_width=True)

    # Visualisasi bar chart
    st.markdown("**Visualisasi Per-Class Accuracy**")

    df_pca_plot = df_pca.set_index("Class")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("MobileNetV2")
        st.bar_chart(df_pca_plot[["MobileNetV2 (%)"]])

    with col_p2:
        st.markdown("MobileNetV3")
        st.bar_chart(df_pca_plot[["MobileNetV3 (%)"]])
    # ======================================================
    # PER-CLASS DIFFERENCE (V3 - V2)
    # ======================================================
    st.subheader("Per-Class Accuracy Difference (MobileNetV3 − MobileNetV2)")

    df_diff = pd.DataFrame({
        "Class": class_names,
        "Difference (%)": (pca_v3 - pca_v2) * 100
    }).set_index("Class")

    st.bar_chart(df_diff)

    st.caption("Nilai positif menunjukkan peningkatan performa MobileNetV3 dibanding MobileNetV2")
    # ======================================================
    # AUTO-HIGHLIGHT KELAS TERLEMAH
    # ======================================================
    st.subheader("Analisis Kelas Terlemah")

    weakest_v2_idx = np.argmin(pca_v2)
    weakest_v3_idx = np.argmin(pca_v3)

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.warning(
            f"📉 **MobileNetV2**\n\n"
            f"Kelas terlemah: **{class_names[weakest_v2_idx]}**\n\n"
            f"Akurasi: **{pca_v2[weakest_v2_idx]*100:.2f}%**"
        )

    with col_w2:
        st.warning(
            f"📉 **MobileNetV3**\n\n"
            f"Kelas terlemah: **{class_names[weakest_v3_idx]}**\n\n"
            f"Akurasi: **{pca_v3[weakest_v3_idx]*100:.2f}%**"
        )
      
st.markdown("""
<div class="footer-text">
Klasifikasi Citra Batik Indonesia<br>
Transfer Learning MobileNetV2 & MobileNetV3<br>
dibuat oleh Tania Fara Sayyidina -- 202210715156
</div>
""", unsafe_allow_html=True)