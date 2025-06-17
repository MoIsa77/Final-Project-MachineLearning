import streamlit as st
import numpy as np
import tempfile
import os
import cv2
from ultralytics import YOLO
from PIL import Image

import shutil

# Page config
st.set_page_config(
    page_title="Deteksi Clinker Particle", page_icon="🪨", layout="centered"
)

# Title and description
st.title("🪨 Deteksi Clinker Particle")
st.markdown(
    "Upload gambar untuk melakukan segmentasi partikel clinker menggunakan model YOLOv8."
)




def clear_previous_results(folder_path="runs/segment/exp"):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            st.warning(f"Gagal menghapus hasil sebelumnya: {e}")
            

def interpret_boulder_detection(results, class_id_boulder=0):
    """
    Fungsi ini menghitung jumlah Boulder yang terdeteksi dan memberikan interpretasi kualitas semen.

    Parameters:
    - results: hasil deteksi YOLOv8
    - class_id_boulder: ID kelas untuk Boulder (default: 0)

    Returns:
    - interpretasi: string hasil interpretasi
    - total_boulders: jumlah total partikel Boulder terdeteksi
    """

    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy() if boxes is not None else []

    total_boulders = sum(cls == class_id_boulder for cls in classes)

    if total_boulders > 5:
        interpretasi = f"""
        **Total Partikel Boulder Terdeteksi:** {total_boulders}

        **Kategori:** Jumlah Boulder > 5

        **Interpretasi Kualitas Semen:**
        - Kualitas semen cenderung **MENURUN dan TIDAK KONSISTEN**
        - Sulit digiling dan boros energi
        - Distribusi ukuran partikel buruk, berpotensi menyebabkan masalah kekuatan dan waktu ikat beton

        **Rekomendasi Tindakan:**
        - Cek sistem pendingin: aliran udara, grate speed, lapisan klinker
        - Evaluasi proses pembakaran: suhu, nyala api, putaran tanur
        - Tinjau ulang komposisi kimia: LSF, SM, AM
        """

    elif total_boulders > 0:
        interpretasi = f"""
        **Total Partikel Boulder Terdeteksi:** {total_boulders}

        **Kategori:** Jumlah Boulder ≤ 5

        **Interpretasi Kualitas Semen:**
        - Kualitas cukup baik, tapi masih ada gangguan minor
        - Bisa menyebabkan inefisiensi kecil

        **Rekomendasi Tindakan:**
        - Lakukan pemantauan parameter proses lebih ketat
        - Lakukan fine-tuning aliran udara atau kecepatan tanur
        """

    else:
        interpretasi = f"""
        **Total Partikel Boulder Terdeteksi:** {total_boulders}

        **Kategori:** Ideal (Boulder = 0)

        **Interpretasi Kualitas Semen:**
        - Proses produksi optimal
        - Konsumsi energi efisien
        - Produk akhir berkualitas tinggi dan konsisten

        **Rekomendasi Tindakan:**
        - Pertahankan stabilitas proses sesuai SOP
        - Jadwalkan maintenance rutin dan quality control berkelanjutan
        """

    return interpretasi, total_boulders


# Load YOLOv8 segmentation model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Path to your trained YOLOv8 segmentation model
    return model


model = load_model()

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "interpretasi" not in st.session_state:
    st.session_state["interpretasi"] = None
if "result_image_path" not in st.session_state:
    st.session_state["result_image_path"] = None

# Image upload
uploaded_file = st.file_uploader(
    "📤 Upload gambar (.jpg/.png)", type=["jpg", "jpeg", "png"]
)


# Run detection
if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang Diupload", use_container_width=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🚀 Jalankan Segmentasi", use_container_width=True):
            with st.spinner("Model sedang mendeteksi partikel..."):
                clear_previous_results()  # ⛔ Auto-clear sebelum segmentasi baru

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                results = model.predict(
                    source=tmp_path,
                    save=True,
                    project="runs/segment",
                    name="exp",
                    exist_ok=True
                )
                (
                 # 👇 [BARU] Hitung waktu selesai dan durasi
                end_time = time.time()
                processing_time = end_time - start_time

                # 👇 [BARU] Hitung rata-rata akurasi (confidence score)
                conf_scores = results[0].boxes.conf.cpu().numpy()
                avg_confidence = np.mean(conf_scores) if len(conf_scores) > 0 else 0

                interpretasi, total = interpret_boulder_detection(results)
                st.session_state["interpretasi"] = interpretasi
                st.session_state["result_image_path"] = os.path.join("runs", "segment", "exp", os.path.basename(tmp_path))
                
                # 👇 [BARU] Simpan statistik ke session_state
                st.session_state["stats"] = {
                    "time": processing_time,
                    "confidence": avg_confidence
                }

                os.remove(tmp_path)
                st.rerun()
                
                interpretasi, total = interpret_boulder_detection(results)
                st.session_state["interpretasi"] = interpretasi
                st.session_state["result_image_path"] = os.path.join("runs", "segment", "exp", os.path.basename(tmp_path))

                # result_image_path = os.path.join("runs", "segment", "exp", os.path.basename(tmp_path))
                # if os.path.exists(result_image_path):
                #     result_image = cv2.imread(result_image_path)
                #     result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                #     st.markdown("### ✅ Hasil Segmentasi:")
                #     st.image(result_image, caption="Hasil Segmentasi YOLOv8", use_container_width=True)
                # else:
                #     st.warning("❗ Gagal menemukan hasil segmentasi di direktori `runs/segment/exp`.")

                os.remove(tmp_path)
                st.rerun()
                )
    with col2:
        if st.button("Clear"):
            keys_to_clear = ["uploaded_file", "interpretasi", "result_image_path"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            clear_previous_results()
            st.rerun()

    
    if st.session_state["interpretasi"] is not None:
        st.markdown("---") # Visual separator
        st.markdown("### Interpretasi Deteksi:")
        st.markdown(st.session_state["interpretasi"])

    if st.session_state["result_image_path"] is not None:
        result_image_path = st.session_state["result_image_path"]
        if os.path.exists(result_image_path):
            result_image = cv2.imread(result_image_path)
            if result_image is not None:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                st.markdown("### ✅ Hasil Segmentasi:")
                st.image(result_image, caption="Hasil Segmentasi YOLOv8", use_container_width=True)
            else:
                st.warning(f"❗ Gagal memuat gambar hasil segmentasi dari {result_image_path}.")
        else:
            st.warning(f"❗ Gagal menemukan hasil segmentasi di direktori `{result_image_path}`. Pastikan model menyimpan hasilnya dengan benar.")



else:
    st.info("Silakan upload gambar terlebih dahulu.")

st.markdown("---")
st.caption("Model segmentasi partikel clinker berbasis YOLOv8")
