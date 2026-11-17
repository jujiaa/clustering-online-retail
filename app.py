# ==========================================
# APLIKASI STREAMLIT
# Segmentasi Pelanggan Online Retail
# ==========================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Memuat model dan scaler
package = joblib.load(
    "deployment_model/clustering_deployment.joblib"
)

model = package["model"]
scaler = package["scaler"]
feature_columns = package["feature_columns"]

# Nama segmen dan tindakan sesuai profil RFM K-Means k=3
nama_segmen = {
    2: "Aktif & Bernilai Tinggi",
    0: "Aktif Belanja Menengah",
    1: "Lama Tidak Aktif"
}

tindakan = {
    2: "Program loyalitas, penawaran eksklusif, prioritas stok produk favorit",
    0: "Rekomendasi produk dan promosi pembelian ulang",
    1: "Kampanye reaktivasi dengan anggaran terukur"
}

# Judul aplikasi
st.title("Segmentasi Pelanggan Online Retail")
st.write("Prediksi cluster pelanggan menggunakan K-Means")

# Input data pelanggan
recency = st.number_input(
    "Recency (hari)", min_value=0, value=15
)

frequency = st.number_input(
    "Frequency (transaksi)", min_value=1, value=8
)

monetary = st.number_input(
    "Monetary (GBP)", min_value=0.01, value=2500.0
)

# Tombol prediksi
if st.button("Prediksi Cluster"):

    # Membentuk data pelanggan
    data = pd.DataFrame({
        "Recency": [recency],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log": [np.log1p(monetary)]
    })

    # Scaling menggunakan scaler lama
    data_scaled = scaler.transform(
        data[feature_columns]
    )

    # Prediksi cluster
    cluster = int(model.predict(data_scaled)[0])

    # Menampilkan hasil
    st.success(
        f"Pelanggan masuk ke Cluster {cluster}: {nama_segmen[cluster]}"
    )
    st.write("Tindakan yang disarankan:", tindakan[cluster])