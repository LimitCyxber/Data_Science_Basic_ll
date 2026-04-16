
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

# --- Fungsi untuk memuat model dan encoders ---
@st.cache_resource # Cache resource to load model/encoders only once
def load_resources():
    model_filename = 'best_regression_model.pkl'
    encoders_filename = 'preprocessing_encoders.pkl'

    try:
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        with open(encoders_filename, 'rb') as file:
            loaded_encoders = pickle.load(file)
        return loaded_model, loaded_encoders
    except FileNotFoundError:
        st.error(f"Error: File '{model_filename}' atau '{encoders_filename}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model atau encoders: {e}")
        st.stop()

loaded_model, loaded_encoders = load_resources()

loaded_ohe = loaded_encoders.get('OneHotEncoder')
loaded_le_pendidikan = loaded_encoders.get('LabelEncoder_Pendidikan')
loaded_le_jurusan = loaded_encoders.get('LabelEncoder_Jurusan')
loaded_scaler = loaded_encoders.get('StandardScaler')

# Define expected columns based on training data (important for consistent feature order)
# This list should match X_train.columns used during model training.
expected_columns = ['Usia', 'Pendidikan', 'Jurusan', 'Durasi_Jam', 'Nilai_Ujian',
                    'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

# --- Fungsi untuk Preprocessing dan Prediksi ---
def predict_gaji_pertama(new_data):
    # Pastikan input adalah DataFrame
    processed_data = pd.DataFrame([new_data])

    # a. One-Hot Encoding untuk Jenis_Kelamin dan Status_Bekerja
    one_hot_cols = ['Jenis_Kelamin', 'Status_Bekerja']
    if loaded_ohe:
        # Ensure all columns exist, fill with empty string if missing for consistent transformation
        for col in one_hot_cols:
            if col not in processed_data.columns:
                processed_data[col] = '' # OneHotEncoder will handle 'unknown' if configured

        encoded_features = loaded_ohe.transform(processed_data[one_hot_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=loaded_ohe.get_feature_names_out(one_hot_cols), index=processed_data.index)

        processed_data = pd.concat([processed_data, encoded_df], axis=1)
        processed_data.drop(columns=one_hot_cols, inplace=True)

    # b. Label Encoding untuk Pendidikan dan Jurusan
    label_cols_map = {
        'Pendidikan': loaded_le_pendidikan,
        'Jurusan': loaded_le_jurusan
    }
    for col, le in label_cols_map.items():
        if le and col in processed_data.columns:
            # Handle potential unknown labels: if a category in new_data was not seen during fit,
            # LabelEncoder.transform will raise an error. We use the existing logic where it's assumed
            # valid options are selected from selectboxes.
            processed_data[col] = le.transform(processed_data[col])
        elif col not in processed_data.columns:
            # Assign a default encoded value, e.g., 0, but this assumes 0 is a safe default
            # and might need more robust handling in a production environment.
            processed_data[col] = 0

    # Ensure processed_data has all expected columns, adding missing ones with a default value (e.g., 0)
    for col in expected_columns:
        if col not in processed_data.columns:
            processed_data[col] = 0.0 # Use 0.0 for consistency with scaled numerical features

    # Drop any extra columns not in the expected_columns list
    # And reorder the columns to match the training data's order
    processed_data = processed_data[expected_columns]

    # c. Scaling Fitur Numerik (and encoded categorical features) - apply to the entire DataFrame
    if loaded_scaler:
        processed_data_scaled_array = loaded_scaler.transform(processed_data)
        processed_data = pd.DataFrame(processed_data_scaled_array, columns=expected_columns, index=processed_data.index)

    # Lakukan prediksi
    prediction = loaded_model.predict(processed_data)
    return prediction[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Prediksi Gaji Awal Lulusan Pelatihan Vokasi")

st.title("💰 Prediksi Gaji Awal Lulusan Pelatihan Vokasi")
st.markdown("Aplikasi ini memprediksi gaji awal (dalam **juta Rupiah**) seorang peserta pelatihan vokasi berdasarkan beberapa faktor.")

st.sidebar.header("Input Data Peserta")

# Get options for selectboxes from loaded encoders
pendidikan_options = loaded_le_pendidikan.classes_.tolist() if loaded_le_pendidikan else []
jurusan_options = loaded_le_jurusan.classes_.tolist() if loaded_le_jurusan else []

# Assuming 'Jenis_Kelamin' and 'Status_Bekerja' were the first two columns in ohe.fit_transform
jenis_kelamin_options = loaded_ohe.categories_[0].tolist() if loaded_ohe and len(loaded_ohe.categories_) > 0 else []
status_bekerja_options = loaded_ohe.categories_[1].tolist() if loaded_ohe and len(loaded_ohe.categories_) > 1 else []

# Input Form in sidebar
with st.sidebar.form("prediction_form"):
    st.subheader("Data Numerik")
    usia = st.number_input("Usia", min_value=18, max_value=65, value=25, help="Usia peserta pelatihan.")
    durasi_jam = st.number_input("Durasi Pelatihan (Jam)", min_value=30, max_value=1000, value=60, help="Total durasi jam pelatihan yang diikuti.")
    nilai_ujian = st.number_input("Nilai Ujian", min_value=0.0, max_value=100.0, value=85.0, step=0.1, help="Nilai akhir ujian pelatihan.")

    st.subheader("Data Kategorikal")
    pendidikan = st.selectbox("Pendidikan Terakhir", options=pendidikan_options, index=pendidikan_options.index('SMK') if 'SMK' in pendidikan_options else 0, help="Tingkat pendidikan terakhir peserta.")
    jurusan = st.selectbox("Jurusan Pelatihan", options=jurusan_options, index=jurusan_options.index('Otomotif') if 'Otomotif' in jurusan_options else 0, help="Jurusan pelatihan vokasi yang diikuti.")
    jenis_kelamin = st.selectbox("Jenis Kelamin", options=jenis_kelamin_options, index=jenis_kelamin_options.index('Laki-laki') if 'Laki-laki' in jenis_kelamin_options else 0, help="Jenis kelamin peserta.")
    status_bekerja = st.selectbox("Status Bekerja Setelah Lulus", options=status_bekerja_options, index=status_bekerja_options.index('Belum Bekerja') if 'Belum Bekerja' in status_bekerja_options else 0, help="Apakah peserta sudah bekerja setelah lulus pelatihan.")

    submitted = st.form_submit_button("Prediksi Gaji")

if submitted:
    input_data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }

    st.write("---")
    st.subheader("Hasil Prediksi")

    with st.spinner("Memproses prediksi..."):
        try:
            predicted_gaji = predict_gaji_pertama(input_data)
            st.success(f"Berdasarkan input yang diberikan, prediksi **Gaji Awal** adalah: **{predicted_gaji:.2f} Juta Rupiah**")
            st.balloons()
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.write("Mohon pastikan semua input sudah benar dan model/encoders dimuat dengan tepat.")

st.markdown("---")
st.caption("Dibuat dengan Streamlit dan Model Machine Learning")
