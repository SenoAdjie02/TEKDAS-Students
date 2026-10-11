"""Dashboard StudentDOA: data akademik, BI, ML, dan interpretasi LLM.
Jalankan dengan: streamlit run app.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from data_prep import FEATURE_COLUMNS, ID_COLUMN, TARGET, prepare_student_features
from vai_analyst import DEFAULT_MODEL, generate_student_strategy, build_prompt

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "dataset_studentDOA.csv"
MODEL_FILE = BASE_DIR / "student_outcome_pipeline.pkl"

st.set_page_config(page_title="StudentDOA | Analisis Status Akademik", layout="wide")
st.title("StudentDOA — dari Data ke ML dan LLM")
st.caption("Dashboard pembelajaran untuk memahami data mahasiswa, ringkasan status akademik, dan prediksi model.")


@st.cache_data
def load_data():
    source = pd.read_csv(SOURCE_FILE)
    return source, prepare_student_features(source)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None


@st.cache_data
def load_optional_outputs():
    ip, mp, rp = BASE_DIR / "feature_importance.csv", BASE_DIR / "model_metrics.json", BASE_DIR / "data_quality_report.json"
    importance = pd.read_csv(ip) if ip.exists() else pd.DataFrame()
    metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    return importance, metrics, report


if not SOURCE_FILE.exists():
    st.error(f"Dataset tidak ditemukan: {SOURCE_FILE.name}")
    st.stop()

source, students = load_data()
model = load_model()
feature_importance, model_metrics, quality_report = load_optional_outputs()
labels = sorted(students[TARGET].dropna().unique().tolist())

st.sidebar.header("Pilih mahasiswa")
selected_id = st.sidebar.selectbox("ID mahasiswa", students[ID_COLUMN].tolist())
student = students.loc[students[ID_COLUMN].eq(selected_id)].iloc[0]
student_features = students.loc[students[ID_COLUMN].eq(selected_id), FEATURE_COLUMNS]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baris sumber", f"{len(source):,}")
c2.metric("Data mahasiswa", f"{len(students):,}")
c3.metric("Jumlah fitur", str(len(FEATURE_COLUMNS)))
c4.metric("Jumlah status", str(len(labels)))

tab_data, tab_bi, tab_ml, tab_llm = st.tabs(["1 · Data Mahasiswa", "2 · Business Intelligence", "3 · Machine Learning", "4 · LLM Analyst"])

with tab_data:
    st.subheader("Dataset StudentDOA")
    st.markdown(f"Sumber: **{SOURCE_FILE.name}**. Setiap baris mewakili data mahasiswa; **{TARGET}** adalah label status akademik. Kolom kategori disimpan sebagai kode numerik.")
    st.write(f"Ukuran data: {source.shape[0]:,} baris × {source.shape[1]} kolom")
    view = st.selectbox("Pilih tampilan", ["Semua data sumber", "Data siap analisis"])
    st.dataframe((source if view == "Semua data sumber" else students).head(100), use_container_width=True)
    with st.expander("Kualitas dan persiapan data"):
        if quality_report:
            st.json(quality_report)
        else:
            st.info("Laporan kualitas belum tersedia. Jalankan python data_prep.py.")

with tab_bi:
    st.subheader("Ringkasan status akademik")
    counts = students[TARGET].value_counts().reindex(labels, fill_value=0)
    left, right = st.columns(2)
    with left:
        st.markdown("**Jumlah mahasiswa per status**")
        st.bar_chart(counts)
    with right:
        st.markdown("**Proporsi status**")
        st.dataframe(counts.rename("Jumlah").to_frame().assign(Proporsi=(counts / counts.sum()).map(lambda value: f"{value:.1%}")), use_container_width=True)
    st.markdown("**Rata-rata usia saat pendaftaran per status**")
    st.bar_chart(students.groupby(TARGET)["Age at enrollment"].mean().reindex(labels))
    st.markdown(f"**Data mahasiswa {selected_id}**")
    st.dataframe(pd.DataFrame([student.to_dict()]), use_container_width=True)

with tab_ml:
    st.subheader("Prediksi status akademik")
    st.dataframe(pd.DataFrame([student.to_dict()]), use_container_width=True)
    if model is None:
        st.warning("Model belum ditemukan. Siapkan data dan latih model dengan python data_prep.py lalu python ml_model.py.")
    else:
        probabilities = model.predict_proba(student_features)[0]
        probability_map = {str(label): float(probability) for label, probability in zip(model.classes_, probabilities)}
        predicted = max(probability_map, key=probability_map.get)
        a, b, c = st.columns(3)
        a.metric("Prediksi model", predicted)
        b.metric("Status pada dataset", str(student[TARGET]))
        c.metric("Probabilitas kelas teratas", f"{probability_map[predicted]:.1%}")
        st.markdown("**Probabilitas untuk setiap status**")
        st.bar_chart(pd.Series(probability_map, name="Probabilitas"))
        st.info("Probabilitas adalah keluaran model, bukan kepastian atau penyebab status. Kode kategori numerik tidak ditafsirkan sebagai ukuran.")
        if model_metrics:
            st.markdown("**Evaluasi model pada data uji**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.3f}")
            m2.metric("Macro precision", f"{model_metrics.get('precision_macro', 0):.3f}")
            m3.metric("Macro F1", f"{model_metrics.get('f1_macro', 0):.3f}")
            matrix, metric_labels = model_metrics.get("confusion_matrix"), model_metrics.get("classes", [])
            if matrix and metric_labels:
                st.caption("Confusion matrix (baris = aktual, kolom = prediksi)")
                st.dataframe(pd.DataFrame(matrix, index=[f"Aktual: {x}" for x in metric_labels], columns=[f"Prediksi: {x}" for x in metric_labels]), use_container_width=True)
        if not feature_importance.empty:
            st.markdown("**Fitur global teratas menurut model**")
            st.bar_chart(feature_importance.head(15).sort_values("importance").set_index("feature")["importance"])

with tab_llm:
    st.subheader("LLM sebagai analyst — membantu menjelaskan hasil")
    st.caption(f"Model Ollama: {DEFAULT_MODEL} (dapat diubah melalui OLLAMA_MODEL)")
    if model is None:
        st.info("Latih model terlebih dahulu agar interpretasi menyertakan probabilitas prediksi.")
    else:
        probabilities = model.predict_proba(student_features)[0]
        probability_map = {str(label): float(probability) for label, probability in zip(model.classes_, probabilities)}
        factors = feature_importance.head(8).to_dict("records") if not feature_importance.empty else []
        prompt = build_prompt(student.to_dict(), probability_map, factors)
        with st.expander("Lihat prompt"):
            st.code(prompt, language="text")
        if st.button("Buat ringkasan dan saran dukungan"):
            with st.spinner("Meminta interpretasi berbasis data kepada Ollama..."):
                answer = generate_student_strategy(student.to_dict(), probability_map, factors)
            st.markdown(answer)

st.markdown("---")
st.caption("Gunakan prediksi sebagai bahan telaah bersama data dan konteks akademik; keputusan dukungan tetap memerlukan penilaian staf.")
