import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import time

# =========================
# FUNCTION WARNA SENTIMEN
# =========================
def get_color(label):
    if str(label).lower() == "positif":
        return "green"
    elif str(label).lower() == "netral":
        return "yellow"
    else:
        return "red"

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="📊",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
model_nb = joblib.load('model_nb.pkl')
model_svm = joblib.load('model_svm.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# =========================
# CSS UI + HOVER EFFECT
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #1e3a8a, #2563eb);
}
[data-testid="stSidebar"] * {
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #1e3a8a;
}

/* Header */
.header-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.05);
    text-align: center;
    margin-bottom: 20px;
}

/* Card */
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.08);
    border-left: 5px solid #3b82f6;
    margin-top: 15px;
    transition: all 0.3s ease;
}

/* 🔥 Hover effect */
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0px 12px 25px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.selectbox("Menu", ["Home", "Input Teks", "Upload Excel"])

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-box">
    <h1 class="title">📊 Analisis Sentimen Opini Publik Terhadap Kinerja Presiden Prabowo</h1>
    <p style="color:gray;">Naive Bayes vs SVM</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HOME (DENGAN GAMBAR + HOVER)
# =========================================================
if menu == "Home":

    st.markdown("## 🏠 Selamat Datang")
    st.write("Aplikasi ini digunakan untuk menganalisis sentimen teks menggunakan Machine Learning.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>📊 Naive Bayes</h3>
        <p>
        Algoritma probabilistik berbasis Teorema Bayes.
        Cepat dan sangat cocok untuk analisis teks.
        </p>
        <ul>
            <li>✔️ Cepat</li>
            <li>✔️ Ringan</li>
            <li>✔️ Cocok NLP</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🤖 Support Vector Machine</h3>
        <p>
        Algoritma klasifikasi yang mencari pemisah terbaik antar kelas.
        Sangat akurat untuk data kompleks.
        </p>
        <ul>
            <li>✔️ Akurasi tinggi</li>
            <li>✔️ Stabil</li>
            <li>✔️ Powerful</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.info("📌 Naive Bayes cepat, SVM lebih akurat untuk data kompleks.")

# =========================================================
# INPUT TEKS
# =========================================================
elif menu == "Input Teks":

    teks = st.text_area("Masukkan teks:")

    if st.button("🔍 Analisis Sekarang"):
        if teks.strip() != "":

            with st.spinner("⏳ Sedang menganalisis..."):
                time.sleep(1)

                teks_vector = vectorizer.transform([teks])

                hasil_nb = label_encoder.inverse_transform(model_nb.predict(teks_vector))[0]
                confidence_nb = model_nb.predict_proba(teks_vector).max() * 100

                hasil_svm = label_encoder.inverse_transform(model_svm.predict(teks_vector))[0]
                confidence_svm = model_svm.predict_proba(teks_vector).max() * 100

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Naive Bayes")
                if hasil_nb == "positif":
                    st.success(hasil_nb.upper())
                else:
                    st.error(hasil_nb.upper())
                st.write(f"{confidence_nb:.2f}%")
                st.progress(confidence_nb/100)

            with col2:
                st.subheader("SVM")
                if hasil_svm == "positif":
                    st.success(hasil_svm.upper())
                else:
                    st.error(hasil_svm.upper())
                st.write(f"{confidence_svm:.2f}%")
                st.progress(confidence_svm/100)

        else:
            st.warning("Masukkan teks!")

# =========================================================
# UPLOAD EXCEL
# =========================================================
elif menu == "Upload Excel":

    file = st.file_uploader("Upload Excel", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.dataframe(df.head())

        kolom = st.selectbox("Pilih kolom teks", df.columns)

        if st.button("🔍 Analisis Data Excel"):

            with st.spinner("⏳ Memproses data..."):
                time.sleep(1)

                teks_vector = vectorizer.transform(df[kolom].astype(str))

                df["NB_Hasil"] = label_encoder.inverse_transform(model_nb.predict(teks_vector))
                df["SVM_Hasil"] = label_encoder.inverse_transform(model_svm.predict(teks_vector))

            st.success("✅ Analisis selesai!")
            st.dataframe(df.head())

            col1, col2 = st.columns(2)

            with col1:
                nb = df["NB_Hasil"].value_counts()
                colors = [get_color(i) for i in nb.index]
                fig1, ax1 = plt.subplots()
                ax1.bar(nb.index, nb.values, color=colors)
                st.pyplot(fig1)

            with col2:
                svm = df["SVM_Hasil"].value_counts()
                colors = [get_color(i) for i in svm.index]
                fig2, ax2 = plt.subplots()
                ax2.bar(svm.index, svm.values, color=colors)
                st.pyplot(fig2)

            fig3, ax3 = plt.subplots()
            ax3.pie(nb, labels=nb.index, autopct='%1.1f%%', colors=colors)
            st.pyplot(fig3)

            df.to_excel("hasil.xlsx", index=False)
            with open("hasil.xlsx", "rb") as f:
                st.download_button("📥 Download Excel", f, "hasil.xlsx")