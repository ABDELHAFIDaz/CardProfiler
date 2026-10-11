from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_pipeline.joblib"

COLS = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

st.title("Segmentation des clients carte de crédit")
st.write("Entrez le profil d'un client, ou chargez un fichier CSV, pour prédire son segment.")

tab_form, tab_csv = st.tabs(["Saisir un profil", "Charger un CSV"])

with tab_form:
    col1, col2 = st.columns(2)
    values = {
        "BALANCE": col1.number_input("BALANCE", min_value=0.0, value=870.0),
        "PURCHASES": col1.number_input("PURCHASES", min_value=0.0, value=360.0),
        "ONEOFF_PURCHASES": col1.number_input("ONEOFF_PURCHASES", min_value=0.0, value=40.0),
        "INSTALLMENTS_PURCHASES": col1.number_input("INSTALLMENTS_PURCHASES", min_value=0.0, value=90.0),
        "CASH_ADVANCE": col2.number_input("CASH_ADVANCE", min_value=0.0, value=0.0),
        "CREDIT_LIMIT": col2.number_input("CREDIT_LIMIT", min_value=0.0, value=3000.0),
        "PAYMENTS": col2.number_input("PAYMENTS", min_value=0.0, value=860.0),
    }

    if st.button("Prédire le segment"):
        client = pd.DataFrame([values])[COLS]
        segment = pipeline.predict(client)[0]
        st.success(f"Segment prédit : **{segment}**")

        probas = pipeline.predict_proba(client)[0]
        st.write("Probabilités par segment :")
        st.bar_chart(pd.Series(probas, index=pipeline.classes_))

with tab_csv:
    uploaded = st.file_uploader("CSV avec les colonnes : " + ", ".join(COLS), type="csv")
    if uploaded is not None:
        df_new = pd.read_csv(uploaded)
        missing = [c for c in COLS if c not in df_new.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
        else:
            df_new["segment_predit"] = pipeline.predict(df_new[COLS])
            st.dataframe(df_new)
            st.write(df_new["segment_predit"].value_counts())