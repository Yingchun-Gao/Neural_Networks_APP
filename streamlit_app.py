import streamlit as st
import pandas as pd
import pickle
import os


st.set_page_config(page_title="Graduate Admission Predictor")

st.title("Graduate Admission Predictor")

st.write(
    "Predict whether a student has a high chance of admission "
    "based on academic profile."
)

# Check if trained model exists
if not os.path.exists("models/mlp_model.pkl"):
    st.error("Model not found. Run main.py first.")
    st.stop()

# load model objects
with open("models/mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.subheader("Student Profile")

gre = st.slider("GRE Score", 260, 340, 310)
toefl = st.slider("TOEFL Score", 80, 120, 100)
cgpa = st.slider("CGPA", 6.0, 10.0, 8.0)

sop = st.slider("SOP Strength", 1.0, 5.0, 3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)

rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
research = st.selectbox("Research Experience", [0, 1])

if st.button("Predict Admission Chance"):

    input_df = pd.DataFrame([[0.0] * len(feature_columns)], columns=feature_columns)

    input_df.loc[0, "GRE_Score"] = gre
    input_df.loc[0, "TOEFL_Score"] = toefl
    input_df.loc[0, "CGPA"] = cgpa
    input_df.loc[0, "SOP"] = sop
    input_df.loc[0, "LOR"] = lor

    rating_col = f"University_Rating_{rating}"
    if rating_col in input_df.columns:
        input_df.loc[0, rating_col] = 1

    research_col = f"Research_{research}"
    if research_col in input_df.columns:
        input_df.loc[0, research_col] = 1

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"High chance of admission (probability {prob:.2f})")
    else:
        st.warning(f"Lower chance of admission (probability {prob:.2f})")
