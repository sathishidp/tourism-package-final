import os
import streamlit as st
import pandas as pd
import numpy as np
from predict_utils import download_model_from_hf, load_model, inputs_to_dataframe

st.set_page_config(page_title="Tourism Package Purchase Predictor", layout="centered")

st.title("🎯 Wellness Tourism Package - Purchase Predictor")
st.markdown("Enter customer & interaction details and click **Predict** to get probability and label.")

# -----------------------
# Configuration (set these as Space variables or leave defaults)
# -----------------------
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "sathishaiuse/wellness-classifier-model")  # change to your model repo
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", None)  # optional, fallback logic will attempt candidates
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# The feature order must match training pipeline
FEATURE_ORDER = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

# -----------------------
# Download & load model (on first run)
# -----------------------
@st.cache_resource(ttl=60*60)
def get_model():
    try:
        local_path = download_model_from_hf(HF_MODEL_REPO, HF_MODEL_FILENAME, token=HF_TOKEN, local_dir="/tmp/model")
        model = load_model(local_path)
        return model, local_path
    except Exception as e:
        st.error(f"Failed to download/load model: {e}")
        return None, None

model, model_path = get_model()
if model is None:
    st.warning("Model not loaded. Check HF_MODEL_REPO, HF_MODEL_FILENAME and HF_TOKEN (if private repo).")
    st.stop()

st.caption(f"Using model file: `{model_path}`")

# -----------------------
# Build input form
# -----------------------
with st.form("predict_form"):
    st.subheader("Customer Details")
    col1, col2, col3 = st.columns(3)
    Age = col1.number_input("Age", min_value=18, max_value=100, value=30)
    CityTier = col1.selectbox("CityTier", options=[1,2,3], index=0)
    NumberOfPersonVisiting = col1.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2)
    PreferredPropertyStar = col2.selectbox("PreferredPropertyStar", options=[1,2,3,4,5], index=3)
    NumberOfTrips = col2.number_input("NumberOfTrips (annually)", min_value=0, max_value=20, value=2)
    Passport = col2.selectbox("Passport (0=No, 1=Yes)", options=[0,1], index=1)
    OwnCar = col3.selectbox("OwnCar (0=No,1=Yes)", options=[0,1], index=1)
    NumberOfChildrenVisiting = col3.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
    MonthlyIncome = col3.number_input("MonthlyIncome", min_value=0, value=30000)

    st.subheader("Interaction Details")
    PitchSatisfactionScore = st.slider("PitchSatisfactionScore (1-10)", 0, 10, 7)
    ProductPitched = st.selectbox("ProductPitched", options=["Wellness","Holiday","Adventure","Relaxation"], index=0)
    NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=20, value=2)
    DurationOfPitch = st.number_input("DurationOfPitch (minutes)", min_value=0, max_value=120, value=15)

    st.subheader("Demographics / Job")
    TypeofContact = st.selectbox("TypeofContact", options=["Company Invited", "Self Inquiry"])
    Occupation = st.text_input("Occupation", value="Salaried")
    Gender = st.selectbox("Gender", options=["Male","Female","Other"])
    MaritalStatus = st.selectbox("MaritalStatus", options=["Single","Married","Divorced"])
    Designation = st.text_input("Designation", value="Employee")

    submitted = st.form_submit_button("Predict")

if submitted:
    # construct single-record dict
    rec = {
        "Age": Age,
        "CityTier": CityTier,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch,
        "TypeofContact": TypeofContact,
        "Occupation": Occupation,
        "Gender": Gender,
        "MaritalStatus": MaritalStatus,
        "Designation": Designation,
        "ProductPitched": ProductPitched
    }

    try:
        df = inputs_to_dataframe(rec, FEATURE_ORDER)
        # The model is expected to be a sklearn Pipeline
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:,1]
            pred = (probs >= 0.5).astype(int)
            st.metric("Predicted Probability (purchase)", f"{probs[0]:.4f}")
            st.write("Predicted Label (ProdTaken):", int(pred[0]))
        else:
            pred = model.predict(df)
            st.write("Predicted Label (ProdTaken):", int(pred[0]))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
