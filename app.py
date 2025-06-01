import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.write("Fill in the loan application details to predict approval status.")

# Input fields
loan_id = st.text_input("Loan ID (any identifier)", value="LN001")

dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)
cibil_score = st.slider("CIBIL Score", 300, 900, 650)
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0)

# Convert categorical inputs to encoded format
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# Prepare input for model
input_data = np.array([[
    dependents,
    education_encoded,
    self_employed_encoded,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]])

# Predict button
if st.button("🔍 Predict Loan Status"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader("Result:")
    st.success(result)

    st.write("### 🔢 Prediction Probabilities:")
    st.write(f"- Approved: `{prob[1]*100:.2f}%`")
    st.write(f"- Rejected: `{prob[0]*100:.2f}%`")

    st.info("Note: This prediction is based on a trained neural network model.")

