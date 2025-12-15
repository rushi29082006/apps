import numpy as np
import joblib
import streamlit as st
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Mini Bank Loan System", layout="centered")
st.title("🏦 Mini Bank Loan Decision System")
st.caption("All amounts are in INR (₹)")

# ---------------- LOAD MODEL SAFELY ----------------
if not os.path.exists("loan_model.pkl"):
    st.error("❌ Model file (loan_model.pkl) not found")
    st.stop()

model = joblib.load("loan_model.pkl")

# ---------------- USER INPUTS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input(
    "Applicant Income (Monthly ₹)", min_value=0, step=500
)
coapplicant_income = st.number_input(
    "Co-applicant Income (Monthly ₹)", min_value=0, step=500
)

loan_amount = st.number_input(
    "Loan Amount (₹)", min_value=1, step=1000
)

loan_term = st.selectbox(
    "Loan Term (Months)", [120, 180, 240, 300, 360]
)

credit_history = st.selectbox("Credit History", ["Good", "Bad"])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# ---------------- BUTTON ----------------
if st.button("Check Loan Eligibility"):

    # ---------------- BASIC VALIDATION ----------------
    total_income = applicant_income + coapplicant_income

    if total_income <= 0:
        st.error("❌ Rejected: No income provided")
        st.stop()

    if self_employed == "Yes" and total_income < 3000:
        st.error("❌ Rejected: Income below minimum for self-employed")
        st.stop()

    if self_employed == "No" and total_income < 2000:
        st.error("❌ Rejected: Income below minimum")
        st.stop()

    # ---------------- EMI CALCULATION ----------------
    annual_rate = 0.09
    monthly_rate = annual_rate / 12

    emi = (
        loan_amount
        * monthly_rate
        * (1 + monthly_rate) ** loan_term
    ) / ((1 + monthly_rate) ** loan_term - 1)

    dti = emi / total_income

    st.subheader("📊 Affordability Check")
    st.write(f"Monthly EMI: **₹{emi:.2f}**")
    st.write(f"DTI Ratio: **{dti * 100:.2f}%**")

    if dti > 0.45:
        st.error("❌ Rejected: EMI too high compared to income")
        st.stop()

    # ---------------- ENCODING ----------------
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Good" else 0
    dependents = 4 if dependents == "3+" else int(dependents)
    property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    applicant_income_log = np.log1p(applicant_income)
    coapplicant_income_log = np.log1p(coapplicant_income)

    # ---------------- MODEL INPUT ----------------
    data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income_log,
        coapplicant_income_log,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])

    # ---------------- PREDICTION ----------------
    approval_prob = model.predict_proba(data)[0][1]

    st.subheader("📈 Risk Assessment")
    st.write(f"Approval Probability: **{approval_prob * 100:.2f}%**")

    # ---------------- FINAL DECISION ----------------
    if approval_prob >= 0.60 and dti <= 0.35:
        st.success("✅ Loan Approved")
    elif approval_prob >= 0.45:
        st.warning("⚠️ Manual Review Required")
    else:
        st.error("❌ Loan Rejected")