import numpy as np
import joblib
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Mini Bank Loan System", layout="centered")
st.title("Mini Bank Loan Decision System")

model = joblib.load("loan_model.pkl")

# ---------------- USER INPUTS ----------------

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# -------- INCOME (MONTHLY) --------
applicant_income = st.number_input(
    "Applicant Monthly Income (₹)",
    min_value=0,
    step=1000
)

coapplicant_income = st.number_input(
    "Co-applicant Monthly Income (₹)",
    min_value=0,
    step=1000
)

total_income = applicant_income + coapplicant_income

#
