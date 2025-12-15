import numpy as np
import joblib
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Mini Bank Loan System", layout="centered")
st.title("Mini Bank Loan Decision System")

# Load trained ML model
model = joblib.load("loan_model.pkl")

# ---------------- USER INPUTS ----------------

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# -------- INCOME (MONTHLY – REAL BANK STANDARD) --------
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

# -------- LOAN DETAILS --------
loan_amount = st.number_input(
    "Requested Loan Amount (₹)",
    min_value=10000,
    step=10000
)

loan_years = st.selectbox(
    "Loan Tenure (Years)",
    [10, 15, 20, 25, 30]
)

loan_term_months = loan_years * 12

# -------- CREDIT SCORE (REALISTIC) --------
credit_score = st.slider(
    "Credit Score (300–900)",
    min_value=300,
    max_value=900,
    value=700,
    step=10
)

property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# ---------------- HELPER FUNCTIONS ----------------

def loan_multiplier(score):
    if score >= 800:
        return 1.0
    elif score >= 750:
        return 0.8
    elif score >= 700:
        return 0.6
    elif score >= 650:
        return 0.5
    else:
        return 0.0


def credit_advice(score, dti):
    advice = []

    if score < 650:
        advice.append("Pay all EMIs and credit card bills on time for at least 6 months.")
        advice.append("Keep credit card usage below 30% of the limit.")
        advice.append("Avoid multiple loan or credit card applications.")

    if score < 700:
        advice.append("Clear small outstanding loans to improve credit mix.")
        advice.append("Do not close old credit accounts; credit history length matters.")

    if dti > 0.40:
        advice.append("Reduce existing debts before reapplying.")
        advice.append("Apply for a lower loan amount or increase income.")

    advice.append("Maintain stable employment for 6–12 months.")

    return advice

# ---------------- PROCESS ----------------

if st.button("Check Loan Eligibility"):

    rejection_reasons = []

    # -------- INCOME VALIDATION --------
    total_income = applicant_income + coapplicant_income

    if total_income == 0:
        rejection_reasons.append("No verifiable monthly income")

    if total_income < 15000:
        rejection_reasons.append("Total monthly income too low for loan eligibility")

    # -------- CREDIT-BASED LOAN SCALING --------
    multiplier = loan_multiplier(credit_score)

    if multiplier == 0:
        rejection_reasons.append("Credit score too low")

    approved_loan_amount = loan_amount * multiplier

    # -------- EMI CALCULATION (MONTHLY) --------
    annual_rate = 0.09
    monthly_rate = annual_rate / 12

    if approved_loan_amount > 0:
        emi = (
            approved_loan_amount * monthly_rate *
            (1 + monthly_rate) ** loan_term_months
        ) / ((1 + monthly_rate) ** loan_term_months - 1)
    else:
        emi = 0

    dti = emi / total_income if total_income > 0 else 1

    if dti > 0.45:
        rejection_reasons.append("EMI burden too high compared to monthly income")

    # -------- ML FEATURE PREPARATION --------
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    dependents_val = 4 if dependents == "3+" else int(dependents)
    property_area_val = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    credit_score_norm = (credit_score - 300) / 600

    data = np.array([[
        gender_val,
        married_val,
        dependents_val,
        education_val,
        self_employed_val,
        np.log1p(applicant_income),
        np.log1p(coapplicant_income),
        approved_loan_amount,
        loan_term_months,
        credit_score_norm,
        property_area_val
    ]])

    approval_prob = model.predict_proba(data)[0][1]

    if approval_prob < 0.45:
        rejection_reasons.append("High predicted default risk")

    # ---------------- OUTPUT ----------------

    st.subheader("Affordability Analysis")
    st.write(f"Approved Loan Amount: ₹{approved_loan_amount:,.0f}")
    st.write(f"Monthly EMI: ₹{emi:,.0f}")
    st.write(f"DTI Ratio: {dti*100:.2f}%")
    st.write(f"Loan Tenure: {loan_years} years ({loan_term_months} months)")

    st.subheader("Risk Assessment")
    st.write(f"Approval Probability: {approval_prob*100:.2f}%")

    # ---------------- FINAL DECISION ----------------
    if rejection_reasons:
        st.error("Loan Rejected")

        st.subheader("Reasons for Rejection")
        for r in rejection_reasons:
            st.write(f"- {r}")

        st.subheader("How to Improve Your Chances Next Time")
        for tip in credit_advice(credit_score, dti):
            st.write(f"• {tip}")

    else:
        if approval_prob >= 0.65:
            st.success(f"Loan Approved: ₹{approved_loan_amount:,.0f}")
        else:
            st.warning(
                f"Partial Approval / Manual Review: ₹{approved_loan_amount:,.0f}"
            )
