import numpy as np
import joblib
import streamlit as st

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Mini Bank Loan System", layout="centered")
st.title("Mini Bank Loan Decision System")

model = joblib.load("loan_model.pkl")

# ---------------- INPUTS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income (monthly)", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0)

loan_amount = st.number_input("Requested Loan Amount", min_value=10000, step=10000)
loan_term = st.selectbox("Loan Term (months)", [120, 180, 240, 300, 360])

credit_score = st.slider(
    "Credit Score (300–900)",
    min_value=300,
    max_value=900,
    value=700,
    step=10
)

property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# ---------------- FUNCTIONS ----------------
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
        advice.append("Avoid applying for multiple loans or credit cards.")

    if score < 700:
        advice.append("Clear small outstanding loans to improve credit mix.")
        advice.append("Do not close old credit accounts; credit age matters.")

    if dti > 0.40:
        advice.append("Reduce existing debts before reapplying.")
        advice.append("Increase income or apply for a smaller loan amount.")

    advice.append("Maintain stable employment for the next 6–12 months.")
    return advice

# ---------------- PROCESS ----------------
if st.button("Check Loan Eligibility"):

    rejection_reasons = []

    total_income = applicant_income + coapplicant_income
    if total_income == 0:
        rejection_reasons.append("No verifiable income")

    multiplier = loan_multiplier(credit_score)
    if multiplier == 0:
        rejection_reasons.append("Credit score too low")

    approved_loan_amount = loan_amount * multiplier

    # EMI calculation
    annual_rate = 0.09
    monthly_rate = annual_rate / 12

    emi = (
        approved_loan_amount * monthly_rate *
        (1 + monthly_rate) ** loan_term
    ) / ((1 + monthly_rate) ** loan_term - 1)

    dti = emi / total_income if total_income > 0 else 1

    if dti > 0.45:
        rejection_reasons.append("High EMI burden compared to income")

    # ---------------- ML FEATURES ----------------
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    dependents = 4 if dependents == "3+" else int(dependents)
    property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    credit_score_norm = (credit_score - 300) / 600

    data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        np.log1p(applicant_income),
        np.log1p(coapplicant_income),
        approved_loan_amount,
        loan_term,
        credit_score_norm,
        property_area
    ]])

    approval_prob = model.predict_proba(data)[0][1]

    if approval_prob < 0.45:
        rejection_reasons.append("High predicted default risk")

    # ---------------- OUTPUT ----------------
    st.subheader("Affordability Analysis")
    st.write(f"Approved Loan Amount: ₹{approved_loan_amount:,.0f}")
    st.write(f"Monthly EMI: ₹{emi:,.0f}")
    st.write(f"DTI Ratio: {dti*100:.2f}%")

    st.subheader("Risk Assessment")
    st.write(f"Approval Probability: {approval_prob*100:.2f}%")

    # ---------------- FINAL DECISION ----------------
    if rejection_reasons:
        st.error("Loan Rejected")

        st.subheader("Reasons for Rejection")
        for r in rejection_reasons:
            st.write(f"- {r}")

        st.subheader("How to Improve Your Chances Next Time")
        tips = credit_advice(credit_score, dti)
        for tip in tips:
            st.write(f"• {tip}")

    else:
        if approval_prob >= 0.65:
            st.success(f"Loan Approved: ₹{approved_loan_amount:,.0f}")
        else:
            st.warning(
                f"Partial Approval / Manual Review: ₹{approved_loan_amount:,.0f}"
            )
