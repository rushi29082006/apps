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

# -------- TENURE (MONTHS: 12–360) --------
loan_term_months = st.slider(
    "Loan Tenure (Months)",
    min_value=12,
    max_value=360,
    value=240,
    step=12
)

# -------- CREDIT SCORE --------
credit_score = st.slider(
    "Credit Score (300–900)",
    min_value=300,
    max_value=900,
    value=700,
    step=10
)

property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# -------- LOAN AMOUNT (USER INPUT) --------
loan_amount = st.number_input(
    "Requested Loan Amount (₹)",
    min_value=10000,
    step=10000
)

# ---------------- CONSTANTS ----------------
ANNUAL_RATE = 0.09
MONTHLY_RATE = ANNUAL_RATE / 12
MAX_DTI = 0.40   # 40% income rule (real bank standard)

# ---------------- FUNCTIONS ----------------

def credit_multiplier(score):
    if score >= 800:
        return 1.0
    elif score >= 750:
        return 0.85
    elif score >= 700:
        return 0.70
    elif score >= 650:
        return 0.55
    else:
        return 0.0


def calculate_max_loan(max_emi, months):
    return (
        max_emi *
        ((1 + MONTHLY_RATE) ** months - 1)
    ) / (
        MONTHLY_RATE * (1 + MONTHLY_RATE) ** months
    )


def calculate_emi(amount, months):
    return (
        amount * MONTHLY_RATE *
        (1 + MONTHLY_RATE) ** months
    ) / ((1 + MONTHLY_RATE) ** months - 1)


def credit_advice(score, dti):
    advice = []

    if score < 650:
        advice.append("Pay EMIs and credit card bills on time for 6 months.")
        advice.append("Reduce credit card usage below 30% of limit.")
        advice.append("Avoid multiple loan or credit card applications.")

    if score < 700:
        advice.append("Clear small outstanding loans.")
        advice.append("Do not close old credit accounts.")

    if dti > 0.40:
        advice.append("Reduce existing debts.")
        advice.append("Apply for a lower loan amount or longer tenure.")

    advice.append("Maintain stable employment for 6–12 months.")
    return advice

# ---------------- PROCESS ----------------

if st.button("Check Loan Eligibility"):

    rejection_reasons = []

    # -------- BASIC INCOME CHECK --------
    if total_income == 0:
        rejection_reasons.append("No verifiable monthly income")

    if total_income < 15000:
        rejection_reasons.append("Monthly income too low for loan eligibility")

    # -------- EMI-BASED ELIGIBILITY --------
    max_emi = total_income * MAX_DTI
    max_loan_by_income = calculate_max_loan(max_emi, loan_term_months)

    # -------- CREDIT-BASED SCALING --------
    credit_factor = credit_multiplier(credit_score)

    if credit_factor == 0:
        rejection_reasons.append("Credit score too low")

    max_eligible_loan = max_loan_by_income * credit_factor

    # -------- FINAL ELIGIBLE LOAN --------
    eligible_loan_amount = min(loan_amount, max_eligible_loan)

    if eligible_loan_amount <= 0:
        rejection_reasons.append("Loan amount not eligible based on income and credit")

    # -------- EMI ON ELIGIBLE AMOUNT --------
    emi = calculate_emi(eligible_loan_amount, loan_term_months)
    dti = emi / total_income if total_income > 0 else 1

    # -------- ML FEATURES --------
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    dependents_val = 4 if dependents == "3+" else int(dependents)
    property_area_val = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    credit_score_norm = (credit_score - 300) / 600

    ml_input = np.array([[
        gender_val,
        married_val,
        dependents_val,
        education_val,
        self_employed_val,
        np.log1p(applicant_income),
        np.log1p(coapplicant_income),
        eligible_loan_amount,
        loan_term_months,
        credit_score_norm,
        property_area_val
    ]])

    approval_prob = model.predict_proba(ml_input)[0][1]

    if approval_prob < 0.45:
        rejection_reasons.append("High predicted default risk")

    # ---------------- OUTPUT ----------------

    st.subheader("Loan Eligibility Summary")
    st.write(f"Maximum Eligible Loan: ₹{max_eligible_loan:,.0f}")
    st.write(f"Requested Loan: ₹{loan_amount:,.0f}")
    st.write(f"Approved / Eligible Loan: ₹{eligible_loan_amount:,.0f}")
    st.write(f"Loan Tenure: {loan_term_months} months")
    st.write(f"Monthly EMI: ₹{emi:,.0f}")
    st.write(f"DTI Ratio: {dti*100:.2f}%")
    st.write(f"Approval Probability: {approval_prob*100:.2f}%")

    # ---------------- FINAL DECISION ----------------
    if rejection_reasons:
        st.error("Loan Rejected")

        st.subheader("Reasons")
        for r in rejection_reasons:
            st.write(f"- {r}")

        st.subheader("How to Improve Next Time")
        for tip in credit_advice(credit_score, dti):
            st.write(f"• {tip}")

    else:
        if loan_amount > max_eligible_loan:
            st.warning(
                f"Requested amount reduced to eligible limit: ₹{eligible_loan_amount:,.0f}"
            )

        if approval_prob >= 0.65:
            st.success(f"Loan Approved: ₹{eligible_loan_amount:,.0f}")
        else:
            st.warning(f"Manual Review Required: ₹{eligible_loan_amount:,.0f}")
