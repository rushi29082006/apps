import numpy as np
import joblib
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Mini Bank Loan System", layout="centered")
st.title("Mini Bank Loan Eligibility Calculator")

model = joblib.load("loan_model.pkl")

# ---------------- USER INPUTS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# -------- MONTHLY INCOME --------
applicant_income = st.number_input(
    "Applicant Monthly Income (₹)", min_value=0, step=1000
)
coapplicant_income = st.number_input(
    "Co-applicant Monthly Income (₹)", min_value=0, step=1000
)

total_income = applicant_income + coapplicant_income

# -------- TENURE (MONTHS) --------
loan_term_months = st.slider(
    "Loan Tenure (Months)", min_value=12, max_value=360, value=240, step=12
)

# -------- CREDIT SCORE --------
credit_score = st.slider(
    "Credit Score (300–900)", min_value=300, max_value=900, value=700, step=10
)

property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# -------- REQUESTED LOAN --------
loan_amount = st.number_input(
    "Requested Loan Amount (₹)", min_value=10000, step=10000
)

# ---------------- CONSTANTS ----------------
ANNUAL_RATE = 0.09
MONTHLY_RATE = ANNUAL_RATE / 12
MAX_DTI = 0.40  # 40% EMI rule

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
        return 0.40  # minimum eligibility, NOT rejection


def calculate_max_loan(max_emi, months):
    return (
        max_emi * ((1 + MONTHLY_RATE) ** months - 1)
    ) / (MONTHLY_RATE * (1 + MONTHLY_RATE) ** months)


def calculate_emi(amount, months):
    return (
        amount * MONTHLY_RATE * (1 + MONTHLY_RATE) ** months
    ) / ((1 + MONTHLY_RATE) ** months - 1)

# ---------------- PROCESS ----------------
if st.button("Calculate Eligibility"):

    # -------- EMI CAPACITY --------
    max_emi = total_income * MAX_DTI

    max_loan_by_income = calculate_max_loan(
        max_emi, loan_term_months
    )

    # -------- CREDIT SCALING (NO REJECTION) --------
    credit_factor = credit_multiplier(credit_score)

    max_eligible_loan = max_loan_by_income * credit_factor

    # -------- FINAL ELIGIBLE LOAN --------
    eligible_loan_amount = min(loan_amount, max_eligible_loan)

    # -------- EMI ON ELIGIBLE LOAN --------
    emi = calculate_emi(eligible_loan_amount, loan_term_months)

    dti = emi / total_income if total_income > 0 else 0

    # ---------------- OPTIONAL ML (NO BLOCKING) ----------------
    gender_v = 1 if gender == "Male" else 0
    married_v = 1 if married == "Yes" else 0
    education_v = 1 if education == "Graduate" else 0
    self_emp_v = 1 if self_employed == "Yes" else 0
    dependents_v = 4 if dependents == "3+" else int(dependents)
    property_v = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    credit_norm = (credit_score - 300) / 600

    ml_input = np.array([[
        gender_v,
        married_v,
        dependents_v,
        education_v,
        self_emp_v,
        np.log1p(applicant_income),
        np.log1p(coapplicant_income),
        eligible_loan_amount,
        loan_term_months,
        credit_norm,
        property_v
    ]])

    approval_prob = model.predict_proba(ml_input)[0][1]

    # ---------------- OUTPUT (ONLY WHAT YOU ASKED) ----------------
    st.subheader("Eligibility Result")

    st.write(f"**Eligible Loan Amount:** ₹{eligible_loan_amount:,.0f}")
    st.write(f"**Monthly EMI:** ₹{emi:,.0f}")
    st.write(f"**Loan Tenure:** {loan_term_months} months")
    st.write(f"**DTI Ratio:** {dti*100:.2f}%")

    if loan_amount > max_eligible_loan:
        st.info(
            f"Requested amount adjusted to maximum eligible limit."
        )
