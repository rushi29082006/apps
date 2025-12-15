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
MAX_DTI = 0.40

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
        return 0.40


def calculate_max_loan(max_emi, months):
    return (
        max_emi * ((1 + MONTHLY_RATE) ** months - 1)
    ) / (MONTHLY_RATE * (1 + MONTHLY_RATE) ** months)


def calculate_emi(amount, months):
    return (
        amount * MONTHLY_RATE * (1 + MONTHLY_RATE) ** months
    ) / ((1 + MONTHLY_RATE) ** months - 1)


def improvement_advice(score, dti, income, tenure):
    advice = []

    if score < 750:
        advice.append("Improve credit score by paying EMIs and credit card bills on time.")

    if dti > 0.30:
        advice.append("Reduce EMI burden by choosing a longer tenure or lower loan amount.")

    if income < 30000:
        advice.append("Increase monthly income or add a co-applicant.")

    if tenure < 180:
        advice.append("Opt for a longer tenure to increase eligible loan amount.")

    advice.append("Maintain stable employment and avoid multiple loan applications.")

    return advice

# ---------------- PROCESS ----------------
if st.button("Calculate Eligibility"):

    # -------- EMI CAPACITY --------
    max_emi = total_income * MAX_DTI
    max_loan_by_income = calculate_max_loan(max_emi, loan_term_months)

    # -------- CREDIT SCALING --------
    credit_factor = credit_multiplier(credit_score)
    max_eligible_loan = max_loan_by_income * credit_factor

    # -------- FINAL ELIGIBLE LOAN --------
    eligible_loan_amount = min(loan_amount, max_eligible_loan)

    # -------- EMI --------
    emi = calculate_emi(eligible_loan_amount, loan_term_months)
    dti = emi / total_income if total_income > 0 else 0

    # ---------------- OPTIONAL ML (INFORMATIVE ONLY) ----------------
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

    # ---------------- OUTPUT ----------------
    st.subheader("Eligibility Result")

    st.write(f"**Eligible Loan Amount:** ₹{eligible_loan_amount:,.0f}")
    st.write(f"**Estimated Monthly EMI:** ₹{emi:,.0f}")
    st.write(f"**Loan Tenure:** {loan_term_months} months")
    st.write(f"**DTI Ratio:** {dti*100:.2f}%")

    # -------- WHY THIS AMOUNT IS ELIGIBLE --------
    st.subheader("Why This Amount Is Eligible")

    st.write(
        f"- Based on 40% EMI rule, your maximum affordable EMI is ₹{max_emi:,.0f}."
    )
    st.write(
        f"- With a tenure of {loan_term_months} months, this supports a loan of up to ₹{max_loan_by_income:,.0f}."
    )
    st.write(
        f"- Your credit score scales the eligible amount to ₹{max_eligible_loan:,.0f}."
    )

    if loan_amount > max_eligible_loan:
        st.write(
            "- Requested loan exceeded eligibility, so it was adjusted to the eligible limit."
        )
    else:
        st.write(
            "- Requested loan is within the eligible limit."
        )

    # -------- IMPROVEMENT ADVICE --------
    st.subheader("How to Improve Eligibility Next Time")

    for tip in improvement_advice(
        credit_score, dti, total_income, loan_term_months
    ):
        st.write(f"• {tip}")
