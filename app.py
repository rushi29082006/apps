import streamlit as st
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Eligibility (70% EMI Rule)", layout="centered")
st.title("Loan Eligibility System – EMI 70% Rule")

# ---------------- INPUTS ----------------
st.subheader("Applicant Details")

applicant_name = st.text_input("Applicant Name")
applicant_income = st.number_input(
    "Applicant Monthly Income (₹)", min_value=0, step=1000
)

coapplicant_name = st.text_input("Co-applicant Name (Optional)")
coapplicant_income = st.number_input(
    "Co-applicant Monthly Income (₹)", min_value=0, step=1000
)

total_income = applicant_income + coapplicant_income

st.subheader("Loan Details")

loan_term_months = st.slider(
    "Loan Tenure (Months)", min_value=12, max_value=360, value=120, step=12
)

requested_loan = st.number_input(
    "Requested Loan Amount (₹)", min_value=10000, step=10000
)

# ---------------- CONSTANTS ----------------
ANNUAL_INTEREST_RATE = 0.12   # higher rate (realistic for 70% EMI loans)
MONTHLY_RATE = ANNUAL_INTEREST_RATE / 12
MAX_DTI = 0.70   # 🔥 70% EMI RULE

# ---------------- FUNCTIONS ----------------
def calculate_max_loan(max_emi, months):
    return (
        max_emi * ((1 + MONTHLY_RATE) ** months - 1)
    ) / (MONTHLY_RATE * (1 + MONTHLY_RATE) ** months)


def calculate_emi(amount, months):
    return (
        amount * MONTHLY_RATE * (1 + MONTHLY_RATE) ** months
    ) / ((1 + MONTHLY_RATE) ** months - 1)

# ---------------- PROCESS ----------------
if st.button("Check Eligibility"):

    if applicant_name.strip() == "":
        st.error("Applicant name is required")
        st.stop()

    if total_income <= 0:
        st.error("Total monthly income must be greater than zero")
        st.stop()

    # -------- EMI CAPACITY (70%) --------
    max_emi = total_income * MAX_DTI

    max_eligible_loan = calculate_max_loan(
        max_emi, loan_term_months
    )

    eligible_loan = min(requested_loan, max_eligible_loan)

    emi = calculate_emi(eligible_loan, loan_term_months)
    dti = emi / total_income

    # ---------------- OUTPUT ----------------
    st.subheader("Eligibility Result")

    st.write(f"**Applicant:** {applicant_name}")
    if coapplicant_name.strip():
        st.write(f"**Co-applicant:** {coapplicant_name}")

    st.write(f"**Total Monthly Income:** ₹{total_income:,.0f}")
    st.write(f"**Loan Tenure:** {loan_term_months} months")

    st.write(f"**Maximum EMI Allowed (70%):** ₹{max_emi:,.0f}")
    st.write(f"**Maximum Eligible Loan:** ₹{max_eligible_loan:,.0f}")
    st.write(f"**Requested Loan:** ₹{requested_loan:,.0f}")
    st.write(f"**Approved / Eligible Loan:** ₹{eligible_loan:,.0f}")

    st.write(f"**Estimated Monthly EMI:** ₹{emi:,.0f}")
    st.write(f"**DTI Ratio:** {dti*100:.2f}%")

    if requested_loan > max_eligible_loan:
        st.warning(
            "Requested loan exceeds eligibility. "
            "Loan amount adjusted to maximum eligible limit."
        )

    st.success("Loan is ELIGIBLE under the 70% EMI rule")
