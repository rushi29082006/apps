import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="Loan Risk Assessment", layout="centered")
st.title("Loan Risk Assessment System")

model = joblib.load("loan_model.pkl")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=1)
loan_term = st.selectbox("Loan Term (months)", [120, 180, 240, 300, 360, 480])

credit_history = st.selectbox("Credit History", ["Good", "Bad"])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

if st.button("Assess Risk"):

    # HARD RULES (BANK STYLE)
    if credit_history == "Bad":
        st.error("Rejected: Poor credit history")
        st.stop()

    if applicant_income == 0 and coapplicant_income == 0:
        st.error("Rejected: No income")
        st.stop()

    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1
    dependents = 4 if dependents == "3+" else int(dependents)
    property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    applicant_income = np.log1p(applicant_income)
    coapplicant_income = np.log1p(coapplicant_income)

    data = np.array([[
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income,
        loan_amount, loan_term, credit_history, property_area
    ]])

    pd_risk = model.predict_proba(data)[0][1]

    st.write(f"Default Risk: **{pd_risk*100:.2f}%**")

    if pd_risk < 0.35:
        st.success("Approved (Low Risk)")
    elif pd_risk < 0.50:
        st.warning("Manual Review Required")
    else:
        st.error("Rejected (High Risk)")
