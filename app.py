import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("Loan Approval Prediction System")

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

gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0
dependents = 4 if dependents == "3+" else int(dependents)
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

if st.button("Predict Loan Status"):

    data = np.array([[
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income,
        loan_amount, loan_term, credit_history, property_area
    ]])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.write(f"Approval Probability: **{probability*100:.2f}%**")

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    # ---------------- EXPLANATION ----------------
    st.subheader("Why this decision?")

    reasons = []

    if credit_history == 0:
        reasons.append("Poor credit history")
    if applicant_income < 3000:
        reasons.append("Low applicant income")
    if loan_amount > applicant_income * 10:
        reasons.append("High loan amount compared to income")
    if dependents >= 3:
        reasons.append("High number of dependents")
    if self_employed == 1:
        reasons.append("Self-employed income risk")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("• Strong financial profile")
