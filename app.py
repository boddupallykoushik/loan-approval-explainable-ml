import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("loan_model.pkl", "rb"))

st.title("💰 Loan Approval Prediction (AI + Explainable AI)")

# -------------------------------
# INPUT FIELDS
# -------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income = st.text_input("Applicant Income (₹)")
co_income = st.text_input("Coapplicant Income (₹)")
loan_amount = st.text_input("Loan Amount (₹)")
loan_term = st.text_input("Loan Term (in days)")

credit_history = st.selectbox("Credit History", ["1", "0"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Feature names
feature_names = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict"):
    
    if income == "" or co_income == "" or loan_amount == "" or loan_term == "":
        st.warning("Please fill all numeric fields")
        st.stop()

    try:
        # -------------------------------
        # INPUT CONVERSION
        # -------------------------------
        income_val = float(income)
        co_income_val = float(co_income)
        loan_amount_val = float(loan_amount)
        loan_term_val = float(loan_term)

        dependents_val = int(dependents.replace("+", ""))

        gender_val = 1 if gender == "Male" else 0
        married_val = 1 if married == "Yes" else 0
        education_val = 1 if education == "Graduate" else 0
        self_employed_val = 1 if self_employed == "Yes" else 0

        credit_val = int(credit_history)
        property_val = ["Rural", "Semiurban", "Urban"].index(property_area)

        # 🔥 DEFINE INPUT DATA HERE
        input_data = np.array([
            gender_val,
            married_val,
            dependents_val,
            education_val,
            self_employed_val,
            income_val,
            co_income_val,
            loan_amount_val,
            loan_term_val,
            credit_val,
            property_val
        ], dtype=float).reshape(1, -1)

        # -------------------------------
        # PREDICTION
        # -------------------------------
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

        # -------------------------------
        # SHAP
        # -------------------------------
        input_df = pd.DataFrame(input_data, columns=feature_names)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
            base_val = explainer.expected_value[1]
        else:
            shap_vals = shap_values
            base_val = explainer.expected_value

        for i in range(len(feature_names)):
            impact = float(np.array(shap_vals[0][i]).flatten()[0])

            if impact > 0:
                st.write(f"🔼 {feature_names[i]} increased approval chance ({impact:.4f})")
            else:
                st.write(f"🔽 {feature_names[i]} decreased approval chance ({impact:.4f})")

    except Exception as e:
        st.error(f"Error: {e}")


