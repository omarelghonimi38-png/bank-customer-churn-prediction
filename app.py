import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ==========================================
# PAGE
# ==========================================

st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦"
)

st.title("🏦 Bank Customer Churn Predictor")

st.write(
    "Enter the customer's information to predict "
    "whether they are likely to leave the bank."
)


# ==========================================
# USER INPUT
# ==========================================

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

country = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

products_number = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

credit_card = st.selectbox(
    "Has Credit Card?",
    [0, 1]
)

active_member = st.selectbox(
    "Active Member?",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Churn"):

    customer = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    customer["credit_score"] = credit_score
    customer["age"] = age
    customer["tenure"] = tenure
    customer["balance"] = balance
    customer["products_number"] = products_number
    customer["credit_card"] = credit_card
    customer["active_member"] = active_member
    customer["estimated_salary"] = estimated_salary

    # Country Encoding
    if country == "Germany":
        customer["country_Germany"] = 1

    elif country == "Spain":
        customer["country_Spain"] = 1

    # Gender Encoding
    if gender == "Male":
        customer["gender_Male"] = 1

    # Prediction
    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    # Result
    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN.")
    else:
        st.success("✅ Customer is likely to STAY.")

    st.write(
        f"Churn Probability: **{probability * 100:.2f}%**"
    )