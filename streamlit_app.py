import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="centered")

model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
categorical_columns = joblib.load("categorical_columns.pkl")

st.title("📊 Employee Attrition Predictor")
st.write("Fill in employee details to check retention risk.")

st.subheader("Personal Info")
col1, col2 = st.columns(2)
age = col1.number_input("Age", value=30)
marital_status = col2.selectbox("Marital Status", ["Single", "Married", "Divorced"])
distance_from_home = st.number_input("Distance From Home (km)", value=10)

st.subheader("Job Details")
col1, col2 = st.columns(2)
department = col1.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = col2.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager", "Healthcare Representative"])

col1, col2 = st.columns(2)
years_at_company = col1.number_input("Years At Company", value=3)
total_working_years = col2.number_input("Total Working Years", value=5)

col1, col2 = st.columns(2)
num_companies_worked = col1.number_input("Num Companies Worked", value=2)
overtime = col2.selectbox("OverTime", ["No", "Yes"])

st.subheader("Satisfaction & Growth")
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
work_life_balance = st.slider("Work Life Balance", 1, 4, 3)

col1, col2 = st.columns(2)
training_times_last_year = col1.number_input("Training Times Last Year", value=2)
years_since_last_promotion = col2.number_input("Years Since Last Promotion", value=1)

st.subheader("Compensation")
col1, col2 = st.columns(2)
monthly_income = col1.number_input("Monthly Income", value=5000)
percent_salary_hike = col2.number_input("Percent Salary Hike", value=15)

if st.button("Check Attrition Risk", type="primary"):
    input_dict = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "DistanceFromHome": distance_from_home,
        "YearsAtCompany": years_at_company,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "TotalWorkingYears": total_working_years,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "TrainingTimesLastYear": training_times_last_year,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "OverTime": overtime,
        "Department": department,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ WILL LEAVE — Attrition Risk: {proba*100:.1f}%")
    else:
        st.success(f"✅ WILL STAY — Attrition Risk: {proba*100:.1f}%")

    st.progress(float(proba))