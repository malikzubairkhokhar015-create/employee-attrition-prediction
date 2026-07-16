from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Employee Attrition Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/attrition_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
categorical_columns = joblib.load("model/categorical_columns.pkl")

class Employee(BaseModel):
    Age: float
    MonthlyIncome: float
    DistanceFromHome: float
    YearsAtCompany: float
    JobSatisfaction: float
    WorkLifeBalance: float
    TotalWorkingYears: float
    NumCompaniesWorked: float
    PercentSalaryHike: float
    TrainingTimesLastYear: float
    YearsSinceLastPromotion: float
    OverTime: str
    Department: str
    JobRole: str
    MaritalStatus: str

@app.post("/predict")
def predict(employee: Employee):
    input_df = pd.DataFrame([employee.dict()])
    input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    return {
        "prediction": "WILL LEAVE" if pred == 1 else "WILL STAY",
        "attrition_probability": round(float(proba), 4)
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")