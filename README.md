# 📊 Employee Attrition Predictor

A machine learning model that predicts whether an employee is likely to leave a company, and explains *why* — turning a black-box prediction into an actionable HR insight.

🔗 **Live App:** https://employee-attrition-prediction-rjujkpsmfnzjw6vburaa8e.streamlit.app

## Overview
This project uses classic ML algorithms (Logistic Regression and Random Forest) trained on employee HR data to predict attrition risk. Built with Streamlit for a clean, interactive interface — no technical setup needed to try it out.

## Input Features the Model Uses
👤 Age & Marital Status
🏢 Department & Job Role
📅 Years At Company
💼 Total Working Years
🔄 Number of Companies Worked
⏰ OverTime
😊 Job Satisfaction
⚖️ Work-Life Balance
📚 Training Times Last Year
📈 Years Since Last Promotion
💰 Monthly Income & Salary Hike %

## How It Works
The model was trained on employee HR data, learning the statistical relationship between an employee's work conditions and their likelihood of leaving. It recognized that overtime, poor work-life balance, low job satisfaction, and low income consistently correlate with higher attrition risk.

## What It Predicts
Given any combination of employee details, the model instantly estimates the attrition risk percentage and classifies the employee as **Will Stay** or **Will Leave**.

## Top Factors Driving Attrition (Feature Importance)
1. OverTime
2. Work-Life Balance
3. Monthly Income
4. Job Satisfaction

## Model Performance
| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.44 | 0.74 | 0.56 |
| Random Forest | 0.55 | 0.55 | 0.55 |

Trained and evaluated using an 80/20 train-test split with `class_weight="balanced"` to handle class imbalance.

## Built With
Python · Scikit-learn · Pandas · NumPy · Streamlit

## How to Run Locally
```bash
git clone https://github.com/malikzubairkhokhar015-create/employee-attrition-prediction
cd employee-attrition-prediction
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `streamlit_app.py` — Streamlit web application
- `attrition_model.pkl` — Trained Random Forest model
- `scaler.pkl` — Feature scaler used during training
- `feature_columns.pkl` — Column structure used by the model
- `categorical_columns.pkl` — List of categorical columns for encoding
- `app.py` — FastAPI backend (alternative deployment)
- `index.html` — Web interface for FastAPI version

