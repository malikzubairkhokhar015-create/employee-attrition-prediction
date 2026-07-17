# Employee Attrition Prediction

A machine learning system that predicts whether an employee is likely to leave a company, and explains *why* using feature importance analysis.

## Overview
This project uses classic ML algorithms (Logistic Regression and Random Forest) trained on employee HR data to predict attrition risk. Unlike a black-box prediction, this model also shows which factors matter most — helping HR teams take real, actionable retention steps.

## Key Features
- Compares Logistic Regression vs Random Forest with hyperparameter tuning
- Handles class imbalance using `class_weight="balanced"`
- **Feature importance visualization** — shows the top reasons employees leave (e.g., overtime, poor work-life balance, low income)
- Full end-to-end deployment: trained model → FastAPI backend → interactive web UI

## Top Factors Driving Attrition (from Feature Importance)
1. OverTime (Yes/No)
2. Work-Life Balance
3. Monthly Income
4. Job Satisfaction

## Tech Stack
- **Model:** scikit-learn (Logistic Regression, Random Forest)
- **Backend:** FastAPI
- **Frontend:** HTML/CSS/JavaScript
- **Training:** Google Colab (Jupyter Notebook)

## How to Run Locally
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python -m uvicorn app:app --reload`
4. Open `http://127.0.0.1:8000` in your browser

## Files
- `app.py` — FastAPI backend serving predictions
- `index.html` — Web interface for entering employee details
- `attrition_model.pkl` — Trained model
- `scaler.pkl` — Feature scaler used during training
- `feature_columns.pkl` — Column structure used by the model
- `categorical_columns.pkl` — List of categorical columns for encoding

## Results
| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.44 | 0.74 | 0.56 |
| Random Forest | 0.55 | 0.55 | 0.55 |
