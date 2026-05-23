#============================================================

import pandas as pd

#============================================================
#KPI CALCULATIONS
#============================================================

def calculate_dashboard_kpis(prediction_df):
    total_patients = len(prediction_df)

    predicted_readmissions = prediction_df[
    prediction_df["predicted_readmission"] == 1
    ].shape[0]

    readmission_rate = (predicted_readmissions / total_patients) * 100

    average_risk_score = prediction_df[ "readmission_risk_score" ].mean()

    high_risk_patients = prediction_df[prediction_df["risk_category"] == "High Risk"].shape[0]

    average_length_of_stay = prediction_df["length_of_stay_days"].mean()

    return {
    "total_patients": total_patients,
    "predicted_readmissions": predicted_readmissions,
    "readmission_rate": round(readmission_rate, 2),
    "average_risk_score": round(average_risk_score, 2),
    "high_risk_patients": high_risk_patients,
    "average_length_of_stay": round(average_length_of_stay, 2)
    }

