import pandas as pd
import joblib
import numpy as np


# ============================================================
# LOAD MODEL + FEATURE SCHEMA
# ============================================================

MODEL_PATH = "models/readmission_model.pkl"

FEATURE_PATH = "models/readmission_features.pkl"


readmission_model = joblib.load(MODEL_PATH)

expected_features = joblib.load(FEATURE_PATH)


# ============================================================
# PREPARE INPUT DATA
# ============================================================

def prepare_prediction_input(input_df):

    # ---------------------------------
    # Create copy
    # ---------------------------------

    prepared_df = input_df.copy()

    # ---------------------------------
    # Remove target column
    # ---------------------------------

    if "readmitted_within_30day" in prepared_df.columns:

        prepared_df.drop(
            columns=["readmitted_within_30day"],
            inplace=True
        )

    # ---------------------------------
    # Remove prediction columns
    # ---------------------------------

    removable_columns = [

        "readmission_risk_score",

        "predicted_readmission",

        "risk_category"

    ]

    for col in removable_columns:

        if col in prepared_df.columns:

            prepared_df.drop(
                columns=[col],
                inplace=True
            )

    # ---------------------------------
    # Add missing training columns
    # ---------------------------------

    for col in expected_features:

        if col not in prepared_df.columns:

            prepared_df[col] = 0

    # ---------------------------------
    # Keep only expected features
    # ---------------------------------

    prepared_df = prepared_df[expected_features]

    return prepared_df


# ============================================================
# GENERATE READMISSION PREDICTIONS
# ============================================================

def generate_readmission_predictions(input_df):

    # ---------------------------------
    # Prepare input data
    # ---------------------------------

    prepared_df = prepare_prediction_input(
        input_df
    )

    # ---------------------------------
    # Predict probabilities
    # ---------------------------------

    risk_probabilities = (
        readmission_model.predict_proba(
            prepared_df
        )[:, 1]
    )

    # ---------------------------------
    # Predict classes
    # ---------------------------------

    predictions = readmission_model.predict(
        prepared_df
    )

    # ---------------------------------
    # Create output dataframe
    # ---------------------------------

    results_df = input_df.copy()

    results_df["readmission_risk_score"] = np.round(
        risk_probabilities * 100,
        2
    )

    results_df["predicted_readmission"] = predictions

    # ---------------------------------
    # Risk categories
    # ---------------------------------

    results_df["risk_category"] = pd.cut(
        results_df["readmission_risk_score"],
        bins=[0, 30, 70, 100],
        labels=[
            "Low Risk",
            "Moderate Risk",
            "High Risk"
        ]
    )

    return results_df