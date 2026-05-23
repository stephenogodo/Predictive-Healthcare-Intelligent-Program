import pandas as pd
import numpy as np
import joblib


# ====================================================
# LOAD MODEL + FEATURES
# ====================================================

FORECAST_MODEL_PATH = "models/ed_forecast_model.pkl"

FEATURE_PATH = "models/ed_forecast_features.pkl"


ed_forecast_model = joblib.load(
    FORECAST_MODEL_PATH
)

expected_forecast_features = joblib.load(
    FEATURE_PATH
)


# ====================================================
# GENERATE FORECAST INPUT
# ====================================================

def build_forecast_input():

    forecast_df = pd.DataFrame()

    # Create dummy future rows
    for col in expected_forecast_features:

        forecast_df[col] = [0] * 7

    return forecast_df


# ====================================================
# GENERATE ED FORECAST
# ====================================================

def generate_ed_forecast():

    forecast_input = build_forecast_input()

    predictions = ed_forecast_model.predict(
        forecast_input
    )

    results_df = pd.DataFrame({

        "forecast_day": pd.date_range(
            start=pd.Timestamp.today(),
            periods=7,
            freq="D"
        ),

        "predicted_ed_demand": np.round(
            predictions,
            0
        )

    })

    return results_df