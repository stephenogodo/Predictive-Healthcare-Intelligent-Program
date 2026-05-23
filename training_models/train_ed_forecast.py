import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_PATH = "datasets/processed/daily_ed_forecasting.csv"
TARGET = "ed_arrivals"
FORECAST_HORIZON = 7

EXPERIMENT_NAME = "ED_Demand_Forecasting"

mlflow.set_experiment(EXPERIMENT_NAME)

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred)
    }


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
logger.info("Loading forecasting dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)

df = df.sort_index()

# --------------------------------------------------
# Create Supervised Dataset
# --------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

leak_prone = [
    "wait_time_minutes",
    "door_to_doctor_min",
    "ed_los_minutes",
    "labs_ordered",
    "imaging_ordered",
    "severity_index"
]

X = X.drop(columns=leak_prone)

# Temporal split (NO RANDOM SPLIT)
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

logger.info(f"Train size: {X_train.shape}")
logger.info(f"Test size: {X_test.shape}")

joblib.dump(
    X_train.columns.tolist(),
    "models/ed_forecast_features.pkl"
)
#===============================================================
# Testing for leakage by checking correlation of features with target   
#===============================================================
for col in X.columns:
    corr = X[col].corr(y_train)
    if abs(corr) > 0.95:
        logger.info(f"Potential leakage detected: {col} has correlation {corr:.2f} with target.")    
    else:
        logger.info(f"No leakage detected: {col} has correlation {corr:.2f} with target.")
# ==================================================
# BASELINE MODEL (FR-19)
# Naive persistence forecast
# ==================================================
with mlflow.start_run(run_name="Naive_Baseline"):

    logger.info("Training Naive Baseline")
    baseline_pred = y_test.shift(1).bfill()

    baseline_metrics = evaluate_model(y_test, baseline_pred)

    mlflow.log_params({
        "model_type": "naive_persistence",
        "forecast_horizon": FORECAST_HORIZON
    })

    mlflow.log_metrics(baseline_metrics)

    logger.info(f"Baseline Metrics: {baseline_metrics}")

# ==================================================
# PRODUCTION MODEL (LightGBM)
# ==================================================
with mlflow.start_run(run_name="LightGBM_ED_Forecaster" ):

    logger.info("Training LightGBM Forecaster")

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    model_metrics = evaluate_model(y_test, preds)

    # --------------------------------------------------
    # Log parameters
    # --------------------------------------------------
    mlflow.log_params({
        "model_type": "LightGBM",
        "forecast_horizon": FORECAST_HORIZON,
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 31
    })

    # --------------------------------------------------
    # Log metrics
    # --------------------------------------------------
    mlflow.log_metrics(model_metrics)

    # --------------------------------------------------
    # Log feature list
    # --------------------------------------------------
    feature_file = "feature_list.txt"
    Path(feature_file).write_text("\n".join(X.columns))
    mlflow.log_artifact(feature_file)

    # --------------------------------------------------
    # Log Model
    # --------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        name="ed_forecasting_model"
    )

    joblib.dump(
    model,
    "models/ed_forecast_model.pkl"
    )
    logger.info(f"Model Metrics: {model_metrics}")

# ==================================================
# VALIDATION CHECKPOINT (FR-21)
# ==================================================
improvement = (
    baseline_metrics["RMSE"] - model_metrics["RMSE"]
) / baseline_metrics["RMSE"]

logger.info(f"Improvement over baseline: {improvement:.2%}")

if improvement > 0:
    logger.info("✅ Model satisfies FR-19 requirement.")
else:
    logger.warning("⚠ Model does NOT outperform baseline.")

logger.info("ED Forecast Training Pipeline Completed Successfully.")