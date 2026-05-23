#=========================================================
# Predictive Healthcare Intelligent Program (PHIP)
# Readmission Risk Prediction Model Pipeline
# This module defines the data preprocessing and modeling 
# pipeline for predicting patient readmission risk.
#=========================================================


#==========================================================
# STEP 1:  Importing necessary libraries
#==========================================================

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import os


#=========================================================
# STEP 2:  LOAD DATA (WBS 2.3 – Clean Dataset)
#=========================================================
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = PROJECT_ROOT / "datasets" / "readmission_feature_store.csv"

print(f"Loading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)


#=========================================================
# STEP 3:  Further CLEANING + LEAKAGE PREVENTION
#=========================================================

df["discharge_date"] = pd.to_datetime(df["discharge_date"])

# Remove IDs and post-outcome leakage
drop_cols = [
    "patient_id",
    "admission_id",
    "readmission_date",
    "admission_date",
    "attending_physician_id"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])


#===================================================================
# STEP 4: TEMPORAL TRAIN / TEST SPLIT (Hospital Standard)
#===================================================================
# We will use a temporal split based on discharge date to mimic 
# real-world deployment

df = df.sort_values("discharge_date")

split_date = df["discharge_date"].quantile(0.80)

train_df = df[df.discharge_date < split_date]
test_df  = df[df.discharge_date >= split_date]

train_df = train_df.drop(columns=["discharge_date"])
test_df  = test_df.drop(columns=["discharge_date"])

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

#=========================================================
# STEP 5: FEATURE / TARGET SEPARATION
#=========================================================
TARGET = "readmitted_within_30d"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


#=========================================================
# STEP 6: AUTOMATIC FEATURE TYPE DETECTION
#=========================================================

# Identify categorical and numerical features
categorical_cols = X_train.select_dtypes(
    include=["object","category"]
).columns.tolist()

numeric_cols = X_train.select_dtypes(
    exclude=["object","category"]
).columns.tolist()

#=========================================================
# STEP 7: ONE-HOT ENCODING PIPELINE
#=========================================================

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

#=========================================================
# STEP 8: CLASS IMBALANCE HANDLING
#=========================================================

# We will use class weights in the logistic regression model
# to handle imbalance
scale_pos_weight = (
    y_train.value_counts()[0] /
    y_train.value_counts()[1]
)

#========================================================
# STEP 9: SETUP MLFLOW (WBS 2.6)
#=========================================================
mlflow.set_experiment("MHN_Readmission_Modelling")


#=========================================================
# STEP 10: BASELINE MODEL — LOGISTIC REGRESSION (WBS 2.4)
#=========================================================

import datetime

run_name = f"Baseline_Logistic_Regression_{datetime.datetime.now():%Y%m%d_%H%M}"

#with mlflow.start_run(run_name=run_name):

with mlflow.start_run(run_name=run_name):

    baseline = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])



    baseline.fit(X_train, y_train)

    probs = baseline.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)

    print(classification_report(y_test, preds))

    mlflow.log_param("model","LogisticRegression")
    mlflow.log_metric("ROC_AUC", auc)

    mlflow.sklearn.log_model(baseline, "baseline_model")


#=========================================================
# STEP 11: XGBOOST MODEL (INITIAL VERSION)
#=========================================================

run_name1 = f"XGBoost_{datetime.datetime.now():%Y%m%d_%H%M}"


with mlflow.start_run(run_name=run_name1):

    xgb_model = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    xgb_model.fit(X_train, y_train)

    probs = xgb_model.predict_proba(X_test)[:,1]


#=========================================================
# STEP 12: THRESHOLD OPTIMISATION (BRD Precision ≥ 0.80)
#=========================================================
precision, recall, thresholds = precision_recall_curve(y_test, probs)

results = pd.DataFrame({
    "threshold": thresholds,
    "precision": precision[:-1],
    "recall": recall[:-1]
})

TARGET_PRECISION = 0.80

filtered = results.query("precision >= @TARGET_PRECISION")

best = (
    filtered if not filtered.empty else results
).sort_values("precision", ascending=False).iloc[0]
optimal_threshold = best.threshold
preds = (probs >= optimal_threshold).astype(int)

print("Optimal Threshold:", optimal_threshold)
print(classification_report(y_test, preds))

#=========================================================
# STEP 13: MODEL EVALUATION ARTIFACTS (WBS 2.7)
#=========================================================

# Confusion Matrix

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")

os.makedirs("evaluation_outputs", exist_ok=True)
plt.savefig("evaluation_outputs/confusion_matrix.png")
plt.close()

#===================================================================
#STEP 14                   ROC Curve
#===================================================================

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, probs)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")

plt.savefig("evaluation_outputs/roc_curve.png")
plt.close()



#=========================================================
#STEP 15:     LOG FINAL METRICS TO MLFLOW
#=========================================================

auc = roc_auc_score(y_test, probs)

mlflow.log_param("model","XGBoost")
mlflow.log_param("optimal_threshold", float(optimal_threshold))

mlflow.log_metric("ROC_AUC", auc)
mlflow.log_metric("precision", float(best.precision))
mlflow.log_metric("recall", float(best.recall))

mlflow.log_artifact("evaluation_outputs/confusion_matrix.png")
mlflow.log_artifact("evaluation_outputs/roc_curve.png")

mlflow.sklearn.log_model(xgb_model, "xgboost_model")


#=========================================================
#STEP 16:     PM VALIDATION CHECKPOINT OUTPUT (WBS 2.10)
#=========================================================

evaluation_summary = pd.DataFrame({
    "Metric":["ROC_AUC","Precision","Recall","Threshold"],
    "Value":[auc,best.precision,best.recall,optimal_threshold]
})

evaluation_summary.to_csv(
    "evaluation_outputs/model_evaluation_summary.csv",
    index=False
)


#=========================================================
#STEP 17:     MODEL REFINEMENT LOOP READY (WBS 2.11)
#=========================================================
'''If PM requests improvement,
The following are things to be done iteratively:

change hyperparameters
rerun script
MLflow automatically records new experiment

Example run names:

XGBoost_v1
XGBoost_PM_refined
XGBoost_final '''


#=========================================================
# FINAL DEPLOYABLE OUTPUT (Week 3 Dependency)
#=========================================================

# The final output of this script is the trained XGBoost model
# and the associated evaluation metrics/artifacts logged in MLflow.

results = X_test.copy()

results["readmission_probability"] = probs

results["risk_level"] = pd.cut(
    probs,
    bins=[0,0.4,0.65,1],
    labels=["Low","Medium","High"]
)

results.to_csv("readmission_predictions.csv", index=False)