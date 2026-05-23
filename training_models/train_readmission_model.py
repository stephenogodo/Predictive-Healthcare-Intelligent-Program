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
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns


import mlflow.sklearn
import os
import mlflow

mlflow.end_run()   #  kill any orphan active run

#=========================================================
# LOGGER CONFIGURATION
#=========================================================

import logging
import sys





logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_pipeline.log")
    ]
)
logger = logging.getLogger("PHIP_Readmission_Model")



#=========================================================
# STEP 2:  LOAD DATA (WBS 2.3 – Clean Dataset)
#=========================================================
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = PROJECT_ROOT / "datasets" /"processed"/ "read_ml.csv"

logger.info(f"Loading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

logger.info(f"Dataset Shape: {df.shape}")

#=========================================================
# STEP 3:  Further CLEANING + LEAKAGE PREVENTION
#=========================================================
logger.info("Performing leakage prevention and cleaning")

df["discharge_date"] = pd.to_datetime(df["discharge_date"])

# Remove IDs and post-outcome leakage
drop_cols = [
    
    "admission_id",
    "readmission_date",
    "admission_date",
    "attending_physician_id",
    "readmitted_30d",
    "total_charges_usd",
    "insurance_paid_usd",
    "readmission_reason",
    "mm",
    "ed_visit_last_90d_y",
    "readmitted-within_30d",
    "adm_last_90d_y",
    "ed_visits_last_360d_y",
    "labs_last_90d_y",
    "meds_90d_y",
    "meds_360d_y",
    "adm_last_360d_y",
    "admission_date",
    "adm_velocity_y",
    "first_name",
    "last_name",
    "date_of_birth",
    "age",
    "zip_code",
    "admission_id"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])


#===================================================================
# STEP 4: TEMPORAL TRAIN / TEST SPLIT (Hospital Standard)
#===================================================================
# We will use a temporal split based on discharge date to mimic 
# real-world deployment

logger.info("Performing temporal train-test split")

df = df.sort_values("discharge_date")

split_date = df["discharge_date"].quantile(0.80)

train_df = df[df.discharge_date < split_date]
test_df  = df[df.discharge_date >= split_date]

train_df = train_df.drop(columns=["discharge_date"])
test_df  = test_df.drop(columns=["discharge_date"])

logger.info(f"Train Shape: {train_df.shape}")
logger.info(f"Test Shape: {test_df.shape}")

#=========================================================
#           TEMPORARY STEP — LEAKAGE AUDIT
#=========================================================
logger.info("Running leakage correlation audit")


'''for col in X.columns:
    corr = X[col].corr(y_train)
    if abs(corr) > 0.95:
        logger.info(f"Potential leakage detected: {col} has correlation {corr:.2f} with target.")    
    else:
        logger.info(f"No leakage detected: {col} has correlation {corr:.2f} with target.")
'''
leak_check = (
    train_df
    .corr(numeric_only=True)["readmitted_within_30days"]
    .abs()
    .sort_values(ascending=False)
)

print(leak_check.head(15))


train_patients = set(train_df["patient_id"])
test_patients = set(test_df["patient_id"])

logger.info(
    f"Patient overlap: {len(train_patients.intersection(test_patients))}"
)

train_df = train_df.drop(columns=["patient_id"])
test_df  = test_df.drop(columns=["patient_id"])
#=========================================================
# STEP 5: FEATURE / TARGET SEPARATION
#=========================================================
TARGET = "readmitted_within_30days"
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



logger.info(
    f"Detected {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")

# Define the column transformer for preprocessing
#===========================================================================================
# STEP 7: PREPROCESSING PIPELINE (FIXED VERSION)
#===========================================================================================

#numeric_transformer = Pipeline([
#    ("scaler", StandardScaler())
#])

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

#categorical_transformer = Pipeline([
#    ("onehot", OneHotEncoder(handle_unknown="ignore"))
#])

from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

#preprocessor = ColumnTransformer([
 #   ("cat", categorical_transformer, categorical_cols),
#    ("num", numeric_transformer, numeric_cols)
#])

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

#=========================================================
# STEP 8: CLASS IMBALANCE HANDLING
#=========================================================

# We will use class weights in the logistic regression model
# to handle imbalance
false_positive_penalty = 2  #  precision booster
scale_pos_weight = (
    y_train.value_counts()[0] /
    y_train.value_counts()[1]
) * false_positive_penalty
logger.info(f"Computed scale_pos_weight = {scale_pos_weight:.2f}")
#objective="binary:logistic"
#========================================================
# STEP 9: SETUP MLFLOW (WBS 2.6)
#=========================================================
mlflow.set_experiment("MHN_Readmission_Modelling")


#=========================================================
# STEP 10: BASELINE MODEL — LOGISTIC REGRESSION (WBS 2.4)
#=========================================================
import datetime
logger.info("Starting the PHIP Readmission Master run Training Pipeline")
run_name0 = f"PHIP_Readmission_Master_Run_{datetime.datetime.now():%Y%m%d_%H%M}"
with mlflow.start_run(run_name=run_name0):

    run_name = f"Baseline_Logistic_Regression_Calibrated_{datetime.datetime.now():%Y%m%d_%H%M}"


    logger.info("Training Baseline Logistic Regression")
    with mlflow.start_run(run_name=run_name, nested=True):

        #baseline_pipe = Pipeline([
        #    ("prep", preprocessor),
        #    ("model", LogisticRegression(
        #        max_iter=4000,
        #        solver="saga",
        #        class_weight="balanced",
        #        random_state=42
        #    ))
        #])
    

        baseline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
            ))
            ]
            )


        calibrated_baseline = CalibratedClassifierCV(
        baseline_model,
        method="isotonic",
        cv=5
        )

        calibrated_baseline.fit(X_train, y_train)
        ## ------------------------------------------------------------------
        # GET PREDICTED PROBABILITIES
        # -------------------------------------------------------------------

        probs = calibrated_baseline.predict_proba(X_test)[:,1]
        y_pred = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)

        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_test, probs)

        logger.info("Baseline Model Results:")
        logger.info(classification_report(y_test, y_pred))

        mlflow.log_param("model","Calibrated_LogisticRegression")
        mlflow.log_metric("ROC_AUC", auc)
        mlflow.log_metric("PR_AUC", pr_auc)

        mlflow.sklearn.log_model(calibrated_baseline, name="baseline_model")

        #=========================================================
        # STEP 11: XGBOOST MODEL (INITIAL VERSION)
        #========================================================= 

 
        run_name1 = f"XGBoost_{datetime.datetime.now():%Y%m%d_%H%M}"


    with mlflow.start_run(run_name=run_name1, nested=True):

        logger.info("Building XGBoost pipeline")

        xgb_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        objective="binary:logistic",
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",   # Changed to AUC-PR for better imbalance handling
        tree_method="hist",
        random_state=42
        )
        )])
    


        logger.info("Applying probability calibration")

        calibrated_xgb = CalibratedClassifierCV(
        xgb_pipe,
        method="isotonic",
        cv=5
      )

        logger.info("Training calibrated XGBoost model")

        calibrated_xgb.fit(X_train, y_train)

        probs = calibrated_xgb.predict_proba(X_test)[:,1]

        logger.info("Generating predictions completed")

        # ===== Log Model =====
        mlflow.sklearn.log_model(
        calibrated_xgb,
        name="xgboost_model"
        )

        
        # =============================================================
        # SAVE MODEL AND FEATURE SCHEMA LOCALLY FOR STREAMLIT APP
        # ==============================================================
        joblib.dump(
        calibrated_xgb  ,
        "models/readmission_model.pkl"
        )

        # save feature schema
        joblib.dump(
        calibrated_xgb.feature_names_in_,
        "models/readmission_features.pkl"
        )

        print("✅ Model and feature schema saved!")


      
    #=========================================================
    # STEP 12: THRESHOLD OPTIMISATION (BRD Precision ≥ 0.80)
    #=========================================================
    with mlflow.start_run(run_name=f"Threshold_Optimization_{datetime.datetime.now():%Y%m%d_%H%M}", nested=True):
        from sklearn.metrics import precision_score, recall_score
        import numpy as np

        logger.info("Running BRD Precision Threshold Optimization")

        y_prob = calibrated_xgb.predict_proba(X_test)[:, 1]

        TARGET_PRECISION = 0.80
        MIN_RECALL = 0.15

        best_threshold = None
        best_precision = 0
        best_recall = 0

        candidate_found = False

        # -----------------------------------
        # 1. BRD-CONSTRAINED SEARCH
        # -----------------------------------
        for t in np.arange(0.01, 0.99, 0.01):

            preds = (y_prob >= t).astype(int)

            # prevent undefined precision
            if preds.sum() == 0:
                continue

        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)

        if precision >= TARGET_PRECISION and recall >= MIN_RECALL:

            candidate_found = True

        if recall > best_recall:
            best_threshold = t
            best_precision = precision
            best_recall = recall


        # -----------------------------------
        # 2. SAFETY FALLBACK
        # -----------------------------------
        if not candidate_found:

            logger.warning(
            "No threshold achieved 80% precision — selecting highest precision available."
            )

            for t in np.arange(0.01, 0.99, 0.01):

                preds = (y_prob >= t).astype(int)

                if preds.sum() == 0:
                    continue

                precision = precision_score(y_test, preds)
                recall = recall_score(y_test, preds)

                if precision > best_precision:
                    best_threshold = t
                    best_precision = precision
                    best_recall = recall


        # -----------------------------------
        # 3. FINAL PREDICTIONS
        # -----------------------------------
        y_pred = (y_prob >= best_threshold).astype(int)

        logger.info(f"Selected Threshold = {best_threshold:.3f}")
        logger.info(f"Precision = {best_precision:.3f}")
        logger.info(f"Recall = {best_recall:.3f}")
        #=========================================================
        # STEP 13: MODEL EVALUATION ARTIFACTS (WBS 2.7)
        #=========================================================

        # Confusion Matrix

        final_preds = (y_prob >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, final_preds)


        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")

        logger.info("Saving evaluation artifacts")

        os.makedirs("evaluation_outputs", exist_ok=True)
        plt.savefig("evaluation_outputs/confusion_matrix.png")
        plt.close()

        logger.info(f"Baseline ROC-AUC: {auc:.4f}")

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
        logger.info("ROC curve saved")


        #=========================================================
        #STEP 15:      FINAL DECISION POLICY LOGGING TO MLFLOW
        #=========================================================
        from sklearn.metrics import average_precision_score

        pr_auc = average_precision_score(y_test, probs)

        mlflow.log_metric("PR_AUC", pr_auc)
        auc = roc_auc_score(y_test, probs)

        final_preds = (probs >= best_threshold).astype(int)

        from sklearn.metrics import precision_score, recall_score

        best_precision = precision_score(y_test, final_preds)
        best_recall = recall_score(y_test, final_preds)

        # ---- LOG MODEL CONFIGURATION ----
        mlflow.log_param("model", "Calibrated_XGBoost")
        mlflow.log_param("decision_threshold", float(best_threshold))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("n_features", X_train.shape[1])


        # ---- LOG CLINICAL PERFORMANCE ----
        mlflow.log_metric("final_precision", float(best_precision))
        mlflow.log_metric("final_recall", float(best_recall))
        mlflow.log_metric("ROC_AUC", auc)
        mlflow.log_metric("PR_AUC", pr_auc)


        mlflow.log_artifact("evaluation_outputs/confusion_matrix.png")
        mlflow.log_artifact("evaluation_outputs/roc_curve.png")
        np.save("evaluation_outputs/decision_threshold.npy", best_threshold)

        mlflow.log_artifact("evaluation_outputs/decision_threshold.npy")

        mlflow.sklearn.log_model(calibrated_xgb, name="xgboost_model")
        logger.info(Pipeline.named_steps)
    

        #=========================================================
        #STEP 16:     PM VALIDATION CHECKPOINT OUTPUT (WBS 2.10)
        #=========================================================

        evaluation_summary = pd.DataFrame({
        "Metric":["ROC_AUC","Precision","Recall","Threshold"],
        "Value":[auc,best_precision, best_recall, best_threshold]
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
        # STEP 18: FINAL DEPLOYABLE OUTPUT (Week 3 Dependency)
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

        



        #=========================================================
        # STEP 19: SHAP EXPLAINABILITY (CLINICAL TRUST)
        #=========================================================
     
        #----------------------------------------------------------------------------
        #SHAP EXPLAINABILITY WITH REAL COLUMNS' NAMES INSTEAD OF GENERIC NAMES
        #-----------------------------------------------------------------------------
       
        logger.info("Preparing feature names for SHAP explainability")

        # ---------------------------------------------------
        # 1. Unwrap calibrated model safely
        # ---------------------------------------------------
        if hasattr(calibrated_xgb, "estimator"):
            pipeline = calibrated_xgb.estimator
        elif hasattr(calibrated_xgb, "base_estimator"):
            pipeline = calibrated_xgb.base_estimator
        else:
            pipeline = calibrated_xgb

        logger.info(f"Pipeline steps detected: {pipeline.named_steps}")

        # ---------------------------------------------------
        # 2. Automatically find preprocessor + model
        # ---------------------------------------------------
        from sklearn.compose import ColumnTransformer
        from xgboost import XGBClassifier

        preprocessor = None
        model = None

        for name, step in pipeline.named_steps.items():

            if isinstance(step, ColumnTransformer):
                preprocessor = step

            if isinstance(step, XGBClassifier):
                 model = step

            # Safety check
        if preprocessor is None:
            raise ValueError("No ColumnTransformer found in pipeline")

        if model is None:
            raise ValueError("No XGBClassifier found in pipeline")

        # ---------------------------------------------------
        # 3. Extract feature names
        # ---------------------------------------------------
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [f.split("__")[-1] for f in feature_names]

        # ---------------------------------------------------
        # 4. Transform test data
        # --------------------------------------------------- 

        X_test_transformed = preprocessor.transform(X_test)

        X_test_shap = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index
        )
       
        # Explain the XGBoost model, NOT the calibrator
        logger.info("Generating SHAP explanations...")
        import shap
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test_shap)

        # Plot With Real Feature Names
        import matplotlib.pyplot as plt
        logger.info("Generating SHAP bar importance plot...")
        plt.figure()
        shap.summary_plot(
        shap_values,
        X_test_shap,
        plot_type="bar",
        show=False
        )
        plt.tight_layout()
        output_path = "evaluation_outputs/shap_feature_importance_bar.png"

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        # Log to MLflow
        mlflow.log_artifact(output_path)

        logger.info("SHAP bar feature importance plot saved")
        logger.info("PHIP Readmission Model Pipeline Completed Successfully")
        