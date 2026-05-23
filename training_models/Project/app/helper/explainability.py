import shap
import matplotlib.pyplot as plt


# ============================================================
# GENERATE SHAP SUMMARY
# ============================================================

def generate_shap_summary(
    calibrated_model,
    sample_data
):

    # ====================================================
    # REMOVE NON-FEATURE COLUMNS
    # ====================================================

    shap_sample = sample_data.copy()

    removable_columns = [

        "readmission_risk_score",

        "predicted_readmission",

        "risk_category"

    ]

    for col in removable_columns:

        if col in shap_sample.columns:

            shap_sample.drop(
                columns=[col],
                inplace=True
            )

    # Remove target
    if "readmitted_within_30day" in shap_sample.columns:

        shap_sample.drop(
            columns=["readmitted_within_30day"],
            inplace=True
        )

    # ====================================================
    # EXTRACT FITTED PIPELINE
    # ====================================================

    fitted_pipeline = (
        calibrated_model
        .calibrated_classifiers_[0]
        .estimator
    )

    # ====================================================
    # EXTRACT COMPONENTS
    # ====================================================

    preprocessor = fitted_pipeline.named_steps["prep"]

    xgb_model = fitted_pipeline.named_steps["model"]

    # ====================================================
    # TRANSFORM DATA
    # ====================================================

    transformed_data = preprocessor.transform(
        shap_sample
    )

    # ====================================================
    # FEATURE NAMES
    # ====================================================

    try:

        feature_names = (
            preprocessor.get_feature_names_out()
        )

    except:

        feature_names = [

            f"feature_{i}"

            for i in range(
                transformed_data.shape[1]
            )
        ]

    # ====================================================
    # SHAP EXPLAINER
    # ====================================================

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer.shap_values(
        transformed_data
    )

    # Binary classification handling
    if isinstance(shap_values, list):

        shap_values = shap_values[1]

    # ====================================================
    # SHAP BAR CHART
    # ====================================================

    plt.figure(figsize=(12,8))

    shap.summary_plot(
        shap_values,
        transformed_data,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    fig = plt.gcf()

    return fig