# Predictive-Healthcare-Intelligent-Program
The Predictive Healthcare Intelligence Programme aims to transform hospital operations from reactive to proactive through data science and analytics. The programme will: Predict patient readmission risks, identify high-risk patients early, forecast emergency department demand, and enable data-driven decision-making.


# 🏥 Predictive Healthcare Intelligent System

## AI-Powered Clinical Decision Support & Operational Intelligence Platform

---

# 📌 Project Overview

The **Predictive Healthcare Intelligent System** is an end-to-end Artificial Intelligence and Healthcare Analytics platform developed to support:

* 30-Day Hospital Readmission Risk Prediction
* Emergency Department (ED) Demand Forecasting
* Explainable AI (XAI) for clinician transparency
* Interactive Healthcare Intelligence Dashboards

The system integrates:

* Machine Learning
* Forecasting
* Explainable AI
* Healthcare Operational Analytics
* Interactive Business Intelligence Dashboards

into a unified clinical decision-support ecosystem.

---

# 🎯 Project Objectives

The primary objectives of the project were to:

* Predict patient readmission risk within 30 days of discharge
* Forecast Emergency Department demand over a 7-day horizon
* Improve proactive discharge planning
* Support hospital operational readiness
* Provide explainable and transparent AI predictions
* Deliver an interactive dashboard usable by non-technical healthcare stakeholders

---

# 🏥 Business Problem

Hospital readmissions significantly increase:

* healthcare operational costs
* patient care burden
* resource utilisation pressure
* emergency department congestion

Healthcare organisations require predictive and operational intelligence systems capable of:

* identifying high-risk patients early
* forecasting operational demand
* improving discharge planning
* supporting evidence-based decision-making

This project addresses those challenges using Artificial Intelligence and interactive healthcare analytics.

---

# 🧠 Core Features

## ✅ 30-Day Readmission Risk Prediction

Predicts the probability of patient readmission using machine learning models trained on healthcare operational and clinical data.

### Risk Categories

* Low Risk
* Moderate Risk
* High Risk

---

## ✅ Emergency Department Demand Forecasting

Forecasts future ED demand over a 7-day horizon to support:

* staffing planning
* resource allocation
* operational preparedness
* patient flow management

---

## ✅ Explainable AI with SHAP

Integrated SHAP explainability to provide transparency into machine learning predictions.

The system identifies influential features such as:

* previous admissions
* comorbidities
* length of stay
* abnormal laboratory indicators

---

## ✅ Interactive Streamlit Dashboard

Interactive healthcare intelligence dashboard featuring:

* dynamic filtering
* KPI monitoring
* patient-level analytics
* hospital operational analytics
* forecasting visualisations
* SHAP explainability visualisations

---

# 🏗️ System Architecture

## Project Structure

```text
Predictive-Healthcare-Intelligent-Program/
│
├── app/
│   ├── streamlit_app.py
│   └── helper/
│       ├── prediction.py
│       ├── forecasting.py
│       ├── explainability.py
│       └── metrics.py
│
├── datasets/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── training_models/
│
├── mlruns/
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Dataset Overview

The project utilised multiple healthcare datasets containing:

* patient demographics
* admissions
* ICU stays
* laboratory records
* operational healthcare events
* discharge records
* transactional clinical data

The datasets were:

* cleaned
* aggregated
* merged
* engineered into predictive healthcare features

to create a unified machine learning dataset.

---

# ⚙️ Data Engineering Pipeline

## Data Loading

Healthcare datasets were loaded into the Python workspace using Pandas.

```python
pd.read_csv()
```

---

## Data Cleaning

The preprocessing pipeline included:

* missing value handling
* datatype correction
* duplicate removal
* invalid value correction
* feature consistency validation

---

## Transactional Data Aggregation

Transactional healthcare records were aggregated into patient-level features using:

* counts
* means
* maximum values
* operational summaries

---

## Feature Engineering

Engineered features included:

### Clinical Features

* number of comorbidities
* previous admissions
* ICU indicators
* abnormal lab counts
* length of stay

### Operational Features

* admission type
* discharge indicators
* hospital identifiers
* temporal operational features

### Forecasting Features

* lag variables
* rolling averages
* cyclical calendar encoding

---

# 🤖 Machine Learning Pipeline

## Primary Readmission Model

### XGBoost Classifier

The main prediction model used:

* XGBoost
* Scikit-learn Pipelines
* ColumnTransformer preprocessing

---

## Probability Calibration

Implemented using:

```python
CalibratedClassifierCV
```

to improve probability reliability and clinical trustworthiness.

---

## Explainable AI

SHAP (SHapley Additive Explanations) was integrated to provide:

* feature importance
* prediction transparency
* clinician interpretability

---

# 📈 Model Evaluation

The model was evaluated using:

* ROC-AUC
* Recall
* Precision
* F1-Score

### Healthcare Priority

Recall was prioritised to minimise the risk of missing truly high-risk patients.

---

# 📦 MLflow Experiment Tracking

MLflow was integrated for:

* experiment tracking
* parameter logging
* metric logging
* model artifact management

Benefits included:

* reproducibility
* model governance
* experiment comparison

---

# 🖥️ Streamlit Dashboard

The interactive dashboard includes:

---

## 🏥 Readmission Dashboard

Features:

* patient-level risk scores
* KPI monitoring
* risk distribution analytics
* hospital comparison charts
* admission-type analytics

---

## 🚑 ED Forecasting Dashboard

Displays:

* 7-day demand forecasts
* operational trend charts
* forecasting tables

---

## 🧠 SHAP Explainability Dashboard

Displays:

* SHAP feature importance
* explainable AI visualisations
* clinician interpretability insights

---

# 🔎 Dynamic Filtering

The dashboard supports real-time filtering by:

* hospital
* admission type
* risk category
* patient ID

All tables and charts update dynamically.

---

# 📋 Functional Requirement Fulfilment

| Requirement | Status                              |
| ----------- | ----------------------------------- |
| FR-22       | ✅ Patient-level risk dashboard      |
| FR-23       | ✅ ED forecasting dashboard          |
| FR-24       | ✅ KPI operational analytics         |
| FR-25       | ✅ Non-technical dashboard usability |
| FR-26       | ✅ PM review and sign-off ready      |
| FR-27       | ✅ Stakeholder demo workflow         |
| FR-28       | ✅ Stakeholder presentation ready    |

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Predictive-Healthcare-Intelligent-System.git
```

---

## Navigate to Project

```bash
cd Predictive-Healthcare-Intelligent-System
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Dashboard

Run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

Expected output:

```text
Local URL: http://localhost:8501
```

---

# 📌 Technologies Used

## Programming & Data Science

* Python
* Pandas
* NumPy

---

## Machine Learning

* Scikit-learn
* XGBoost
* SHAP

---

## Dashboard & Visualisation

* Streamlit
* Plotly
* Matplotlib
* Seaborn

---

## Experiment Tracking

* MLflow

---

# 📈 Business & Clinical Impact

The system demonstrates significant operational value by supporting:

* proactive discharge planning
* high-risk patient identification
* healthcare operational readiness
* resource optimisation
* explainable AI decision support
* data-driven healthcare intelligence

---

# ⚠️ Project Limitations

Several limitations were identified:

* precision target not fully achieved
* simplified future forecasting feature generation
* lack of real-time EHR integration
* limited socioeconomic variables
* prototype-scale deployment architecture

These limitations provide opportunities for future enhancement.

---

# 🔮 Future Enhancements

Potential future improvements include:

* cloud deployment (AWS/Azure/GCP)
* real-time EHR integration
* patient-specific SHAP explanations
* clinician intervention recommendation engine
* Docker containerisation
* API deployment
* advanced time-series forecasting
* enterprise authentication

---

# 👨‍💻 Author

### Stephen Ogodo

Focus Areas:

* Healthcare AI
* Predictive Analytics
* Explainable AI
* Operational Intelligence
* Clinical Decision Support Systems

---

# 📜 License

This project is intended for academic and research purposes.

---

# ⭐ Acknowledgements

Special acknowledgement to:

* healthcare data science methodologies
* Scikit-learn community
* XGBoost developers
* SHAP explainability framework
* Streamlit open-source ecosystem

for enabling rapid development of explainable healthcare AI systems.

