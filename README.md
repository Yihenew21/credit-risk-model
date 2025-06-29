# Credit Scoring Business Understanding

### How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord emphasizes accurate risk measurement and transparency to ensure financial institutions maintain adequate capital for credit risk. An interpretable model, such as Logistic Regression with Weight of Evidence (WoE), allows regulators and stakeholders to understand how predictions are made, ensuring compliance with Basel II's supervisory review process. Well-documented models facilitate audits and validation, reducing regulatory risks and ensuring alignment with capital adequacy requirements.

### Why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

Since the dataset lacks a direct "default" label, a proxy variable is necessary to categorize customers as high or low risk. This proxy approximates default likelihood based on behavioral patterns. For this challenge, the key innovation lies in transforming behavioral data into a predictive risk signal by analyzing customer Recency, Frequency, and Monetary (RFM) patterns. This allows for the training of a model that outputs a risk probability score, a vital metric that can be used to inform loan approvals and terms.

However, making predictions based on this proxy carries potential business risks:

- **Misclassification**: The proxy may not accurately reflect true default risk, leading to incorrect loan approvals or rejections.
- **Bias**: If RFM metrics are skewed or incomplete, the model may unfairly penalize certain customer segments.
- **Regulatory Scrutiny**: Regulators may question the validity of the proxy, requiring robust justification and validation.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

- **Simple Models (Logistic Regression with WoE)**:
  - **Pros**: Highly interpretable, easier to explain to regulators, aligns with Basel II's transparency requirements, and is computationally efficient.
  - **Cons**: May have lower predictive power, potentially missing complex patterns in the data.
- **Complex Models (Gradient Boosting)**:
  - **Pros**: Higher predictive accuracy, captures non-linear relationships and complex patterns.
  - **Cons**: Less interpretable, challenging to justify to regulators, higher computational cost, and risk of overfitting.
    In a regulated financial context, interpretability often outweighs marginal performance gains, making a simple, well-understood model like Logistic Regression with WoE a preferred choice unless a complex model can be rigorously validated and explained to regulators.

---

# 📊 Credit Risk Model Project

## 🔎 Project Overview

The **Credit Risk Model Project** aims to build an interpretable and regulatory-compliant machine learning pipeline for predicting transaction-based fraud risk. Aligned with the **Basel II Accord**, the project emphasizes **transparency**, **explainability**, and **auditability** in model development.

The project is developed with a modular pipeline architecture, ensuring reproducibility, traceability, and iterative development. It includes robust data preprocessing, exploratory data analysis, feature engineering, model training, and an API interface for deployment.

---

## ⚙️ Project Structure

```bash
Credit_Risk_Model/
├── data/
│   ├── raw/                     # Raw dataset (e.g., from Kaggle)
│   ├── processed/               # Processed data (after feature engineering)
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering and preprocessing scripts
│   ├── model_training.py        # Model training and evaluation scripts
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── pydantic_models.py   # Pydantic models for API validation
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py  # Unit tests for data processing
├── notebooks/
│   ├── eda.ipynb                # Exploratory Data Analysis notebook
├── Dockerfile                   # Dockerfile for containerizing the API
├── docker-compose.yml           # Docker Compose for running the service
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               # GitHub Actions CI/CD workflow
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
├── setup.py                     # Optional: For packaging the project
```

---

## 📊 Technical Progress

### ✅ Task 1: Business Understanding & EDA

- Loaded dataset with 95,662 rows and 16 columns.
- Initial hypothesis developed around `TransactionStartTime`, `Amount`, and `FraudResult`.
- Explored:
  - Distribution of target and features.
  - Missing values.
  - Behavioral aggregation patterns.
- Insights from EDA shaped our proxy fraud signal design.

### 🛠 Task 3: Feature Engineering Pipeline

- Preprocessing:
  - `KNNImputer`, `StandardScaler` for numerical data.
  - One-hot encoding for categorical fields.
- Custom features:
  - Temporal breakdown (`transhour`, `transday`)
  - Aggregate customer behaviors (`transcount`, `totalamount`)
- Feature Selection:
  - Used **Weight of Evidence (WoE)** and **Information Value (IV)** for regulatory-compliant variable filtering.

The pipeline is modular using `scikit-learn`’s `Pipeline` and `ColumnTransformer` with `remainder='passthrough'`.

---

## 📈 Modeling Plan

Upcoming phases include:

- Model training (starting with Logistic Regression and Gradient Boosting).
- Evaluation using AUC-ROC, confusion matrix, and cross-validation.
- Interpretation using SHAP, LIME, and WoE-based dashboards.
- Deployment via FastAPI.

---

## 📦 Setup & Usage

### 🔧 Installation

```bash
git clone https://github.com/yourusername/Credit_Risk_Model.git
cd Credit_Risk_Model
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 🚀 Running the API

```bash
uvicorn src.api.main:app --reload
```

---

## 🔍 Future Improvements

- Integrate visual dashboards for EDA and SHAP insights.
- Automate model interpretability reports.
- Extend pipeline for real-time fraud detection.

---

## 📑 Acknowledgements

- KAIM Solar Challenge - W5 Credit Risk Modeling Track
- scikit-learn, pandas, WoE encoder, SHAP, LIME, FastAPI

---

## 📬 Contact

Project lead: **Yihenew Animut**  
For questions or collaboration: `yihenew@example.com` _(replace with real contact)_
