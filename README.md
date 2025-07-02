# Credit Scoring Business Understanding

### How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord emphasizes accurate risk measurement and transparency to ensure financial institutions maintain adequate capital for credit risk. An interpretable model, such as Logistic Regression with Weight of Evidence (WoE), allows regulators and stakeholders to understand how predictions are made, ensuring compliance with Basel II's supervisory review process. Well-documented models facilitate audits and validation, reducing regulatory risks and ensuring alignment with capital adequacy requirements.

### Why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

Since the dataset lacks a direct "default" label, a proxy variable is necessary to categorize customers as high or low risk. This proxy approximates default likelihood based on behavioral patterns, specifically using Recency, Frequency, and Monetary (RFM) metrics derived from transaction data. The key innovation lies in transforming these behavioral patterns into a predictive risk signal, enabling the training of a model that outputs a risk probability score to inform loan approvals and terms.

However, predictions based on this proxy carry potential business risks:

- **Misclassification**: The proxy may not accurately reflect true default risk, leading to incorrect loan approvals or rejections.
- **Bias**: If RFM metrics are skewed or incomplete, the model may unfairly penalize certain customer segments.
- **Regulatory Scrutiny**: Regulators may question the proxy's validity, requiring robust justification and validation.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

- **Simple Models (Logistic Regression with WoE)**:
  - **Pros**: Highly interpretable, easier to explain to regulators, aligns with Basel II's transparency requirements, computationally efficient.
  - **Cons**: May have lower predictive power, potentially missing complex patterns.
- **Complex Models (Gradient Boosting)**:
  - **Pros**: Higher predictive accuracy, captures non-linear relationships and complex patterns.
  - **Cons**: Less interpretable, challenging to justify to regulators, higher computational cost, risk of overfitting.
    In a regulated financial context, interpretability often outweighs marginal performance gains, making Logistic Regression with WoE a preferred choice unless a complex model can be rigorously validated and explained.

---

# 📊 Credit Risk Model Project

## 🔎 Project Overview

The **Credit Risk Model Project** aims to build an interpretable and regulatory-compliant machine learning pipeline for predicting transaction-based fraud risk, aligned with the **Basel II Accord**. The project emphasizes **transparency**, **explainability**, and **auditability** through a modular pipeline architecture, ensuring reproducibility, traceability, and iterative development. It includes robust data preprocessing, exploratory data analysis (EDA), feature engineering, model training, unit testing, and a deployable FastAPI interface. The model has been successfully tested locally, with predictions working as expected.

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
│   │   ├── main.py              # FastAPI application with MLflow fallback
│   │   ├── pydantic_models.py   # Pydantic models for API validation
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py  # Unit tests for data processing with MLflow mocking
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
- Developed initial hypothesis around `TransactionStartTime`, `Amount`, and `FraudResult`.
- Explored:
  - Distribution of target and features.
  - Missing values analysis.
  - Behavioral aggregation patterns (RFM).
- Insights shaped the proxy fraud signal design based on RFM metrics.

### ✅ Task 2: Data Preprocessing & Proxy Variable Creation

- Implemented preprocessing pipeline using `scikit-learn`:
  - `KNNImputer` for handling missing values.
  - `StandardScaler` for numerical data normalization.
  - One-hot encoding for categorical variables.
- Created a proxy variable for risk classification using RFM patterns (Recency, Frequency, Monetary).

### ✅ Task 3: Feature Engineering Pipeline

- Developed a modular feature engineering pipeline with `ColumnTransformer` and `Pipeline`:
  - Temporal features: `transhour`, `transday`.
  - Aggregate features: `transcount`, `totalamount`, `avgamount`, `amountstd`.
- Applied **Weight of Evidence (WoE)** and **Information Value (IV)** for feature selection, ensuring regulatory compliance.
- Pipeline outputs processed data ready for modeling.

### ✅ Task 4: Model Training & Evaluation

- Trained initial models:
  - **Logistic Regression with WoE**: Chosen for interpretability and Basel II compliance.
  - **Gradient Boosting**: Explored for higher accuracy, with SHAP for interpretability.
- Evaluated using AUC-ROC, confusion matrix, and 5-fold cross-validation.
- Logged metrics and models using MLflow locally (disabled in CI).

### ✅ Task 5: Unit Testing

- Implemented unit tests in `tests/test_data_processing.py`:
  - Tested `load_processed_data`, `split_data`, and `train_and_tune_models`.
  - Mocked MLflow calls to avoid network dependencies in CI, ensuring robust test coverage.

### ✅ Task 6: CI/CD Pipeline & Deployment Setup

- Configured a GitHub Actions CI pipeline (`.github/workflows/ci.yml`):
  - Linting with Flake8.
  - Unit tests with pytest.
  - Docker build and container startup with `docker-compose`.
  - Logs verification (API health check temporarily skipped due to MLflow dependency).
- Containerized the FastAPI app with Docker:
  - Used `docker-compose.yml` to set `working_dir: /app` and `PYTHONPATH=/app`.
  - Added MLflow fallback in `src/api/main.py` with `USE_MLFLOW=false` for CI.
- Ensured reproducibility and traceability with modular code and documentation.

---

## 📈 Modeling Plan

- **Completed**: Initial model training and evaluation, with successful local predictions.
- **Next Steps**: Refine models with hyperparameter tuning, enhance interpretability with SHAP and WoE dashboards, and deploy to a staging environment.

---

## 📦 Setup & Usage

### 🔧 Installation

```bash
git clone https://github.com/Yihenew21/Credit_Risk_Model.git
cd Credit_Risk_Model
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 🚀 Running the API Locally

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

- Test a prediction locally:
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{"Amount_woe": -0.34, "trans_hour_woe": 0.12, "total_amount": 500.0, "avg_amount": 50.0, "trans_count": 10.0, "amount_std": 20.0, "trans_day": 15.0, "trans_month": 6.0, "trans_year": 2025.0, "CurrencyCode_UGX": 1.0, "ProviderId_ProviderId_1": 1.0, "ProductId_ProductId_1": 1.0, "ProductCategory_airtime": 1.0, "ChannelId_ChannelId_1": 1.0}'
  ```

### 🐳 Running with Docker

```bash
docker-compose up --build
```

- Access the API at `http://localhost:8000` and test with the same `curl` command above.

---

## 🔍 Future Improvements

- Integrate visual dashboards for EDA, SHAP, and WoE insights.
- Automate model interpretability reports for regulatory compliance.
- Enable real-time fraud detection with streaming data.
- Reintroduce API health checks in CI after resolving MLflow dependency or adding an MLflow server.

---

## 📑 Acknowledgements

- KAIM Solar Challenge - W5 Credit Risk Modeling Track
- scikit-learn, pandas, WoE encoder, SHAP, LIME, FastAPI, MLflow, Docker, GitHub Actions

---

## 📬 Contact

Project lead: **Yihenew Animut**  
For questions or collaboration: `birukanimut1234@gmail.com`
