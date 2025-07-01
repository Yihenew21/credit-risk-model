import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging
import os
from datetime import datetime
from pytz import timezone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(processed_data_dir='data/processed'):
    """Loads the latest processed dataset from the specified directory."""
    logger.info(f"Loading latest processed data from {processed_data_dir}...")
    if not os.path.exists(processed_data_dir):
        raise FileNotFoundError(f"Processed data directory not found at {processed_data_dir}")
    files = os.listdir(processed_data_dir)
    if not files:
        raise FileNotFoundError(f"No processed data files found in {processed_data_dir}")
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(processed_data_dir, x)))
    file_path = os.path.join(processed_data_dir, latest_file)
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data from {latest_file} with shape: {df.shape}")
    return df

def split_data(df, target_col='is_high_risk', test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    logger.info("Splitting data into training and testing sets...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_and_tune_models(X_train, X_test, y_train, y_test, is_test=False):
    """Trains and tunes Logistic Regression, Decision Tree, and Random Forest models, tracks with MLflow."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Credit_Risk_Modeling")

    models = {
        "LogisticRegression": LogisticRegression,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier
    }

    best_model = None
    best_score = 0

    for name, model_class in models.items():
        logger.info(f"Training and tuning {name}...")
        with mlflow.start_run(run_name=f"{name}_Run"):

            # Define hyperparameter grids
            if name == "LogisticRegression":
                param_grid = {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'max_iter': [100, 200, 300],
                    'solver': ['lbfgs']  # Explicitly set solver to avoid deprecation warning
                }
                model = model_class(random_state=42, class_weight='balanced')
            elif name == "DecisionTree":
                param_grid = {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
                model = model_class(random_state=42, class_weight='balanced')
            else:  # RandomForest
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
                model = model_class(random_state=42, class_weight='balanced')

            # Perform GridSearchCV with increased cross-validation
            grid_search = GridSearchCV(model, param_grid, cv=10, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Best model from grid search
            best_model_instance = grid_search.best_estimator_
            y_pred = best_model_instance.predict(X_test)
            y_pred_proba = best_model_instance.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            logger.info(f"{name} Metrics: {metrics}")
            logger.info(f"Best parameters: {grid_search.best_params_}")

            # Log to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model_instance, f"{name}_model")

            # Track best model
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model = best_model_instance

    # Register the best model in MLflow Model Registry only if not in test mode
    if best_model is not None and not is_test:
        with mlflow.start_run(run_name="Best_Model_Registration"):
            mlflow.sklearn.log_model(best_model, "best_model")
            model_info = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/best_model",
                "CreditRiskModel"
            )
            logger.info(f"Registered model with version {model_info.version}")

    return best_model

def main():
    """Main function to orchestrate model training and tracking."""
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)
    best_model = train_and_tune_models(X_train, X_test, y_train, y_test)
    logger.info("Model training and tracking completed successfully!")

if __name__ == "__main__":
    main()