import mlflow.pyfunc
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import logging

# Load environment variables
load_dotenv()
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
model_name = "CreditRiskModel"
model_version = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model from MLflow registry
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
app = FastAPI()

# Include Pydantic models from separate file
from .pydantic_models import CustomerData

@app.get("/")
async def read_root():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(customer_data: CustomerData):
    """Endpoint to predict credit risk probability for a new customer."""
    try:
        # Convert Pydantic model to dictionary and extract all 57 features
        data_dict = customer_data.dict()
        features = np.array([[
            data_dict.get('Value', 0.0),
            data_dict.get('CountryCode', 0.0),
            data_dict.get('PricingStrategy', 0.0),
            data_dict.get('CurrencyCode_UGX', 0.0),
            data_dict.get('ProviderId_ProviderId_1', 0.0),
            data_dict.get('ProviderId_ProviderId_2', 0.0),
            data_dict.get('ProviderId_ProviderId_3', 0.0),
            data_dict.get('ProviderId_ProviderId_4', 0.0),
            data_dict.get('ProviderId_ProviderId_5', 0.0),
            data_dict.get('ProviderId_ProviderId_6', 0.0),
            data_dict.get('ProductId_ProductId_1', 0.0),
            data_dict.get('ProductId_ProductId_10', 0.0),
            data_dict.get('ProductId_ProductId_11', 0.0),
            data_dict.get('ProductId_ProductId_12', 0.0),
            data_dict.get('ProductId_ProductId_13', 0.0),
            data_dict.get('ProductId_ProductId_14', 0.0),
            data_dict.get('ProductId_ProductId_15', 0.0),
            data_dict.get('ProductId_ProductId_16', 0.0),
            data_dict.get('ProductId_ProductId_19', 0.0),
            data_dict.get('ProductId_ProductId_2', 0.0),
            data_dict.get('ProductId_ProductId_20', 0.0),
            data_dict.get('ProductId_ProductId_21', 0.0),
            data_dict.get('ProductId_ProductId_22', 0.0),
            data_dict.get('ProductId_ProductId_23', 0.0),
            data_dict.get('ProductId_ProductId_24', 0.0),
            data_dict.get('ProductId_ProductId_27', 0.0),
            data_dict.get('ProductId_ProductId_3', 0.0),
            data_dict.get('ProductId_ProductId_4', 0.0),
            data_dict.get('ProductId_ProductId_5', 0.0),
            data_dict.get('ProductId_ProductId_6', 0.0),
            data_dict.get('ProductId_ProductId_7', 0.0),
            data_dict.get('ProductId_ProductId_8', 0.0),
            data_dict.get('ProductId_ProductId_9', 0.0),
            data_dict.get('ProductCategory_airtime', 0.0),
            data_dict.get('ProductCategory_data_bundles', 0.0),
            data_dict.get('ProductCategory_financial_services', 0.0),
            data_dict.get('ProductCategory_movies', 0.0),
            data_dict.get('ProductCategory_other', 0.0),
            data_dict.get('ProductCategory_ticket', 0.0),
            data_dict.get('ProductCategory_transport', 0.0),
            data_dict.get('ProductCategory_tv', 0.0),
            data_dict.get('ProductCategory_utility_bill', 0.0),
            data_dict.get('ChannelId_ChannelId_1', 0.0),
            data_dict.get('ChannelId_ChannelId_2', 0.0),
            data_dict.get('ChannelId_ChannelId_3', 0.0),
            data_dict.get('ChannelId_ChannelId_5', 0.0),
            data_dict.get('trans_day', 0.0),
            data_dict.get('trans_month', 0.0),
            data_dict.get('trans_year', 0.0),
            data_dict.get('total_amount', 0.0),
            data_dict.get('avg_amount', 0.0),
            data_dict.get('trans_count', 0.0),
            data_dict.get('amount_std', 0.0),
            data_dict.get('min_amount', 0.0),
            data_dict.get('max_amount', 0.0),
            data_dict.get('Amount_woe', 0.0),
            data_dict.get('trans_hour_woe', 0.0)
        ]])
        
        # Use predict since predict_proba is not available
        prediction = model.predict(features)
        logger.info(f"Prediction for customer data: {prediction}")
        return {"risk_probability": float(prediction[0])}  # Convert to float, assuming single prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")