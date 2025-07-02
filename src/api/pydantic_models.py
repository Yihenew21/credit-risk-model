from pydantic import BaseModel
from typing import Optional


class CustomerData(BaseModel):
    """
    Pydantic model for validating incoming customer data with selected
    features.
    """

    Amount_woe: Optional[float] = 0.0
    trans_hour_woe: Optional[float] = 0.0
    total_amount: Optional[float] = 0.0
    avg_amount: Optional[float] = 0.0
    trans_count: Optional[float] = 0.0
    amount_std: Optional[float] = 0.0
    trans_day: Optional[float] = 0.0
    trans_month: Optional[float] = 0.0
    trans_year: Optional[float] = 0.0
    CurrencyCode_UGX: Optional[float] = 0.0
    ProviderId_ProviderId_1: Optional[float] = 0.0
    ProductId_ProductId_1: Optional[float] = 0.0
    ProductCategory_airtime: Optional[float] = 0.0
    ChannelId_ChannelId_1: Optional[float] = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "Amount_woe": -0.34,
                "trans_hour_woe": 0.12,
                "total_amount": 500.0,
                "avg_amount": 50.0,
                "trans_count": 10.0,
                "amount_std": 20.0,
                "trans_day": 15.0,
                "trans_month": 6.0,
                "trans_year": 2025.0,
                "CurrencyCode_UGX": 1.0,
                "ProviderId_ProviderId_1": 1.0,
                "ProductId_ProductId_1": 1.0,
                "ProductCategory_airtime": 1.0,
                "ChannelId_ChannelId_1": 1.0,
            }
        }


class PredictionResponse(BaseModel):
    """Pydantic model for the prediction response."""

    risk_probability: float
