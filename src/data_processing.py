import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
import os
import logging
from woe.feature_process import process_woe_trans, calulate_iv
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Transformer for Temporal Features
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transaction_idx = None
        self.total_cols = None
    
    def fit(self, X, y=None):
        # Indices are set in DataProcessor, no need to check columns here
        if self.transaction_idx is None:
            logger.error("TransactionStartTime index not set")
            raise ValueError("TransactionStartTime index not set")
        logger.info(f"Using pre-set TransactionStartTime index: {self.transaction_idx}")
        self.total_cols = X.shape[1]  # Update total columns based on transformed data
        logger.info(f"Total columns after transformation: {self.total_cols}")
        return self
    
    def transform(self, X):
        if self.transaction_idx is None or self.total_cols is None:
            raise ValueError("Transformer not fitted or indices not set")
        if self.transaction_idx >= X.shape[1]:
            logger.error(f"TransactionStartTime index {self.transaction_idx} exceeds array shape {X.shape}")
            raise ValueError("TransactionStartTime index out of bounds")
        transaction_col = X[:, self.transaction_idx]
        df = pd.DataFrame({'TransactionStartTime': transaction_col})
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['trans_hour'] = df['TransactionStartTime'].dt.hour
        df['trans_day'] = df['TransactionStartTime'].dt.day
        df['trans_month'] = df['TransactionStartTime'].dt.month
        df['trans_year'] = df['TransactionStartTime'].dt.year
        df['trans_day_name'] = df['TransactionStartTime'].dt.day_name()
        # Drop the Timestamp and trans_day_name columns, keep only numerical features
        numerical_df = df.drop(columns=['TransactionStartTime', 'trans_day_name'])
        return numerical_df.values.astype(float)  # Convert to numpy array with float type

# Custom Transformer for Aggregate Features
class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.customer_idx = None
        self.amount_idx = None
        self.total_cols = None
    
    def fit(self, X, y=None):
        # Indices are set in DataProcessor, no need to check columns here
        if self.customer_idx is None or self.amount_idx is None:
            logger.error("CustomerId or Amount indices not set")
            raise ValueError("CustomerId or Amount indices not set")
        logger.info(f"Using pre-set CustomerId index: {self.customer_idx}, Amount index: {self.amount_idx}")
        self.total_cols = X.shape[1]  # Update total columns based on transformed data
        logger.info(f"Total columns after transformation: {self.total_cols}")
        return self
    
    def transform(self, X):
        if self.customer_idx is None or self.amount_idx is None or self.total_cols is None:
            raise ValueError("Transformer not fitted or indices not set")
        if self.customer_idx >= X.shape[1] or self.amount_idx >= X.shape[1]:
            logger.error(f"CustomerId index {self.customer_idx} or Amount index {self.amount_idx} exceeds array shape {X.shape}")
            raise ValueError("CustomerId or Amount indices out of bounds")
        customer_col = X[:, self.customer_idx]
        amount_col = X[:, self.amount_idx]
        df = pd.DataFrame({'CustomerId': customer_col, 'Amount': amount_col})
        if 'CustomerId' in df.columns and 'Amount' in df.columns:
            agg_features = df.groupby('CustomerId').agg({
                'Amount': ['sum', 'mean', 'count', 'std']
            }).reset_index()
            agg_features.columns = ['CustomerId', 'total_amount', 'avg_amount', 'trans_count', 'amount_std']
            df = df.merge(agg_features, on='CustomerId', how='left')
        return df.values  # Return as numpy array

class DataProcessor:
    def __init__(self, raw_data_path='data/raw/data.csv', processed_data_path='data/processed/processed_data.csv',
                 k_best_features=5, target_col='FraudResult'):
        """Initialize with paths and parameters."""
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.k_best_features = k_best_features
        self.target_col = target_col
        self.df = None
        self.pipeline = None
        self.label_encoders = {}

    def load_data(self):
        """Load the raw dataset with validation."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded data with shape: {self.df.shape}")
        return self.df

    def create_pipeline(self):
        """Define the sklearn Pipeline with advanced transformations."""
        numerical_cols = ['Amount', 'Value', 'CountryCode', 'PricingStrategy']
        categorical_cols = ['ProductCategory', 'CurrencyCode', 'ProviderId', 'ProductId', 'ChannelId']
        drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId']

        # Initialize transformers with column indices from original DataFrame
        temporal_transformer = TemporalFeatureExtractor()
        aggregate_transformer = AggregateFeatureExtractor()

        # Set indices based on original DataFrame
        if self.df is not None:
            if 'TransactionStartTime' in self.df.columns:
                temporal_transformer.transaction_idx = self.df.columns.get_loc('TransactionStartTime')
                logger.info(f"Set TransactionStartTime index to {temporal_transformer.transaction_idx}")
            else:
                logger.error("TransactionStartTime not found in original data")
                raise ValueError("TransactionStartTime not found in original data")
            if 'CustomerId' in self.df.columns and 'Amount' in self.df.columns:
                aggregate_transformer.customer_idx = self.df.columns.get_loc('CustomerId')
                aggregate_transformer.amount_idx = self.df.columns.get_loc('Amount')
                logger.info(f"Set CustomerId index to {aggregate_transformer.customer_idx}, Amount index to {aggregate_transformer.amount_idx}")
            else:
                logger.error("CustomerId or Amount not found in original data")
                raise ValueError("CustomerId or Amount not found in original data")

        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        feature_engineering = FeatureUnion(transformer_list=[
            ('temporal', temporal_transformer),
            ('aggregate', aggregate_transformer)
        ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols),
                    ('drop', 'drop', drop_cols)
                ],
                remainder='passthrough'  # Keeps TransactionStartTime, CustomerId, FraudResult, etc.
            )),
            ('feature_engineering', feature_engineering),
            ('woe_iv', FunctionTransformer(self._apply_woe_iv, kw_args={'target_col': self.target_col})),
            ('feature_selection', SelectKBest(score_func=f_classif, k=self.k_best_features))
        ])
        logger.info("Pipeline created with preprocessing, feature engineering, WoE/IV, and selection")

    def _apply_woe_iv(self, X, target_col):
        """Apply WoE and IV transformation using woe.feature_process."""
        df = pd.DataFrame(X)  # Convert back to DataFrame
        if target_col in df.columns:
            woe_features = ['Amount', 'trans_hour']
            for feature in woe_features:
                if feature in df.columns:
                    df[f'{feature}_bin'] = pd.qcut(df[feature], q=4, duplicates='drop')
                    df[f'{feature}_woe'] = process_woe_trans(df[f'{feature}_bin'], df[target_col], max_interval=5)
                    iv = calulate_iv(df[f'{feature}_bin'], df[target_col])
                    logger.info(f"IV for {feature}: {iv}")
        return df.values  # Return as numpy array

    def process(self):
        """Run the full processing pipeline and save results."""
        self.load_data()
        self.create_pipeline()
        # Separate features (X) and target (y)
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        self.df = self.pipeline.fit_transform(X, y)
        self.df = pd.DataFrame(self.df, columns=self._get_feature_names())
        self.save_processed_data()

    def _get_feature_names(self):
        """Extract feature names from the pipeline."""
        feature_names = []
        preprocessor = self.pipeline.named_steps['preprocessor']
        numerical_cols = preprocessor.transformers_[0][2]
        categorical_cols = preprocessor.transformers_[1][2]
        feature_names.extend(numerical_cols)
        feature_names.extend(preprocessor.named_transformers_['cat']
                           .named_steps['onehot'].get_feature_names_out(categorical_cols))
        # Add passthrough columns based on original dataset order
        passthrough_cols = ['TransactionStartTime', 'CustomerId', 'FraudResult']
        feature_names.extend(passthrough_cols)
        feature_names.extend(['trans_hour', 'trans_day', 'trans_month', 'trans_year',
                            'total_amount', 'avg_amount', 'trans_count', 'amount_std'])
        feature_names.extend([f'{feat}_woe' for feat in ['Amount', 'trans_hour']])
        return feature_names

    def save_processed_data(self):
        """Save the processed dataset."""
        if self.df is None:
            raise ValueError("Data not processed. Call process() first.")
        
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        self.df.to_csv(self.processed_data_path, index=False)
        logger.info(f"Processed data saved to {self.processed_data_path}")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process()