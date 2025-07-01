import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom Transformer for WoE Calculation ---
class WoETransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to calculate and apply Weight of Evidence (WoE) transformation.
    """
    def __init__(self, feature_name, n_bins=5):
        self.feature_name = feature_name
        self.n_bins = n_bins
        self.woe_map = None
        self.iv_value = None
        self.bins = None
    
    def fit(self, X, y):
        """
        Fits the transformer by calculating WoE and IV for each bin of the feature.
        """
        df = pd.DataFrame({'feature': X[self.feature_name], 'target': y}).dropna()
        
        if df.empty or len(df['target'].unique()) < 2:
            logger.warning(f"Insufficient data or target classes for WoE calculation on '{self.feature_name}'.")
            self.woe_map = {}
            self.iv_value = 0
            return self
            
        # Bin the feature using qcut
        try:
            # Use rank to handle duplicates gracefully
            ranked_feature = df['feature'].rank(method='first')
            labels, self.bins = pd.qcut(ranked_feature, q=self.n_bins, duplicates='drop', retbins=True)
            df['bin'] = labels.astype(str)
        except ValueError as ve:
            logger.warning(f"Could not create {self.n_bins} bins for '{self.feature_name}' (Error: {ve}). Skipping WoE/IV.")
            self.woe_map = {}
            self.iv_value = 0
            return self

        # Check if enough bins were created
        if df['bin'].nunique() < 2:
            logger.warning(f"Only {df['bin'].nunique()} bin(s) created for '{self.feature_name}'. Skipping WoE/IV.")
            self.woe_map = {}
            self.iv_value = 0
            return self

        # Calculate WoE and IV for each bin
        bin_summary = df.groupby('bin')['target'].agg(
            total_count='count',
            bad_count='sum',
            good_count=lambda x: (x == 0).sum()
        ).reset_index()

        global_bad = df['target'].sum()
        global_good = len(df) - global_bad

        if global_good == 0 or global_bad == 0:
            logger.warning(f"Global good or bad count is zero for '{self.feature_name}'. WoE/IV cannot be calculated.")
            self.woe_map = {}
            self.iv_value = 0
            return self

        # Calculate WoE and IV, handling division by zero
        bin_summary['bad_rate'] = bin_summary['bad_count'] / global_bad
        bin_summary['good_rate'] = bin_summary['good_count'] / global_good
        
        # Avoid log(0)
        bin_summary['woe'] = np.log(bin_summary['good_rate'] / bin_summary['bad_rate']).replace([np.inf, -np.inf], 0)
        bin_summary['iv'] = (bin_summary['good_rate'] - bin_summary['bad_rate']) * bin_summary['woe']
        
        self.woe_map = bin_summary.set_index('bin')['woe'].to_dict()
        self.iv_value = bin_summary['iv'].sum()
        
        logger.info(f"IV for {self.feature_name}: {self.iv_value:.4f}")
        
        return self

    def transform(self, X):
        """
        Transforms the feature using the fitted WoE map.
        """
        X_copy = X.copy()
        
        if not self.woe_map: # If WoE calculation failed in fit
            logger.warning(f"WoE mapping not available for '{self.feature_name}', assigning 0.")
            return pd.DataFrame({f"{self.feature_name}_woe": np.zeros(len(X))}, index=X.index)

        # Apply the same binning logic as in fit using the stored bin edges
        ranked_feature = X_copy[self.feature_name].rank(method='first')
        binned_data = pd.cut(ranked_feature, bins=self.bins, include_lowest=True, labels=list(self.woe_map.keys()))
        
        # Map the WoE values and fill NaNs with a default value (e.g., 0)
        # --- FIX: Convert to float before filling NaNs to avoid Categorical error ---
        woe_transformed = binned_data.map(self.woe_map).astype(float).fillna(0)
        
        return pd.DataFrame({f"{self.feature_name}_woe": woe_transformed}, index=X.index)


# --- Custom Transformer for Temporal Features ---
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to extract temporal features from a datetime column.
    """
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        if self.datetime_col not in X.columns:
            raise ValueError(f"Column '{self.datetime_col}' not found in the input DataFrame.")
        return self

    def transform(self, X):
        logger.info(f"Extracting temporal features from '{self.datetime_col}'...")
        X_copy = X.copy()
        X_copy[self.datetime_col] = pd.to_datetime(X_copy[self.datetime_col], errors='coerce')
        X_copy['trans_hour'] = X_copy[self.datetime_col].dt.hour
        X_copy['trans_day'] = X_copy[self.datetime_col].dt.day
        X_copy['trans_month'] = X_copy[self.datetime_col].dt.month
        X_copy['trans_year'] = X_copy[self.datetime_col].dt.year
        logger.info(f"Successfully extracted: 'trans_hour', 'trans_day', 'trans_month', 'trans_year'")
        return X_copy[['trans_hour', 'trans_day', 'trans_month', 'trans_year']]

# --- Custom Transformer for Aggregate Features ---
class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to create aggregate features based on CustomerId and Amount.
    """
    def __init__(self, customer_col='CustomerId', amount_col='Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.agg_features_df = None
    
    def fit(self, X, y=None):
        logger.info("Calculating aggregate features for fitting...")
        if self.customer_col not in X.columns or self.amount_col not in X.columns:
            raise ValueError(f"Required columns '{self.customer_col}' or '{self.amount_col}' not found.")
            
        agg_features = X.groupby(self.customer_col).agg(
            total_amount=(self.amount_col, 'sum'),
            avg_amount=(self.amount_col, 'mean'),
            trans_count=(self.amount_col, 'count'),
            amount_std=(self.amount_col, 'std'),
            min_amount=(self.amount_col, 'min'),
            max_amount=(self.amount_col, 'max')
        ).reset_index()
        
        agg_features['amount_std'] = agg_features['amount_std'].fillna(0)
        self.agg_features_df = agg_features
        logger.info("Aggregate features calculated and stored for transformation.")
        return self
    
    def transform(self, X):
        logger.info("Merging pre-calculated aggregate features...")
        X_copy = X.copy()
        merged_df = X_copy[[self.customer_col]].merge(self.agg_features_df, on=self.customer_col, how='left')
        return merged_df.drop(columns=[self.customer_col])

# --- Main Data Processing Class ---
class DataProcessor:
    """
    Manages the end-to-end data processing pipeline for the credit risk model.
    """
    def __init__(self, raw_data_path='data/raw/data.csv', processed_data_path='data/processed/processed_data.csv'):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.df = None
        self.target_col = 'FraudResult'
        self.id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId']
        self.temporal_col = 'TransactionStartTime'
        self.customer_col = 'CustomerId'
        
        self.numerical_cols = ['Amount', 'Value', 'CountryCode', 'PricingStrategy']
        self.categorical_cols = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
        self.woe_features = ['Amount', 'trans_hour']

    def load_data(self):
        """Loads data from the input CSV file."""
        logger.info(f"Loading data from {self.raw_data_path}...")
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        logger.info("Data loaded successfully.")
        logger.info(f"Original data shape: {self.df.shape}")
        logger.info(f"Columns: {self.df.columns.tolist()}")

    def build_preprocessor(self):
        """
        Builds the scikit-learn preprocessing pipeline for numerical and categorical data.
        """
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols),
            ],
            remainder='passthrough'
        )
        
        return preprocessor

    def process_data(self):
        """
        Executes the full data processing pipeline.
        """
        if self.df is None:
            self.load_data()
        
        y = self.df[self.target_col]
        X = self.df.drop(columns=[self.target_col] + self.id_cols, errors='ignore')
        
        temporal_extractor = TemporalFeatureExtractor(datetime_col=self.temporal_col)
        temporal_features_df = temporal_extractor.fit_transform(X)
        
        aggregate_extractor = AggregateFeatureExtractor(customer_col=self.customer_col, amount_col='Amount')
        aggregate_features_df = aggregate_extractor.fit_transform(X)

        X = X.drop(columns=[self.customer_col, self.temporal_col], errors='ignore')

        logger.info("Applying main preprocessing pipeline...")
        preprocessor = self.build_preprocessor()
        X_processed_array = preprocessor.fit_transform(X)
        transformed_feature_names = preprocessor.get_feature_names_out()
        X_processed_df = pd.DataFrame(X_processed_array, columns=transformed_feature_names, index=X.index)
        
        rename_mapping = {name: name[5:] if name.startswith('num__') or name.startswith('cat__') else name for name in transformed_feature_names}
        X_processed_df.rename(columns=rename_mapping, inplace=True)
        
        logger.info("Combining all processed features...")
        combined_features_df = pd.concat([X_processed_df, temporal_features_df, aggregate_features_df], axis=1)
        
        # --- NEW STEP: Apply custom WoE transformation ---
        logger.info(f"Calculating WoE for features: {self.woe_features} with custom transformer...")
        woe_transformed_df = pd.DataFrame(index=combined_features_df.index)

        for feature in self.woe_features:
            woe_transformer = WoETransformer(feature_name=feature)
            woe_transformer.fit(combined_features_df, y)
            woe_feature_df = woe_transformer.transform(combined_features_df)
            woe_transformed_df = pd.concat([woe_transformed_df, woe_feature_df], axis=1)

        # --- Finalize the DataFrame and save ---
        final_df = pd.concat([combined_features_df.drop(columns=self.woe_features, errors='ignore'), woe_transformed_df, y], axis=1)
        self.df = final_df
        self.save_processed_data()

    def save_processed_data(self):
        """Saves the processed DataFrame to a CSV file."""
        if self.df is None:
            raise ValueError("Data not processed. Call process_data() first.")
            
        logger.info(f"Saving processed data to {self.processed_data_path}...")
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        self.df.to_csv(self.processed_data_path, index=False)
        logger.info(f"Processed data saved successfully. Final shape: {self.df.shape}")

# --- Main execution block ---
if __name__ == "__main__":
    processor = DataProcessor(
        raw_data_path='data/raw/data.csv',
        processed_data_path='data/processed/processed_data.csv'
    )
    try:
        processor.process_data()
        logger.info("Data processing completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")