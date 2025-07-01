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
from sklearn.cluster import KMeans
from pytz import timezone

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom Transformer for WoE Calculation ---
class WoETransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to calculate and apply Weight of Evidence (WoE) transformation
    for a specified feature, useful for encoding categorical or continuous variables
    in credit risk modeling.

    Parameters:
    -----------
    feature_name : str
        The name of the feature column to transform.
    n_bins : int, optional (default=5)
        The number of bins to use for discretizing the feature during WoE calculation.

    Attributes:
    -----------
    woe_map : dict or None
        Mapping of bins to their corresponding WoE values.
    iv_value : float or None
        The Information Value (IV) calculated for the feature.
    bins : array-like or None
        The bin edges used for discretizing the feature.
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

        Parameters:
        -----------
        X : pandas.DataFrame
            Input DataFrame containing the feature to transform.
        y : pandas.Series
            Target variable for WoE calculation.

        Returns:
        --------
        self : WoETransformer
            The fitted transformer instance.
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

        Parameters:
        -----------
        X : pandas.DataFrame
            Input DataFrame containing the feature to transform.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with the transformed WoE feature.
        """
        X_copy = X.copy()
        
        if not self.woe_map: # If WoE calculation failed in fit
            logger.warning(f"WoE mapping not available for '{self.feature_name}', assigning 0.")
            return pd.DataFrame({f"{self.feature_name}_woe": np.zeros(len(X))}, index=X.index)

        # Apply the same binning logic as in fit using the stored bin edges
        ranked_feature = X_copy[self.feature_name].rank(method='first')
        binned_data = pd.cut(ranked_feature, bins=self.bins, include_lowest=True, labels=list(self.woe_map.keys()))
        
        # Map the WoE values and fill NaNs with a default value (e.g., 0)
        woe_transformed = binned_data.map(self.woe_map).astype(float).fillna(0)
        
        return pd.DataFrame({f"{self.feature_name}_woe": woe_transformed}, index=X.index)


# --- Custom Transformer for Temporal Features ---
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to extract temporal features (hour, day, month, year)
    from a datetime column for use in the data processing pipeline.

    Parameters:
    -----------
    datetime_col : str, optional (default='TransactionStartTime')
        The name of the datetime column to extract features from.
    """
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        """
        Fits the transformer by validating the presence of the datetime column.

        Parameters:
        -----------
        X : pandas.DataFrame
            Input DataFrame containing the datetime column.
        y : pandas.Series, optional
            Target variable (not used but included for compatibility).

        Returns:
        --------
        self : TemporalFeatureExtractor
            The fitted transformer instance.
        """
        if self.datetime_col not in X.columns:
            raise ValueError(f"Column '{self.datetime_col}' not found in the input DataFrame.")
        return self

    def transform(self, X):
        """
        Extracts temporal features from the datetime column.

        Parameters:
        -----------
        X : pandas.DataFrame
            Input DataFrame containing the datetime column.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with extracted temporal features (trans_hour, trans_day, trans_month, trans_year).
        """
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
    A custom transformer to create aggregate features (e.g., sum, mean, count)
    based on CustomerId and Amount for enhanced data analysis.

    Parameters:
    -----------
    customer_col : str, optional (default='CustomerId')
        The column name identifying unique customers.
    amount_col : str, optional (default='Amount')
        The column name for transaction amounts to aggregate.
    """
    def __init__(self, customer_col='CustomerId', amount_col='Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.agg_features_df = None
    
    def fit(self, X, y=None):
        """
        Fits the transformer by calculating aggregate features for each customer.

        Parameters:
        -----------
        X : pandas.DataFrame
            Input DataFrame containing customer and amount columns.
        y : pandas.Series, optional
            Target variable (not used but included for compatibility).

        Returns:
        --------
        self : AggregateFeatureExtractor
            The fitted transformer instance.
        """
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
        """
        Merges pre-calculated aggregate features into the input DataFrame.

        Parameters:
        -----------
        X : pandas.DataFrame
            Input DataFrame containing the customer column.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with merged aggregate features.
        """
        logger.info("Merging pre-calculated aggregate features...")
        X_copy = X.copy()
        merged_df = X_copy[[self.customer_col]].merge(self.agg_features_df, on=self.customer_col, how='left')
        return merged_df.drop(columns=[self.customer_col])

# --- Main Data Processing Class ---
class DataProcessor:
    """
    Manages the end-to-end data processing pipeline for creating a proxy target
    variable (is_high_risk) based on RFM analysis and clustering for credit risk modeling.

    Parameters:
    -----------
    raw_data_path : str, optional (default='data/raw/data.csv')
        Path to the raw input CSV file.
    processed_data_path : str, optional (default='data/processed/processed_data_task4_{timestamp}.csv')
        Path template for saving the processed data, with {timestamp} replaced by the current date and time.
    """
    def __init__(self, raw_data_path='data/raw/data.csv', processed_data_path='data/processed/processed_data_task4_{timestamp}.csv'):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path.format(timestamp=datetime.now(timezone('Africa/Nairobi')).strftime('%Y%m%d_%H%M'))
        self.df = None
        self.target_col = 'FraudResult'
        self.id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId']
        self.temporal_col = 'TransactionStartTime'
        self.customer_col = 'CustomerId'
        
        self.numerical_cols = ['Amount', 'Value', 'CountryCode', 'PricingStrategy']
        self.categorical_cols = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
        self.woe_features = ['Amount', 'trans_hour']

    def load_data(self):
        """Loads the raw dataset from the specified CSV file."""
        logger.info(f"Loading data from {self.raw_data_path}...")
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        logger.info("Data loaded successfully.")
        logger.info(f"Original data shape: {self.df.shape}")
        logger.info(f"Columns: {self.df.columns.tolist()}")

    def build_preprocessor(self):
        """
        Builds a scikit-learn preprocessing pipeline for numerical and categorical features.

        Returns:
        --------
        ColumnTransformer
            The configured preprocessing pipeline.
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

    def calculate_rfm(self):
        """
        Calculates Recency, Frequency, and Monetary (RFM) metrics for each CustomerId.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing RFM metrics (Recency, Frequency, Monetary) for each customer.
        """
        logger.info("Calculating RFM metrics...")
        # Set snapshot date (current date + 1 day, timezone-aware)
        eat = timezone('Africa/Nairobi')
        snapshot_date = pd.to_datetime('2025-07-01 05:13:00').replace(tzinfo=eat) + pd.Timedelta(days=1)
        
        # Convert TransactionStartTime to timezone-aware datetime
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'], utc=True).dt.tz_convert('Africa/Nairobi')
        
        # Calculate RFM
        rfm = self.df.groupby(self.customer_col).agg({
            self.temporal_col: lambda x: (snapshot_date - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            'Amount': 'sum'  # Monetary
        }).rename(columns={
            self.temporal_col: 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        }).reset_index()
        
        logger.info(f"RFM calculated for {len(rfm)} customers.")
        return rfm

    def cluster_customers(self, rfm):
        """
        Clusters customers into 3 groups using K-Means based on RFM profiles.

        Parameters:
        -----------
        rfm : pandas.DataFrame
            DataFrame containing Recency, Frequency, and Monetary metrics.

        Returns:
        --------
        tuple
            A tuple (rfm, kmeans) where rfm is the updated DataFrame with cluster labels
            and kmeans is the fitted KMeans model.
        """
        logger.info("Clustering customers based on RFM...")
        # Scale RFM features
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        logger.info(f"Customers clustered into {kmeans.n_clusters} groups.")
        return rfm, kmeans

    def assign_high_risk_label(self, rfm):
        """
        Analyzes clusters and assigns the high-risk label based on the least engaged group.

        Parameters:
        -----------
        rfm : pandas.DataFrame
            DataFrame containing RFM metrics and cluster labels.

        Returns:
        --------
        pandas.DataFrame
            Updated DataFrame with the is_high_risk binary column.
        """
        logger.info("Analyzing clusters to assign high-risk labels...")
        # Analyze cluster characteristics (e.g., low Frequency and low Monetary indicate disengagement)
        cluster_summary = rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).reset_index()
        
        # Identify the high-risk cluster (highest Recency, lowest Frequency, lowest Monetary)
        high_risk_cluster = cluster_summary.loc[
            (cluster_summary['Recency'].idxmax()) &
            (cluster_summary['Frequency'].idxmin()) &
            (cluster_summary['Monetary'].idxmin())
        ]['Cluster']
        
        # Assign binary label: 1 for high-risk cluster, 0 for others
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
        logger.info(f"High-risk cluster identified as {high_risk_cluster}. Assigned 'is_high_risk' labels.")
        return rfm

    def integrate_target_variable(self):
        """
        Merges the is_high_risk column back into the main processed dataset.
        """
        logger.info("Integrating high-risk target variable into the main dataset...")
        rfm = self.calculate_rfm()
        rfm, _ = self.cluster_customers(rfm)
        rfm = self.assign_high_risk_label(rfm)
        
        # Merge with the main DataFrame
        self.df = self.df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        logger.info(f"Target variable 'is_high_risk' integrated. New shape: {self.df.shape}")

    def process_data(self):
        """
        Executes the full data processing pipeline including proxy target variable engineering.

        This method orchestrates the preprocessing of numerical and categorical features,
        extracts temporal and aggregate features, applies WoE transformation, and integrates
        the is_high_risk proxy target variable.
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

        # --- Integrate proxy target variable ---
        self.integrate_target_variable()
        
        # --- Finalize the DataFrame and save ---
        final_df = pd.concat([combined_features_df.drop(columns=self.woe_features, errors='ignore'), woe_transformed_df, self.df['is_high_risk']], axis=1)
        self.df = final_df
        self.save_processed_data()

    def save_processed_data(self):
        """
        Saves the processed DataFrame to a CSV file with an enhanced filename
        including the task name and timestamp.

        Raises:
        -------
        ValueError
            If the data has not been processed before saving.
        """
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
        processed_data_path='data/processed/processed_data_task4_{timestamp}.csv'
    )
    try:
        processor.process_data()
        logger.info("Data processing completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")