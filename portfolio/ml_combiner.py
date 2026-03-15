import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from typing import Dict

class TreeFactorCombiner:
    """
    A machine learning factor combiner that uses LightGBM to predict forward returns
    based on multiple orthogonal input factors.
    """

    def __init__(self, train_end_date: str, forward_period: int = 5):
        """
        Initialize the TreeFactorCombiner.

        Args:
            train_end_date (str): The cutoff date for the training set (e.g., '2025-01-01').
                                  Data after this date will be used for testing/prediction.
            forward_period (int): The number of periods to look forward for the target return. Defaults to 5.
        """
        self.train_end_date = pd.to_datetime(train_end_date)
        self.forward_period = forward_period

    def prepare_data(self, factor_dict: Dict[str, pd.DataFrame], price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the master dataset by stacking factors and aligning them with forward returns.

        Args:
            factor_dict (Dict[str, pd.DataFrame]): Dictionary mapping factor names to their wide-format DataFrames.
            price_df (pd.DataFrame): Wide-format DataFrame of historical prices (Dates x Tickers).

        Returns:
            pd.DataFrame: A merged DataFrame with a MultiIndex (date, ticker), containing
                          all factor features and the target 'forward_return'.
        """
        stacked_factors = []
        
        # Stack each factor dataframe into a MultiIndex series
        for factor_name, df in factor_dict.items():
            stacked_series = df.stack()
            stacked_series.index.names = ['date', 'ticker']
            stacked_series.name = factor_name
            stacked_factors.append(stacked_series)
            
        # Concat them into a feature DataFrame X
        X = pd.concat(stacked_factors, axis=1)
        
        # Calculate the forward return from price_df
        # Formula: price_{t+forward_period} / price_t - 1
        forward_returns = price_df.shift(-self.forward_period) / price_df - 1
        
        # Stack into a Series y
        y = forward_returns.stack()
        y.index.names = ['date', 'ticker']
        y.name = 'forward_return'
        
        # Merge X and y on the (date, ticker) index
        merged_df = X.join(y, how='inner')
        
        # Drop any rows where y or all features are NaN
        merged_df = merged_df.dropna(subset=['forward_return'])
        merged_df = merged_df.dropna(subset=X.columns, how='all')
        
        return merged_df

    def train_and_predict(self, factor_dict: Dict[str, pd.DataFrame], price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Trains the LightGBM model on historical data and predicts factor scores for the out-of-sample period.

        Args:
            factor_dict (Dict[str, pd.DataFrame]): Dictionary mapping factor names to their wide-format DataFrames.
            price_df (pd.DataFrame): Wide-format DataFrame of historical prices (Dates x Tickers).

        Returns:
            pd.DataFrame: A wide-format DataFrame (Dates x Tickers) representing the final 
                          combined ML factor scores for the out-of-sample period.
        """
        # Call prepare_data to get the master dataset
        master_df = self.prepare_data(factor_dict, price_df)
        
        # Split the dataset based on self.train_end_date
        # Index level 0 is 'date'
        train_df = master_df[master_df.index.get_level_values('date') <= self.train_end_date]
        test_df = master_df[master_df.index.get_level_values('date') > self.train_end_date]
        
        if test_df.empty:
            raise ValueError(f"No test data available after train_end_date: {self.train_end_date}")
            
        if train_df.empty:
            raise ValueError(f"No training data available before train_end_date: {self.train_end_date}")

        feature_cols = list(factor_dict.keys())
        target_col = 'forward_return'
        
        # Separate X_train, y_train and X_test, y_test
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_test = test_df[feature_cols]
        # y_test is not strictly needed for prediction, but good practice
        
        # Instantiate LightGBM Regressor
        self.model = LGBMRegressor(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.05, 
            random_state=42, 
            n_jobs=-1
        )
        
        # Fit the model on (X_train, y_train)
        print(f"Training LightGBM model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Predict on X_test
        print(f"Predicting on {len(X_test)} out-of-sample periods...")
        predictions = self.model.predict(X_test)
        
        # Create a Series with the predictions, keeping the (date, ticker) index from X_test
        pred_series = pd.Series(predictions, index=X_test.index, name='ml_combined_score')
        
        # Unstack this Series back into a wide DataFrame (Dates x Tickers)
        final_scores_wide = pred_series.unstack(level='ticker')
        
        return final_scores_wide
