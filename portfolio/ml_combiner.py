import pandas as pd
import numpy as np
from lightgbm import LGBMRanker
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
        
        # Calculate TWO separate return columns:
        # 1) forward_return_1d: strict 1-day physical return (for RL environment)
        forward_returns_1d = price_df.shift(-1) / price_df - 1
        y_1d = forward_returns_1d.stack()
        y_1d.index.names = ['date', 'ticker']
        y_1d.name = 'forward_return_1d'

        # 2) forward_return_nd: N-day return (for LightGBM target)
        forward_returns_nd = price_df.shift(-self.forward_period) / price_df - 1
        y_nd = forward_returns_nd.stack()
        y_nd.index.names = ['date', 'ticker']
        y_nd.name = 'forward_return_nd'

        # Merge X and both return columns on the (date, ticker) index
        merged_df = X.join(y_1d, how='inner').join(y_nd, how='inner')

        # Drop any rows where either return is NaN or all features are NaN
        merged_df = merged_df.dropna(subset=['forward_return_1d', 'forward_return_nd'])
        merged_df = merged_df.dropna(subset=X.columns, how='all')
        
        return merged_df

    def train_and_predict(self, factor_dict: Dict[str, pd.DataFrame], price_df: pd.DataFrame) -> pd.DataFrame:
        master_df = self.prepare_data(factor_dict, price_df)
        
        train_df = master_df[master_df.index.get_level_values('date') <= self.train_end_date].copy()

        if train_df.empty:
            raise ValueError("Insufficient data after train/test split.")

        feature_cols = list(factor_dict.keys())
        target_col = 'forward_return_nd'
        
        # --- LambdaRank 核心改造开始 ---
        # 1. 必须严格按日期排序，这是 LightGBM 分组的前提
        train_df = train_df.sort_index(level='date')
        
        # 2. 将连续收益率转化为横截面上的 0-4 档整数标签 (Relevance Score)
        # 0 是最差，4 是最好
        y_train_labels = train_df.groupby(level='date')[target_col].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')
        ).fillna(2).astype(int) # 如果出现极少量的 NaN，填入中间档位 2
        
        # 3. 计算每天截面上的样本数 (Group Array)
        group_train = train_df.groupby(level='date').size().values
        
        X_train = train_df[feature_cols]
        
        # 实例化 Ranker
        self.model = LGBMRanker(
            n_estimators=150,
            learning_rate=0.02,
            max_depth=4,
            num_leaves=15,
            min_child_samples=300,
            random_state=42,
            n_jobs=-1,
            importance_type='gain'
        )
        
        print(f"Training LGBMRanker on {len(X_train)} samples across {len(group_train)} cross-sections...")
        self.model.fit(X_train, y_train_labels, group=group_train)
        
        print(f"Predicting ranks on full dataset ({len(master_df)} samples)...")
        predictions_all = self.model.predict(master_df[feature_cols])
        master_df['ml_combined_score'] = predictions_all

        return master_df
