from typing import Any

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class HistoricalVolatility(BaseFactor):
    """
    Historical Volatility factor.
    
    This factor captures the Low Volatility Anomaly. It calculates the annualized
    rolling standard deviation of daily returns and inverts it (multiplies by -1),
    so that assets with lower volatility receive higher factor scores.
    """

    def __init__(self, lookback_period: int = 20, **kwargs: Any) -> None:
        """
        Initialize the HistoricalVolatility factor.

        Args:
            lookback_period (int): The number of periods to look back for volatility calculation.
                                   Defaults to 20.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(name="HistoricalVolatility", lookback_period=lookback_period, **kwargs)
        self.lookback_period = lookback_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the historical volatility factor.

        Formula: -1 * (std(daily_returns) * sqrt(252))

        Args:
            data (pd.DataFrame): DataFrame containing 'close' prices.

        Returns:
            pd.Series: The computed volatility factor values (inverted).
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain a 'close' column.")

        # Calculate daily percentage returns
        returns = data['close'].pct_change()

        # Calculate rolling standard deviation of daily returns
        rolling_std = returns.rolling(window=self.lookback_period).std()

        # Annualize the volatility (assuming 252 trading days)
        annualized_vol = rolling_std * np.sqrt(252)

        # Invert the volatility to favor low volatility (lower vol = higher score)
        volatility_score = annualized_vol * -1

        return volatility_score
