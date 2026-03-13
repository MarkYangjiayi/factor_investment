from typing import Any

import pandas as pd

from factors.base import BaseFactor


class ShortTermReversal(BaseFactor):
    """
    Short-Term Reversal factor.
    
    This factor assumes mean reversion in the short term. It calculates the percentage
    return over a specified lookback period and inverts it (multiplies by -1), 
    so that assets with lower past returns receive higher factor scores.
    """

    def __init__(self, lookback_period: int = 5, **kwargs: Any) -> None:
        """
        Initialize the ShortTermReversal factor.

        Args:
            lookback_period (int): The number of periods to look back for return calculation.
                                   Defaults to 5.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(name="ShortTermReversal", lookback_period=lookback_period, **kwargs)
        self.lookback_period = lookback_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the short-term reversal factor.

        Formula: -1 * (Close_t / Close_{t-lookback} - 1)

        Args:
            data (pd.DataFrame): DataFrame containing 'close' prices.

        Returns:
            pd.Series: The computed reversal factor values.
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain a 'close' column.")

        # Calculate percentage return over the lookback period
        # pct_change(n) calculates (price_t / price_{t-n}) - 1
        returns = data['close'].pct_change(periods=self.lookback_period)
        
        # Invert the returns to capture reversal (buy losers, sell winners)
        reversal_score = returns * -1
        
        return reversal_score
