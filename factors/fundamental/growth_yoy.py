from typing import Any
import pandas as pd
from factors.base import BaseFactor

class ProfitabilityGrowth(BaseFactor):
    """
    Earnings Growth (YoY) factor.
    
    Measures the year-over-year change in profitability.
    
    Formula: Current ROA - ROA 1 year ago (approx 252 trading days)
    Higher growth is considered a positive signal.
    """

    def __init__(self, lag: int = 252, **kwargs: Any) -> None:
        """
        Initialize the ProfitabilityGrowth factor.
        
        Args:
            lag (int): Number of trading days to look back for YoY comparison. Default is 252.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(name="ProfitabilityGrowth", lag=lag, **kwargs)
        self.lag = lag

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the Profitability Growth factor.

        Args:
            data (pd.DataFrame): DataFrame containing the 'roa' column.

        Returns:
            pd.Series: The computed YoY growth values.
        """
        if 'roa' not in data.columns:
            raise ValueError("Data must contain a 'roa' column.")

        # Calculate Year-over-Year change in ROA
        growth = data['roa'] - data['roa'].shift(self.lag)
        
        return growth