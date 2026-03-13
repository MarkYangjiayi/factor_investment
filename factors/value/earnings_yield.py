import numpy as np
import pandas as pd

from factors.base import BaseFactor


class EarningsYield(BaseFactor):
    """
    Classic Value Factor: Earnings Yield (EP).
    
    Calculated as the inverse of the Price-to-Earnings (PE) ratio: 1 / PE.
    Includes robust fundamental data cleaning:
    - Forward-fills quarterly reported PE ratios for up to 90 days.
    - Masks negative or zero PE ratios (as they are generally meaningless in this context).
    - Winsorizes extreme outliers at the 1st and 99th percentiles.
    """

    def __init__(self, name: str = "earnings_yield", **kwargs):
        """
        Initialize the Earnings Yield factor.
        
        Args:
            name (str): Factor name. Default is "earnings_yield".
            **kwargs: Additional parameters.
        """
        super().__init__(name=name, **kwargs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the Earnings Yield factor using strictly vectorized operations.
        
        Args:
            data (pd.DataFrame): Standardized input data. Must contain a 'pe_ratio' column.
            
        Returns:
            pd.Series: The cleaned and computed Earnings Yield factor values.
        """
        if 'pe_ratio' not in data.columns:
            raise KeyError("The input DataFrame must contain a 'pe_ratio' column.")
            
        # Extract the PE ratio column
        pe_ratio = data['pe_ratio']
        
        # 1. Handle missing data: Forward-fill quarterly data up to 90 days limit
        pe_ratio = pe_ratio.ffill(limit=90)
        
        # 2. Handle invalid values: Replace zero or negative PE ratios with NaN
        pe_ratio = pe_ratio.where(pe_ratio > 0, np.nan)
        
        # 3. Calculate Earnings Yield (EP)
        ep = 1.0 / pe_ratio
        
        # 4. Outlier Handling (Winsorization)
        # Calculate the 1st and 99th percentiles
        p01 = ep.quantile(0.01)
        p99 = ep.quantile(0.99)
        
        # Clip the series to these percentiles
        ep = ep.clip(lower=p01, upper=p99)
        
        # Name the output series
        ep.name = self.name
        
        return ep
