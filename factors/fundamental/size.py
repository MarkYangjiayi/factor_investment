from typing import Any
import pandas as pd
import numpy as np
from factors.base import BaseFactor

class SmallCap(BaseFactor):
    """
    Size Factor (Inverse Market Cap).
    
    Favors smaller companies based on market capitalization, capturing the classic 
    "Size Premium" (small-cap stocks outperforming large-cap stocks over the long term).
    
    Formula: -1 * log(Market Cap)
    Market Cap is calculated as close price * shares_out.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="SmallCap", **kwargs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the Small Cap (Size) factor.

        Args:
            data (pd.DataFrame): DataFrame containing 'close' and 'shares_out' columns.

        Returns:
            pd.Series: The computed size factor values (inverted log market cap).
        """
        required_cols = ['close', 'shares_out']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data is missing required columns: {missing_cols}")

        # Calculate Market Cap
        market_cap = data['close'] * data['shares_out']
        
        # Apply log transformation and invert to favor small caps
        # Using np.log to handle scale, and clipping at 0 to avoid log(0) issues just in case
        safe_market_cap = market_cap.replace(0, np.nan)
        size_factor = -1 * np.log(safe_market_cap)
        
        return size_factor