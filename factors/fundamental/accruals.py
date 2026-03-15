from typing import Any
import pandas as pd
from factors.base import BaseFactor

class Accruals(BaseFactor):
    """
    Accruals Ratio factor.
    
    Measures the non-cash component of earnings. High accruals are often a sign of 
    lower earnings quality or potential earnings manipulation.
    
    Formula: (Net Income - Operating Cash Flow) / Total Assets
    Since we have roa = Net Income / Total Assets and cf_ops = Operating Cash Flow / Total Assets,
    the ratio is simply: roa - cf_ops.
    
    We multiply by -1 because lower accruals are generally considered better (higher quality).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="Accruals", **kwargs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the Accruals factor.

        Args:
            data (pd.DataFrame): DataFrame containing 'roa' and 'cf_ops' columns.

        Returns:
            pd.Series: The computed Accruals factor values.
        """
        required_cols = ['roa', 'cf_ops']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data is missing required columns: {missing_cols}")

        # Calculate Accruals Ratio
        accruals_ratio = data['roa'] - data['cf_ops']
        
        # Invert so lower accruals score higher
        return accruals_ratio * -1