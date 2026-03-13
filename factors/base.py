from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFactor(ABC):
    """
    Abstract base class for all alpha factors in the alpha_vibe framework.
    
    All specific factor implementations must inherit from this class and implement
    the `compute` method.
    
    CRITICAL PERFORMANCE REQUIREMENT:
    The `compute` method strictly forbids using native Python `for` loops or 
    slow Pandas iterative methods like `df.iterrows()`, `df.apply()`, or `df.itertuples()`. 
    All calculations MUST rely entirely on NumPy and Pandas vectorized operations 
    (e.g., `df.rolling()`, `df.shift()`, `np.where()`). This ensures high-performance 
    evaluation during backtesting.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the factor with a name and optional parameters.
        
        Args:
            name (str): The name identifier of the factor.
            **kwargs: Any specific parameters required for the factor's calculation
                      (e.g., `lookback_period=20`).
        """
        self.name = name
        self.params = kwargs

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Compute the factor values based on the provided standardized data.
        
        Args:
            data (pd.DataFrame): The standardized input data containing standard EOD columns 
                                 ('open', 'high', 'low', 'close', 'adjusted_close', 'volume') 
                                 with a properly set DatetimeIndex.
                                 
        Returns:
            pd.Series | pd.DataFrame: The computed factor values.
        """
        pass
