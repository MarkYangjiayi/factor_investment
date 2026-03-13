import pandas as pd

from factors.base import BaseFactor


class TimeSeriesMomentum(BaseFactor):
    """
    Classic Time-Series Momentum (TSOM) Factor.
    
    Computes the percentage return of an asset over a specified lookback period.
    """
    
    def __init__(self, lookback_period: int = 252, name: str = "ts_momentum", **kwargs):
        """
        Initialize the Time-Series Momentum factor.
        
        Args:
            lookback_period (int): Lookback period in days. Default is 252 (approx. 1 trading year).
            name (str): Factor name. Default is "ts_momentum".
            **kwargs: Additional parameters.
        """
        super().__init__(name=name, lookback_period=lookback_period, **kwargs)
        self.lookback_period = lookback_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the percentage return over the lookback period using the 'close' price.
        
        Uses strictly vectorized operations as mandated by BaseFactor.
        
        Args:
            data (pd.DataFrame): Standardized input data.
            
        Returns:
            pd.Series: The computed momentum factor values, sharing the DatetimeIndex of the input.
        """
        if 'close' not in data.columns:
            raise KeyError("The input DataFrame must contain a 'close' column.")
            
        # Strictly vectorized percentage change calculation over the lookback window
        momentum: pd.Series = data['close'].pct_change(periods=self.lookback_period)
        
        # Name the output series for clarity
        momentum.name = self.name
        
        return momentum
