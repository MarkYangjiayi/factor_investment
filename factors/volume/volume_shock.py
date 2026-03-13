from typing import Any

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class VolumeShock(BaseFactor):
    """
    Volume Shock factor.
    
    This factor identifies stocks experiencing abnormal trading volume.
    It calculates the ratio of the current day's volume to the rolling mean volume
    over a specified period. A higher ratio indicates a volume shock.
    """

    def __init__(self, ma_period: int = 20, **kwargs: Any) -> None:
        """
        Initialize the VolumeShock factor.

        Args:
            ma_period (int): The moving average period for volume. Defaults to 20.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(name="VolumeShock", ma_period=ma_period, **kwargs)
        self.ma_period = ma_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the volume shock factor.

        Formula: Volume_t / Mean(Volume_{t-ma_period} ... Volume_t)

        Args:
            data (pd.DataFrame): DataFrame containing 'volume' column.

        Returns:
            pd.Series: The computed volume shock factor values.
        """
        if 'volume' not in data.columns:
            raise ValueError("Data must contain a 'volume' column.")

        volume = data['volume']
        
        # Calculate rolling mean volume
        rolling_mean_volume = volume.rolling(window=self.ma_period).mean()

        # Handle division by zero by adding a small epsilon to the denominator
        # Alternatively, we could replace 0s with NaNs, but epsilon is safer for vectorized ops
        epsilon = 1e-8
        
        # Calculate volume shock ratio
        volume_shock = volume / (rolling_mean_volume + epsilon)
        
        # Replace any potential infinities with NaN (though epsilon should prevent this)
        volume_shock = volume_shock.replace([np.inf, -np.inf], np.nan)

        return volume_shock
