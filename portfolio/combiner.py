import numpy as np
import pandas as pd
from typing import Dict


class LinearFactorCombiner:
    """
    Synthesizes multiple individual alpha factors into a single composite score.
    
    Factors are cross-sectionally standardized (Z-scored) daily to handle 
    different scales and distributions before applying static weights.
    """

    def combine(
        self, 
        aligned_factors: Dict[str, pd.DataFrame], 
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Combines a dictionary of aligned factor DataFrames into a single, 
        weighted composite factor DataFrame.
        
        Args:
            aligned_factors (Dict[str, pd.DataFrame]): Dictionary mapping factor 
                                                       names to their aligned, wide-format DataFrames 
                                                       (index: dates, columns: tickers).
            weights (Dict[str, float]): Dictionary mapping factor names to their 
                                        desired synthesized weight (e.g. {'momentum': 0.4}).
                                        
        Returns:
            pd.DataFrame: The z-scored, composite factor DataFrame ready for Alphalens.
        """
        if not aligned_factors or not weights:
            raise ValueError("Both aligned_factors and weights must contain data.")
            
        composite_df = None
        
        for name, factor_df in aligned_factors.items():
            if name not in weights:
                continue
                
            weight = weights[name]
            
            # 1. Cross-Sectional Standardization (Z-score)
            # Subtract the row (daily cross-section) mean
            row_mean = factor_df.mean(axis=1)
            # Divide by the row (daily cross-section) standard deviation
            row_std = factor_df.std(axis=1)
            
            # Vectorized broadcasting handles the alignment perfectly. 
            # Note: mean/std automatically ignore NaN by default in Pandas,
            # and dividing NaN by anything or dividing by NaN will remain NaN.
            z_scored_factor = factor_df.sub(row_mean, axis=0).div(row_std, axis=0)
            
            # 2. Apply Weighting
            weighted_factor = z_scored_factor * weight
            
            # 3. Aggregation (Summation)
            if composite_df is None:
                composite_df = weighted_factor
            else:
                # Use .add() to ensure proper alignment and avoid NaN propagation where possible,
                # but if a ticker is missing completely on a day for one factor, 
                # we prefer it to be NaN in the composite (hence fill_value=None or default).
                # To be conservative and strict: only sum where both exist.
                composite_df = composite_df + weighted_factor
                
        if composite_df is None:
            raise ValueError("No matching factors found in the provided weights.")
            
        return composite_df
