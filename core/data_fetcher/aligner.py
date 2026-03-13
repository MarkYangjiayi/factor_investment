import pandas as pd
from typing import Dict


class CrossSectionAligner:
    """
    Module to handle cross-sectional alignment of factor data for multiple tickers.
    
    Factors for individual tickers often have misaligned DatetimeIndexes due to 
    differing trading histories, IPOs, and delistings. This class aligns those 
    individual series into a unified cross-sectional DataFrame.
    """
    
    def align_factors(self, factor_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Aligns a dictionary of factor Series into a single cross-sectional DataFrame.
        
        This method uses a vectorized outer join (pd.concat with axis=1) on the dates. 
        Where a ticker has missing data for a date, the value will correctly be 
        represented as NaN, allowing cross-sectional operations (e.g., ranking) 
        to ignore them properly.
        
        Args:
            factor_dict (Dict[str, pd.Series]): A mapping where keys are ticker symbols (e.g. 'AAPL')
                                                and values are the computed factor values (pd.Series) 
                                                indexed by DatetimeIndex.
                                                
        Returns:
            pd.DataFrame: A unified DataFrame indexed by a global DatetimeIndex.
                          The columns represent the ticker symbols.
        """
        if not factor_dict:
            return pd.DataFrame()
            
        # Efficiently concat all the series side-by-side using the outer join via axis=1.
        # Naming the columns directly from the dict keys.
        df_aligned = pd.concat(factor_dict.values(), axis=1, keys=factor_dict.keys())
        
        # Sort the DatetimeIndex chronologically
        df_aligned.sort_index(inplace=True)
        
        return df_aligned
