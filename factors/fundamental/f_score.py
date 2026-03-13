from typing import Any

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class PiotroskiFScore(BaseFactor):
    """
    Piotroski F-Score fundamental factor.
    
    This factor measures the financial strength of a company using 9 criteria across
    Profitability, Leverage/Liquidity, and Operating Efficiency. Each met criterion
    adds 1 point to the score, resulting in a total score between 0 and 9.
    
    The input data is assumed to be daily forward-filled fundamental data.
    Year-over-Year (YoY) comparisons are performed using a 252-day lag.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the PiotroskiFScore factor.
        
        Args:
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(name="PiotroskiFScore", **kwargs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the Piotroski F-Score.

        Args:
            data (pd.DataFrame): DataFrame containing the following columns:
                - roa (Return on Assets)
                - cf_ops (Operating Cash Flow)
                - leverage (Long-term Debt / Assets)
                - current_ratio (Current Assets / Current Liabilities)
                - shares_out (Shares Outstanding)
                - gross_margin (Gross Margin)
                - asset_turnover (Asset Turnover)

        Returns:
            pd.Series: The computed F-Score (integer 0-9).
        """
        required_cols = [
            'roa', 'cf_ops', 'leverage', 'current_ratio', 
            'shares_out', 'gross_margin', 'asset_turnover'
        ]
        
        # Check for missing columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data is missing required columns: {missing_cols}")

        # Extract columns for cleaner access
        roa = data['roa']
        cf_ops = data['cf_ops']
        leverage = data['leverage']
        current_ratio = data['current_ratio']
        shares_out = data['shares_out']
        gross_margin = data['gross_margin']
        asset_turnover = data['asset_turnover']

        # Define lag for Year-over-Year comparison (approx 252 trading days)
        lag = 252

        # --- Profitability Signals (4 points) ---
        # 1. ROA > 0
        score_roa_pos = (roa > 0).astype(int)
        
        # 2. Operating Cash Flow > 0
        score_cf_pos = (cf_ops > 0).astype(int)
        
        # 3. Change in ROA > 0 (Current ROA > Previous Year ROA)
        score_roa_growth = (roa > roa.shift(lag)).astype(int)
        
        # 4. Accruals: Operating Cash Flow > ROA
        score_accrual = (cf_ops > roa).astype(int)

        # --- Leverage, Liquidity, and Source of Funds Signals (3 points) ---
        # 5. Change in Leverage < 0 (Current Leverage < Previous Year Leverage)
        # Note: Lower leverage is better
        score_deleveraging = (leverage < leverage.shift(lag)).astype(int)
        
        # 6. Change in Current Ratio > 0 (Current Ratio > Previous Year Ratio)
        score_liquidity = (current_ratio > current_ratio.shift(lag)).astype(int)
        
        # 7. Change in Shares Outstanding <= 0 (No new equity issued)
        # Note: shares_out <= previous shares_out means no dilution or buybacks (positive signal)
        score_no_dilution = (shares_out <= shares_out.shift(lag)).astype(int)

        # --- Operating Efficiency Signals (2 points) ---
        # 8. Change in Gross Margin > 0 (Current GM > Previous Year GM)
        score_margin_growth = (gross_margin > gross_margin.shift(lag)).astype(int)
        
        # 9. Change in Asset Turnover > 0 (Current AT > Previous Year AT)
        score_turnover_growth = (asset_turnover > asset_turnover.shift(lag)).astype(int)

        # --- Total F-Score ---
        f_score = (
            score_roa_pos + 
            score_cf_pos + 
            score_roa_growth + 
            score_accrual + 
            score_deleveraging + 
            score_liquidity + 
            score_no_dilution + 
            score_margin_growth + 
            score_turnover_growth
        )

        # Handle NaNs created by shifting (first 252 days will be invalid for growth metrics)
        # We can fill them with 0 or keep as NaN. Keeping as NaN is safer to avoid false signals.
        # However, for the non-growth metrics (ROA > 0, CFO > 0, CFO > ROA), they are valid from day 1.
        # A common approach is to return NaN if any component is NaN, or treat missing growth data as 0.
        # Given this is a factor score, having NaNs for the first year is standard.
        
        # Re-introduce NaNs where shift caused them (if strictness is required)
        # But simpler: the boolean comparison with NaN usually results in False (0), 
        # except for inequalities where behavior can vary. 
        # Let's ensure we return NaN for the initial lookback period to be safe.
        valid_mask = roa.shift(lag).notna() # Use one shifted series as proxy for validity
        f_score = f_score.where(valid_mask, np.nan)

        return f_score
