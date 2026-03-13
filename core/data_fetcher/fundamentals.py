import asyncio
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from core.config import EODHD_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Asynchronously fetches fundamental data for a given ticker from the EODHD API.

    Args:
        ticker (str): The ticker symbol (e.g., 'AAPL.US').

    Returns:
        Dict[str, Any]: The JSON response containing fundamental data.
                        Returns an empty dictionary if the request fails.
    """
    if not EODHD_API_KEY:
        logger.error("EODHD_API_KEY is not set in the environment variables.")
        return {}

    url = f"https://eodhd.com/api/fundamentals/{ticker}"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Failed to fetch fundamentals for {ticker}. Status: {response.status}")
                    return {}
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        return {}


def process_and_merge_fundamentals(ticker: str, raw_price_df: pd.DataFrame, fundamental_json: dict) -> pd.DataFrame:
    """
    Safely extract nested JSON fundamental data, align by filing_date, and forward-fill onto daily prices.
    """
    financials = fundamental_json.get('Financials', {})
    bs = financials.get('Balance_Sheet', {}).get('quarterly', {})
    inc = financials.get('Income_Statement', {}).get('quarterly', {})
    cf = financials.get('Cash_Flow', {}).get('quarterly', {})

    records = []
    
    for date_str, bs_data in bs.items():
        inc_data = inc.get(date_str, {})
        cf_data = cf.get(date_str, {})
        
        filing_date = bs_data.get('filing_date')
        if not filing_date or filing_date == 'null':
            filing_date = (pd.to_datetime(date_str) + pd.Timedelta(days=45)).strftime('%Y-%m-%d')
            
        def safe_float(val):
            try:
                return float(val) if val and val != 'null' else np.nan
            except:
                return np.nan

        records.append({
            'filing_date': pd.to_datetime(filing_date),
            'netIncome': safe_float(inc_data.get('netIncome')),
            'totalAssets': safe_float(bs_data.get('totalAssets')),
            'totalOperatingCashFlows': safe_float(cf_data.get('totalCashFromOperatingActivities', cf_data.get('totalOperatingCashFlows'))),
            'longTermDebt': safe_float(bs_data.get('longTermDebt')),
            'totalCurrentAssets': safe_float(bs_data.get('totalCurrentAssets')),
            'totalCurrentLiabilities': safe_float(bs_data.get('totalCurrentLiabilities')),
            'commonStockSharesOutstanding': safe_float(bs_data.get('commonStockSharesOutstanding')),
            'grossProfit': safe_float(inc_data.get('grossProfit')),
            'totalRevenue': safe_float(inc_data.get('totalRevenue'))
        })
        
    if not records:
        logger.warning(f"No valid fundamental records extracted for {ticker}.")
        cols = ['roa', 'cf_ops', 'leverage', 'current_ratio', 'shares_out', 'gross_margin', 'asset_turnover']
        for col in cols:
            raw_price_df[col] = np.nan
        return raw_price_df
        
    fund_df = pd.DataFrame(records)
    fund_df = fund_df.dropna(subset=['filing_date']).drop_duplicates(subset=['filing_date']).set_index('filing_date').sort_index()
    
    if not isinstance(raw_price_df.index, pd.DatetimeIndex):
        raw_price_df.index = pd.to_datetime(raw_price_df.index)
    
    # Safe Left Join ensures no trading days are dropped, even if a filing date was on a weekend
    merged_df = raw_price_df.join(fund_df, how='left')
    
    fund_cols = ['netIncome', 'totalAssets', 'totalOperatingCashFlows', 'longTermDebt', 
                 'totalCurrentAssets', 'totalCurrentLiabilities', 'commonStockSharesOutstanding', 
                 'grossProfit', 'totalRevenue']
    
    # Forward fill the fundamentals
    merged_df[fund_cols] = merged_df[fund_cols].ffill(limit=120)
    
    # Calculate ratios safely
    merged_df['roa'] = merged_df['netIncome'] / merged_df['totalAssets']
    merged_df['cf_ops'] = merged_df['totalOperatingCashFlows'] / merged_df['totalAssets']
    merged_df['leverage'] = merged_df['longTermDebt'] / merged_df['totalAssets']
    merged_df['current_ratio'] = merged_df['totalCurrentAssets'] / merged_df['totalCurrentLiabilities']
    merged_df['shares_out'] = merged_df['commonStockSharesOutstanding']
    merged_df['gross_margin'] = merged_df['grossProfit'] / merged_df['totalRevenue']
    merged_df['asset_turnover'] = merged_df['totalRevenue'] / merged_df['totalAssets']
    
    # Drop absolute columns to save memory
    merged_df = merged_df.drop(columns=fund_cols)
    
    return merged_df
