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

    def safe_float(val):
        try:
            return float(val) if val and val != 'null' else np.nan
        except:
            return np.nan

    # Build a unified date set across all three statements so that income/cashflow
    # entries at dates not present in the balance sheet (e.g. annual-only filings)
    # are also captured, and bs entries with no matching inc/cf row fall back
    # gracefully to NaN rather than silently dropping the data.
    all_period_dates = set(bs.keys()) | set(inc.keys()) | set(cf.keys())

    records = []

    for date_str in all_period_dates:
        bs_data  = bs.get(date_str, {})
        inc_data = inc.get(date_str, {})
        cf_data  = cf.get(date_str, {})

        filing_date = bs_data.get('filing_date') or inc_data.get('filing_date') or cf_data.get('filing_date')
        # filing_date == date_str means EODHD is returning the fiscal period-end as the
        # "filing date" — a known data quality gap for older quarters. In reality, SEC
        # quarterly filings (10-Q/10-K) require 40-45 days after period end, so Δ=0
        # is impossible and introduces pure lookahead bias. Apply the same +45d fallback.
        if not filing_date or filing_date in ('null', date_str):
            filing_date = (pd.to_datetime(date_str) + pd.Timedelta(days=45)).strftime('%Y-%m-%d')

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
    
    fund_cols = ['netIncome', 'totalAssets', 'totalOperatingCashFlows', 'longTermDebt',
                 'totalCurrentAssets', 'totalCurrentLiabilities', 'commonStockSharesOutstanding',
                 'grossProfit', 'totalRevenue']

    # Many tickers use period-end dates (Mar 31, Jun 30, Sep 30, Dec 31) as filing dates.
    # These frequently fall on weekends, so an exact-date left join would silently drop those
    # quarters, and ffill would eventually run out, producing long NaN gaps.
    # Fix: expand to the union of price and filing dates, ffill across that combined index,
    # then select back to trading days only.
    all_dates = raw_price_df.index.union(fund_df.index).sort_values()
    fund_aligned = fund_df.reindex(all_dates).ffill(limit=130).reindex(raw_price_df.index)
    merged_df = raw_price_df.join(fund_aligned, how='left')
    
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
