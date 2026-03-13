import os
from typing import List, Dict, Any

import pandas as pd

from core.config import RAW_DIR
from core.data_fetcher.eodhd_client import fetch_historical_data


def process_and_save_eod_data(raw_data: List[Dict[str, Any]], ticker: str) -> None:
    """
    Process raw JSON EOD data into a standardized Pandas DataFrame and save it as a Parquet file.
    
    Args:
        raw_data (list): The raw JSON data from the EODHD API.
        ticker (str): The ticker symbol.
    """
    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return
    
    # Ensure 'date' column is DateTime index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # Convert standard columns to float64
    standard_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    for col in standard_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
            
    # Save to Parquet in the RAW_DIR using pyarrow engine
    file_path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    df.to_parquet(file_path, engine='pyarrow')


async def update_ticker_data(ticker: str, start_date: str = "2000-01-01", end_date: str = "2100-01-01") -> None:
    """
    Wrapper function to fetch historical data and save it.
    
    Args:
        ticker (str): The ticker symbol.
        start_date (str): The start date for data fetching.
        end_date (str): The end date for data fetching.
    """
    # Fetch data asynchronously from the EODHD API
    raw_data = await fetch_historical_data(ticker, start_date, end_date)
    
    # Process and save the data synchronously (since it's CPU-bound and file I/O)
    process_and_save_eod_data(raw_data, ticker)
