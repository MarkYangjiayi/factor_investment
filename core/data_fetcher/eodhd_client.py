import logging
from typing import Any, Dict

import aiohttp

from core.config import EODHD_API_KEY

logger = logging.getLogger(__name__)

async def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Fetch historical End-Of-Day (EOD) data for a given ticker asynchronously using the EODHD API.
    
    Args:
        ticker (str): The ticker symbol (e.g., 'AAPL.US').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        
    Returns:
        dict: The raw JSON data returned by the API.
    """
    if not EODHD_API_KEY:
        raise ValueError("EODHD_API_KEY is not defined in the configuration.")

    url = f"https://eodhd.com/api/eod/{ticker}"
    params = {
        "api_token": EODHD_API_KEY,
        "from": start_date,
        "to": end_date,
        "fmt": "json"
    }

    try:
        # Architecture Rule: Initialize the session inside the function
        # to ensure thread safety and avoid event-loop binding issues.
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                # The API usually returns a list for JSON EOD data, 
                # but type hint dict is used per requirements.
                # If it's a list, it will naturally cast/return as parsed JSON.
                return data
    except aiohttp.ClientError as e:
        logger.error(f"Network error occurred while fetching data for {ticker}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
