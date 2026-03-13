import asyncio
import logging
from typing import List

import aiohttp

from core.config import EODHD_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_index_constituents(index_ticker: str = "DJI.INDX") -> List[str]:
    """Fetches the constituents of a given index from EODHD Fundamentals API.

    Args:
        index_ticker (str): The ticker symbol of the index (e.g., "DJI.INDX", "GSPC.INDX").
                            Defaults to "DJI.INDX".

    Returns:
        List[str]: A list of constituent stock tickers formatted as "{Code}.{Exchange}"
                   (e.g., ["AAPL.US", "MSFT.US"]). Returns an empty list if fetching fails
                   or no components are found.

    Raises:
        ValueError: If EODHD_API_KEY is not set.
    """
    if not EODHD_API_KEY:
        logger.error("EODHD_API_KEY is missing in core.config.")
        raise ValueError("EODHD_API_KEY is not set in configuration.")

    url = f"https://eodhd.com/api/fundamentals/{index_ticker}"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json"
    }

    async with aiohttp.ClientSession() as session:
        try:
            logger.info(f"Fetching constituents for index: {index_ticker}")
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch data for {index_ticker}. Status: {response.status}")
                    return []

                try:
                    data = await response.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON response for {index_ticker}: {e}")
                    return []

                if not isinstance(data, dict):
                    logger.error(f"Unexpected response format for {index_ticker}: Expected dict, got {type(data)}")
                    return []

                components = data.get("Components", {})
                if not components:
                    logger.warning(f"No components found for index: {index_ticker}")
                    return []

                # Handle both dictionary (common in EODHD) and list formats for Components
                iterator = components.values() if isinstance(components, dict) else components
                
                constituents: List[str] = []
                for component in iterator:
                    # Ensure component is a dictionary before accessing fields
                    if not isinstance(component, dict):
                        continue
                        
                    code = component.get("Code")
                    exchange = component.get("Exchange")

                    if code and exchange:
                        constituents.append(f"{code}.{exchange}")
                    else:
                        logger.debug(f"Skipping incomplete component data: {component}")

                logger.info(f"Successfully fetched {len(constituents)} constituents for {index_ticker}")
                return constituents

        except aiohttp.ClientError as e:
            logger.error(f"Network error while fetching index constituents for {index_ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_index_constituents: {e}")
            return []
