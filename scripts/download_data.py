import asyncio
import logging
import time
from datetime import datetime

from core.data_fetcher.processor import update_ticker_data
from core.data_fetcher.universe import get_index_constituents

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def run_pipeline():
    """
    Main asynchronous pipeline to batch download and process EOD data for multiple tickers.
    """
    tickers = await get_index_constituents()# ["AAPL.US", "MSFT.US", "GOOGL.US"]  # EODHD usually expects exchange suffixes like .US
    start_date = "2010-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    logger.info(f"Starting batch download for {len(tickers)} tickers from {start_date} to {end_date}.")
    
    start_time = time.time()
    
    # Create tasks for all tickers to be executed concurrently
    tasks = [
        update_ticker_data(ticker=ticker, start_date=start_date, end_date=end_date)
        for ticker in tickers
    ]
    
    # Execute all tasks concurrently.
    # We use return_exceptions=True so that one failing request doesn't crash the entire batch.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    # Check for exceptions in the results
    success_count = 0
    fail_count = 0
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch or process data for {ticker}: {result}")
            fail_count += 1
        else:
            logger.info(f"Successfully processed and saved data for {ticker}.")
            success_count += 1
            
    total_time = end_time - start_time
    logger.info(f"Pipeline completed in {total_time:.2f} seconds.")
    logger.info(f"Summary: {success_count} succeeded, {fail_count} failed.")

if __name__ == "__main__":
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        logger.info("Pipeline was interrupted by the user.")
