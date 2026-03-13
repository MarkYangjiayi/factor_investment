import asyncio
import pandas as pd
from pathlib import Path
import logging
import sys
import os

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.append(project_root)

from core.config import RAW_DIR, PROCESSED_DIR
from core.data_fetcher.fundamentals import fetch_fundamentals, process_and_merge_fundamentals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def build_all():
    """
    Iterates through all raw price data, fetches corresponding fundamental data,
    merges them, and saves the result to the processed directory.
    """
    raw_path = Path(RAW_DIR)
    processed_path = Path(PROCESSED_DIR)
    
    # Ensure processed directory exists
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Get all parquet files
    parquet_files = list(raw_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {RAW_DIR}")
        return

    logger.info(f"Found {len(parquet_files)} tickers to process.")

    for file_path in parquet_files:
        ticker = file_path.stem  # e.g., 'AAPL.US'
        
        try:
            logger.info(f"Processing {ticker}...")
            
            # 1. Load Raw Price Data
            raw_price_df = pd.read_parquet(file_path)
            
            if raw_price_df.empty:
                logger.warning(f"Raw data for {ticker} is empty. Skipping.")
                continue

            # 2. Fetch Fundamentals
            fundamentals_data = await fetch_fundamentals(ticker)
            
            if not fundamentals_data:
                logger.warning(f"No fundamental data fetched for {ticker}. Skipping merge.")
                # Depending on requirements, we might want to save just the price data 
                # or skip it. Here we'll skip to ensure data quality.
                continue

            # 3. Process and Merge
            merged_df = process_and_merge_fundamentals(ticker, raw_price_df, fundamentals_data)
            
            # 4. Save to Processed Directory
            output_path = processed_path / f"{ticker}.parquet"
            merged_df.to_parquet(output_path)
            
            logger.info(f"Successfully processed and saved {ticker} to {output_path}")

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}", exc_info=True)

    logger.info("Data build process completed.")

if __name__ == "__main__":
    asyncio.run(build_all())
