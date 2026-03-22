import asyncio
import time
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
    Fetches fundamental data for all raw tickers in parallel with a semaphore-bounded
    concurrency of 15, matching the rate-limiting strategy used in download_data.py.
    """
    raw_path = Path(RAW_DIR)
    processed_path = Path(PROCESSED_DIR)

    processed_path.mkdir(parents=True, exist_ok=True)

    parquet_files = list(raw_path.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {RAW_DIR}")
        return

    logger.info(f"Found {len(parquet_files)} tickers to process.")
    start_time = time.time()

    semaphore = asyncio.Semaphore(15)

    async def bounded_process(file_path: Path):
        ticker = file_path.stem  # e.g., 'AAPL.US'
        async with semaphore:
            await asyncio.sleep(0.05)  # stagger TCP connection bursts
            raw_price_df = pd.read_parquet(file_path)
            if raw_price_df.empty:
                logger.warning(f"Raw data for {ticker} is empty. Skipping.")
                return None

            fundamentals_data = await fetch_fundamentals(ticker)
            if not fundamentals_data:
                logger.warning(f"No fundamental data fetched for {ticker}. Skipping.")
                return None

        # CPU-bound work runs outside the semaphore so the slot is freed for the next fetch
        merged_df = process_and_merge_fundamentals(ticker, raw_price_df, fundamentals_data)
        output_path = processed_path / f"{ticker}.parquet"
        merged_df.to_parquet(output_path)
        return ticker

    tasks = [bounded_process(f) for f in parquet_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    fail_count = 0
    for file_path, result in zip(parquet_files, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {file_path.stem}: {result}")
            fail_count += 1
        elif result is not None:
            success_count += 1
            if success_count % 50 == 0:
                logger.info(f"Progress: {success_count} tickers processed...")

    total_time = time.time() - start_time
    logger.info(f"Build completed in {total_time:.2f}s. {success_count} succeeded, {fail_count} failed.")

if __name__ == "__main__":
    try:
        asyncio.run(build_all())
    except KeyboardInterrupt:
        logger.info("Build was interrupted by the user.")
