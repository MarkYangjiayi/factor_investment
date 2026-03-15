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
    Main asynchronous pipeline to batch download and process EOD data for multiple tickers
    with strictly controlled concurrency to prevent HTTP 429 errors.
    """
    tickers = await get_index_constituents("GSPC.INDX")
    start_date = "2010-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    logger.info(f"Starting batch download for {len(tickers)} tickers from {start_date} to {end_date}.")
    
    start_time = time.time()
    
    # 核心修改 1：创建一个容量为 15 的信号量
    # 这意味着同一时刻最多只有 15 个协程在向服务器发请求
    semaphore = asyncio.Semaphore(15)

    # 核心修改 2：创建一个包装器函数，强制任务在拿不到“令牌”时排队等待
    async def bounded_update(ticker):
        async with semaphore:
            # 加上一个极其微小的延迟，进一步打散 TCP 连接的瞬间波峰
            await asyncio.sleep(0.05)
            return await update_ticker_data(ticker=ticker, start_date=start_date, end_date=end_date)
    
    # 创建受限的任务列表
    tasks = [bounded_update(ticker) for ticker in tickers]
    
    # Execute all tasks concurrently.
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
            success_count += 1
            # 减少成功日志的打印频率，防止终端被淹没
            if success_count % 50 == 0:
                logger.info(f"Progress: Successfully processed {success_count} tickers...")
            
    total_time = end_time - start_time
    logger.info(f"Pipeline completed in {total_time:.2f} seconds.")
    logger.info(f"Summary: {success_count} succeeded, {fail_count} failed.")

if __name__ == "__main__":
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        logger.info("Pipeline was interrupted by the user.")
