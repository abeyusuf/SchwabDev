# Optimized code - Please test before using
import datetime
import os
import pandas as pd
import logging
from dotenv import load_dotenv
import schwabdev
import time
import pytz
import threading

logger = logging.getLogger(__name__)
TIMEZONE = pytz.timezone('America/New_York')

class HistoricalDataDownloader:
    """Handles downloading and processing of historical price data from Schwab API."""
    
    def __init__(self, app_key: str, app_secret: str):
        """Initialize the downloader with API credentials."""
        self.app_key = app_key
        self.app_secret = app_secret

    def fetch_historical_data(self, symbol: str, start_date=None, end_date=None, retries=3, delay=1) -> pd.DataFrame:
        """
        Fetch historical price data for a given symbol with a retry mechanism.
        
        Args:
            symbol (str): The stock symbol to fetch data for
            start_date (datetime): Start date for the data
            end_date (datetime): End date for the data
            retries (int): Number of retries
            delay (int): Delay between retries in seconds
                
        Returns:
            pd.DataFrame: Processed historical price data
        """
        attempt = 0
        while attempt < retries:
            try:
                logger.info(f"Fetching historical price data for {symbol}, attempt {attempt+1}")
                
                # Initialize the client inside the method to ensure thread safety
                client = schwabdev.Client(self.app_key, self.app_secret)

                # If start_date or end_date is None, use default values
                if start_date is None:
                    start_date = datetime.datetime(2000, 1, 1)
                if end_date is None:
                    end_date = datetime.datetime.now()

                response = client.price_history(
                    symbol=symbol,
                    periodType="year",
                    frequencyType="daily",
                    startDate=start_date,
                    endDate=end_date,
                    needExtendedHoursData=False
                )

                return self._process_response(response, symbol)
            except Exception as e:
                attempt += 1
                logger.error(f"Error fetching data for {symbol}: {e}. Attempt {attempt}/{retries}")
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch data for {symbol} after {retries} attempts.")
                    return pd.DataFrame()  # Return empty DataFrame if all retries fail

    def _process_response(self, response, symbol: str) -> pd.DataFrame:
        """Process API response and convert to formatted DataFrame."""
        data = response.json()
        candles = data.get("candles", [])
        
        if not candles:
            logger.warning(f"No data received for {symbol}")
            return pd.DataFrame()
                
        df = pd.DataFrame(candles)
        
        # Convert timestamp to datetime and normalize
        df['Date'] = pd.to_datetime(df['datetime'], unit='ms').dt.normalize()
        
        # Rename columns to standard format
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Add 'Adj Close' column
        df['Adj Close'] = df['Close']
        
        # Reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        df.set_index('Date', inplace=True)
        
        # Normalize the index to ensure time components are set to 00:00:00
        df.index = df.index.normalize()
        
        return df


def get_stock_data(tickers, start_date="2000-01-01", end_date="2030-12-31"):
    """
    Fetch stock data from Schwab for a fixed date range and return a dict of DataFrames.
    """
    stock_data = {}

    # Initialize the downloader
    load_dotenv()
    app_key = os.getenv("app_key")
    app_secret = os.getenv("app_secret")
    if not app_key or not app_secret:
        raise ValueError("API credentials not found. Check your .env file.")
    downloader = HistoricalDataDownloader(app_key, app_secret)

    # Convert string dates to datetime objects
    start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    data_lock = threading.Lock()

    def fetch_data_for_ticker(ticker):
        print(f"Fetching data for {ticker}...")
        data = downloader.fetch_historical_data(ticker, start_date_dt, end_date_dt)
        if data.empty:
            logger.warning(f"No historical data loaded for {ticker}, skipping.")
            return
        with data_lock:
            stock_data[ticker] = data

    # Create and start threads
    threads = []
    for ticker in tickers:
        thread = threading.Thread(target=fetch_data_for_ticker, args=(ticker,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return stock_data


