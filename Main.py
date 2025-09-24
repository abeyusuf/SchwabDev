import concurrent.futures
import schwabdev  # Ensure this is imported
import pandas as pd
import threading
import numpy as np
import quantstats_lumi as qs
import re
import matplotlib.pyplot as plt
import csv
import alpaca_trade_api as tradeapi
import alpaca_trade_api as alpaca
import time
import logging
from datetime import datetime, datetime as dt, timedelta, time as dtime
import sys
import collections  
import pytz
import os
import data_download8, indicatorcalc_testenv3  # Ensure these modules are correctly implemented
from dotenv import load_dotenv
import itertools
from pymongo import MongoClient
import copy  # <--- FOR DEEP COPYING DATA SAFELY
import gc
gc.disable()

# Load environment variables
load_dotenv()

def safe_mongo_call(operation, operation_name, max_retries=3):
    """Safely execute MongoDB operations with retry logic"""
    import time
    import pymongo.errors
    
    for attempt in range(max_retries):
        try:
            return operation()
        except (pymongo.errors.ServerSelectionTimeoutError, 
                pymongo.errors.NetworkTimeout,
                pymongo.errors.ConnectionFailure) as e:
            logger.warning(f"{operation_name} failed (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"{operation_name} failed after {max_retries} attempts")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    return None
# ------------------ Configuration Parameters ------------------
# mode: 'backtest' or 'live_trade'
mode = 'backtest'

BENCHMARK_TICKER = 'TECL'  # Replace 'SPY' with your desired benchmark ticker

# Remove the old backtest log file if it exists
if mode == 'backtest':
    if os.path.exists('backtest_log.csv'):
        os.remove('backtest_log.csv')

# Timezone configuration
TIMEZONE = pytz.timezone('America/New_York')

# Initial capital for backtesting
initial_capital = 100000  # Update as per your account balance

# Stock tickers to trade
start_date = "2020-01-01" if mode == 'backtest' else datetime.now(TIMEZONE).strftime("%Y-%m-%d")
end_date = "2030-12-31"
tickers = ["IBIT", "AGG", "BIL", "BITI", "BND","BDRY", "BTAL", "FDN", "FNGD", "GBTC", "KMLM", "MSTX", "MSTR","WEBL",
          "BSV", "QLD", "QQQ", "QQQE", "SOXL", "SOXS", "SPXL", "SPY", "SQQQ", "SSO", "SVIX", "SVXY",'BULZ','FNGO',
          "TECL", "BITX", "TECS", "TQQQ", "UVXY", "VOOG", "VOOV", "VOX", "VTV", "SOXX", "XLI", "XLK",
          "XLP", "XLU", "TMF", "XLY", "XTN", "PSQ", "XLF", "VIXY", "SPHB", "UVIX", "IYW", "SPIB", "IEF",
          "VXX", "SMH", "RETL", "FNGU", "QQQU", "SSO", "SPXL", "USD", "ROM", "RETL", "UPRO", "NVDL","TNA","SPYU","IOO","TLT","LQD","VGIT","VCSH"]

# ------------------ Account Configuration ------------------
accounts_info = [
   
       {
        "account_name": "Alpaca_Account",
        "broker": "Alpaca",
        "alpaca_api_key": os.getenv('ALPACA_API_KEY'),
        "alpaca_secret_key": os.getenv('ALPACA_SECRET_KEY'),
        "alpaca_base_url": os.getenv('ALPACA_BASE_URL'),
        "account_balance_multiplier": 0.95,
        "mongo_uri": "mongodb://localhost:27017/",
        "mongo_db": "trade_logs",
        "mongo_collection": "Homegrown_live_trades",
        "log_filename_prefix": "Alpaca"
    },
    {
        "account_name": "Schwab_Test",
        "broker": "Schwab",
        "schwab_account_hash": os.getenv('account_hash_AY'),
        "schwab_app_key": os.getenv('app_key'),
        "schwab_app_secret": os.getenv('app_secret'),
        "schwab_callback_url": os.getenv('callback_url'),
        "account_balance_multiplier": 1.0,
        "mongo_uri": "mongodb://localhost:27017/",
        "mongo_db": "Schwab_AY_IRA_trade_logs",
        "mongo_collection": "Homegrown_live_trades",
        "log_filename_prefix": "Schwab_Test"
    }

]

# ------------------ Logging Setup ------------------
import logging, os
from datetime import datetime

loggers = {}

if mode == 'live_trade':
    for account in accounts_info:
        name   = account['account_name']
        prefix = account.get('log_filename_prefix', 'default_prefix')

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()          # wipe any stale handlers

        os.makedirs("logs", exist_ok=True)
        filename = os.path.join(
            "logs",
            f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        fmt = '%(asctime)s %(levelname)s: %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'

        fh = logging.FileHandler(filename, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        fh.setLevel(logging.DEBUG)       # everything goes to the file
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt, datefmt))
        sh.setLevel(logging.INFO)        # keep console a bit quieter
        logger.addHandler(sh)

        logger.propagate = False
        loggers[name] = logger

else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ------------------ MongoDB Setup ------------------
if mode == 'live_trade':
    client = MongoClient(accounts_info[0]['mongo_uri'])  # Use the first account's URI or any
    trades_collections = {}
    for account in accounts_info:
        db = client[account['mongo_db']]
        trades_collections[account['account_name']] = db[account['mongo_collection']]
else:
    trades_collections = None

# ------------------ Alpaca API Initialization ------------------
api_clients = {}
if mode == 'live_trade':
    for account in accounts_info:
        if account['broker'] == 'Alpaca':
            api_clients[account['account_name']] = alpaca.REST(
                account['alpaca_api_key'],
                account['alpaca_secret_key'],
                base_url=account['alpaca_base_url']
            )
        elif account['broker'] == 'Schwab':
            api_clients[account['account_name']] = schwabdev.Client(
                app_key=account['schwab_app_key'],
                app_secret=account['schwab_app_secret'],
                callback_url=account['schwab_callback_url'],
                tokens_file='tokens.json',
                update_tokens_auto=True
            )
else:
    # In backtest, no real API clients needed
    pass

# ------------------ Trade Logging Functions ------------------

def log_trade(trade_details, account_name):
    if mode == 'backtest':
        # CSV logging for backtest
        log_file = 'backtest_log.csv'
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode='a', newline='',encoding='utf-8',errors='replace') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['Trade Type','Strategy','Ticker','Date','Price','Quantity','Reason','Order ID','Profit $','Profit %']
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_details)
    else:
        # MongoDB logging
        trades_collections[account_name].insert_one(trade_details)

def log_completed_trade(trade_details, account_name):
    # Similar approach as log_trade
    if mode == 'backtest':
        log_file = 'backtest_log.csv'
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode='a', newline='',
          encoding='utf-8',    # <- ADD THIS LINE
          errors='replace') as file:
            writer = csv.DictWriter(file, fieldnames=trade_details.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_details)
    else:
        trades_collections[account_name].insert_one(trade_details)

# ------------------ Strategy Class ------------------
class Strategy:
    def __init__(self, name, entry_conditions=None, exit_conditions=None, tickers=None,
                 max_symbols=1, sort_by=None, sort_order='nlargest', equity_fraction=1.0,
                 allow_same_day_reentry=False, atr_multiplier=None, percent_exit=None,
                 max_holding_days=None, **kwargs):
        """
        - max_symbols: Maximum number of symbols to trade at any given time.
        - sort_by: Column name to sort tickers by (e.g., 'NATR', 'volatility').
        - sort_order: 'nlargest' or 'nsmallest' to choose sorting order for indicator values.
        - equity_fraction: Fraction of total equity to allocate to this strategy.
        - allow_same_day_reentry: Boolean flag to allow same-day re-entry after exit.
        - atr_multiplier: Multiplier for ATR-based exit.
        - percent_exit: Exit if price moves by this percentage (e.g., 1.05 for 5% gain).
        - max_holding_days: Maximum number of days to hold a position.
        """
        self.name = name
        self.entry_conditions = entry_conditions
        self.exit_conditions = exit_conditions
        self.tickers = tickers
        self.max_symbols = max_symbols
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.equity_fraction = equity_fraction
        self.allow_same_day_reentry = allow_same_day_reentry
        self.atr_multiplier = atr_multiplier
        self.percent_exit = percent_exit
        self.max_holding_days = max_holding_days

        # Process entry rules dynamically based on the kwargs
        self.entry_rules = []
        entry_indices = set()
        for key in kwargs:
            if key.startswith('entry_conditions_'):
                index = key[len('entry_conditions_'):]
                entry_indices.add(index)
        if entry_indices:
            for index in sorted(entry_indices, key=int):
                conditions = kwargs.get(f'entry_conditions_{index}', ())
                tickers = kwargs.get(f'Ticker_Entry_{index}', [])
                self.entry_rules.append({'conditions': conditions, 'tickers': tickers})
        else:
            if self.entry_conditions and self.tickers:
                self.entry_rules.append({'conditions': self.entry_conditions, 'tickers': self.tickers})

        # Process exit rules dynamically based on kwargs
        self.exit_rules = []
        exit_indices = set()
        for key in kwargs:
            if key.startswith('exit_conditions_'):
                index = key[len('exit_conditions_'):]
                exit_indices.add(index)
        if exit_indices:
            for index in sorted(exit_indices, key=int):
                conditions = kwargs.get(f'exit_conditions_{index}', ())
                tickers = kwargs.get(f'Ticker_Exit_{index}', [])
                self.exit_rules.append({'conditions': conditions, 'tickers': tickers})
        else:
            if self.exit_conditions and self.tickers:
                self.exit_rules.append({'conditions': self.exit_conditions, 'tickers': self.tickers})


    def get_all_tickers(self):
        tickers = set()
        for rule in self.entry_rules:
            tickers.update(rule['tickers'])
        for rule in self.exit_rules:
            tickers.update(rule['tickers'])
        return list(tickers)

    def _evaluate_sort_expression(self, data_row):
        def safe_eval(expr, local_dict):
            allowed_names = {"np": np}
            code = compile(expr, "<string>", "eval")
            for name in code.co_names:
                if name not in allowed_names and name not in local_dict:
                    raise NameError(f"Use of {name} not allowed")
            # Add a small epsilon to avoid division by zero
            local_dict = {k: (v if v != 0 else 1e-6) for k, v in local_dict.items()}
            return eval(code, {"__builtins__": None, 'np': np}, local_dict)


        if isinstance(self.sort_by, str):
            expr = self.sort_by
            # Prepare local variables: map column names to their values
            local_dict = data_row.to_dict()
            try:
                return safe_eval(expr, local_dict)
            except Exception as e:
                print(f"Error evaluating expression '{self.sort_by}': {str(e)}")
                return np.nan
        else:
            return data_row[self.sort_by]


    def check_entry(self, stock_data, date):
        date = pd.to_datetime(date).tz_localize(None).normalize()
        entries = []
        tickers_to_consider = []

        for rule in self.entry_rules:
            tickers = rule['tickers']
            conditions_list = rule['conditions']
            for ticker in tickers:
                if ticker in stock_data and not stock_data[ticker].empty:
                    # Check if there is data on or before this date, and get the last available
                    date = pd.to_datetime(date).tz_localize(None).normalize()
                    relevant_data = stock_data[ticker].loc[stock_data[ticker].index <= date] 
                    if not relevant_data.empty:
                        row = relevant_data.iloc[-1]
                    else:
                        print(f"No available data to evaluate entry for {ticker} on or before {date}. Skipping.")
                        continue
                    logging.debug(f"Evaluating entry conditions for {ticker} on {date}")
                    conditions = entry_conditions(row,ticker)
                    for cond in conditions_list:
                        if isinstance(cond, int):
                            cond_id = cond
                            if cond_id in conditions:
                                condition_met, reason = conditions[cond_id]
                                logging.debug(f"Condition {cond_id} for {ticker}: {condition_met}, Reason: {reason}")
                                if condition_met:
                                    tickers_to_consider.append((ticker, reason))
                                    break  # Break if one condition is met
                        elif isinstance(cond, tuple):
                            all_conditions_met = True
                            reasons = []
                            for cond_id in cond:
                                if cond_id in conditions:
                                    condition_met, reason = conditions[cond_id]
                                    logging.debug(f"Condition {cond_id} for {ticker}: {condition_met}, Reason: {reason}")
                                    if condition_met:
                                        reasons.append(reason)
                                    else:
                                        all_conditions_met = False
                                        break
                                else:
                                    all_conditions_met = False
                                    break
                            if all_conditions_met:
                                combined_reason = " & ".join(reasons)
                                tickers_to_consider.append((ticker, combined_reason))
                                break
                        else:
                            logging.error("Invalid condition type in entry_conditions.")
                else:
                    print(f"Data not available for {ticker} on {date}. Skipping.")
                    continue

        if self.sort_by and tickers_to_consider:
            sort_values = {}
            for ticker, _ in tickers_to_consider:
                try:
                    # Ensure we are getting the correct data when sorting
                    date = pd.to_datetime(date).tz_localize(None).normalize()
                    row = stock_data[ticker].loc[stock_data[ticker].index <= date].iloc[-1]
                    sort_value = self._evaluate_sort_expression(row)
                    sort_values[ticker] = sort_value
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    continue

            tickers_sorted = pd.Series(sort_values)
            if self.sort_order == 'nlargest':
                tickers_sorted = tickers_sorted.nlargest(len(tickers_to_consider))
            else:
                tickers_sorted = tickers_sorted.nsmallest(len(tickers_to_consider))

            tickers_to_consider = [
                (ticker, dict(tickers_to_consider)[ticker])
                for ticker in tickers_sorted.index
                if ticker in dict(tickers_to_consider)
            ]

        return tickers_to_consider


    def check_exit(self, stock_data, positions, date):
            date = pd.to_datetime(date).tz_localize(None).normalize()
            exits = []
            exited_tickers = set()

            for rule in self.exit_rules:
                tickers = rule['tickers']
                conditions_list = rule['conditions']
                for ticker in tickers:
                    if ticker in exited_tickers:
                        continue  # Skip if already processed within this strategy
                    if ticker in positions and ticker in stock_data and not stock_data[ticker].empty:
                        #Check if there is data on or before this date, if we do not have any, skip
                        if stock_data[ticker].index[stock_data[ticker].index <= date].empty: # Changed <= to <
                            print(f"No available data to evaluate exit for {ticker} on or before {date}. Skipping.")
                            continue
                        date = pd.to_datetime(date).tz_localize(None).normalize()
                        row = stock_data[ticker].loc[stock_data[ticker].index <= date].iloc[-1] # Changed <= to <
                        logging.debug(f"Evaluating exit conditions for {ticker} on {date}")
                        conditions = exit_conditions(row,ticker)
                        pos = positions[ticker]
                        entry_price = pos['entry_price']
                        entry_date = pos['entry_date']
                        holding_days = np.busday_count(entry_date.date(), pd.to_datetime(date).date())
                        # Pass holding_days and self.max_holding_days into exit_conditions
                        conditions = exit_conditions(row, ticker, holding_days=holding_days, max_holding_days=self.max_holding_days)

                        exited = False

                        for cond in conditions_list:
                            if isinstance(cond, int):
                                cond_id = cond
                                if cond_id in conditions:
                                    condition_met, reason = conditions[cond_id]
                                    logging.debug(f"Condition {cond_id} for {ticker}: {condition_met}, Reason: {reason}")
                                    if condition_met:
                                        exits.append((ticker, reason))
                                        exited_tickers.add(ticker)
                                        exited = True
                                        break
                            elif isinstance(cond, tuple):
                                all_conditions_met = True
                                reasons = []
                                for cond_id in cond:
                                    if cond_id in conditions:
                                        condition_met, reason = conditions[cond_id]
                                        logging.debug(f"Condition {cond_id} for {ticker}: {condition_met}, Reason: {reason}")
                                        if condition_met:
                                            reasons.append(reason)
                                        else:
                                            all_conditions_met = False
                                            break
                                    else:
                                        all_conditions_met = False
                                        break
                                if all_conditions_met:
                                    combined_reason = " & ".join(reasons)
                                    exits.append((ticker, combined_reason))
                                    exited_tickers.add(ticker)
                                    exited = True
                                    break                        
                            else:
                                logging.error("Invalid condition type in exit_conditions.")

                        if not exited:
                            # Check ATR multiplier exit
                            if self.atr_multiplier is not None:
                                if 'atr_14' in stock_data[ticker].columns:
                                    date = pd.to_datetime(date).tz_localize(None).normalize()
                                    atr_value = stock_data[ticker]['atr_14'].loc[stock_data[ticker].index <= date].iloc[-1] # Changed <= to <
                                    stop_price = entry_price + self.atr_multiplier * atr_value
                                    if row['Close'] >= stop_price:
                                        reason = f"ATR Exit: Close ({row['Close']:.2f}) >= Entry Price ({entry_price:.2f}) + {self.atr_multiplier} * ATR({atr_value:.2f})"
                                        exits.append((ticker, reason))
                                        exited_tickers.add(ticker)
                                        exited = True
                                else:
                                    logging.error(f"ATR data not available for {ticker} on {date}")

                        if not exited:
                            # Check percent exit
                            if self.percent_exit is not None:
                                target_price = entry_price * self.percent_exit
                                if row['Close'] >= target_price:
                                    reason = f"Percent Exit: Close ({row['Close']:.2f}) >= {self.percent_exit*100:.2f}% of Entry Price ({entry_price:.2f})"
                                    exits.append((ticker, reason))
                                    exited_tickers.add(ticker)
                                    exited = True

                        if not exited:
                            # Check max holding days
                            if self.max_holding_days is not None:
                                if holding_days >= self.max_holding_days:
                                    reason = f"Max Holding Days Exit: Holding Days ({holding_days}) >= Max Holding Days ({self.max_holding_days})"
                                    exits.append((ticker, reason))
                                    exited_tickers.add(ticker)
                                    exited = True
                    else:
                        print(f"Data not available for {ticker} on {date} or not in positions. Skipping.")
                        continue
            return exits 
# ------------------ Entry and Exit Condition Blocks ----------------------
def entry_conditions(row,ticker):
    """
    Defines entry conditions with condition IDs.
    Returns a dictionary mapping condition IDs to (boolean, reason) tuples.
    """
    conditions = {
        1: (row['rsi_2'] < 10, f"Entry 2: RSI_2 ({row['rsi_2']:.2f}) < 10"),#2
        2: (row['ibs'] < 0.05, f"Entry 7: IBS ({row['ibs']:.2f}) < 0.05 "),#7
        3: ((row['kmlm_close'] < row['kmlm_sma_20']), f"Entry 13: KMLM_Close ({row['kmlm_close']:.2f}) < SMA(20)@KMLM ({row['kmlm_sma_20']:.2f})"),#13
        4: (row['uo_1,3,5'] < 10, f"Entry 15: UO(1,3,5) ({row['uo_1,3,5']:.2f}) < 10"),#15
        5: (row['bnd_roc7'] > 0, f"Entry 18: ROC(7)@BND ({row['bnd_roc7']:.2f}) > 0"),#18
        6: (row['Close'] < row['Close_shift1'] < row['Close_shift2']< row['Close_shift3'] , f"Entry 33: Close({row['Close']:.2f}) < Close@1 ({row['Close_shift1']:.2f})< Close@2 ({row['Close_shift2']:.2f})< Close@3 ({row['Close_shift3']:.2f})"),#33
        7: ((row['rsi_2'] <=15 and row['rsi_3'] <=20 and row['rsi_4'] <=20 and row['rsi_5'] <=20) or row['ibs']<0.05, f"Entry 34: RSI(2)({row['rsi_2']:.2f}) <=15 and RSI(3)({row['rsi_3']:.2f}) <=20 and RSI(4)({row['rsi_4']:.2f}) <=20 and RSI(5)({row['rsi_5']:.2f}) <=20 or IBS({row['ibs']:.2f}) <=0.05 "),#34
        8: (row['rsi_5_hidden_bullish_divergence'] == True, f"Entry 41: RSI(5)_Hidden_Bullish_Divergence is True"),#41
        9: ((row['sqqq_rsi2'] < 10 and row['sqqq_rsi6'] < 10) or (row['sqqq_rsi2'] < 10 and row['sqqq_rsi10'] < 20) or (row['spy_rsi10'] > 80) or 
            (row['tqqq_rsi3'] > 90 and row['tqqq_rsi5'] > 90 and row['tqqq_rsi10'] > 80) or (row['tqqq_rsi5'] > 90 and row['tqqq_rsi10'] > 80) or 
            (row['sqqq_rsi10'] < 20 and row['tqqq_rsi10'] > 90) or (row['sqqq_rsi4'] < 20 and (row['spy_close'] / row['spy_sma_200']) > 1.1) or 
            (row['sqqq_rsi5'] < 20 and (row['spy_close'] / row['spy_sma_200']) > 1.1) or (row['sqqq_rsi6'] < 20 and (row['spy_close'] / row['spy_sma_200']) > 1.1) or 
            (row['sqqq_rsi7'] < 20 and (row['spy_close'] / row['spy_sma_200']) > 1.1) or (row['tqqq_rsi5'] > 80 and (row['spy_close'] / row['spy_sma_200']) > 1.1),
            f"Entry 42: RSI(2)@SQQQ({row['sqqq_rsi2']:.2f})<10 and RSI(6)@SQQQ({row['sqqq_rsi6']:.2f})<10 or "
            f"RSI(2)@SQQQ({row['sqqq_rsi2']:.2f})<10 and RSI(10)@SQQQ({row['sqqq_rsi10']:.2f})<20 or "
            f"RSI(10)@SPY({row['spy_rsi10']:.2f})>80 or "f"RSI(3)@TQQQ({row['tqqq_rsi3']:.2f})>90 and RSI(5)@TQQQ({row['tqqq_rsi5']:.2f})>90 and RSI(10)@TQQQ({row['tqqq_rsi10']:.2f})>80 or "
            f"RSI(5)@TQQQ({row['tqqq_rsi5']:.2f})>90 and RSI(10)@TQQQ({row['tqqq_rsi10']:.2f})>80 or "f"RSI(10)@SQQQ({row['sqqq_rsi10']:.2f})<20 and RSI(10)@TQQQ({row['tqqq_rsi10']:.2f})>90 or "
            f"RSI(4)@SQQQ({row['sqqq_rsi4']:.2f})<20 and SPY_CLOSE/200SMA({row['spy_close']/row['spy_sma_200']:.2f})>1.1 or "f"RSI(5)@SQQQ({row['sqqq_rsi5']:.2f})<20 and SPY_CLOSE/200SMA({row['spy_close']/row['spy_sma_200']:.2f})>1.1 or "
            f"RSI(6)@SQQQ({row['sqqq_rsi6']:.2f})<20 and SPY_CLOSE/200SMA({row['spy_close']/row['spy_sma_200']:.2f})>1.1 or "f"RSI(7)@SQQQ({row['sqqq_rsi7']:.2f})<20 and SPY_CLOSE/200SMA({row['spy_close']/row['spy_sma_200']:.2f})>1.1 or "
            f"RSI(5)@TQQQ({row['tqqq_rsi5']:.2f})>80 and SPY_CLOSE/200SMA({row['spy_close']/row['spy_sma_200']:.2f})>1.1"),#42
        10: ((row['bnd_rsi2'] > 90 and row['gbtc_rsi2'] < 30) 
             or (row['bnd_rsi3'] > 90 and row['gbtc_rsi3'] < 30)
            or ((row['bnd_close'] / row['bnd_sma_200']) > 1 and row['tqqq_rsi10'] > 90)
            or ((row['bnd_close'] / row['bnd_sma_200']) > 1 and row['bnd_rsi3'] > 90 and (row['gbtc_close'] / row['gbtc_sma_100']) > 1)
            or ((row['bnd_close'] / row['bnd_sma_200']) > 1.1 and row['gbtc_rsi2'] < 10 and (row['gbtc_close'] / row['gbtc_sma_100']) > 1)
            or ((row['spy_close'] / row['spy_sma_200']) > 1.1 and row['spy_rsi6'] > 90),
             f"Entry 48: Intermarket rotation signal - BND RSI_2: {row['bnd_rsi2']:.2f}, BND RSI_3: {row['bnd_rsi3']:.2f}, "
                f"GBTC RSI_2: {row['gbtc_rsi2']:.2f}, TQQQ RSI_10: {row['tqqq_rsi10']:.2f}, SPY RSI_6: {row['spy_rsi6']:.2f}, "
                f"BND/SMA200: {(row['bnd_close'] / row['bnd_sma_200']):.3f}, GBTC/SMA100: {(row['gbtc_close'] / row['gbtc_sma_100']):.3f}, "
                f"SPY/SMA200: {(row['spy_close'] / row['spy_sma_200']):.3f}"),#48

        11: ((row['tqqq_close'] > row['tqqq_sma_200']) and (row['tqqq_rsi10'] > 79 or row['qqq_rsi10'] > 79 or row['qqqe_rsi10'] > 79 or row['soxl_rsi10'] > 79 
            or row['spxl_rsi10'] > 79 or row['tecl_rsi10'] > 79 or row['voog_rsi10'] > 79 or row['voov_rsi10'] > 79 or row['vox_rsi10'] > 79 or row['vtv_rsi10'] > 79 or row['xlf_rsi10'] > 79 
            or row['xlp_rsi10'] > 79 or row['xli_rsi10'] > 79 or row['xlk_rsi10'] > 79 or row['xly_rsi10'] > 79 or row['xlu_rsi10'] > 79 or row['xtn_rsi10'] > 79), 
            f"Entry 49: TQQQ uptrend + sector rotation signal - TQQQ/SMA200: {(row['tqqq_close'] / row['tqqq_sma_200']):.3f}, "
            f"Overbought RSI_10 values: "
            f"TQQQ: {row['tqqq_rsi10']:.1f}, QQQ: {row['qqq_rsi10']:.1f}, QQQE: {row['qqqe_rsi10']:.1f}, SOXL: {row['soxl_rsi10']:.1f}, "
            f"SPXL: {row['spxl_rsi10']:.1f}, TECL: {row['tecl_rsi10']:.1f}, XLF: {row['xlf_rsi10']:.1f}, XLK: {row['xlk_rsi10']:.1f}, "
            f"XLY: {row['xly_rsi10']:.1f}, XLU: {row['xlu_rsi10']:.1f}"),#49
        12: ((row['tqqq_close'] > row['tqqq_sma_200']) and row['tqqq_rsi10'] < 79, 
            f"Entry 50: TQQQ_Close ({row['tqqq_close']:.2f}) > SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}) and RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) < 79"),#50
        13: ((row['tqqq_close'] < row['tqqq_sma_200']) and row['tqqq_rsi10'] < 31, 
            f"Entry 51: TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}) and RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) < 31"),#51
        14: ((row['tqqq_close'] < row['tqqq_sma_200']) and (row['tqqq_rsi10'] > 31 and row['soxl_rsi10'] < 30), 
            f"Entry 52: TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}), RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) > 31, and RSI(10)@SOXL ({row['soxl_rsi10']:.2f}) < 30"),#52
        15: ((row['tqqq_close'] < row['tqqq_sma_200']) and (row['tqqq_rsi10'] > 31 and row['soxl_rsi10'] > 30) and (row['tqqq_close'] < row['tqqq_sma_20']), 
            f"Entry 53: TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}), RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) > 31, RSI(10)@SOXL ({row['soxl_rsi10']:.2f}) > 30, and TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(20)@TQQQ ({row['tqqq_sma_20']:.2f})"),#53
        16: ((row['tqqq_close'] < row['tqqq_sma_200']) and (row['tqqq_rsi10'] > 31 and row['soxl_rsi10'] > 30) and (row['tqqq_close'] > row['tqqq_sma_20']), 
            f"Entry 54: TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}), RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) > 31, RSI(10)@SOXL ({row['soxl_rsi10']:.2f}) > 30, and TQQQ_Close ({row['tqqq_close']:.2f}) > SMA(20)@TQQQ ({row['tqqq_sma_20']:.2f})"),#54
        17: ((row['spy_close'] > row['spy_sma_200']) and row['tqqq_rsi10'] > 79, 
            f"Entry 57: SPY_Close ({row['spy_close']:.2f}) > SMA(200)@SPY ({row['spy_sma_200']:.2f}) and RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) > 79"),#57                                                 
        18: ((row['spy_close'] > row['spy_sma_200']) and row['tqqq_rsi10'] < 79 and row['spxl_rsi10'] > 80, 
            f"Entry 58: SPY_Close ({row['spy_close']:.2f}) > SMA(200)@SPY ({row['spy_sma_200']:.2f}) and RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) <79 and RSI(10)@SPXL ({row['spxl_rsi10']:.2f}) >80"), #58
        19: (((row['qqqe_rsi10'] > 79) or (row['vtv_rsi10'] > 79) or (row['vox_rsi10'] > 79) or (row['tecl_rsi10'] > 79) or (row['xlp_rsi10'] > 79) or (row['voov_rsi10'] > 79) or
                (row['voog_rsi10'] > 79) or (row['tqqq_rsi10'] > 79) or (row['xly_rsi10'] > 79) or (row['spy_rsi10'] > 79)),
            f"Entry 59: (RSI(10)@QQQE ({row['qqqe_rsi10']:.2f}) > 79 or " f"RSI(10)@VTV ({row['vtv_rsi10']:.2f}) > 79 or "f"RSI(10)@VOX ({row['vox_rsi10']:.2f}) > 79 or "
            f"RSI(10)@TECL ({row['tecl_rsi10']:.2f}) > 79 or "f"RSI(10)@XLP ({row['xlp_rsi10']:.2f}) > 79 or "f"RSI(10)@VOOV ({row['voov_rsi10']:.2f}) > 79 or "f"RSI(10)@VOOG ({row['voog_rsi10']:.2f}) > 79 or "
            f"RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) > 79 or "f"RSI(10)@XLY ({row['xly_rsi10']:.2f}) > 79 or "
            f"RSI(10)@SPY ({row['spy_rsi10']:.2f}) > 79)"),#59
        20: ((row['spy_close'] > row['spy_sma_200']) and row['tqqq_rsi10'] < 79 and row['spxl_rsi10'] < 80, 
            f"Entry 60: SPY_Close ({row['spy_close']:.2f}) > SMA(200)@SPY ({row['spy_sma_200']:.2f}), TQQQ_RSI10 ({row['tqqq_rsi10']:.2f}) < 79, and SPXL_RSI10 ({row['spxl_rsi10']:.2f}) < 80"),#60
        61: ((row['spy_close'] > row['spy_sma_200']) and (row['tqqq_rsi10'] < 79 and row['spxl_rsi10'] < 80) and 
            (row['rsi_2_bullish_divergence'] == True or row['rsi_2_hidden_bullish_divergence'] == True or 
            row['rsi_3_bullish_divergence'] == True or row['rsi_3_hidden_bullish_divergence'] == True), 
            f"Entry 61: SPY_Close ({row['spy_close']:.2f}) > SMA(200)@SPY ({row['spy_sma_200']:.2f}), TQQQ_RSI10 ({row['tqqq_rsi10']:.2f}) < 79, SPXL_RSI10 ({row['spxl_rsi10']:.2f}) < 80, and RSI(2/3) Divergences present"),#61
        62: ((row['spy_close'] < row['spy_sma_200']) and row['tqqq_rsi10'] < 31, 
            f"Entry 62: SPY_Close ({row['spy_close']:.2f}) < SMA(200)@SPY ({row['spy_sma_200']:.2f}) and TQQQ_RSI10 ({row['tqqq_rsi10']:.2f}) < 31"),#62
        63: ((row['spy_close'] < row['spy_sma_200']) and row['tqqq_rsi10'] > 31 and row['spy_rsi10'] < 30, 
            f"Entry 63: SPY_Close ({row['spy_close']:.2f}) < SMA(200)@SPY ({row['spy_sma_200']:.2f}), TQQQ_RSI10 ({row['tqqq_rsi10']:.2f}) > 31, and SPY_RSI10 ({row['spy_rsi10']:.2f}) < 30"),#63 
        64: ((row['spy_close'] < row['spy_sma_200']) and row['tqqq_rsi10'] > 31 and row['spy_rsi10'] > 30 and row['uvxy_rsi10'] < 74 and 
            (row['tqqq_close'] > row['tqqq_sma_20']), 
            f"Entry 64: SPY_Close ({row['spy_close']:.2f}) < SMA(200)@SPY ({row['spy_sma_200']:.2f}), TQQQ_RSI10 ({row['tqqq_rsi10']:.2f}) > 31, SPY_RSI10 ({row['spy_rsi10']:.2f}) > 30, UVXY_RSI10 ({row['uvxy_rsi10']:.2f}) < 74, and TQQQ_Close ({row['tqqq_close']:.2f}) > SMA(20)@TQQQ ({row['tqqq_sma_20']:.2f})"),#64
        65: ((row['spy_close'] < row['spy_sma_200']) and row['tqqq_rsi10'] > 31 and row['spy_rsi10'] > 30 and row['uvxy_rsi10'] < 74 and 
            (row['tqqq_close'] < row['tqqq_sma_20']), 
            f"Entry 65: SPY_Close ({row['spy_close']:.2f}) < SMA(200)@SPY ({row['spy_sma_200']:.2f}), TQQQ_RSI10 ({row['tqqq_rsi10']:.2f}) > 31, SPY_RSI10 ({row['spy_rsi10']:.2f}) > 30, UVXY_RSI10 ({row['uvxy_rsi10']:.2f}) < 74, and TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(20)@TQQQ ({row['tqqq_sma_20']:.2f})"),#65
        71: ((row['Close'] > row['Close_rolling_max_25_yesterday']), 
            f"Entry 71: Close({row['Close']:.2f}) > MaxClose25DaysAgo({row['Close_rolling_max_25_yesterday']:.2f})"),#71
        114: (row['spy_rsi2']<20 and row['spy_rsi3']<30, f"Entry 114: RSI(2) ({row['spy_rsi2']:.2f}) <20 and RSI(3) ({row['spy_rsi3']:.2f}) <30"),#114
        115: (row['spy_rsi4'] <20, f"Entry 115: RSI(4) ({row['spy_rsi4']:.2f}) <20"),#115
        116: (row['spy_rsi5'] <20, f"Entry 116: RSI(5) ({row['spy_rsi5']:.2f}) <20"),#116
        117: (row['spy_rsi6'] <20, f"Entry 117: RSI(6) ({row['spy_rsi6']:.2f}) <20"),#117

        118: (row['bil_rsi10'] <row['ief_rsi10'] and row['spy_rsi10'] > 75, f"Entry 118: RSI(5)@BIL ({row['bil_rsi5']:.2f}) < RSI(5)@BIL({row['ief_rsi5']:.2f}) and RSI(5)@SPY({row['spy_rsi5']:.2f})>75 "),#118
        126: (row['spy_rsi3'] < 20 and row['spy_rsi4'] < 30 or row['rsi_5'] < 10 or row['rsi_6'] < 10 , f"Entry 126: (SPY_RSI3({row['spy_rsi3']:.2f}) < 20 AND SPY_RSI4({row['spy_rsi4']:.2f}) < 30) OR "
            f"RSI5({row['rsi_5']:.2f}) < 10 OR RSI6({row['rsi_6']:.2f}) < 10"),#126
        127: (row['spy_rsi10'] >80 and (row['spy_close']/ row['spy_sma_200'])>1, f"Entry 127: SPY_RSI10({row['spy_rsi10']:.2f}) > 80 AND SPY Close({row['spy_close']:.2f}) > SPY SMA200({row['spy_sma_200']:.2f})"),#127
        128: (row['spy_rsi2']<20 and row['spy_rsi3']<30 and (row['spy_close']/ row['spy_sma_200'])>1, f"Entry 128: SPY_RSI2({row['spy_rsi2']:.2f}) < 20 AND SPY_RSI3({row['spy_rsi3']:.2f}) < 30 AND SPY Close({row['spy_close']:.2f}) > SPY SMA200({row['spy_sma_200']:.2f})"),#128
         130: (row['tqqq_rsi10'] > 80 or row['vtv_rsi10'] > 80 or row['xlk_rsi10'] > 80 or row['spy_rsi10'] > 80, f"Entry 130: RSI(10)TQQQ ({row['tqqq_rsi10']:.2f}) > 80 or RSI(10)VTV ({row['vtv_rsi10']:.2f}) > 80 or RSI(10)XLK ({row['xlk_rsi10']:.2f}) > 80 or RSI(10)SPY ({row['spy_rsi10']:.2f}) > 80"),#130
        131: (row['tqqq_rsi10'] < 30 or row['vtv_rsi10'] <30 or row['xlk_rsi10'] <30 or row['spy_rsi10'] <30, f"Entry 131: "f"RSI(10)TQQQ ({row['tqqq_rsi10']:.2f}) < 30 or RSI(10)VTV  ({row['vtv_rsi10']:.2f}) < 30 or RSI(10)XLK  ({row['xlk_rsi10']:.2f}) < 30 or RSI(10)SPY  ({row['spy_rsi10']:.2f}) < 30"),#131
        132: (row['tqqq_rsi10'] < 80 and row['tqqq_rsi10'] > 30, f"Entry 132: RSI(10)QQQ ({row['tqqq_rsi10']:.2f}) < 80 and RSI(10)QQQ ({row['tqqq_rsi10']:.2f}) > 30"),#132


        3800: (row['ibs'] < 0.06 
		or (row['price_above_supertrend_12_4']==True and row['ibs'] < 0.1)
		or (row['macdh_hdiv_bull_slope']==True and row['ibs'] < 0.1)
		or (row['ibs_hdiv_bull_slope']==True and row['spy_close']/row['spy_sma_20'] > 1)
		or (row['ibs_10_div_bull_slope']==True and row['ibs'] < 0.1)
		or (row['bop_div_bull_slope']==True and row['ibs'] < 0.1)		
		or (row['rsi4_hdiv_bull_slope']==True and row['ibs'] < 0.1)	
		or (row['spy_rsi2_wilder']<20 and row['spy_rsi3_wilder']<30 )		
		or (row['higher_highs_5']==True and row['ibs'] < 0.1)
		or ((row['tqqq_rsi10_wilder']<30)
		or (row['qqq_rsi10_wilder']<30)		
		or (row['xlp_rsi10_wilder']<30)		
		or (row['xlk_rsi10_wilder']<30)	
		or (row['soxl_rsi10_wilder']<30)	
		or (row['tecl_rsi10_wilder']<30)
		or (row['spy_rsi10_wilder']<30)) and row['ibs'] < 0.2,f"Entry 3800: IBS ({row['ibs']:.3f}) < 0.06 or price_above_supertrend_12_4 ({row['price_above_supertrend_12_4']}) and IBS < 0.1 or macdh_hdiv_bull_slope ({row['macdh_hdiv_bull_slope']}) and IBS({row['ibs']:.3f}) < 0.1 or ibs_hdiv_bull_slope ({row['ibs_hdiv_bull_slope']}) and spy_close/spy_sma_20({row['spy_close']/row['spy_sma_20']}) > 1 or ibs_10_div_bull_slope ({row['ibs_10_div_bull_slope']}) and IBS < 0.1 or bop_div_bull_slope ({row['bop_div_bull_slope']}) and IBS < 0.1 or rsi4_hdiv_bull_slope ({row['rsi4_hdiv_bull_slope']}) and IBS < 0.1 or spy_rsi2_wilder ({row['spy_rsi2_wilder']:.2f}) < 20 and spy_rsi3_wilder ({row['spy_rsi3_wilder']:.2f}) < 30 or higher_highs_5 ({row['higher_highs_5']}) and IBS < 0.1 or tqqq_rsi10_wilder ({row['tqqq_rsi10_wilder']:.2f}) < 30 or qqq_rsi10_wilder ({row['qqq_rsi10_wilder']:.2f}) < 30 or xlp_rsi10_wilder ({row['xlp_rsi10_wilder']:.2f}) < 30 or xlk_rsi10_wilder ({row['xlk_rsi10_wilder']:.2f}) < 30 or soxl_rsi10_wilder ({row['soxl_rsi10_wilder']:.2f}) < 30 or tecl_rsi10_wilder ({row['tecl_rsi10_wilder']:.2f}) < 30 or spy_rsi10_wilder ({row['spy_rsi10_wilder']:.2f}) < 30 and IBS < 0.2"),	#3800		

        3900:((row['tqqq_rsi10_adaptive']>79)
		or (row['qqq_rsi10_adaptive']>79)		
		or (row['xlp_rsi10_adaptive']>79)		
		or (row['spy_rsi10_adaptive']>79)
		or (row['tecl_rsi10_adaptive']>79),f"Entry 3900: RSI(10)TQQQ_adaptive ({row['tqqq_rsi10_adaptive']:.2f}) > 79 or RSI(10)QQQ_adaptive ({row['qqq_rsi10_adaptive']:.2f}) > 79 or RSI(10)XLP_adaptive ({row['xlp_rsi10_adaptive']:.2f}) > 79 or RSI(10)SPY_adaptive ({row['spy_rsi10_adaptive']:.2f}) > 79 or RSI(10)TECL_adaptive ({row['tecl_rsi10_adaptive']:.2f}) > 79"),#3900

                 #REI Test
        4000: (row['rei_2'] < -95 and (row['ibs'] < 0.1) and ((row['fdn_rsi20'] > row['xlu_rsi20'] and row['fdn_rsi20_shift1'] > row['xlu_rsi20_shift1']) or (row['spy_close']  > row['spy_sma_200'])
        or (row['ibs'] < 0.1) or (row['kmlm_close'] < row['kmlm_sma_20']) or (row['xlk_rsi10'] > row['kmlm_rsi10'] )
        or (row['uo_1,3,5'] < 20 and row['cmf_2'] < 0) or (row['uo_1,3,5'] < 10 and row['uo_1,3,5_shift1'] > 10)
        or (row['wr_2_shift1'] > -90 and row['wr_2'] < -90) ), f"Entry 4000: REI(2) ({row['rei_2']:.2f}) < -95 and IBS ({row['ibs']:.3f}) < 0.1 and (FDN_RSI20 ({row['fdn_rsi20']:.2f}) > XLU_RSI20 ({row['xlu_rsi20']:.2f}) and FDN_RSI20_shift1 ({row['fdn_rsi20_shift1']:.2f}) > XLU_RSI20_shift1 ({row['xlu_rsi20_shift1']:.2f}) OR SPY_close ({row['spy_close']:.2f}) > SPY_SMA200 ({row['spy_sma_200']:.2f}) OR IBS ({row['ibs']:.3f}) < 0.1 OR KMLM_close ({row['kmlm_close']:.2f}) < KMLM_SMA20 ({row['kmlm_sma_20']:.2f}) OR XLK_RSI10 ({row['xlk_rsi10']:.2f}) > KMLM_RSI10 ({row['kmlm_rsi10']:.2f}) OR UO_1,3,5 ({row['uo_1,3,5']:.2f}) < 20 and CMF_2 ({row['cmf_2']:.2f}) < 0 OR UO_1,3,5 ({row['uo_1,3,5']:.2f}) < 10 and UO_1,3,5_shift1 ({row['uo_1,3,5_shift1']:.2f}) > 10 OR WR_2_shift1 ({row['wr_2_shift1']:.2f}) > -90 and WR_2 ({row['wr_2']:.2f}) < -90)"),#4000

    }
    return conditions


def exit_conditions(row, ticker, holding_days=0, max_holding_days=0):
    """
    Defines exit conditions with condition IDs.
    Returns a dictionary mapping condition IDs to (boolean, reason) tuples.
    Now it accepts two additional parameters: holding_days and max_holding_days.

    """
    conditions = {
        7: (row['ibs'] > 0.75, f"Exit 7: IBS ({row['ibs']:.2f}) > 0.75 "),
        14: ((row['rsi_2'] > 90), f"Exit 14: RSI_2 ({row['rsi_2']:.2f}) >= 90"),
        33: ((row['ibs'] > 0.75)  or (row['rsi_5_bearish_divergence'] == True) or (row['rsi_5_hidden_bearish_divergence'] == True) , f"Exit 33: IBS ({row['ibs']:.2f}) > 0.5 or RSI Bearish Divergence is True or RSI Hidden Bearish Divergence is True "),
        62: (((row['tqqq_rsi10'] < 70) or (row['spxl_rsi10'] < 70) or not ((row['qqqe_rsi10'] > 79) or (row['vtv_rsi10'] > 79) or (row['vox_rsi10'] > 79) or (row['tecl_rsi10'] > 79) or 
                (row['xlp_rsi10'] > 79) or (row['voov_rsi10'] > 79) or (row['voog_rsi10'] > 79) or (row['tqqq_rsi10'] > 79) or (row['xly_rsi10'] > 79) or 
                (row['spy_rsi10'] > 79))) and ticker in ['UVXY','UVIX','VXX'], f"Exit 62: TQQQ_RSI10 ({row['tqqq_rsi10']:.2f}) or SPXL_RSI10 ({row['spxl_rsi10']:.2f}) < 70, "f"or no indicator RSI10 > 79: [QQQE: {row['qqqe_rsi10']:.2f}, VTV: {row['vtv_rsi10']:.2f}, "
            f"VOX: {row['vox_rsi10']:.2f}, TECL: {row['tecl_rsi10']:.2f}, XLP: {row['xlp_rsi10']:.2f}, "f"VOOV: {row['voov_rsi10']:.2f}, VOOG: {row['voog_rsi10']:.2f}, TQQQ: {row['tqqq_rsi10']:.2f}, "
            f"XLY: {row['xly_rsi10']:.2f}, SPY: {row['spy_rsi10']:.2f}]"),
        63: (((row['spxl_rsi10'] > 70) or (row['spy_rsi10'] > 50) or (row['spy_rsi10'] > 50)) and ticker in ['IBIT','MSTR','MSTX','GBTC','BITX','STCE'],f"Exit 63: SPXL RSI(10) ({row['spxl_rsi10']:.2f}) >70 or SPY RSI(10) ({row['spy_rsi10']:.2f}) > 50 or TQQQ RSI(10) ({row['tqqq_rsi10']:.2f}) >70"),
        64: (((row['tqqq_rsi10'] < 30) or (row['spy_rsi10'] < 30) or (row['tqqq_close'] > row['tqqq_sma_20']) or (row['spy_close'] > row['spy_sma_200']) or (row['uvxy_rsi10'] > 74)),
            (f"Exit 64: TQQQ RSI(10) ({row['tqqq_rsi10']:.2f}) < 30 or "f"SPY RSI(10) ({row['spy_rsi10']:.2f}) < 30 or "f"TQQQ_Close ({row['tqqq_close']:.2f}) > TQQQ_SMA(20) ({row['tqqq_sma_20']:.2f}) or "
                f"SPY_Close ({row['spy_close']:.2f}) > SMA(200)@SPY ({row['spy_sma_200']:.2f}) or "f"UVXY RSI(10) ({row['uvxy_rsi10']:.2f}) > 74")),
        77: ( holding_days >= 3 and ticker in ['UVXY','UVIX','SVXY','SVIX'],f"Exit 77: {ticker} forced exit after 1 day (holding_days={holding_days})"),

        81: (row['spy_rsi2'] >10, f"Exit 81: RSI(2)@SPY ({row['spy_rsi2']:.2f}) >10"),
        85: (row['bil_rsi5'] <row['ief_rsi5'] and row['spy_rsi5'] < 75, f"Exit 85: RSI(5)@BIL ({row['bil_rsi5']:.2f}) < RSI(5)@BIL({row['ief_rsi5']:.2f}) and RSI(5)@SPY({row['spy_rsi5']:.2f})<75 "),
        86: (row['bil_rsi5'] >row['ief_rsi5'] and row['spib_rsi10'] > row['sphb_rsi10'] and row['soxx_rsi8'] > row['iyw_rsi8'], f"Exit 86: RSI(5)@BIL ({row['bil_rsi5']:.2f}) > RSI(5)@IEF({row['ief_rsi5']:.2f}) and RSI(5)@SPIB({row['spib_rsi10']:.2f})>RSI(5)@SPHB({row['sphb_rsi10']:.2f}) and  RSI(8)@SOXX({row['soxx_rsi8']:.2f})>RSI(8)@IYW({row['iyw_rsi8']:.2f})"),
        87: (row['bil_rsi5'] >row['ief_rsi5'] and row['spib_rsi10'] > row['sphb_rsi10'] and row['soxx_rsi8'] < row['iyw_rsi8'], f"Exit 87: RSI(5)@BIL ({row['bil_rsi5']:.2f}) > RSI(5)@IEF({row['ief_rsi5']:.2f}) and RSI(5)@SPIB({row['spib_rsi10']:.2f})>RSI(5)@SPHB({row['sphb_rsi10']:.2f}) and  RSI(8)@SOXX({row['soxx_rsi8']:.2f})<RSI(8)@IYW({row['iyw_rsi8']:.2f})"),
        88: (row['bil_rsi5'] >row['ief_rsi5'] and row['spib_rsi10'] < row['sphb_rsi10'] , f"Exit 88: RSI(5)@BIL ({row['bil_rsi5']:.2f}) > RSI(5)@IEF({row['ief_rsi5']:.2f}) and RSI(5)@SPIB({row['spib_rsi10']:.2f})<RSI(5)@SPHB({row['sphb_rsi10']:.2f}) and  RSI(10)@SOXX({row['soxx_rsi10']:.2f})<RSI(10)@IYW({row['iyw_rsi10']:.2f})"),
        91: ( holding_days >= 1 and ticker in ['UVXY','UVIX','SVXY','SVIX'],f"Exit 91: {ticker} forced exit after 1 day (holding_days={holding_days})"),
     
        130: (row['tqqq_rsi10'] > 80, f"Entry 130: RSI(10)TQQQ ({row['tqqq_rsi10']:.2f}) > 80"),
        131: (row['tqqq_rsi10'] < 80 and row['tqqq_rsi10'] > 30, f"Exit 131: RSI(10)QQQ ({row['tqqq_rsi10']:.2f}) < 80 and RSI(10)QQQ ({row['tqqq_rsi10']:.2f}) < 30"),
        132: ( row['tqqq_rsi10'] < 30,f"Exit 132: RSI(10)QQQ ({row['tqqq_rsi10']:.2f}) > 30"),
        490:((row['tqqq_close'] > row['tqqq_sma_200']) and (row['tqqq_rsi10'] > 79 or row['qqq_rsi10'] > 79 or row['qqqe_rsi10'] > 79 or row['soxl_rsi10'] > 79 
            or row['spxl_rsi10'] > 79 or row['tecl_rsi10'] > 79 or row['voog_rsi10'] > 79 or row['voov_rsi10'] > 79 or row['vox_rsi10'] > 79 or row['vtv_rsi10'] > 79 or row['xlf_rsi10'] > 79 
            or row['xlp_rsi10'] > 79 or row['xli_rsi10'] > 79 or row['xlk_rsi10'] > 79 or row['xly_rsi10'] > 79 or row['xlu_rsi10'] > 79 or row['xtn_rsi10'] > 79), 
            f"Exit 490: At least one RSI_10 is above 79. This suggests potential overbought conditions for one or more tickers, which may indicate a possible reversal or mean reversion. RSI above 79 often signals that the asset's price is extended to the upside and may need to consolidate or pull back."),
        500: ((row['tqqq_close'] > row['tqqq_sma_200']) and row['tqqq_rsi10'] < 79, 
            f"Exit 500: TQQQ_Close ({row['tqqq_close']:.2f}) > SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}) and RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) < 79"),
        510: ((row['tqqq_close'] < row['tqqq_sma_200']) and row['tqqq_rsi10'] < 31, 
            f"Exit 510: TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}) and RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) < 31"),
        520: ((row['tqqq_close'] < row['tqqq_sma_200']) and (row['tqqq_rsi10'] > 31 and row['soxl_rsi10'] < 30), 
            f"Exit 520: TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}), RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) > 31, and RSI(10)@SOXL ({row['soxl_rsi10']:.2f}) < 30"),
        530: ((row['tqqq_close'] < row['tqqq_sma_200']) and (row['tqqq_rsi10'] > 31 and row['soxl_rsi10'] > 30) and (row['tqqq_close'] < row['tqqq_sma_20']), 
            f"Exit 530: TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}), RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) > 31, RSI(10)@SOXL ({row['soxl_rsi10']:.2f}) < 30, and TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(20)@TQQQ ({row['tqqq_sma_20']:.2f})"),
        540: ((row['tqqq_close'] < row['tqqq_sma_200']) and (row['tqqq_rsi10'] > 31 and row['soxl_rsi10'] > 30) and (row['tqqq_close'] > row['tqqq_sma_20']), 
            f"Exit 540: TQQQ_Close ({row['tqqq_close']:.2f}) < SMA(200)@TQQQ ({row['tqqq_sma_200']:.2f}), RSI(10)@TQQQ ({row['tqqq_rsi10']:.2f}) > 31, RSI(10)@SOXL ({row['soxl_rsi10']:.2f}) < 30, and TQQQ_Close ({row['tqqq_close']:.2f}) > SMA(20)@TQQQ ({row['tqqq_sma_20']:.2f})"),
        2000: (row['spy_rsi10'] < 79, 
            f"Exit 20: RSI(10)@SPY ({row['spy_rsi10']:.2f}) < 79)"),
           }
    return conditions


# ----------------- Alpaca/Schwab Trading Integration -----------------
def equity_buy_market(symbol, quantity):
    order = {
        "orderType": "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol,
                    "assetType": "EQUITY"
                }
            }
        ]
    }
    return order

def equity_sell_market(symbol, quantity):
    order = {
        "orderType": "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": "SELL",
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol,
                    "assetType": "EQUITY"
                }
            }
        ]
    }
    return order

def place_order_threadsafe(symbol, qty, side, account_name, price=None, max_retries=10):
    """
    Places a buy or sell order via Alpaca or Schwab API.
    In backtest mode, simulates the order.
    Returns the order ID and submission time if successful, else (None, None).
    """
    logger = loggers[account_name]
    if mode == 'backtest':
        # Simulate
        order_submission_time = datetime.now()
        order_id = f"simulated_order_{symbol}_{side}_{order_submission_time.strftime('%Y%m%d%H%M%S')}"
        logger.info(f"Simulated {side.capitalize()} order for {qty} shares of {symbol} at ${price:.2f}.")
        print(f"Simulated {side.capitalize()} order for {qty} shares of {symbol} at ${price:.2f}.")
        return order_id, order_submission_time

    # LIVE mode
    api = api_clients[account_name]
    account_info = next(acc for acc in accounts_info if acc['account_name'] == account_name)

    if account_info['broker'] == 'Alpaca':
        for attempt in range(max_retries):
            try:
                qty_str = str(int(qty))
                order_submission_time = datetime.now()
                logger.info(f"Attempting to place {side.upper()} order for {qty_str} shares of {symbol}. Attempt {attempt+1}/{max_retries}")
                print(f"Attempting to place {side.upper()} order for {qty_str} shares of {symbol}. Attempt {attempt+1}/{max_retries}")
                if side.lower() == 'buy':
                    order = api.submit_order(
                        symbol=symbol,
                        qty=qty_str,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                elif side.lower() == 'sell':
                    order = api.submit_order(
                        symbol=symbol,
                        qty=qty_str,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                else:
                    raise ValueError("Invalid order side. Use 'buy' or 'sell'.")

                logger.info(f"Placed {side.upper()} order for {qty} shares of {symbol}. Order ID: {order.id}")
                return order.id, order_submission_time

            except tradeapi.rest.APIError as api_error:
                logger.error(f"APIError placing order for {symbol}: {api_error}. Attempt {attempt + 1}/{max_retries}")
                print(f"APIError placing order for {symbol}: {api_error}. Attempt {attempt + 1}/{max_retries}")
                error_message = str(api_error).lower()
                if 'insufficient day trading buying power' in error_message:
                    logger.error(f"Insufficient day trading buying power for {symbol}. Attempt {attempt + 1}/{max_retries}")
                    return None, None
            except Exception as e:
                logger.error(f"Error placing order for {symbol}: {e}. Attempt {attempt + 1}/{max_retries}")
                print(f"Error placing order for {symbol}: {e}. Attempt {attempt + 1}/{max_retries}")
            time.sleep(1)
        logger.error(f"Failed to place order for {symbol} after {max_retries} attempts.")
        return None, None

    elif account_info['broker'] == 'Schwab':
        for attempt in range(max_retries):
            try:
                qty = int(qty)
                if side.lower() == 'buy':
                    order_spec = equity_buy_market(symbol, qty)
                elif side.lower() == 'sell':
                    order_spec = equity_sell_market(symbol, qty)
                else:
                    raise ValueError("Invalid order side. Use 'buy' or 'sell'.")

                schwab_response = api.order_place(account_info['schwab_account_hash'], order_spec)
                if schwab_response.status_code in [200, 201]:
                    order_url = schwab_response.headers.get('Location', None)
                    if order_url:
                        order_id = order_url.split('/')[-1]
                    else:
                        order_id = None
                    logger.info(f"Schwab order placed: {order_id}")
                    print(f"Placed {side.upper()} order for {qty} shares of {symbol} via Schwab.")
                    return order_id, datetime.now()
                else:
                    error_message = schwab_response.text
                    logger.error(f"Error placing Schwab order: {error_message}. Attempt {attempt + 1}/{max_retries}")
                    print(f"Error placing Schwab order: {error_message}. Attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                logger.error(f"Exception placing Schwab order: {e}. Attempt {attempt + 1}/{max_retries}")
                print(f"Exception placing Schwab order: {e}. Attempt {attempt + 1}/{max_retries}")
            time.sleep(1)
        logger.error(f"Failed to place order for {symbol} after {max_retries} attempts.")
        return None, None

def get_account_balance(account_name):
    """
    Retrieves the current account balance.
    Returns the balance as a float.
    """
    global portfolio_value
    logger = loggers[account_name]
    if mode == 'backtest':  # Simulate balance in backtest mode
        return portfolio_value
    api = api_clients[account_name]
    account_info = next(acc for acc in accounts_info if acc['account_name'] == account_name)
    if account_info['broker'] == 'Alpaca':
        try:
            account = api.get_account()
            balance = float(account.equity)
            logger.info(f"Current Alpaca account balance: ${balance:.2f}")
            return balance
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            print(f"Error fetching account balance: {e}")
            return 0.0
    elif account_info['broker'] == 'Schwab':
        try:
            account_details_response = api.account_details(account_info['schwab_account_hash'])
            account_data = account_details_response.json()
            # Extract the equity from account details
            balance = float(account_data['securitiesAccount']['currentBalances']['liquidationValue'])
            logger.info(f"Current Schwab account balance: ${balance:.2f}")
            return balance
        except Exception as e:
            logger.error(f"Error fetching Schwab account balance: {e}")
            print(f"Error fetching account balance: {e}")
            return 0.0

def calculate_quantity(price, strategy_cash, positions_to_fill):
    """
    Calculates the number of shares to buy based on the strategy's allocated cash and the stock price.
    Ensures that the quantity does not exceed the number of positions to fill.
    """
    if positions_to_fill <= 0:
        logging.info("No positions to fill.")
        return 0
    
    if pd.isna(strategy_cash) or pd.isna(price) or price <= 0:
        logging.warning(f"Cannot calculate quantity because amount_per_position or price is NaN. amount_per_position={amount_per_position}, price={price}")
        return 0
    
    amount_per_position = strategy_cash / positions_to_fill

    if pd.isna(amount_per_position):  # Keep this for extra safety
        logging.warning(f"amount_per_position is NaN: {amount_per_position}")
        return 0

    qty = int(amount_per_position // price)
    logging.info(f"Calculating quantity: strategy_cash=${strategy_cash:.2f}, positions_to_fill={positions_to_fill}, amount_per_position=${amount_per_position:.2f}, price=${price:.2f}, qty={qty}")
    if qty == 0 and amount_per_position >= price:
        qty = 1
        logging.info(f"Adjusted quantity to 1 because amount_per_position >= price")
    return qty

def sync_with_broker(account_name):
    """
    Synchronizes current positions with the broker and retrieves trade history from MongoDB (live mode)
    or returns empty positions for backtest mode.
    Returns a dictionary of current positions {strategy_name: {ticker: position_dict}}
    """
    logger = loggers[account_name]
    if mode == 'backtest':
        # For backtesting, retain the original behavior: return empty dicts
        return {strategy.name: {} for strategy in strategies}

    # For live mode, we proceed below
    positions_dict = {strategy.name: {} for strategy in strategies}

    # Initialize broker_positions_dict
    broker_positions_dict = {}

    # Fetch positions from the broker (unchanged)
    api = api_clients[account_name]
    account_info = next(acc for acc in accounts_info if acc['account_name'] == account_name)
    if account_info['broker'] == 'Alpaca':
        try:
            alpaca_positions = api.list_positions()
            broker_positions_dict = {position.symbol: int(float(position.qty)) for position in alpaca_positions}
            logger.info("Successfully synced positions from Alpaca.")
        except Exception as e:
            logger.error(f"Error syncing with Alpaca: {e}")
            print(f"Error syncing with Alpaca: {e}")
            broker_positions_dict = {}
    elif account_info['broker'] == 'Schwab':
        try:
            response = api.account_details(account_info['schwab_account_hash'], fields='positions')
            account_data = response.json()
            positions = account_data['securitiesAccount'].get('positions', [])
            broker_positions_dict = {}
            for position in positions:
                symbol = position['instrument']['symbol']
                qty = int(position.get('longQuantity', 0)) - int(position.get('shortQuantity', 0))
                broker_positions_dict[symbol] = qty
            logger.info("Successfully synced positions from Schwab.")
        except Exception as e:
            logger.error(f"Error syncing with Schwab: {e}")
            print(f"Error syncing with Schwab: {e}")
            broker_positions_dict = {}

    # Now instead of reading from CSV, we fetch trades from MongoDB
    if mode == 'live_trade':
        try:
            # Fetch all trades from the MongoDB collection
            # Assuming these include both entries and exits
            all_trades = list(trades_collections[account_name].find({}))

            if not all_trades:
                # No trades recorded yet
                print("No trades found in MongoDB for live mode.")
                print(f"Current Positions: {positions_dict}")
                return positions_dict

            trades_df = pd.DataFrame(all_trades)

            # Ensure we have the columns expected. If some columns are missing, handle accordingly
            # Required columns: 'Trade Type', 'Strategy', 'Ticker', 'Date', 'Price', 'Quantity', 'Reason', 'Order ID'
            for col in ['Trade Type', 'Strategy', 'Ticker', 'Date', 'Price', 'Quantity', 'Reason', 'Order ID']:
                if col not in trades_df.columns:
                    trades_df[col] = None  # Fill missing columns with None if necessary

            # Convert 'Date' column to datetime
            if 'Date' in trades_df.columns and not trades_df.empty:
                trades_df['Date'] = pd.to_datetime(trades_df['Date'])

            # Identify open positions: entries without corresponding exits
            for strategy_name in trades_df['Strategy'].dropna().unique():
                strategy_trades = trades_df[trades_df['Strategy'] == strategy_name]
                entries = strategy_trades[strategy_trades['Trade Type'] == 'Entry']
                exits = strategy_trades[strategy_trades['Trade Type'] == 'Exit']

                # Calculate net positions per ticker
                for ticker in entries['Ticker'].unique():
                    entry_qty = entries[entries['Ticker'] == ticker]['Quantity'].sum()
                    exit_qty = exits[exits['Ticker'] == ticker]['Quantity'].sum() if not exits.empty else 0
                    net_qty = entry_qty - exit_qty

                    if net_qty != 0:
                        # Get the last entry details for this ticker in this strategy
                        ticker_entries = entries[entries['Ticker'] == ticker].sort_values('Date')
                        last_entry = ticker_entries.iloc[-1]

                        position_details = {
                            'qty': net_qty,
                            'entry_price': float(last_entry['Price']),
                            'entry_date': pd.to_datetime(last_entry['Date']),
                            'entry_reason': last_entry['Reason'],
                            'entry_order_id': last_entry['Order ID']
                        }
                        positions_dict[strategy_name][ticker] = position_details

            # Cross-validate with broker positions
            broker_total_positions = sum(broker_positions_dict.values())
            local_total_positions = sum(
                sum(strategy_positions[ticker]['qty'] for ticker in strategy_positions)
                for strategy_positions in positions_dict.values()
            )

            if broker_total_positions != local_total_positions:
                logger.warning(f"Mismatch between broker positions ({broker_total_positions}) and local positions ({local_total_positions}).")
                print(f"Warning: Mismatch between broker positions ({broker_total_positions}) and local positions ({local_total_positions}).")
            else:
                logger.info("Broker positions and local positions are in sync.")

            logger.info("Successfully synced positions from MongoDB.")

        except Exception as e:
            logger.error(f"Error reading trades from MongoDB: {e}")
            print(f"Error reading trades from MongoDB: {e}")

    print(f"Current Positions: {positions_dict}")
    return positions_dict

# ---------------- Performance Metrics Calculation -----------------
def calculate_metrics(portfolio_series):
    """
    Calculates Sharpe and Sortino ratios for the portfolio.
    """
    daily_returns = portfolio_series.pct_change().dropna()

    # Sharpe Ratio
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252) if downside_returns.std() != 0 else np.inf

    return sharpe_ratio, sortino_ratio

# ----------------- Backtesting Execution -----------------
import concurrent.futures

def backtest_strategies(stock_data, strategies, mode='backtest', initial_capital=100000):
    """
    Executes strategies in backtest mode.
    """
    logging.info("Starting backtest execution.")
    portfolio_cash = initial_capital
    positions = {strategy.name: {} for strategy in strategies}
    strategy_portfolios = {strategy.name: [] for strategy in strategies}
    strategy_dates = {strategy.name: [] for strategy in strategies}
    total_portfolio = []
    portfolio_dates = []

    # Get all unique dates from stock data
    all_dates = set()
    for ticker in stock_data:
        all_dates.update(stock_data[ticker].index)
    sorted_dates = sorted(all_dates)

    # Print all dates being processed
    print("Dates being processed in backtest:")
    for date in sorted_dates:
        print(date)

    for date in sorted_dates:
        # Step 1: Calculate total positions value and total portfolio value
        total_positions_value = 0
        strategy_total_values = {}
        for strategy in strategies:
            strategy_positions = positions[strategy.name]
            strategy_positions_value = 0
            for ticker, pos in strategy_positions.items():
                # Ensure ticker has data
                if ticker in stock_data:
                    if date in stock_data[ticker].index:
                        current_price = stock_data[ticker]['Close'].loc[date]
                        if pd.isna(current_price):
                            # Attempt to use last available price
                            last_available_date = stock_data[ticker].index[stock_data[ticker].index < date]
                            if not last_available_date.empty:
                                last_date = last_available_date[-1]
                                current_price = stock_data[ticker]['Close'].loc[last_date]
                                if pd.isna(current_price):
                                    print(f"All prices are NaN for {ticker}, cannot calculate position value.")
                                    continue  # Skip this position
                            else:
                                print(f"No available price data for {ticker} before {date}, cannot calculate position value.")
                                continue  # Skip this position
                    else:
                        # Use last available price
                        last_available_date = stock_data[ticker].index[stock_data[ticker].index < date]
                        if not last_available_date.empty:
                            last_date = last_available_date[-1]
                            current_price = stock_data[ticker]['Close'].loc[last_date]
                            if pd.isna(current_price):
                                print(f"All prices are NaN for {ticker}, cannot calculate position value.")
                                continue  # Skip this position
                        else:
                            print(f"No available price data for {ticker} before {date}, cannot calculate position value.")
                            continue  # Skip this position
                    position_value = pos['qty'] * current_price
                    strategy_positions_value += position_value
                else:
                    print(f"{ticker} not in stock data.")
            strategy_total_values[strategy.name] = strategy_positions_value
            total_positions_value += strategy_positions_value

        total_portfolio_value = total_positions_value + portfolio_cash
        total_portfolio.append(total_portfolio_value)
        portfolio_dates.append(date)

        # Step 2: Calculate each strategy's cash allocation
        strategy_cash = {}
        for strategy in strategies:
            strategy_cash[strategy.name] = total_portfolio_value * strategy.equity_fraction

        # Initialize a list to store cash adjustments for strategies
        strategy_cash_adjustments = {strategy.name: 0 for strategy in strategies}

        # Step 3: Process exits for all strategies
        for strategy in strategies:
            strategy_positions = positions[strategy.name]
            strategy_positions_value = strategy_total_values[strategy.name]
            strategy_allocated_cash = strategy_cash[strategy.name] - strategy_positions_value

            exits = strategy.check_exit(stock_data, strategy_positions, date)
            
            # Process exits to group reasons by ticker
            exits_dict = {}
            for ticker, reason in exits:
                if ticker in exits_dict:
                    exits_dict[ticker].append(reason)
                else:
                    exits_dict[ticker] = [reason]

            # Now process each ticker only once
            for ticker, reasons in exits_dict.items():
                combined_reason = "; ".join(reasons)
                print(f"{date.date()} - {strategy.name} - Exiting {ticker}: {combined_reason}")

                position = strategy_positions[ticker]
                qty = position['qty']
                entry_price = position['entry_price']
                entry_date = position['entry_date']
                entry_reason = position['entry_reason']
                entry_order_id = position['entry_order_id']

                exit_price = stock_data[ticker]['Close'].loc[date]
                exit_order_id = f"exit_order_{ticker}_{strategy.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                trade_return = (exit_price / entry_price) - 1
                profit_amount = qty * (exit_price - entry_price)

                # --- Calculate MAE% and MFE% ---
                exit_date = date  # Fix: define exit_date to the current 'date'
                
                # 1) Get data from entry_date *through* exit_date
                trade_period_data = stock_data[ticker].loc[entry_date:exit_date].copy()
                trade_period_data.sort_index(inplace=True)

                # 2) Exclude the entry row itself (the 0th row),
                #    so calculations reflect only post-entry price action
                if not trade_period_data.empty:
                    trade_period_data = trade_period_data.iloc[1:]  # Drops the entry date row

                # 3) Compute lowest low and highest high from POST-entry to exit_date
                if not trade_period_data.empty:
                    lowest_low = trade_period_data['Low'].min()
                    highest_high = trade_period_data['High'].max()
                else:
                    # If there's no data after the entry day (e.g. same-day exit),
                    # define these in a way that yields 0 for MAE/MFE
                    lowest_low = entry_price
                    highest_high = entry_price

                # 4) Compute MAE% and MFE% relative to the entry price
                if entry_price != 0:
                    MAE = ((entry_price - lowest_low) / entry_price) * 100
                    MFE = ((highest_high - entry_price) / entry_price) * 100
                else:
                    MAE = 0.0
                    MFE = 0.0

                # Prepare trade_details and log them
                trade_details = {
                    'Strategy': strategy.name,
                    'Ticker': ticker,
                    'Quantity': qty,
                    'Entry Date': entry_date.strftime("%Y-%m-%d"),
                    'Entry Price': entry_price,
                    'Entry Reason': entry_reason,
                    'Entry Order ID': entry_order_id,
                    'Exit Date': date.strftime("%Y-%m-%d"),
                    'Exit Price': exit_price,
                    'Exit Reason': combined_reason,
                    'Exit Order ID': exit_order_id,
                    'Profit $': profit_amount,
                    'Profit %': trade_return * 100,
                    'Entry Amount $': qty * entry_price,
                    'Exit Amount $': qty * exit_price,
                    'MAE %': MAE,
                    'MFE %': MFE,
                }

                log_completed_trade(trade_details, 'backtest')  # or whichever logging function

                # Update portfolio cash and strategy cash
                proceeds = qty * exit_price
                portfolio_cash += proceeds
                strategy_positions_value -= qty * exit_price
                strategy_allocated_cash = strategy_cash[strategy.name] - strategy_positions_value

                # Remove the position
                del strategy_positions[ticker]

                # Update positions and cash adjustments
                positions[strategy.name] = strategy_positions
                strategy_cash_adjustments[strategy.name] += proceeds

        # Step 4: Process entries for all strategies
        for strategy in strategies:
            strategy_positions = positions[strategy.name]
            strategy_positions_value = 0
            for ticker, pos in strategy_positions.items():
                if date in stock_data[ticker].index:
                    current_price = stock_data[ticker]['Close'].loc[date]
                    position_value = pos['qty'] * current_price
                    strategy_positions_value += position_value
            strategy_allocated_cash = strategy_cash[strategy.name] - strategy_positions_value

            entries = strategy.check_entry(stock_data, date)
            current_positions_count = len(strategy_positions)
            available_positions = strategy.max_symbols - current_positions_count

            filtered_entries = []
            for ticker, reason in entries:
                already_entered = ticker in strategy_positions
                if strategy.allow_same_day_reentry:
                    if not already_entered:
                        filtered_entries.append((ticker, reason))
                else:
                    if not already_entered:
                        filtered_entries.append((ticker, reason))
            logging.debug(f"Filtered entries for {strategy.name} after position check: {[(ticker, reason) for ticker, reason in filtered_entries]}")

            entries_to_process = filtered_entries[:available_positions]

            for ticker, reason in entries_to_process:
                if date in stock_data[ticker].index:
                    price = stock_data[ticker]['Close'].loc[date]
                    if pd.isna(price):
                        print(f"{date.date()} - {strategy.name} - Price data is NaN for {ticker}, skipping.")
                        continue  # Skip this ticker
                    qty = calculate_quantity(price, strategy_allocated_cash, available_positions)
                    if qty > 0:
                        cost = qty * price
                        if cost <= strategy_allocated_cash and cost <= portfolio_cash:
                            # Update portfolio cash and strategy cash
                            portfolio_cash -= cost
                            strategy_positions_value += cost
                            strategy_allocated_cash = strategy_cash[strategy.name] - strategy_positions_value

                            # Record the entry
                            entry_order_id = f"entry_order_{ticker}_{strategy.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            strategy_positions[ticker] = {
                                'qty': qty,
                                'entry_price': price,
                                'entry_date': date,
                                'entry_reason': reason,
                                'entry_order_id': entry_order_id
                            }
                            print(f"{date.date()} - {strategy.name} - Entering {ticker}: {reason} with qty {qty}")

                            # Update positions and cash adjustments
                            positions[strategy.name] = strategy_positions
                            strategy_cash_adjustments[strategy.name] -= cost
                        else:
                            print(f"{date.date()} - {strategy.name} - Not enough cash to enter {ticker}")
                    else:
                        print(f"{date.date()} - {strategy.name} - Quantity calculated is zero for {ticker}")
                else:
                    print(f"{date.date()} - {strategy.name} - Data not available for {ticker}")

        # Step 5: Update total portfolio value after all trades
        total_positions_value = 0
        for strategy in strategies:
            strategy_positions_value = 0
            for ticker, pos in positions[strategy.name].items():
                if date in stock_data[ticker].index:
                    current_price = stock_data[ticker]['Close'].loc[date]
                    position_value = pos['qty'] * current_price
                    strategy_positions_value += position_value
                else:
                    # Handle missing data
                    last_available_date = stock_data[ticker].index[stock_data[ticker].index < date]
                    if not last_available_date.empty:
                        last_date = last_available_date[-1]
                        current_price = stock_data[ticker]['Close'].loc[last_date]
                        position_value = pos['qty'] * current_price
                        strategy_positions_value += position_value
            strategy_total_values[strategy.name] = strategy_positions_value
            total_positions_value += strategy_positions_value

            # Append to strategy portfolios
            strategy_portfolios[strategy.name].append(strategy_positions_value + (strategy_cash[strategy.name] - strategy_positions_value))
            strategy_dates[strategy.name].append(date)

        total_portfolio_value = total_positions_value + portfolio_cash
        total_portfolio[-1] = total_portfolio_value  # Update the last recorded total portfolio value

        # Optionally, log the total portfolio value
        print(f"{date.date()} - Total Portfolio Value: ${total_portfolio_value:,.2f}")

    # Final portfolio and strategy performance
    print(f"\nFinal Combined Portfolio Value: ${total_portfolio_value:,.2f}")
    # Close any open positions at the end of the backtest period
    last_date = sorted_dates[-1]

    print("\nProcessing remaining open positions at the end of backtest period...")
    for strategy in strategies:
        strategy_positions = positions[strategy.name]
        if strategy_positions:
            for ticker, pos in list(strategy_positions.items()):
                # Close the position
                if ticker in stock_data_with_indicators and last_date in stock_data_with_indicators[ticker].index:
                    exit_price = stock_data_with_indicators[ticker]['Close'].loc[last_date]
                    qty = pos['qty']
                    entry_price = pos['entry_price']
                    entry_date = pos['entry_date']
                    entry_reason = pos['entry_reason']
                    entry_order_id = pos['entry_order_id']
                    exit_order_id = f"forced_exit_{ticker}_{strategy.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    trade_return = (exit_price / entry_price) - 1
                    profit_amount = qty * (exit_price - entry_price)

                    # Prepare trade details
                    trade_details = {
                        'Strategy': strategy.name,
                        'Ticker': ticker,
                        'Quantity': qty,
                        'Entry Date': entry_date.strftime("%Y-%m-%d"),
                        'Entry Price': entry_price,
                        'Entry Reason': entry_reason,
                        'Entry Order ID': entry_order_id,
                        'Exit Date': last_date.strftime("%Y-%m-%d"),
                        'Exit Price': exit_price,
                        'Exit Reason': 'period end',
                        'Exit Order ID': exit_order_id,
                        'Profit $': profit_amount,
                        'Profit %': trade_return * 100,
                        'Entry Amount $': qty * entry_price,
                        'Exit Amount $': qty * exit_price,
                    }

                    # Log the completed trade
                    log_completed_trade(trade_details, 'backtest')

                    # Update portfolio cash and positions
                    proceeds = qty * exit_price
                    portfolio_cash += proceeds
                    # Remove the position
                    del strategy_positions[ticker]
                    # Update positions
                    positions[strategy.name] = strategy_positions
                    print(f"Closed position for {ticker} at ${exit_price:.2f} on {last_date.date()}, reason: period end.")
                else:
                    print(f"No price data for {ticker} on {last_date.date()}, cannot close position.")

    # Update total_positions_value and total_portfolio_value
    total_positions_value = 0  # All positions are closed
    total_portfolio_value = portfolio_cash
    total_portfolio[-1] = total_portfolio_value  # Update the last recorded total portfolio value


    # Calculate performance metrics for each strategy
    for strategy in strategies:
        strategy_series = pd.Series(strategy_portfolios[strategy.name], index=strategy_dates[strategy.name])
        sharpe_ratio, sortino_ratio = calculate_metrics(strategy_series)
        trade_log_file = 'backtest_log.csv'
        if os.path.isfile(trade_log_file):
            trade_df = pd.read_csv(trade_log_file)
            strategy_trades = trade_df[trade_df['Strategy'] == strategy.name]
            profitable_trades = len(strategy_trades[strategy_trades['Profit $'] > 0])
            total_trades = len(strategy_trades)
            percentage_profitable = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        else:
            percentage_profitable = 0

        print(f"\nStrategy: {strategy.name}")
        if not strategy_series.empty:
            print(f"Final Strategy Value: ${strategy_series.iloc[-1]:,.2f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Sortino Ratio: {sortino_ratio:.2f}")
            print(f"Percentage of Profitable Trades: {percentage_profitable:.2f}%")
            # Plot strategy equity curve
            plt.plot(strategy_series, label=f"{strategy.name} Equity Curve")
        else:
            print(f"Final Strategy Value: No data available for {strategy.name}")

    # Combined portfolio metrics
    portfolio_series = pd.Series(total_portfolio, index=pd.to_datetime(portfolio_dates))
    sharpe_ratio, sortino_ratio = calculate_metrics(portfolio_series)
    plt.plot(portfolio_series, label="Combined Portfolio")

    print(f"\nFinal Combined Portfolio Value: ${portfolio_series.iloc[-1]:,.2f}")
    print(f"Combined Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Combined Portfolio Sortino Ratio: {sortino_ratio:.2f}")

    # Generate QuantStats report
    portfolio_returns = portfolio_series.pct_change().dropna()
    benchmark_prices = stock_data_with_indicators[BENCHMARK_TICKER]['Close'].loc[portfolio_series.index]
    benchmark_returns = benchmark_prices.pct_change().dropna()
    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')

    qs.reports.html(
        portfolio_returns,
        benchmark=benchmark_returns,
        output='quantstats_report.html',
        title='Backtest Report'
    )

    print(f"\nQuantStats report generated: quantstats_report.html")

    # Show plot of all equity curves
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.show()



# ------------------ Define Strategies ------------------

strategy_2 = Strategy(
    name="Holy_Grail",
    max_symbols=1,
    sort_by='1/natr_14',
    sort_order='nsmallest',
    equity_fraction=0.24,#0.15
    allow_same_day_reentry=True,
    atr_multiplier=2,
    percent_exit=1.05,
    max_holding_days=7,
    entry_conditions_1=[11,], # Holy Grail entry based on RSI
    Ticker_Entry_1=['UVXY','UVIX'],
    entry_conditions_2=[12,], # TQQQ entry based on TQQQ RSI and SMA
    Ticker_Entry_2=['TQQQ' ],#['TQQQ','FNGU','FNGA']
    entry_conditions_3=[13,], # TECL entry based on TQQQ RSI and SMA
    Ticker_Entry_3=['TECL'],
    entry_conditions_4=[14,], # SOXL entry based on TQQQ and SOXL RSI
    Ticker_Entry_4=['SOXL'],
    entry_conditions_5=(15,), # SQQQ entry based on TQQQ RSI and SMA
    Ticker_Entry_5=['SQQQ','BSV'],
    entry_conditions_6=(16,), # TQQQ entry based on TQQQ RSI and SMA
    Ticker_Entry_6=['TQQQ'],
    exit_conditions_1=[490,7],
    Ticker_Exit_1=['TECL','SOXL','TQQQ','SQQQ','BSV'],
    exit_conditions_2=[500,7],
    Ticker_Exit_2=['TECL','SOXL','SQQQ','BSV','UVXY','UVIX','FNGU'],
    exit_conditions_3=[510,7],
    Ticker_Exit_3=['TQQQ','SOXL','SQQQ','BSV','UVXY','UVIX','FNGU'],
    exit_conditions_4=[520,7],
    Ticker_Exit_4=['TQQQ','TECL','SQQQ','BSV','UVXY','UVIX','FNGU'],
    exit_conditions_5=[530,7],
    Ticker_Exit_5=['TQQQ','TECL','SOXL' ,'UVXY','UVIX','FNGU'],
    exit_conditions_6=[540,7],
    Ticker_Exit_6=['TECL','SOXL','SQQQ','BSV','UVXY','UVIX','FNGU'],
    exit_conditions_7=[541,],
    Ticker_Exit_7=['SQQQ','BSV'],
    entry_conditions_9=(114,115,116,117),  
    Ticker_Entry_9=['SVXY','SVIX'],
    exit_conditions_9=(81,77),
    Ticker_Exit_9=['SVXY','SVIX'],

)




# List of strategies
strategies = [strategy_2]

# ========== Configure Start/End for Historical Data ==========
# If you set START_DATE to a future date, many providers return empty data.
# Adjust this to ensure your data provider can give you actual historical data.
START_DATE = "2024-07-31"
END_DATE   = dt.now().strftime("%Y-%m-%d")

# ========== Time Window for Data Refresh ==========
# Example window: from 09:58:45 ET to 10:00:00 ET
DATA_REFRESH_START = dtime(15, 58, 45)#dtime(15, 57, 45)
DATA_REFRESH_END   = dtime(16, 00, 45)#dtime(16, 00, 45)
REFRESH_INTERVAL_SECONDS = 20

# ========== Logging Setup ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== Load Env and Setup MongoDB ==========
load_dotenv()
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    raise ValueError("MongoDB URI not found in .env file. Please set MONGODB_URI.")

db_name = "trading_strategy_db_test"
TIMEZONE = pytz.timezone("US/Eastern")  # Adjust if needed

# ========== Global Data Lock & Data Variable ==========
data_lock = threading.Lock()
stock_data_with_indicators = None  # Shared globally

# --------------------------------------------------------------------
# 1) Insert Ticker Data
# --------------------------------------------------------------------
def insert_ticker_data(ticker, df):
    """
    Inserts historical indicator data for one ticker into the MongoDB 'trading_strategy_db' database.
    """
    if df.empty:
        logger.warning(f"No data to insert for ticker: {ticker}. Skipping.")
        return 0

    client = MongoClient(mongodb_uri)
    db = client[db_name]
    collection_name = f"{ticker}_historical_data"
    collection = db[collection_name]

    data_to_insert = []
    for index, row in df.iterrows():
        record = row.to_dict()
        record["ticker"] = ticker
        # If 'date' not provided in row, fallback to index
        record["date"]   = row.get("date", pd.to_datetime(index).to_pydatetime())
        data_to_insert.append(record)

    inserted_count = 0
    if data_to_insert:
        try:
            result = collection.insert_many(data_to_insert, ordered=False)
            inserted_count = len(result.inserted_ids)
            logger.info(
                f"Thread for {ticker}: Inserted {inserted_count} records into {collection_name}"
            )
        except Exception as db_error:
            logger.error(f"Thread for {ticker}: Error inserting data into MongoDB: {db_error}")
    else:
        logger.warning(f"Thread for {ticker}: No data to insert after processing.")

    client.close()
    return inserted_count

# --------------------------------------------------------------------
# 2) Download, Calculate Indicators, Insert
# --------------------------------------------------------------------
def populate_historical_data_concurrent_threading():
    """
    Downloads data for all tickers, calculates indicators, and inserts into
    ticker-specific collections concurrently.
    Returns a dict { ticker: DataFrame(...) } for all tickers.
    """
    start_time = time.time()

    # 1) Download data
    try:
        stock_data = data_download8.get_stock_data(
            tickers, start_date=START_DATE, end_date=END_DATE
        )
        download_time = time.time() - start_time
        logger.info(f"Data download completed in {download_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during data download: {e}")
        return {}

    # 2) Calculate indicators
    start_time_ind = time.time()
    stock_data_with_indicators = indicatorcalc_testenv3.calculate_indicators_for_all(stock_data)
    indicator_time = time.time() - start_time_ind
    logger.info(f"Indicator calculation completed in {indicator_time:.2f} seconds.")

    # 3) Insert concurrently
    total_inserted_count = 0
    threads = []
    ticker_inserted_counts = {}
    data_lock_insert = threading.Lock()

    def thread_insert_data_wrapper(ticker_inner, df_inner):
        nonlocal total_inserted_count
        try:
            inserted_count = insert_ticker_data(ticker_inner, df_inner)
            with data_lock_insert:
                total_inserted_count += inserted_count
                ticker_inserted_counts[ticker_inner] = inserted_count
        except Exception as err:
            logger.error(f"Thread for {ticker_inner} Exception: {err}")
            with data_lock_insert:
                ticker_inserted_counts[ticker_inner] = 0

    start_time_insert = time.time()
    for tkr, df_tkr in stock_data_with_indicators.items():
        thread = threading.Thread(target=thread_insert_data_wrapper, args=(tkr, df_tkr))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    insert_time = time.time() - start_time_insert
    total_time = time.time() - start_time
    logger.info(f"Concurrent data population finished in {total_time:.2f} sec.")
    logger.info(f"Insertion time: {insert_time:.2f} sec. Total inserted: {total_inserted_count}")

    return stock_data_with_indicators

# --------------------------------------------------------------------
# 3) Aggregate Latest Ticker Records to `all_data`
# --------------------------------------------------------------------
def aggregate_latest_data_to_all_data_collection():
    """
    Finds the single most recent document in each ticker’s collection and places them all
    into the 'all_data' collection as one aggregated document.
    """
    logger.info("Starting aggregation of latest data for all tickers...")
    start_time = time.time()
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    all_data_collection = db["all_data_test"]
    aggregated_data = {}

    for ticker_symbol in tickers:
        collection_name = f"{ticker_symbol}_historical_data"
        ticker_collection = db[collection_name]
        try:
            latest_data_doc = ticker_collection.find().sort([("date", -1)]).limit(1)
            latest_data_list = list(latest_data_doc)
            if latest_data_list:
                latest_data = latest_data_list[0]
                aggregated_data[ticker_symbol] = latest_data
                logger.info(f"Latest data for {ticker_symbol} retrieved.")
            else:
                logger.warning(f"No data for {ticker_symbol} in {collection_name}")
                aggregated_data[ticker_symbol] = None
        except Exception as e:
            logger.error(f"Error retrieving latest data for {ticker_symbol}: {e}")
            aggregated_data[ticker_symbol] = None

    if aggregated_data:
        all_tickers_latest_data_doc = {
            "latest_data_for_tickers": aggregated_data,
            "aggregation_timestamp": dt.now(pytz.utc)
        }
        try:
            result = all_data_collection.insert_one(all_tickers_latest_data_doc)
            logger.info(f"Inserted aggregated latest data into 'all_data' with id: {result.inserted_id}")
        except Exception as e:
            logger.error(f"Error inserting aggregated data: {e}")
    else:
        logger.warning("No data to aggregate.")

    client.close()
    end_time = time.time()
    logger.info(f"Aggregation completed in {end_time - start_time:.2f} seconds.")

# --------------------------------------------------------------------
# 4) Delete the Ticker Collections
# --------------------------------------------------------------------
def delete_ticker_collections():
    """
    Deletes all ticker-specific collections from the DB.
    Use caution—this removes historical data for each ticker.
    """
    logger.info("Deleting ticker-specific collections...")
    start_time = time.time()
    client = MongoClient(mongodb_uri)
    db = client[db_name]

    for ticker_symbol in tickers:
        collection_name = f"{ticker_symbol}_historical_data"
        try:
            if collection_name in db.list_collection_names():
                db.drop_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            else:
                logger.warning(f"Collection {collection_name} not found. Skipping.")
        except Exception as e:
            logger.error(f"Error deleting {collection_name}: {e}")

    client.close()
    end_time = time.time()
    logger.info(f"Deletion completed in {end_time - start_time:.2f} seconds.")

# --------------------------------------------------------------------
# 5) Orchestrate Full Data Cycle
# --------------------------------------------------------------------
def run_data_cycle():
    """
    Runs one cycle: downloads & inserts data, aggregates the latest doc into `all_data`,
    then deletes the ticker collections—leaving only the aggregated documents.
    Returns the new 'stock_data_with_indicators'.
    """
    cycle_start_time = time.time()
    logger.info("--- Starting Data Cycle ---")

    # 1) Populate ticker data
    new_data_with_indicators = populate_historical_data_concurrent_threading()

    # 2) Aggregate the latest doc from each ticker into `all_data`
    aggregate_latest_data_to_all_data_collection()

    # 3) Delete the ticker collections
    delete_ticker_collections()

    cycle_end_time = time.time()
    cycle_duration = cycle_end_time - cycle_start_time
    logger.info(f"--- Data Cycle Completed in {cycle_duration:.2f} seconds ---")
    return new_data_with_indicators

###################################################
#       HELPER FUNCTION: GET LATEST ALL_DATA      #
###################################################
def get_latest_aggregated_data(mongodb_uri, db_name=db_name):
    """
    Returns the 'latest_data_for_tickers' dict from the most recent
    document in the 'all_data' collection, or an empty dict if none found.
    """
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    all_data_collection = db["all_data_test"]
    
    # Find the single most recent doc by sorting aggregation_timestamp desc
    latest_doc = (
        all_data_collection.find({})
        .sort("aggregation_timestamp", -1)
        .limit(1)
    )
    
    docs = list(latest_doc)
    client.close()

    if not docs:
        return {}
    
    doc = docs[0]
    return doc.get("latest_data_for_tickers", {})


###################################################
#    HELPER FUNCTION: CONVERT DICT -> DATAFRAMES   #
###################################################
def dict_to_dataframe(latest_data_for_tickers):
    """
    Converts the big dictionary {ticker -> {fields...}}
    into a dict of DataFrames {ticker -> pd.DataFrame}.
    
    Each ticker’s sub‐object is turned into a 1‐row DataFrame,
    using 'date' as the index if present.
    """
    ticker_dfs = {}
    for ticker, data_dict in latest_data_for_tickers.items():
        if not data_dict:
            continue
        
        # Make a copy to avoid mutating original
        temp = dict(data_dict)
        
        # Remove Mongo _id if it exists
        temp.pop("_id", None)
        
        # Convert date if present in JSON form
        date_val = temp.get("date")
        if isinstance(date_val, dict) and "$date" in date_val:
            date_str = date_val["$date"]
            temp["date"] = pd.to_datetime(date_str)
        elif isinstance(date_val, str):
            temp["date"] = pd.to_datetime(date_val)

        # Create a single-row DataFrame
        df = pd.DataFrame([temp])
        
        # If we have a 'date' column, set it as the index
        if "date" in df.columns:
            df.set_index("date", inplace=True)
        
        ticker_dfs[ticker] = df
    
    return ticker_dfs

# --------------------------------------------------------------------
# 6) Data Refresh Thread (Live Environment)
# --------------------------------------------------------------------
KILL_TIME = dtime(16, 1, 0)  # Example: 4:00 PM ET


def data_refresh_thread():
    global stock_data_with_indicators
    last_refresh_time = None

    while True:
        now = dt.now(TIMEZONE)
        current_time = now.time()
        
        # 1) If past kill time => exit entire script
        if current_time >= KILL_TIME:
            logger.info(f"Now {current_time} >= kill time {KILL_TIME}. Exiting script.")
            os._exit(0)  # <---- forcibly kill *everything* 

        # 2) If in refresh window
        if DATA_REFRESH_START <= current_time <= DATA_REFRESH_END:
            if last_refresh_time is None or (now - last_refresh_time).total_seconds() >= REFRESH_INTERVAL_SECONDS:
                logger.info(f"Starting data refresh at {now}")
                print(f"Starting data refresh at {now}...")

                start_time_cycle = time.time()

                # 1) Run the full data cycle => new data is inserted, aggregated, then ticker collections deleted
                run_data_cycle()

                duration = time.time() - start_time_cycle
                logger.info(f"Data refresh cycle done in {duration:.2f}s.")
                print(f"Data refresh done in {duration:.2f}s.")

                # 2) Now fetch that brand‐new aggregated data from the 'all_data' collection
                latest_dict = get_latest_aggregated_data(mongodb_uri, db_name)

                # 3) Convert each ticker’s latest record into a single‐row DataFrame
                aggregated_dfs = dict_to_dataframe(latest_dict)

                # 4) Store in global for the trading threads
                with data_lock:
                    stock_data_with_indicators = aggregated_dfs

                last_refresh_time = now
            else:
                time.sleep(1)
        else:
            time.sleep(5)


            # ------------------ trading_logic_for_account ------------------
def trading_logic_for_account(account_info):
    """
    Corrected so that we reference the global 'stock_data_with_indicators' directly
    as a DataFrame/dict, not a function.
    """
    account_name = account_info['account_name']
    logger = loggers[account_name]

    # Sync positions once
    positions = sync_with_broker(account_name)

    # Wait for initial data
    while True:
        with data_lock:
            if stock_data_with_indicators is not None:
                break
        logger.debug(f"Waiting for initial data load for {account_name}...")
        print(f"Waiting for initial data load for {account_name}...")
        time.sleep(5)

    while True:
        try:
            now = datetime.now(TIMEZONE)
            current_time = now.time()
            date = pd.Timestamp(now.date())  # just the date portion
            account_balance = get_account_balance(account_name)
            print(f"\nCurrent Time: {now} for {account_name}")
            logger.debug(f"\nCurrent Time: {now} for {account_name}")
            print(f"Account Balance: ${account_balance:,.2f} for {account_name}")
            logger.debug(f"Account Balance: ${account_balance:,.2f} for {account_name}")
            print(f"Existing Positions: {positions} for {account_name}")
            logger.debug(f"Existing Positions: {positions} for {account_name}")

            # Recalculate each strategy's portion
            strategy_cash = {}
            for s in strategies:
                strategy_cash[s.name] = account_info['account_balance_multiplier'] * account_balance * s.equity_fraction

            exit_start, exit_end   = dtime(15,59,0),  dtime(15,59,30)#dtime(15,59,0),  dtime(15,59,30)
            entry_start, entry_end = dtime(15,59,40), dtime(16,0,0)#dtime(15,59,40), dtime(16,0,0)

            within_exit_window  = (exit_start <= current_time <= exit_end)
            within_entry_window = (entry_start <= current_time <= entry_end)

            print(f"Within Exit Window (Print): {within_exit_window} for {account_name}")
            logger.debug(f"Within Exit Window (Log): {within_exit_window} for {account_name}")
            print(f"Within Entry Window (Print): {within_entry_window} for {account_name}")
            logger.debug(f"Within Entry Window (Log): {within_entry_window} for {account_name}")

            # Grab a safe copy of the data
            with data_lock:
                local_data = copy.deepcopy(stock_data_with_indicators)

            if local_data is not None:
                
                # Use ThreadPoolExecutor for concurrent strategy processing
                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #     futures = []
                #     for s in strategies:
                #         future = executor.submit(
                #              process_strategy, s, local_data, date, positions, strategy_cash, account_name, within_exit_window, within_entry_window
                #         )
                #         futures.append(future)
                        
                #     for future in futures:
                #         future.result()  # Wait for all strategies to complete for this cycle
                for s in strategies:
                    logger.info(f"=== PROCESSING STRATEGY: {s.name} ===")
                    # print(f"=== PROCESSING STRATEGY: {s.name} ===")
                    
                    process_strategy(s, local_data, date, positions, strategy_cash, account_name, within_exit_window, within_entry_window)
                    
                    logger.info(f"=== COMPLETED STRATEGY: {s.name} ===")
                    # print(f"=== COMPLETED STRATEGY: {s.name} ===")

            else:
                logger.debug(f"local_data is None, skipping trading logic for {account_name}.")
                # print(f"local_data is None, skipping trading logic for {account_name}.")

            # Sleep time
            if within_exit_window or within_entry_window:
                time.sleep(1)
            else:
                time.sleep(5)

        except Exception as exc:
            logger.exception(f"Exception in trading_logic: {exc} for {account_name}")
            # print(f"Exception in trading_logic: {exc} for {account_name}")
            time.sleep(5)


def process_strategy(
    s, local_data, date, positions, strategy_cash, account_name,
    within_exit_window, within_entry_window
):
    """
    Processes a single strategy, making sure the loggers are specific
    to this strategy and account
    """
    logger = loggers[account_name]
    positions_for_strat = positions.setdefault(s.name, {})

    # Make sure local_data isn't None
    if local_data is None:
        logger.debug(f"No data available for {account_name}, skipping {s.name}.")
        print(f"No data available for {account_name}, skipping {s.name}.")
        return

    # 1) Exits
    if within_exit_window:
        # ENHANCED LOGGING: Show what check_exit returns
        # Safely check for exit method  
        check_exit = getattr(s, "check_exit", None)
        if check_exit is None:
            logger.error("Strategy %s has no check_exit method", s.name)
            return
        exits = check_exit(local_data, positions_for_strat, date)        
        logger.debug(f"Exits with met conditions returned by check_exit for {s.name}: {exits} for {account_name}")
        print(f"Exits with met conditions returned by check_exit for {s.name}: {exits} for {account_name}")
        
        logger.debug(f"Pre-filtered exits (before order placement checks) for {s.name}: {exits} for {account_name}")
        print(f"Pre-filtered exits (before order placement checks) for {s.name}: {exits} for {account_name}")

        # ENHANCED LOGGING: Show detailed evaluation of each ticker in current positions
        logger.debug(f"=== DETAILED EXIT EVALUATION for {s.name} ===")
        # print(f"=== DETAILED EXIT EVALUATION for {s.name} ===")
        
        # Get all tickers currently held in positions for this strategy
        current_position_tickers = list(positions_for_strat.keys())
        logger.debug(f"Current position tickers in strategy {s.name}: {current_position_tickers}")
        # print(f"Current position tickers in strategy {s.name}: {current_position_tickers}")
        
        # Evaluate each ticker's exit conditions
        for ticker in current_position_tickers:
            if ticker in local_data:
                try:
                    # Check if ticker exists first
                    if ticker not in local_data:
                        logger.warning("No data available for ticker %s", ticker)
                        continue

                    # Check if we have data for the date
                    subset = local_data[ticker].loc[local_data[ticker].index <= date]
                    if subset.empty:
                        logger.warning("No data for %s on %s", ticker, date)
                        continue

                    row = subset.iloc[-1]
                    conditions = exit_conditions(row, ticker)
                    
                    logger.debug(f"--- Evaluating {ticker} for EXIT in {s.name} ---")
                    # print(f"--- Evaluating {ticker} for EXIT in {s.name} ---")
                    
                    # Log position details
                    position_info = positions_for_strat[ticker]
                    logger.debug(f"  Position info: qty={position_info['qty']}, entry_price={position_info['entry_price']}")
                    # print(f"  Position info: qty={position_info['qty']}, entry_price={position_info['entry_price']}")
                    
                    # Check each exit rule
                    if hasattr(s, 'exit_rules') and s.exit_rules:
                        for rule_idx, rule in enumerate(s.exit_rules):
                            if ticker in rule['tickers']:
                                logger.debug(f"  Exit Rule {rule_idx}: {ticker} is in tickers {rule['tickers']}")
                                # print(f"  Exit Rule {rule_idx}: {ticker} is in tickers {rule['tickers']}")
                                
                                for condition in rule['conditions']:
                                    if isinstance(condition, int):
                                        if condition in conditions:
                                            condition_met, cond_reason = conditions[condition]
                                            logger.debug(f"    - Exit Condition {condition}: {cond_reason} - Met: {condition_met}")
                                            # print(f"    - Exit Condition {condition}: {cond_reason} - Met: {condition_met}")
                                        else:
                                            logger.debug(f"    - Exit Condition {condition}: NOT FOUND in conditions dict")
                                            # print(f"    - Exit Condition {condition}: NOT FOUND in conditions dict")
                                    elif isinstance(condition, tuple):
                                        logger.debug(f"    - Tuple exit condition: {condition}")
                                        # print(f"    - Tuple exit condition: {condition}")
                                        all_met = True
                                        reasons = []
                                        for cond_id in condition:
                                            if cond_id in conditions:
                                                condition_met, cond_reason = conditions[cond_id]
                                                logger.debug(f"      - Exit Condition {cond_id}: {cond_reason} - Met: {condition_met}")
                                                print(f"      - Exit Condition {cond_id}: {cond_reason} - Met: {condition_met}")
                                                if condition_met:
                                                    reasons.append(cond_reason)
                                                else:
                                                    all_met = False
                                            else:
                                                logger.debug(f"      - Exit Condition {cond_id}: NOT FOUND in conditions dict")
                                                # print(f"      - Exit Condition {cond_id}: NOT FOUND in conditions dict")
                                                all_met = False
                                        
                                        if all_met:
                                            combined_reason = " & ".join(reasons)
                                            logger.debug(f"    - ALL exit conditions in tuple met: {combined_reason}")
                                            # print(f"    - ALL exit conditions in tuple met: {combined_reason}")
                                        else:
                                            logger.debug(f"    - NOT all exit conditions in tuple met")
                                            # print(f"    - NOT all exit conditions in tuple met")
                            else:
                                logger.debug(f"  Exit Rule {rule_idx}: {ticker} NOT in tickers {rule['tickers']}")
                                # print(f"  Exit Rule {rule_idx}: {ticker} NOT in tickers {rule['tickers']}")
                    else:
                        logger.debug(f"No exit_rules attribute for strategy {s.name}")
                        # print(f"No exit_rules attribute for strategy {s.name}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating {ticker} for exit: {e}")
                    # print(f"Error evaluating {ticker} for exit: {e}")
            else:
                logger.debug(f"  {ticker}: NO DATA AVAILABLE for exit evaluation")
                # print(f"  {ticker}: NO DATA AVAILABLE for exit evaluation")
        
        logger.debug(f"=== END DETAILED EXIT EVALUATION for {s.name} ===")
        # print(f"=== END DETAILED EXIT EVALUATION for {s.name} ===")

        if exits:
            # Group by ticker
            exit_dict = {}
            for (ticker, reason) in exits:
                exit_dict.setdefault(ticker, []).append(reason)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for ticker, reasons_list in exit_dict.items():
                    combined_reason = "; ".join(reasons_list)
                    logger.debug(f"{s.name} - Exiting {ticker}: {combined_reason} for {account_name}")
                    # print(f"{s.name} - Exiting {ticker}: {combined_reason} for {account_name}")

                    if ticker in positions_for_strat:
                        qty = positions_for_strat[ticker]['qty']
                        if 'Close' not in local_data[ticker].columns:
                            logger.warning("No Close price for %s", ticker)
                            continue

                        price = local_data[ticker]['Close'].iloc[-1]

                        # (Optional: Log which conditions are met)
                        if ticker in local_data:

                            # Check if we have data for the date
                            subset = local_data[ticker].loc[local_data[ticker].index <= date]
                            if subset.empty:
                                logger.warning("No data for %s on %s", ticker, date)
                                continue  # ✅ CORRECT - inside the for ticker loop

                            row = subset.iloc[-1]
                            conditions = exit_conditions(row, ticker)


                            if hasattr(s,'exit_rules') and s.exit_rules:
                                for rule in s.exit_rules:
                                    if ticker in rule['tickers']:
                                            for condition in rule['conditions']:
                                                if isinstance(condition,int):
                                                    if condition in conditions:
                                                        condition_met, cond_reason = conditions[condition]
                                                        logger.debug(f"  - {ticker} - Exit Condition {condition}: {cond_reason} - Met: {condition_met} for {account_name}")
                                                        # print(f"  - {ticker} - Exit Condition {condition}: {cond_reason} - Met: {condition_met} for {account_name}")
                                                elif isinstance(condition,tuple):
                                                    for cond_id in condition:
                                                        if cond_id in conditions:
                                                            condition_met, cond_reason = conditions[cond_id]
                                                            logger.debug(f"    - {ticker} - Exit Condition {cond_id}: {cond_reason} - Met: {condition_met} for {account_name}")
                                                            # print(f"    - {ticker} - Exit Condition {cond_id}: {cond_reason} - Met: {condition_met} for {account_name}")
                            else:
                                logger.debug(f"No exit_rules attribute for strategy {s.name} for {account_name}")
                        else:
                            logger.debug(f"Data not available for {ticker} in exit check for {account_name}.")
                            # print(f"Data not available for {ticker} in exit check for {account_name}.")
                        future = executor.submit(
                            place_order_threadsafe, ticker, qty, 'sell',
                            account_name, price=price
                        )
                        futures.append((future, ticker, qty, price, combined_reason, s.name))

                for future, ticker, qty, price, combined_reason, strat_name in futures:
                    order_id, order_time = future.result()
                    if order_id:
                        exit_date_str = order_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        exit_trade = {
                            'Trade Type': 'Exit',
                            'Strategy': strat_name,
                            'Ticker': ticker,
                            'Date': exit_date_str,
                            'Price': price,
                            'Quantity': int(qty),
                            'Reason': combined_reason,
                            'Order ID': order_id,
                            'Profit $': (price - positions_for_strat[ticker]['entry_price']) * qty,
                            'Profit %': ((price / positions_for_strat[ticker]['entry_price']) - 1) * 100
                        }
                        log_trade(exit_trade, account_name)
                        del positions_for_strat[ticker]
                    else:
                        logger.error(f"{s.name} - SELL order failed for {ticker} for {account_name}.")
                        # print(f"{s.name} - SELL order failed for {ticker} for {account_name}.")
        else:
             # We are in the exit window, but there's nothing to exit
             logger.debug(f"No exit signals found for {account_name}, skipping exits for {s.name}.")
            #  print(f"No exit signals found for {account_name}, skipping exits for {s.name}.")
    else:
           logger.debug(f"Not in exit window, skipping exits for {account_name} for {s.name}")
        #    print(f"Not in exit window, skipping exits for {account_name} for {s.name}")

    # 2) Entries
    if within_entry_window:
        # The usual entry logic ...
        current_positions_count = len(positions_for_strat)
        available_positions = s.max_symbols - current_positions_count
        strategy_alloc_cash = strategy_cash[s.name]
        
        # ENHANCED LOGGING: Show what check_entry returns
        # Safely check for entry method
        check_entry = getattr(s, "check_entry", None)
        if check_entry is None:
            logger.error("Strategy %s has no check_entry method", s.name)
            return
        entries = check_entry(local_data, date)
        
        logger.debug(f"Entries with met conditions returned by check_entry for {s.name}: {entries} for {account_name}")
        # print(f"Entries with met conditions returned by check_entry for {s.name}: {entries} for {account_name}")
        
        logger.debug(f"Pre-filtered entries (before MongoDB/position checks) for {s.name}: {entries} for {account_name}")
        # print(f"Pre-filtered entries (before MongoDB/position checks) for {s.name}: {entries} for {account_name}")

        # ENHANCED LOGGING: Show detailed evaluation of each ticker in the strategy
        logger.debug(f"=== DETAILED ENTRY EVALUATION for {s.name} ===")
        # print(f"=== DETAILED ENTRY EVALUATION for {s.name} ===")
        
        # Get all tickers that this strategy could potentially trade
        all_strategy_tickers = s.get_all_tickers()
        logger.debug(f"All tickers in strategy {s.name}: {all_strategy_tickers}")
        # print(f"All tickers in strategy {s.name}: {all_strategy_tickers}")
        
        # Evaluate each ticker's conditions
        for ticker in all_strategy_tickers:
            if ticker in local_data:
                try:

                    
                    # Check if we have data for the date
                    subset = local_data[ticker].loc[local_data[ticker].index <= date]
                    if subset.empty:
                        logger.warning("No data for %s on %s", ticker, date)
                        continue  # ✅ CORRECT - inside the for ticker loop

                    row = subset.iloc[-1]
                    conditions = entry_conditions(row, ticker)


                    logger.debug(f"--- Evaluating {ticker} for {s.name} ---")
                    # print(f"--- Evaluating {ticker} for {s.name} ---")
                    
                    # Check each entry rule
                    if hasattr(s, 'entry_rules') and s.entry_rules:
                        for rule_idx, rule in enumerate(s.entry_rules):
                            if ticker in rule['tickers']:
                                logger.debug(f"  Rule {rule_idx}: {ticker} is in tickers {rule['tickers']}")
                                # print(f"  Rule {rule_idx}: {ticker} is in tickers {rule['tickers']}")
                                
                                for condition in rule['conditions']:
                                    if isinstance(condition, int):
                                        if condition in conditions:
                                            condition_met, cond_reason = conditions[condition]
                                            logger.debug(f"    - Condition {condition}: {cond_reason} - Met: {condition_met}")
                                            # print(f"    - Condition {condition}: {cond_reason} - Met: {condition_met}")
                                        else:
                                            logger.debug(f"    - Condition {condition}: NOT FOUND in conditions dict")
                                            # print(f"    - Condition {condition}: NOT FOUND in conditions dict")
                                    elif isinstance(condition, tuple):
                                        logger.debug(f"    - Tuple condition: {condition}")
                                        # print(f"    - Tuple condition: {condition}")
                                        all_met = True
                                        reasons = []
                                        for cond_id in condition:
                                            if cond_id in conditions:
                                                condition_met, cond_reason = conditions[cond_id]
                                                logger.debug(f"      - Condition {cond_id}: {cond_reason} - Met: {condition_met}")
                                                # print(f"      - Condition {cond_id}: {cond_reason} - Met: {condition_met}")
                                                if condition_met:
                                                    reasons.append(cond_reason)
                                                else:
                                                    all_met = False
                                            else:
                                                logger.debug(f"      - Condition {cond_id}: NOT FOUND in conditions dict")
                                                # print(f"      - Condition {cond_id}: NOT FOUND in conditions dict")
                                                all_met = False
                                        
                                        if all_met:
                                            combined_reason = " & ".join(reasons)
                                            logger.debug(f"    - ALL conditions in tuple met: {combined_reason}")
                                            # print(f"    - ALL conditions in tuple met: {combined_reason}")
                                        else:
                                            logger.debug(f"    - NOT all conditions in tuple met")
                                            # print(f"    - NOT all conditions in tuple met")
                            else:
                                logger.debug(f"  Rule {rule_idx}: {ticker} NOT in tickers {rule['tickers']}")
                                # print(f"  Rule {rule_idx}: {ticker} NOT in tickers {rule['tickers']}")
                    else:
                        logger.debug(f"No entry_rules attribute for strategy {s.name}")
                        # print(f"No entry_rules attribute for strategy {s.name}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating {ticker}: {e}")
                    # print(f"Error evaluating {ticker}: {e}")
            else:
                logger.debug(f"  {ticker}: NO DATA AVAILABLE")
                # print(f"  {ticker}: NO DATA AVAILABLE")
        
        logger.debug(f"=== END DETAILED ENTRY EVALUATION for {s.name} ===")
        # print(f"=== END DETAILED ENTRY EVALUATION for {s.name} ===")

        # Log the ticker that are being skipped:
        filtered_entries = []
        for ticker, reason in entries:
            if ticker in local_data:
                 # Check if the ticker exists as an entry in MongoDB for this strategy
                        if ticker in positions_for_strat:
                            logger.debug(f"  - Skipping {ticker} for {s.name}: Already have open position for {account_name}")
                            # print(f"  - Skipping {ticker} for {s.name}: Already have open position for {account_name}")
                            continue  # Skip this ticker since we already have an open position
        
                        # Check if we have data for the date
                        subset = local_data[ticker].loc[local_data[ticker].index <= date]
                        if subset.empty:
                            logger.warning("No data for %s on %s", ticker, date)
                            continue  # ✅ CORRECT - inside the for ticker loop

                        row = subset.iloc[-1]
                        conditions = entry_conditions(row, ticker)




                        logger.debug(f"  Evaluating {ticker} for {s.name}: for {account_name}")
                        # print(f"  Evaluating {ticker} for {s.name}: for {account_name}")
                        if hasattr(s, 'entry_rules') and s.entry_rules:
                            for rule in s.entry_rules:
                                if ticker in rule['tickers']:
                                    for condition in rule['conditions']:
                                            if isinstance(condition,int):
                                                if condition in conditions:
                                                      condition_met, cond_reason = conditions[condition]
                                                      logger.debug(f"    - {ticker} - Entry Condition {condition}: {cond_reason} - Met: {condition_met} for {account_name}")
                                                    #   print(f"    - {ticker} - Entry Condition {condition}: {cond_reason} - Met: {condition_met} for {account_name}")
                                            elif isinstance(condition,tuple):
                                                 for cond_id in condition:
                                                       if cond_id in conditions:
                                                           condition_met, cond_reason = conditions[cond_id]
                                                           logger.debug(f"    - {ticker} - Entry Condition {cond_id}: {cond_reason} - Met: {condition_met} for {account_name}")
                                                        #    print(f"    - {ticker} - Entry Condition {cond_id}: {cond_reason} - Met: {condition_met} for {account_name}")
                        else:
                            logger.debug(f"No entry_rules attribute for strategy {s.name} for {account_name}")
                            
                        filtered_entries.append((ticker, reason))
            else:
                logger.debug(f"Data not available for {ticker} in entry check for {account_name}")
                # print(f"Data not available for {ticker} in entry check for {account_name}")

        # Use ThreadPoolExecutor to place buy orders concurrently
        entries_to_process = filtered_entries[:available_positions]
        if entries_to_process:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for (ticker, reason) in entries_to_process:
                    # Check if Close column exists
                    if 'Close' not in local_data[ticker].columns:
                        logger.warning("No Close price for %s", ticker)
                        continue  # ✅ CORRECT - inside the for loop
                    
                    price = local_data[ticker]['Close'].iloc[-1]


                    qty = calculate_quantity(price, strategy_alloc_cash, available_positions)
                    if qty > 0:
                        total_cost = qty * price
                        future = executor.submit(
                            place_order_threadsafe, ticker, qty, 'buy',
                            account_name, price=price
                        )
                        futures.append((future, ticker, qty, price, reason, s.name))
                    else:
                        logger.debug(f"{s.name} - 0 qty for {ticker}, skipping. for {account_name}")
                        # print(f"{s.name} - 0 qty for {ticker}, skipping. for {account_name}")


                for future, ticker, qty, price, reason, strat_name in futures:
                    order_id, order_time = future.result()
                    if order_id:
                        entry_date_str = order_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        positions_for_strat[ticker] = {
                            'qty': qty,
                            'entry_price': price,
                            'entry_date': date,
                            'entry_reason': reason,
                            'entry_order_id': order_id
                        }
                        entry_trade = {
                            'Trade Type': 'Entry',
                            'Strategy': strat_name,
                            'Ticker': ticker,
                            'Date': entry_date_str,
                            'Price': price,
                            'Quantity': int(qty),
                            'Reason': reason,
                            'Order ID': order_id,
                            'Profit $': None,
                            'Profit %': None
                        }
                        log_trade(entry_trade, account_name)
                        logger.debug(f"{s.name} - Entered {ticker}: {reason} with qty {qty} for {account_name}")
                        # print(f"{s.name} - Entered {ticker}: {reason} with qty {qty} for {account_name}")
                    else:
                        logger.error(f"{s.name} - BUY order failed for {ticker} for {account_name}.")
                        # print(f"{s.name} - BUY order failed for {ticker} for {account_name}.")
        else:
            logger.debug(f"No valid new entries to place for {account_name} in {s.name}.")
            # print(f"No valid new entries to place for {account_name} in {s.name}.")
    else:
        logger.debug(f"Not in entry window, skipping entries for {account_name} for {s.name}")
        # print(f"Not in entry window, skipping entries for {account_name} for {s.name}")
# ------------------ Main Trading Logic ------------------
def trading_logic():
    """
    Spawns one trading_logic_for_account thread per account,
    after ensuring we have data.
    """
    # Wait for data
    global stock_data_with_indicators
    while True:
        with data_lock:
            if stock_data_with_indicators is not None:
                break
        print("Waiting for initial data load in main trading_logic()...")
        logging.debug("Waiting for initial data load in main trading_logic()...")
        time.sleep(5)

    # Start each account's trading thread
    threads = []
    for account_info in accounts_info:
        t = threading.Thread(target=trading_logic_for_account, args=(account_info,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

# ------------------ Backtest vs Live ------------------
if mode == 'backtest':
    # Fetch the stock data
    stock_data = data_download8.get_stock_data(tickers,start_date=start_date,end_date=end_date)

    # Add this print statement to display data ranges for each ticker
    for ticker, df in stock_data.items():
        print(f"{ticker} data range: {df.index.min()} to {df.index.max()}")

    # Calculate indicators using indicatorcalc
    stock_data_with_indicators = indicatorcalc_testenv3.calculate_indicators_for_all(stock_data)

    # Execute strategies
    backtest_strategies(stock_data_with_indicators, strategies, mode=mode, initial_capital=initial_capital)
else:
    # LIVE mode
    data_thread = threading.Thread(target=data_refresh_thread, daemon=True)
    data_thread.start()

    trading_thread = threading.Thread(target=trading_logic)
    trading_thread.start()

    trading_thread.join()