# Trading Automation Framework

## Overview
This repository contains a Python-based trading automation framework that integrates with **Schwab** and **Alpaca** APIs.  
It is designed for **backtesting** and **live trading**, featuring technical indicator computation, historical data downloading, and modular strategy execution with logging support.

## Features
- **Historical Data Downloader**: Fetches and processes price history from Schwab API with retry logic and threading for speed.
- **Technical Indicator Engine**: Numba-optimized calculations for RSI, EMA, SMA, ATR, MACD, SuperTrend, and more. Includes custom divergence detection, peak/trough finding, and symbol-specific indicators.
- **Strategy Execution**: Flexible rule-based entry and exit system with dynamic conditions, ATR-based exits, percent exits, and max-holding-day enforcement.
- **Backtest & Live Modes**: Toggle between simulation and real trading using environment variables. Backtest logs to CSV, live trades log to MongoDB.
- **Account Management**: Load credentials from `.env` files, auto-handle Schwab token refresh, and support multiple accounts.

## Project Structure
```
.
├── Main.py                  # Core execution engine (backtest/live trading)
├── data_download8.py        # Schwab data downloader
├── indicatorcalc_testenv3.py# Technical indicators & calculations
├── Account_Credentials.py   # Account credential management
├── requirements.txt         # Dependencies (to create)
└── README.md                # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-automation.git
   cd trading-automation
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
Create a `.env` file in the project root with the following entries:
```env
# Alpaca Credentials
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Schwab Credentials
app_key=your_schwab_app_key
app_secret=your_schwab_app_secret
callback_url=https://127.0.0.1
token_path=tokens.json

# MongoDB URI
MONGO_URI=mongodb://localhost:27017/
```

## Usage
### Backtesting
Run:
```bash
python Main.py
```
- Logs trades to `backtest_log.csv`
- Uses historical data

### Live Trading
1. Set `mode = 'live_trade'` in `Main.py`.
2. Ensure credentials in `.env` are valid.
3. Run:
   ```bash
   python Main.py
   ```
- Logs trades to MongoDB
- Uses Alpaca/Schwab APIs

## Logging
- **Backtest Mode**: CSV logs (`backtest_log.csv`)
- **Live Mode**: MongoDB collections per account
- Configurable logging with per-account log files

## Dependencies
- `pandas`
- `numpy`
- `numba`
- `matplotlib`
- `pandas-ta`
- `ta`
- `alpaca-trade-api`
- `schwabdev`
- `pymongo`
- `python-dotenv`
- `quantstats_lumi`

## Disclaimer
This project is provided for **educational purposes only**.  
Use at your own risk. The authors are not responsible for financial losses incurred by using this software.
