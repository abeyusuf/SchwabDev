import pandas as pd
import numpy as np
from numba import njit
import pandas_ta as ta
import logging
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import KSTIndicator
# from data import data_download  # Correct relative import
from ta.momentum import (AwesomeOscillatorIndicator, KAMAIndicator,ROCIndicator,RSIIndicator,UltimateOscillator, WilliamsRIndicator)
from ta.trend import (EMAIndicator, KSTIndicator, SMAIndicator )
import data_download8
from ta.volatility import (AverageTrueRange)
from ta.volume import (ChaikinMoneyFlowIndicator,  MFIIndicator, VolumeWeightedAveragePrice)
import time
import datetime
import pytz
import os

TIMEZONE = pytz.timezone('America/New_York')

# Global RSI mode setting - change this to apply to all RSI calculations
RSI_MODE = 'wilder'  # Options: 'wilder', 'ema', 'adaptive'

# =============================================================================
# NUMBA-OPTIMIZED TECHNICAL INDICATORS
# =============================================================================

@njit
def _sma_numba(data, period):
    """Numba-optimized Simple Moving Average"""
    n = len(data)
    result = np.empty(n)
    
    # Fill initial values
    for i in range(n):
        if i < period - 1:
            result[i] = np.mean(data[:i+1])
        else:
            result[i] = np.mean(data[i-period+1:i+1])
    
    return result

@njit
def _ema_numba(data, period):
    """Numba-optimized Exponential Moving Average"""
    n = len(data)
    result = np.empty(n)
    alpha = 2.0 / (period + 1)
    
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result
@njit
def _rsi_ema_numba(data, period):
    """Numba-optimized EMA-based RSI"""
    n = len(data)
    result = np.empty(n)
    result[0] = 50.0  # Default first value
    
    if n < 2:
        return result
    
    # Calculate price changes
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    alpha = 2.0 / (period + 1)
    
    # Initialize with first gain/loss
    if len(gains) > 0:
        avg_gain = gains[0]
        avg_loss = losses[0]
    else:
        avg_gain = 0.0
        avg_loss = 0.0
    
    # Calculate RSI for each point using EMA
    for i in range(len(deltas)):
        if i > 0:
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    return result

def rsi_ema(data, period=14):
    """EMA-based RSI"""
    return pd.Series(_rsi_ema_numba(data.values, period), index=data.index)

@njit
def _rsi_wilder_numba(data, period):
    """Numba-optimized Wilder's RSI"""
    n = len(data)
    result = np.empty(n)
    result[0] = 50.0  # Default first value
    
    if n < 2:
        return result
    
    # Calculate price changes
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    alpha = 1.0 / period
    
    # Initialize with first period simple average
    if len(gains) > 0:
        avg_gain = gains[0]
        avg_loss = losses[0]
    else:
        avg_gain = 0.0
        avg_loss = 0.0
    
    # Calculate RSI for each point
    for i in range(len(deltas)):
        if i > 0:
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    return result

@njit
def _rsi_adaptive_numba(data, period):
    """Numba-optimized Adaptive RSI"""
    n = len(data)
    result = np.empty(n)
    
    if n < period:
        return np.full(n, 50.0)
    
    # Calculate volatility (rolling std)
    volatility = np.empty(n)
    for i in range(n):
        start_idx = max(0, i - period + 1)
        window_data = data[start_idx:i+1]
        volatility[i] = np.std(window_data)
    
    # Normalize volatility
    vol_normalized = np.empty(n)
    for i in range(n):
        if i < 50:
            vol_normalized[i] = 0.5
        else:
            vol_window = volatility[max(0, i-49):i+1]
            vol_min = np.min(vol_window)
            vol_max = np.max(vol_window)
            if vol_max - vol_min == 0:
                vol_normalized[i] = 0.5
            else:
                vol_normalized[i] = (volatility[i] - vol_min) / (vol_max - vol_min)
    
    # Calculate adaptive RSI
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    result[0] = 50.0
    avg_gain = 0.0
    avg_loss = 0.0
    
    for i in range(len(deltas)):
        adaptive_period = period * (2.0 - vol_normalized[i + 1])
        adaptive_period = max(2.0, min(adaptive_period, period * 2.0))
        current_alpha = 2.0 / (adaptive_period + 1.0)
        
        if i == 0:
            avg_gain = gains[i]
            avg_loss = losses[i]
        else:
            avg_gain = current_alpha * gains[i] + (1.0 - current_alpha) * avg_gain
            avg_loss = current_alpha * losses[i] + (1.0 - current_alpha) * avg_loss
        
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    return result

@njit
def _macd_advanced_numba(data, fast, slow, signal):
    """Numba-optimized MACD"""
    ema_fast = _ema_numba(data, fast)
    ema_slow = _ema_numba(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_numba(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

@njit
def _atr_numba(high, low, close, period):
    """Numba-optimized ATR"""
    n = len(high)
    tr = np.empty(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    
    return _sma_numba(tr, period)

@njit
def _supertrend_numba(high, low, close, period, multiplier):
    """Numba-optimized SuperTrend"""
    n = len(close)
    atr = _atr_numba(high, low, close, period)
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    supertrend = np.empty(n)
    
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        if close[i] > upper_band[i-1]:
            supertrend[i] = lower_band[i]
        elif close[i] < lower_band[i-1]:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = supertrend[i-1]
    
    return supertrend

@njit
def _bop_numba(open_price, high, low, close, period):
    """Numba-optimized Balance of Power"""
    n = len(close)
    bop_raw = np.empty(n)
    
    for i in range(n):
        range_val = high[i] - low[i]
        if range_val == 0:
            bop_raw[i] = 0.0
        else:
            bop_raw[i] = (close[i] - open_price[i]) / range_val
    
    return _sma_numba(bop_raw, period)

@njit
def _ultimate_oscillator_numba(high, low, close, period1, period2, period3):
    """Numba-optimized Ultimate Oscillator"""
    n = len(close)
    bp = np.empty(n)
    tr = np.empty(n)
    
    bp[0] = close[0] - min(low[0], close[0])
    tr[0] = max(high[0], close[0]) - min(low[0], close[0])
    
    for i in range(1, n):
        bp[i] = close[i] - min(low[i], close[i-1])
        tr[i] = max(high[i], close[i-1]) - min(low[i], close[i-1])
    
    # Calculate rolling sums
    bp_sum1 = np.empty(n)
    tr_sum1 = np.empty(n)
    bp_sum2 = np.empty(n)
    tr_sum2 = np.empty(n)
    bp_sum3 = np.empty(n)
    tr_sum3 = np.empty(n)
    
    for i in range(n):
        start1 = max(0, i - period1 + 1)
        start2 = max(0, i - period2 + 1)
        start3 = max(0, i - period3 + 1)
        
        bp_sum1[i] = np.sum(bp[start1:i+1])
        tr_sum1[i] = np.sum(tr[start1:i+1])
        bp_sum2[i] = np.sum(bp[start2:i+1])
        tr_sum2[i] = np.sum(tr[start2:i+1])
        bp_sum3[i] = np.sum(bp[start3:i+1])
        tr_sum3[i] = np.sum(tr[start3:i+1])
    
    uo = np.empty(n)
    for i in range(n):
        if tr_sum1[i] == 0 or tr_sum2[i] == 0 or tr_sum3[i] == 0:
            uo[i] = 50.0
        else:
            avg1 = bp_sum1[i] / tr_sum1[i]
            avg2 = bp_sum2[i] / tr_sum2[i]
            avg3 = bp_sum3[i] / tr_sum3[i]
            uo[i] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0
    
    return uo

@njit
def _ibs_numba(high, low, close):
    """Numba-optimized Internal Bar Strength"""
    n = len(close)
    result = np.empty(n)
    
    for i in range(n):
        range_val = high[i] - low[i]
        if range_val == 0:
            result[i] = 0.5
        else:
            result[i] = (close[i] - low[i]) / range_val
    
    return result

@njit
def _higher_highs_numba(data, periods):
    """Numba-optimized higher highs detection"""
    n = len(data)
    result = np.empty(n, dtype=np.bool_)
    
    for i in range(n):
        is_higher = True
        for j in range(1, min(periods + 1, i + 1)):
            if data[i] <= data[i - j]:
                is_higher = False
                break
        result[i] = is_higher and i >= periods
    
    return result

@njit
def _find_peaks_troughs_numba(data, window):
    """Numba-optimized peak/trough detection"""
    n = len(data)
    peaks = np.empty(n, dtype=np.bool_)
    troughs = np.empty(n, dtype=np.bool_)
    
    for i in range(n):
        peaks[i] = False
        troughs[i] = False
        
        if i < window or i >= n - window:
            continue
            
        # Check if current point is a peak
        is_peak = True
        is_trough = True
        
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j != i:
                if data[j] >= data[i]:
                    is_peak = False
                if data[j] <= data[i]:
                    is_trough = False
        
        peaks[i] = is_peak
        troughs[i] = is_trough
    
    return peaks, troughs

@njit
def _detect_divergence_slope_numba(price_data, indicator_data, lookback):
    """Numba-optimized slope-based divergence detection"""
    n = len(price_data)
    threshold = 0.001
    
    bull_div = np.empty(n, dtype=np.bool_)
    bear_div = np.empty(n, dtype=np.bool_)
    hdiv_bull = np.empty(n, dtype=np.bool_)
    hdiv_bear = np.empty(n, dtype=np.bool_)
    
    for i in range(n):
        bull_div[i] = False
        bear_div[i] = False
        hdiv_bull[i] = False
        hdiv_bear[i] = False
        
        if i < lookback:
            continue
        
        # Calculate slopes using simple linear regression
        x_vals = np.arange(lookback, dtype=np.float64)
        
        # Price slope
        y_price = price_data[i-lookback+1:i+1]
        mean_x = np.mean(x_vals)
        mean_y_price = np.mean(y_price)
        
        num_price = np.sum((x_vals - mean_x) * (y_price - mean_y_price))
        den = np.sum((x_vals - mean_x) ** 2)
        
        if den != 0:
            price_slope = num_price / den
        else:
            price_slope = 0.0
        
        # Indicator slope
        y_ind = indicator_data[i-lookback+1:i+1]
        mean_y_ind = np.mean(y_ind)
        
        num_ind = np.sum((x_vals - mean_x) * (y_ind - mean_y_ind))
        
        if den != 0:
            ind_slope = num_ind / den
        else:
            ind_slope = 0.0
        
        # Detect divergences
        bull_div[i] = (price_slope < -threshold) and (ind_slope > threshold)
        bear_div[i] = (price_slope > threshold) and (ind_slope < -threshold)
        hdiv_bull[i] = (price_slope > threshold) and (ind_slope < -threshold)
        hdiv_bear[i] = (price_slope < -threshold) and (ind_slope > threshold)
    
    return bull_div, bear_div, hdiv_bull, hdiv_bear

@njit
def _cross_above_numba(data1, data2):
    """Numba-optimized cross above detection"""
    n = len(data1)
    result = np.empty(n, dtype=np.bool_)
    result[0] = False
    
    for i in range(1, n):
        result[i] = (data1[i] > data2[i]) and (data1[i-1] <= data2[i-1])
    
    return result

@njit
def _cross_below_numba(data1, data2):
    """Numba-optimized cross below detection"""
    n = len(data1)
    result = np.empty(n, dtype=np.bool_)
    result[0] = False
    
    for i in range(1, n):
        result[i] = (data1[i] < data2[i]) and (data1[i-1] >= data2[i-1])
    
    return result

# =============================================================================
# WRAPPER FUNCTIONS (maintain same interface as original)
# =============================================================================

def sma(data, period):
    """Simple Moving Average"""
    return pd.Series(_sma_numba(data.values, period), index=data.index)

def ema(data, period):
    """Exponential Moving Average"""
    return pd.Series(_ema_numba(data.values, period), index=data.index)

def rsi_wilder(data, period=14):
    """Wilder's RSI (Traditional)"""
    return pd.Series(_rsi_wilder_numba(data.values, period), index=data.index)

def rsi_adaptive(data, period=14):
    """Adaptive RSI (period adjusts based on volatility)"""
    return pd.Series(_rsi_adaptive_numba(data.values, period), index=data.index)

def rsi_advanced(data, period=14, method='wilder'):
    """RSI with multiple methods"""
    if method == 'wilder':
        return rsi_wilder(data, period)
    elif method == 'ema':
        return rsi_ema(data, period)
    elif method == 'adaptive':
        return rsi_adaptive(data, period)
    else:
        return rsi_wilder(data, period)  # Default to Wilder


def macd_advanced(data, fast=12, slow=26, signal=9):
    """MACD with histogram"""
    macd_line, signal_line, histogram = _macd_advanced_numba(data.values, fast, slow, signal)
    return (pd.Series(macd_line, index=data.index),
            pd.Series(signal_line, index=data.index),
            pd.Series(histogram, index=data.index))

def atr_advanced(high, low, close, period=14):
    """Average True Range"""
    return pd.Series(_atr_numba(high.values, low.values, close.values, period), index=close.index)

def supertrend_advanced(high, low, close, period=12, multiplier=3):
    """SuperTrend indicator"""
    return pd.Series(_supertrend_numba(high.values, low.values, close.values, period, multiplier), index=close.index)

def bop_advanced(open_price, high, low, close, period=14):
    """Balance of Power"""
    return pd.Series(_bop_numba(open_price.values, high.values, low.values, close.values, period), index=close.index)

def ultimate_oscillator_advanced(high, low, close, period1=2, period2=4, period3=6):
    """Ultimate Oscillator"""
    return pd.Series(_ultimate_oscillator_numba(high.values, low.values, close.values, period1, period2, period3), index=close.index)

def ibs_advanced(high, low, close):
    """Internal Bar Strength"""
    return pd.Series(_ibs_numba(high.values, low.values, close.values), index=close.index)

def higher_highs_advanced(data, periods=5):
    """Check for consecutive higher highs"""
    return pd.Series(_higher_highs_numba(data.values, periods), index=data.index)

def find_peaks_troughs(data, window=5):
    """Find peaks and troughs in data"""
    peaks, troughs = _find_peaks_troughs_numba(data.values, window)
    return pd.Series(peaks, index=data.index), pd.Series(troughs, index=data.index)

def detect_divergence_slope(price_data, indicator_data, lookback=10):
    """Slope-based divergence detection"""
    bull_div, bear_div, hdiv_bull, hdiv_bear = _detect_divergence_slope_numba(
        price_data.values, indicator_data.values, lookback)
    
    return {
        'div_bull': pd.Series(bull_div, index=price_data.index),
        'div_bear': pd.Series(bear_div, index=price_data.index),
        'hdiv_bull': pd.Series(hdiv_bull, index=price_data.index),
        'hdiv_bear': pd.Series(hdiv_bear, index=price_data.index)
    }

def detect_divergence_pivot(price_data, indicator_data, lookback=5):
    """Pivot-based divergence detection (simplified for speed)"""
    peaks_price, troughs_price = find_peaks_troughs(price_data, lookback)
    
    bull_div = pd.Series(False, index=price_data.index)
    bear_div = pd.Series(False, index=price_data.index)
    
    # Simplified pivot detection for speed
    trough_indices = price_data.index[troughs_price]
    for i in range(1, len(trough_indices)):
        curr_idx = trough_indices[i]
        prev_idx = trough_indices[i-1]
        if (price_data.loc[curr_idx] < price_data.loc[prev_idx] and 
            indicator_data.loc[curr_idx] > indicator_data.loc[prev_idx]):
            bull_div.loc[curr_idx] = True
    
    peak_indices = price_data.index[peaks_price]
    for i in range(1, len(peak_indices)):
        curr_idx = peak_indices[i]
        prev_idx = peak_indices[i-1]
        if (price_data.loc[curr_idx] > price_data.loc[prev_idx] and 
            indicator_data.loc[curr_idx] < indicator_data.loc[prev_idx]):
            bear_div.loc[curr_idx] = True
    
    return {
        'div_bull': bull_div,
        'div_bear': bear_div,
        'hdiv_bull': pd.Series(False, index=price_data.index),
        'hdiv_bear': pd.Series(False, index=price_data.index)
    }

def cross_above(data1, data2):
    """Check if data1 crosses above data2"""
    return pd.Series(_cross_above_numba(data1.values, data2.values), index=data1.index)

def cross_below(data1, data2):
    """Check if data1 crosses below data2"""
    return pd.Series(_cross_below_numba(data1.values, data2.values), index=data1.index)

def rei(df: pd.DataFrame, period: int = 8) -> pd.Series:
    """
    Calculate Tom DeMark's Range Expansion Index.

    Parameters
    ----------
    df : DataFrame with columns 'High', 'Low', 'Close'
    period : look-back length (default 8)

    Returns
    -------
    pandas Series (float) ranging -100 .. +100
    """
    h, l, c = df['High'], df['Low'], df['Close']
    dh, dl = h - h.shift(2), l - l.shift(2)

    # boolean filters
    up_cap   = (h.shift(2) < c.shift(7)) & (h.shift(2) < c.shift(8)) & \
               (h          < h.shift(5)) & (h          < h.shift(6))
    dn_cap   = (l.shift(2) > c.shift(7)) & (l.shift(2) > c.shift(8)) & \
               (l          > l.shift(5)) & (l          > l.shift(6))

    num = (1 - up_cap.astype(int)) * (1 - dn_cap.astype(int)) * (dh + dl)
    den = np.abs(dh) + np.abs(dl)

    s1 = num.rolling(period).sum()
    s2 = den.rolling(period).sum().replace(0, np.nan)

    return 100 * s1 / s2



def compute_atr_and_natr_multi(df, periods=[2, 14, 18]):
    """
    Compute both ATR and NATR for multiple periods on a single ticker’s DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['High', 'Low', 'Close'].
    periods : list of int, optional
        The list of ATR/NATR periods to compute, e.g. [2, 14, 18].

    Returns
    -------
    pd.DataFrame
        The same DataFrame with columns:
            'atr_{p}', 'natr_{p}' for each p in periods.
    """
    # 1) Calculate True Range components once (to avoid redoing it for each period)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close  = (df['Low'] - df['Close'].shift(1)).abs()
    
    # Combine into a single "true_range" Series
    tr = high_low.to_frame('HL')
    tr['HC'] = high_close
    tr['LC'] = low_close
    tr['true_range'] = tr.max(axis=1)

    # 2) For each period, compute ATR & NATR
    for period in periods:
        atr_col_name = f'atr_{period}'
        natr_col_name = f'natr_{period}'

        # Simple rolling mean for ATR; 
        # if you prefer an EMA, do .ewm(span=period, adjust=False).mean() 
        df[atr_col_name] = (
            tr['true_range'].rolling(window=period, min_periods=1).mean()
        )

        # NATR = 100 * (ATR / Close)
        df[natr_col_name] = (
            100.0 * df[atr_col_name] / df['Close']
        )

    return df

def build_symbol_specific_indicators(stock_data):
    symbol_indicators = {}

    #BDRY#
    # Create a copy of the BDRY data from stock_data
    if 'BDRY' in stock_data:
        bdry_df = stock_data['BDRY'].copy()

        # Add indicators to the bdry_df DataFrame
        bdry_df['bdry_roc2'] = ROCIndicator(close=bdry_df['Close'], window=2).roc()
        bdry_df['bdry_rsi2'] = RSIIndicator(close=bdry_df['Close'], window=2).rsi()
        bdry_df['bdry_sma_20'] = SMAIndicator(close=bdry_df['Close'], window=20).sma_indicator()
        bdry_df['bdry_sma_50'] = SMAIndicator(close=bdry_df['Close'], window=50).sma_indicator()
        bdry_df['bdry_sma_100'] = SMAIndicator(close=bdry_df['Close'], window=100).sma_indicator()
        bdry_df['bdry_sma_200'] = SMAIndicator(close=bdry_df['Close'], window=200).sma_indicator()
        bdry_df['bdry_kama_20'] = KAMAIndicator(close=bdry_df['Close'], window=20, pow1=2, pow2=14).kama()
        bdry_df['bdry_kama_50'] = KAMAIndicator(close=bdry_df['Close'], window=50, pow1=2, pow2=14).kama()
        bdry_df['bdry_uo135'] = UltimateOscillator(high=bdry_df['High'], low=bdry_df['Low'],close=bdry_df['Close'],window1=1,window2=3,window3=5,weight1=3.0,weight2=6.0,weight3=12.0,fillna=False).ultimate_oscillator()
        # Store the indicators in symbol_indicators dictionary
        symbol_indicators['BDRY'] = bdry_df[
            [
                'bdry_roc2', 'bdry_rsi2', 
                'bdry_sma_20', 'bdry_sma_50', 'bdry_sma_100', 'bdry_sma_200', 
                'bdry_kama_20', 'bdry_kama_50', 'bdry_uo135'
            ]
            ]
    else:
        logging.warning("Ticker 'BDRY' not found in stock data. Skipping symbol-specific indicators.")
    
    # Create a copy of the BIL data from stock_data
    if 'BIL' in stock_data:
        bil_df = stock_data['BIL'].copy()

        # Add indicators to the bil_df DataFrame
        bil_df['bil_rsi7'] = RSIIndicator(close=bil_df['Close'], window=7).rsi()
        bil_df['bil_rsi5'] = RSIIndicator(close=bil_df['Close'], window=5).rsi()
        bil_df['bil_rsi10'] = RSIIndicator(close=bil_df['Close'], window=10).rsi()

        # Store the indicators in symbol_indicators dictionary
        symbol_indicators['BIL'] = bil_df[['bil_rsi7','bil_rsi5','bil_rsi10']]
    else:
        logging.warning("Ticker 'BIL' not found in stock data. Skipping symbol-specific indicators.")


    #BND#
    if 'BND' in stock_data:
        # Create a copy of the BND data from stock_data
        bnd_df = stock_data['BND'].copy()

        # Add indicators to the bnd_df DataFrame
        bnd_df['bnd_close'] = bnd_df['Close']  # Adding the close prices as a separate column for clarity
        bnd_df['bnd_rsi2'] = RSIIndicator(close=bnd_df['Close'], window=2).rsi()
        bnd_df['bnd_rsi3'] = RSIIndicator(close=bnd_df['Close'], window=3).rsi()
        bnd_df['bnd_rsi5'] = RSIIndicator(close=bnd_df['Close'], window=5).rsi()


        bnd_df['bnd_roc2'] = ROCIndicator(close=bnd_df['Close'], window=2).roc()
        bnd_df['bnd_roc7'] = ROCIndicator(close=bnd_df['Close'], window=7).roc()
        bnd_df['bnd_roc7_shift1'] = bnd_df['bnd_roc7'].shift(1)

        bnd_df['bnd_roc14'] = ROCIndicator(close=bnd_df['Close'], window=14).roc()
        bnd_df['bnd_roc20'] = ROCIndicator(close=bnd_df['Close'], window=20).roc()

        bnd_df['bnd_sma_200'] = SMAIndicator(close=bnd_df['Close'], window=200).sma_indicator()

        # Volume Weighted Average Price (VWAP) indicators
        bnd_df['bnd_vwap2'] = VolumeWeightedAveragePrice(
            high=bnd_df['High'], low=bnd_df['Low'], close=bnd_df['Close'], volume=bnd_df['Volume'], window=2
        ).volume_weighted_average_price()

        bnd_df['bnd_vwap4'] = VolumeWeightedAveragePrice(
            high=bnd_df['High'], low=bnd_df['Low'], close=bnd_df['Close'], volume=bnd_df['Volume'], window=4
        ).volume_weighted_average_price()

        bnd_df['bnd_vwap14'] = VolumeWeightedAveragePrice(
            high=bnd_df['High'], low=bnd_df['Low'], close=bnd_df['Close'], volume=bnd_df['Volume'], window=14
        ).volume_weighted_average_price()

        # KST Indicator for BND
        bnd_kst246 = KSTIndicator(close=bnd_df['Close'], roc1=2, roc2=4, roc3=6, roc4=8, 
                                window1=1, window2=4, window3=8, window4=12, nsig=3)
        bnd_df['kst_bnd246'] = bnd_kst246.kst()
        bnd_df['kst_sig_bnd246'] = bnd_kst246.kst_sig()
        bnd_df['kst_diff_bnd246'] = bnd_kst246.kst_diff()

        # Store the indicators in symbol_indicators dictionary
        symbol_indicators['BND'] = bnd_df[
            [
                'bnd_roc2', 'bnd_roc7', 'bnd_roc14', 'bnd_roc20', 
                'bnd_vwap2', 'bnd_vwap4', 'bnd_vwap14', 
                'kst_bnd246', 'kst_sig_bnd246', 'kst_diff_bnd246',
                'bnd_roc7_shift1','bnd_rsi2','bnd_rsi3','bnd_rsi5','bnd_close','bnd_sma_200'
            ]
        ]
    else:
        logging.warning("Ticker 'BND' not found in stock data. Skipping symbol-specific indicators.")



    # FDN
    if 'FDN' in stock_data:
        # Create a copy of the FDN data from stock_data
        fdn_df = stock_data['FDN'].copy()

        # Add RSI indicators with different windows to the fdn_df DataFrame
        fdn_df['fdn_rsi2'] = RSIIndicator(close=fdn_df['Close'], window=2).rsi()
        fdn_df['fdn_rsi7'] = RSIIndicator(close=fdn_df['Close'], window=7).rsi()
        fdn_df['fdn_rsi10'] = RSIIndicator(close=fdn_df['Close'], window=10).rsi()
        fdn_df['fdn_rsi20'] = RSIIndicator(close=fdn_df['Close'], window=20).rsi()
        fdn_df['fdn_rsi200'] = RSIIndicator(close=fdn_df['Close'], window=200).rsi()

        # Create shifted versions for the 20-day RSI indicator
        for i in range(1, 5):  # Creates shifts from 1 to 4
            fdn_df[f'fdn_rsi20_shift{i}'] = fdn_df['fdn_rsi20'].shift(i)

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['FDN'] = fdn_df[
            ['fdn_rsi2', 'fdn_rsi7', 'fdn_rsi10', 'fdn_rsi20', 'fdn_rsi200'] +
            [f'fdn_rsi20_shift{i}' for i in range(1, 5)]
        ]
    else:
        logging.warning("Ticker 'FDN' not found in stock data. Skipping symbol-specific indicators.")


    # Create a copy of the GBTC data from stock_data
    if 'GBTC' in stock_data:
        gbtc_df = stock_data['GBTC'].copy()

        # Add indicators to the gbtc_df DataFrame
        gbtc_df['gbtc_close'] = gbtc_df['Close']  # Adding the close prices as a separate column for clarity
        gbtc_df['gbtc_rsi2'] = RSIIndicator(close=gbtc_df['Close'], window=2).rsi()
        gbtc_df['gbtc_rsi3'] = RSIIndicator(close=gbtc_df['Close'], window=3).rsi()
        gbtc_df['gbtc_rsi5'] = RSIIndicator(close=gbtc_df['Close'], window=5).rsi()

        gbtc_df['gbtc_roc2'] = ROCIndicator(close=gbtc_df['Close'], window=2).roc()
        gbtc_df['gbtc_roc7'] = ROCIndicator(close=gbtc_df['Close'], window=7).roc()
        gbtc_df['gbtc_roc7_shift1'] = gbtc_df['gbtc_roc7'].shift(1)

        gbtc_df['gbtc_roc14'] = ROCIndicator(close=gbtc_df['Close'], window=14).roc()
        gbtc_df['gbtc_roc20'] = ROCIndicator(close=gbtc_df['Close'], window=20).roc()

        gbtc_df['gbtc_sma_100'] = SMAIndicator(close=gbtc_df['Close'], window=100).sma_indicator()
        gbtc_df['gbtc_sma_200'] = SMAIndicator(close=gbtc_df['Close'], window=200).sma_indicator()

        # Volume Weighted Average Price (VWAP) indicators
        gbtc_df['gbtc_vwap2'] = VolumeWeightedAveragePrice(
            high=gbtc_df['High'], low=gbtc_df['Low'], close=gbtc_df['Close'], volume=gbtc_df['Volume'], window=2
        ).volume_weighted_average_price()

        gbtc_df['gbtc_vwap4'] = VolumeWeightedAveragePrice(
            high=gbtc_df['High'], low=gbtc_df['Low'], close=gbtc_df['Close'], volume=gbtc_df['Volume'], window=4
        ).volume_weighted_average_price()

        gbtc_df['gbtc_vwap14'] = VolumeWeightedAveragePrice(
            high=gbtc_df['High'], low=gbtc_df['Low'], close=gbtc_df['Close'], volume=gbtc_df['Volume'], window=14
        ).volume_weighted_average_price()

        # KST Indicator for GBTC
        gbtc_kst246 = KSTIndicator(close=gbtc_df['Close'], roc1=2, roc2=4, roc3=6, roc4=8, 
                                window1=1, window2=4, window3=8, window4=12, nsig=3)
        gbtc_df['kst_gbtc246'] = gbtc_kst246.kst()
        gbtc_df['kst_sig_gbtc246'] = gbtc_kst246.kst_sig()
        gbtc_df['kst_diff_gbtc246'] = gbtc_kst246.kst_diff()

        # Store the indicators in symbol_indicators dictionary
        symbol_indicators['GBTC'] = gbtc_df[
            [
                'gbtc_roc2', 'gbtc_roc7', 'gbtc_roc14', 'gbtc_roc20', 'gbtc_vwap2',
                'gbtc_vwap4', 'gbtc_vwap14', 'kst_gbtc246', 'kst_sig_gbtc246', 
                'kst_diff_gbtc246','gbtc_roc7_shift1','gbtc_rsi2','gbtc_rsi3','gbtc_rsi5','gbtc_sma_100','gbtc_sma_200','gbtc_close'
            ]
        ]
    else:
        logging.warning("Ticker 'GBTC' not found in stock data. Skipping symbol-specific indicators.")

    #IOO#
    if 'IOO' in stock_data:
        # Create a copy of the KMLM data from stock_data
        ioo_df = stock_data['IOO'].copy()

        # Add indicators to the kmlm_df DataFrame
        ioo_df['ioo_close'] = ioo_df['Close']  # Adding the close prices as a separate column for clarity
        ioo_df['ioo_rsi10'] = RSIIndicator(close=ioo_df['Close'], window=10).rsi()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['IOO'] = ioo_df[['ioo_close', 'ioo_rsi10']]
    else:
        logging.warning("Ticker 'IOO' not found in stock data. Skipping symbol-specific indicators.")

    #IEF#
    if 'IEF' in stock_data:
        # Create a copy of the KMLM data from stock_data
        ief_df = stock_data['IEF'].copy()

        # Add indicators to the kmlm_df DataFrame
        ief_df['ief_close'] = ief_df['Close']  # Adding the close prices as a separate column for clarity
        ief_df['ief_rsi5'] = RSIIndicator(close=ief_df['Close'], window=5).rsi()
        ief_df['ief_rsi10'] = RSIIndicator(close=ief_df['Close'], window=10).rsi()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['IEF'] = ief_df[['ief_close','ief_rsi10','ief_rsi5']]
    else:
        logging.warning("Ticker 'IEF' not found in stock data. Skipping symbol-specific indicators.")

    #IYW#
    if 'IYW' in stock_data:
        # Create a copy of the KMLM data from stock_data
        iyw_df = stock_data['IYW'].copy()

        # Add indicators to the kmlm_df DataFrame
        iyw_df['iyw_close'] = iyw_df['Close']  # Adding the close prices as a separate column for clarity
        iyw_df['iyw_rsi8'] = RSIIndicator(close=iyw_df['Close'], window=8).rsi()
        iyw_df['iyw_rsi10'] = RSIIndicator(close=iyw_df['Close'], window=10).rsi()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['IYW'] = iyw_df[['iyw_close', 'iyw_rsi10', 'iyw_rsi8']]
    else:
        logging.warning("Ticker 'IYW' not found in stock data. Skipping symbol-specific indicators.")

    #KMLM#
    if 'KMLM' in stock_data:
        # Create a copy of the KMLM data from stock_data
        kmlm_df = stock_data['KMLM'].copy()

        # Add indicators to the kmlm_df DataFrame
        kmlm_df['kmlm_close'] = kmlm_df['Close']  # Adding the close prices as a separate column for clarity
        kmlm_df['kmlm_sma_20'] = SMAIndicator(close=kmlm_df['Close'], window=20).sma_indicator()
        kmlm_df['kmlm_rsi10'] = RSIIndicator(close=kmlm_df['Close'], window=10).rsi()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['KMLM'] = kmlm_df[['kmlm_close', 'kmlm_sma_20', 'kmlm_rsi10']]
    else:
        logging.warning("Ticker 'KMLM' not found in stock data. Skipping symbol-specific indicators.")



    #LQD#
    if 'LQD' in stock_data:
        # Create a copy of the LQD data from stock_data
        lqd_df = stock_data['LQD'].copy()

        # Add the 7-day RSI indicator to the lqd_df DataFrame
        lqd_df['lqd_rsi7'] = RSIIndicator(close=lqd_df['Close'], window=7).rsi()
        lqd_df['lqd_rsi10'] = RSIIndicator(close=lqd_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['LQD'] = lqd_df[['lqd_rsi7','lqd_rsi10']]
    else:
        logging.warning("Ticker 'LQD' not found in stock data. Skipping symbol-specific indicators.")

    #PSQ#
    if 'PSQ' in stock_data:
        # Create a copy of the PSQ data from stock_data
        psq_df = stock_data['PSQ'].copy()

        # Add the 10-day RSI indicator to the psq_df DataFrame
        psq_df['psq_rsi10'] = RSIIndicator(close=psq_df['Close'], window=10).rsi()
        psq_df['psq_rsi20'] = RSIIndicator(close=psq_df['Close'], window=20).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['PSQ'] = psq_df[['psq_rsi10','psq_rsi20']]
    else:
        logging.warning("Ticker 'PSQ' not found in stock data. Skipping symbol-specific indicators.")

    #QLD Add special QLD IBS indicator (needed for XLP exit conditions)
    if 'QLD' in stock_data:
        qld_df = stock_data['QLD'].copy()
        qld_df['qld_ibs'] = ibs_advanced(qld_df['High'], qld_df['Low'], qld_df['Close'])
        symbol_indicators['QLD'] = qld_df[['qld_ibs']]
    else:
        logging.warning("Ticker 'QLD' not found in stock data. Skipping QLD IBS indicator.")

    #QQQ#
    if 'QQQ' in stock_data:
        # Create a copy of the QQQ data from stock_data
        qqq_df = stock_data['QQQ'].copy()

        # Add the 10-day RSI indicator to the qqq_df DataFrame
        qqq_df['qqq_rsi2'] = RSIIndicator(close=qqq_df['Close'], window=2).rsi()
        qqq_df['qqq_rsi3'] = RSIIndicator(close=qqq_df['Close'], window=3).rsi()
        qqq_df['qqq_rsi10'] = RSIIndicator(close=qqq_df['Close'], window=10).rsi()

        # Add advanced RSI methods for QQQ
        qqq_df['qqq_rsi10_wilder'] = rsi_advanced(qqq_df['Close'], 10, 'wilder')
        # qqq_df['qqq_rsi10_ema'] = rsi_advanced(qqq_df['Close'], 10, 'ema')
        qqq_df['qqq_rsi10_adaptive'] = rsi_advanced(qqq_df['Close'], 10, 'adaptive')
        # qqq_df['qqq_rsi10_average'] = rsi_advanced(qqq_df['Close'], 10, 'average')

        # Add SMA and IBS for QQQ
        qqq_df['qqq_sma20'] = sma(qqq_df['Close'], 20)
        qqq_df['qqq_ibs'] = ibs_advanced(qqq_df['High'], qqq_df['Low'], qqq_df['Close'])

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['QQQ'] = qqq_df[['qqq_rsi10','qqq_rsi2','qqq_rsi3', 'qqq_rsi10_wilder', 
                                           'qqq_rsi10_adaptive',
                                          'qqq_sma20', 'qqq_ibs']]
    else:
        logging.warning("Ticker 'QQQ' not found in stock data. Skipping symbol-specific indicators.")

    #QQQE#
    if 'QQQE' in stock_data:
        # Create a copy of the QQQE data from stock_data
        qqqe_df = stock_data['QQQE'].copy()

        # Add the 10-day RSI indicator to the qqqe_df DataFrame
        qqqe_df['qqqe_rsi10'] = RSIIndicator(close=qqqe_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['QQQE'] = qqqe_df[['qqqe_rsi10']]
    else:
        logging.warning("Ticker 'QQQE' not found in stock data. Skipping symbol-specific indicators.")

    #SOXL#
    if 'SOXL' in stock_data:
        # Create a copy of the SOXL data from stock_data
        soxl_df = stock_data['SOXL'].copy()

        # Add indicators to the soxl_df DataFrame
        soxl_df['soxl_close'] = soxl_df['Close']  # Adding the close prices as a separate column for clarity
        soxl_df['soxl_rsi2'] = RSIIndicator(close=soxl_df['Close'], window=2).rsi()
        soxl_df['soxl_rsi6'] = RSIIndicator(close=soxl_df['Close'], window=6).rsi()
        soxl_df['soxl_rsi7'] = RSIIndicator(close=soxl_df['Close'], window=7).rsi()
        soxl_df['soxl_rsi10'] = RSIIndicator(close=soxl_df['Close'], window=10).rsi()
        soxl_df['soxl_sma_20'] = SMAIndicator(close=soxl_df['Close'], window=20).sma_indicator()
        soxl_df['soxl_sma_200'] = SMAIndicator(close=soxl_df['Close'], window=200).sma_indicator()

        # Add advanced RSI methods for SOXL
        soxl_df['soxl_rsi10_wilder'] = rsi_advanced(soxl_df['Close'], 10, 'wilder')
        soxl_df['soxl_rsi10_ema'] = rsi_advanced(soxl_df['Close'], 10, 'ema')
        soxl_df['soxl_rsi10_adaptive'] = rsi_advanced(soxl_df['Close'], 10, 'adaptive')
        # soxl_df['soxl_rsi10_average'] = rsi_advanced(soxl_df['Close'], 10, 'average')

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['SOXL'] = soxl_df[['soxl_close', 'soxl_rsi2','soxl_rsi6',
                                              'soxl_rsi7', 'soxl_rsi10', 'soxl_sma_20', 'soxl_sma_200',
                                              'soxl_rsi10_wilder',
                                            'soxl_rsi10_adaptive','soxl_rsi10_ema']]
    else:
        logging.warning("Ticker 'SOXL' not found in stock data. Skipping symbol-specific indicators.")

    #SOXX#
    if 'SOXX' in stock_data:
        # Create a copy of the SOXX data from stock_data
        soxx_df = stock_data['SOXX'].copy()

        # Add indicators to the soxx_df DataFrame
        soxx_df['soxx_close'] = soxx_df['Close']  # Adding the close prices as a separate column for clarity
        soxx_df['soxx_rsi2'] = RSIIndicator(close=soxx_df['Close'], window=2).rsi()
        soxx_df['soxx_rsi6'] = RSIIndicator(close=soxx_df['Close'], window=6).rsi()
        soxx_df['soxx_rsi7'] = RSIIndicator(close=soxx_df['Close'], window=7).rsi()
        soxx_df['soxx_rsi8'] = RSIIndicator(close=soxx_df['Close'], window=8).rsi()
        soxx_df['soxx_rsi10'] = RSIIndicator(close=soxx_df['Close'], window=10).rsi()
        soxx_df['soxx_sma_20'] = SMAIndicator(close=soxx_df['Close'], window=20).sma_indicator()
        soxx_df['soxx_sma_200'] = SMAIndicator(close=soxx_df['Close'], window=200).sma_indicator()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['SOXX'] = soxx_df[['soxx_close', 'soxx_rsi2','soxx_rsi6', 'soxx_rsi7', 'soxx_rsi8', 'soxx_rsi10', 'soxx_sma_20', 'soxx_sma_200']]
    else:
        logging.warning("Ticker 'SOXX' not found in stock data. Skipping symbol-specific indicators.")

    #SPHB#
    if 'SPHB' in stock_data:
        # Create a copy of the SPHB data from stock_data
        sphb_df = stock_data['SPHB'].copy()

        # Add indicators to the sphb_df DataFrame
        sphb_df['sphb_close'] = sphb_df['Close']  # Adding the close prices as a separate column for clarity
        sphb_df['sphb_rsi8'] = RSIIndicator(close=sphb_df['Close'], window=8).rsi()
        sphb_df['sphb_rsi10'] = RSIIndicator(close=sphb_df['Close'], window=10).rsi()


        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['SPHB'] = sphb_df[['sphb_close', 'sphb_rsi10', 'sphb_rsi8']]
    else:
        logging.warning("Ticker 'SPHB' not found in stock data. Skipping symbol-specific indicators.")

    #SPIB#
    if 'SPIB' in stock_data:
        # Create a copy of the SPIB data from stock_data
        spib_df = stock_data['SPIB'].copy()

        # Add indicators to the spib_df DataFrame
        spib_df['spib_close'] = spib_df['Close']  # Adding the close prices as a separate column for clarity
        spib_df['spib_rsi8'] = RSIIndicator(close=spib_df['Close'], window=8).rsi()
        spib_df['spib_rsi10'] = RSIIndicator(close=spib_df['Close'], window=10).rsi()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['SPIB'] = spib_df[['spib_close', 'spib_rsi10', 'spib_rsi8']]
    else:
        logging.warning("Ticker 'SPIB' not found in stock data. Skipping symbol-specific indicators.")

     #SPXL#
    if 'SPXL' in stock_data:
        # Create a copy of the SPXL data from stock_data
        spxl_df = stock_data['SPXL'].copy()

        # Add indicators to the spxl_df DataFrame
        spxl_df['spxl_close'] = spxl_df['Close']  # Adding the close prices as a separate column for clarity
        spxl_df['spxl_rsi2'] = RSIIndicator(close=spxl_df['Close'], window=2).rsi()
        spxl_df['spxl_rsi6'] = RSIIndicator(close=spxl_df['Close'], window=6).rsi()
        spxl_df['spxl_rsi7'] = RSIIndicator(close=spxl_df['Close'], window=7).rsi()
        spxl_df['spxl_rsi10'] = RSIIndicator(close=spxl_df['Close'], window=10).rsi()
        spxl_df['spxl_sma_20'] = SMAIndicator(close=spxl_df['Close'], window=20).sma_indicator()
        spxl_df['spxl_sma_200'] = SMAIndicator(close=spxl_df['Close'], window=200).sma_indicator()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['SPXL'] = spxl_df[['spxl_close', 'spxl_rsi2','spxl_rsi6', 'spxl_rsi7', 'spxl_rsi10', 'spxl_sma_20', 'spxl_sma_200']]
    else:
        logging.warning("Ticker 'SPXL' not found in stock data. Skipping symbol-specific indicators.")

    #SPY#
    if 'SPY' in stock_data:
        # Create a copy of the SPY data from stock_data
        spy_df = stock_data['SPY'].copy()

        # Add indicators to the spy_df DataFrame
        spy_df['spy_close'] = spy_df['Close']  # Adding the close prices as a separate column for clarity
        spy_df['spy_rsi2'] = RSIIndicator(close=spy_df['Close'], window=2).rsi()
        spy_df['spy_rsi3'] = RSIIndicator(close=spy_df['Close'], window=3).rsi()

        spy_df['spy_rsi5'] = RSIIndicator(close=spy_df['Close'], window=5).rsi()
        spy_df['spy_rsi4'] = RSIIndicator(close=spy_df['Close'], window=4).rsi()
        spy_df['spy_rsi6'] = RSIIndicator(close=spy_df['Close'], window=6).rsi()
        spy_df['spy_rsi7'] = RSIIndicator(close=spy_df['Close'], window=7).rsi()
        spy_df['spy_rsi8'] = RSIIndicator(close=spy_df['Close'], window=8).rsi()
        spy_df['spy_rsi9'] = RSIIndicator(close=spy_df['Close'], window=9).rsi()
        spy_df['spy_rsi10'] = RSIIndicator(close=spy_df['Close'], window=10).rsi()
        spy_df['spy_sma_20'] = SMAIndicator(close=spy_df['Close'], window=20).sma_indicator()
        spy_df['spy_sma_200'] = SMAIndicator(close=spy_df['Close'], window=200).sma_indicator()

        # Add advanced RSI methods for SPY (based on entry/exit conditions)
        spy_df['spy_rsi2_wilder'] = rsi_advanced(spy_df['Close'], 2, 'wilder')
        # spy_df['spy_rsi2_ema'] = rsi_advanced(spy_df['Close'], 2, 'ema')
        spy_df['spy_rsi2_adaptive'] = rsi_advanced(spy_df['Close'], 2, 'adaptive')
        # spy_df['spy_rsi2_average'] = rsi_advanced(spy_df['Close'], 2, 'average')
        
        spy_df['spy_rsi3_wilder'] = rsi_advanced(spy_df['Close'], 3, 'wilder')
        # spy_df['spy_rsi3_ema'] = rsi_advanced(spy_df['Close'], 3, 'ema')
        spy_df['spy_rsi3_adaptive'] = rsi_advanced(spy_df['Close'], 3, 'adaptive')
        # spy_df['spy_rsi3_average'] = rsi_advanced(spy_df['Close'], 3, 'average')
        
        spy_df['spy_rsi10_wilder'] = rsi_advanced(spy_df['Close'], 10, 'wilder')
        # spy_df['spy_rsi10_ema'] = rsi_advanced(spy_df['Close'], 10, 'ema')
        spy_df['spy_rsi10_adaptive'] = rsi_advanced(spy_df['Close'], 10, 'adaptive')
        # spy_df['spy_rsi10_average'] = rsi_advanced(spy_df['Close'], 10, 'average')

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['SPY'] = spy_df[['spy_close', 'spy_rsi2','spy_rsi3','spy_rsi5','spy_rsi4','spy_rsi6', 'spy_rsi7','spy_rsi8','spy_rsi9', 'spy_rsi10', 'spy_sma_20', 'spy_sma_200',
                                          'spy_rsi2_wilder',  'spy_rsi2_adaptive', 
                                          'spy_rsi3_wilder', 'spy_rsi3_adaptive', 
                                          'spy_rsi10_wilder', 'spy_rsi10_adaptive' ]]
    else:
        logging.warning("Ticker 'SPY' not found in stock data. Skipping symbol-specific indicators.")


    #SQQQ#
    if 'SQQQ' in stock_data:
        # Create a copy of the SQQQ data from stock_data
        sqqq_df = stock_data['SQQQ'].copy()

        # Add RSI indicators with different windows to the sqqq_df DataFrame
        sqqq_df['sqqq_rsi2'] = RSIIndicator(close=sqqq_df['Close'], window=2).rsi()
        sqqq_df['sqqq_rsi3'] = RSIIndicator(close=sqqq_df['Close'], window=3).rsi()
        sqqq_df['sqqq_rsi4'] = RSIIndicator(close=sqqq_df['Close'], window=4).rsi()
        sqqq_df['sqqq_rsi5'] = RSIIndicator(close=sqqq_df['Close'], window=5).rsi()
        sqqq_df['sqqq_rsi6'] = RSIIndicator(close=sqqq_df['Close'], window=6).rsi()
        sqqq_df['sqqq_rsi7'] = RSIIndicator(close=sqqq_df['Close'], window=7).rsi()
        sqqq_df['sqqq_rsi10'] = RSIIndicator(close=sqqq_df['Close'], window=10).rsi()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['SQQQ'] = sqqq_df[['sqqq_rsi2', 'sqqq_rsi3', 'sqqq_rsi4', 'sqqq_rsi5', 'sqqq_rsi6', 'sqqq_rsi7', 'sqqq_rsi10']]
    else:
        logging.warning("Ticker 'SQQQ' not found in stock data. Skipping symbol-specific indicators.")

    #SSO#
    if 'SSO' in stock_data:
        # Create a copy of the SQQQ data from stock_data
        sso_df = stock_data['SSO'].copy()

        # Add RSI indicators with different windows to the sqqq_df DataFrame
        sso_df['sso_rsi10'] = RSIIndicator(close=sso_df['Close'], window=10).rsi()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['SSO'] = sso_df[[ 'sso_rsi10']]
    else:
        logging.warning("Ticker 'SSO' not found in stock data. Skipping symbol-specific indicators.")

    #TECL#
    if 'TECL' in stock_data:
        # Create a copy of the TECL data from stock_data
        tecl_df = stock_data['TECL'].copy()
        # Add the 10-day RSI indicator to the tecl_df DataFrame
        tecl_df['tecl_rsi10'] = RSIIndicator(close=tecl_df['Close'], window=10).rsi()

        # Add advanced RSI methods for TECL
        tecl_df['tecl_rsi10_wilder'] = rsi_advanced(tecl_df['Close'], 10, 'wilder')
        # tecl_df['tecl_rsi10_ema'] = rsi_advanced(tecl_df['Close'], 10, 'ema')
        tecl_df['tecl_rsi10_adaptive'] = rsi_advanced(tecl_df['Close'], 10, 'adaptive')
        # tecl_df['tecl_rsi10_average'] = rsi_advanced(tecl_df['Close'], 10, 'average')
        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['TECL'] = tecl_df[['tecl_rsi10', 'tecl_rsi10_wilder',  
                                            'tecl_rsi10_adaptive' ]]
    else:
        logging.warning("Ticker 'TECL' not found in stock data. Skipping symbol-specific indicators.")

    #TLT#
    if 'TLT' in stock_data:
        # Create a copy of the TYD data from stock_data
        tlt_df = stock_data['TLT'].copy()
        # Add the 7-day RSI indicator to the tyd_df DataFrame
        tlt_df['tlt_close'] = tlt_df['Close']  # Adding the close prices as a separate column for clarity
        tlt_df['tlt_rsi5'] = RSIIndicator(close=tlt_df['Close'], window=5).rsi()
        tlt_df['tlt_rsi7'] = RSIIndicator(close=tlt_df['Close'], window=7).rsi()
        tlt_df['tlt_rsi8'] = RSIIndicator(close=tlt_df['Close'], window=8).rsi()
        tlt_df['tlt_rsi10'] = RSIIndicator(close=tlt_df['Close'], window=10).rsi()
        tlt_df['tlt_sma_20'] = SMAIndicator(close=tlt_df['Close'], window=20).sma_indicator()
        tlt_df['tlt_sma_200'] = SMAIndicator(close=tlt_df['Close'], window=200).sma_indicator()
        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['TLT'] = tlt_df[['tlt_rsi5','tlt_rsi8','tlt_rsi10','tlt_rsi7','tlt_sma_20','tlt_sma_200','tlt_close']]
    else:
        logging.warning("Ticker 'TLT' not found in stock data. Skipping symbol-specific indicators.")


    #TQQQ#
    if 'TQQQ' in stock_data:
        # Create a copy of the TQQQ data from stock_data
        tqqq_df = stock_data['TQQQ'].copy()

        # Add close prices and indicators to the tqqq_df DataFrame
        tqqq_df['tqqq_close'] = tqqq_df['Close']  # Adding the close prices as a separate column for clarity
        tqqq_df['tqqq_rsi2'] = RSIIndicator(close=tqqq_df['Close'], window=2).rsi()
        tqqq_df['tqqq_rsi3'] = RSIIndicator(close=tqqq_df['Close'], window=3).rsi()
        tqqq_df['tqqq_rsi4'] = RSIIndicator(close=tqqq_df['Close'], window=4).rsi()
        tqqq_df['tqqq_rsi5'] = RSIIndicator(close=tqqq_df['Close'], window=5).rsi()
        tqqq_df['tqqq_rsi6'] = RSIIndicator(close=tqqq_df['Close'], window=6).rsi()
        tqqq_df['tqqq_rsi7'] = RSIIndicator(close=tqqq_df['Close'], window=7).rsi()
        tqqq_df['tqqq_rsi8'] = RSIIndicator(close=tqqq_df['Close'], window=8).rsi()
        tqqq_df['tqqq_rsi10'] = RSIIndicator(close=tqqq_df['Close'], window=10).rsi()
        tqqq_df['tqqq_sma_20'] = SMAIndicator(close=tqqq_df['Close'], window=20).sma_indicator()
        tqqq_df['tqqq_sma_200'] = SMAIndicator(close=tqqq_df['Close'], window=200).sma_indicator()

        # Add advanced RSI methods for TQQQ
        tqqq_df['tqqq_rsi10_wilder'] = rsi_advanced(tqqq_df['Close'], 10, 'wilder')
        tqqq_df['tqqq_rsi10_ema'] = rsi_advanced(tqqq_df['Close'], 10, 'ema')
        tqqq_df['tqqq_rsi10_adaptive'] = rsi_advanced(tqqq_df['Close'], 10, 'adaptive')
        # tqqq_df['tqqq_rsi10_average'] = rsi_advanced(tqqq_df['Close'], 10, 'average')
        # Add IBS for TQQQ
        tqqq_df['tqqq_ibs'] = ibs_advanced(tqqq_df['High'], tqqq_df['Low'], tqqq_df['Close'])

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['TQQQ'] = tqqq_df[
            [
                'tqqq_close', 'tqqq_rsi2', 'tqqq_rsi3', 'tqqq_rsi4', 'tqqq_rsi5', 
                'tqqq_rsi6', 'tqqq_rsi7', 'tqqq_rsi8', 'tqqq_rsi10', 'tqqq_sma_20','tqqq_sma_200',
                'tqqq_rsi10_wilder',  'tqqq_rsi10_adaptive' ,
                'tqqq_ibs','tqqq_rsi10_ema'
            ]
        ]
    else:
        logging.warning("Ticker 'TQQQ' not found in stock data. Skipping symbol-specific indicators.")


    #UVXY#
    if 'UVXY' in stock_data:
        # Create a copy of the UVXY data from stock_data
        uvxy_df = stock_data['UVXY'].copy()
        uvxy_df['uvxy_close'] = uvxy_df['Close']  # Adding the close prices as a separate column for clarity

        # Add RSI indicators with different windows to the uvxy_df DataFrame
        uvxy_df['uvxy_rsi2'] = RSIIndicator(close=uvxy_df['Close'], window=2).rsi()
        uvxy_df['uvxy_rsi7'] = RSIIndicator(close=uvxy_df['Close'], window=7).rsi()
        uvxy_df['uvxy_rsi10'] = RSIIndicator(close=uvxy_df['Close'], window=10).rsi()
        uvxy_df['uvxy_10day_high_yesterday'] = uvxy_df['Close'].rolling(window=10).max().shift(1)

    # Calculate and add the 10-day high for UVXY
        uvxy_df['uvxy_10day_high'] = uvxy_df['Close'].rolling(window=10).max()
        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['UVXY'] = uvxy_df[['uvxy_rsi2', 'uvxy_rsi7', 'uvxy_rsi10','uvxy_10day_high','uvxy_10day_high_yesterday','uvxy_close']]
    else:
        logging.warning("Ticker 'UVXY' not found in stock data. Skipping symbol-specific indicators.")


    #VCSH#
    if 'VCSH' in stock_data:
        # Create a copy of the VOOG data from stock_data
        vcsh_df = stock_data['VCSH'].copy()

        # Add the 10-day RSI indicator to the voog_df DataFrame
        vcsh_df['vcsh_rsi10'] = RSIIndicator(close=vcsh_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['VCSH'] = vcsh_df[['vcsh_rsi10']]
    else:
        logging.warning("Ticker 'VCSH' not found in stock data. Skipping symbol-specific indicators.")

    #VGIT#
    if 'VGIT' in stock_data:
        # Create a copy of the VOOG data from stock_data
        vgit_df = stock_data['VGIT'].copy()

        # Add the 10-day RSI indicator to the voog_df DataFrame
        vgit_df['vgit_rsi7'] = RSIIndicator(close=vgit_df['Close'], window=7).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['VGIT'] = vgit_df[['vgit_rsi7']]
    else:
        logging.warning("Ticker 'VGIT' not found in stock data. Skipping symbol-specific indicators.")


    #VOOV#
    if 'VOOG' in stock_data:
        # Create a copy of the VOOG data from stock_data
        voog_df = stock_data['VOOG'].copy()

        # Add the 10-day RSI indicator to the voog_df DataFrame
        voog_df['voog_rsi10'] = RSIIndicator(close=voog_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['VOOG'] = voog_df[['voog_rsi10']]
    else:
        logging.warning("Ticker 'VOOG' not found in stock data. Skipping symbol-specific indicators.")


    #VOOV#
    if 'VOOV' in stock_data:
        # Create a copy of the VOOV data from stock_data
        voov_df = stock_data['VOOV'].copy()

        # Add the 10-day RSI indicator to the voov_df DataFrame
        voov_df['voov_rsi10'] = RSIIndicator(close=voov_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['VOOV'] = voov_df[['voov_rsi10']]
    else:
        logging.warning("Ticker 'VOOV' not found in stock data. Skipping symbol-specific indicators.")



    #VOX#
    if 'VOX' in stock_data:
        # Create a copy of the VOX data from stock_data
        vox_df = stock_data['VOX'].copy()

        # Add the 10-day RSI indicator to the vox_df DataFrame
        vox_df['vox_rsi10'] = RSIIndicator(close=vox_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['VOX'] = vox_df[['vox_rsi10']]
    else:
        logging.warning("Ticker 'VOX' not found in stock data. Skipping symbol-specific indicators.")

    #VTV#
    if 'VTV' in stock_data:
        # Create a copy of the VTV data from stock_data
        vtv_df = stock_data['VTV'].copy()

        # Add the 10-day RSI indicator to the vtv_df DataFrame
        vtv_df['vtv_rsi10'] = RSIIndicator(close=vtv_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['VTV'] = vtv_df[['vtv_rsi10']]
    else:
        logging.warning("Ticker 'VTV' not found in stock data. Skipping symbol-specific indicators.")


    #XLF#
    if 'XLF' in stock_data:
        # Create a copy of the XLF data from stock_data
        xlf_df = stock_data['XLF'].copy()

        # Add the 10-day RSI indicator to the xlp_df DataFrame
        xlf_df['xlf_rsi10'] = RSIIndicator(close=xlf_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['XLF'] = xlf_df[['xlf_rsi10']]
    else:
        logging.warning("Ticker 'XLF' not found in stock data. Skipping symbol-specific indicators.")

    #XLP#
    if 'XLP' in stock_data:
        # Create a copy of the XLP data from stock_data
        xlp_df = stock_data['XLP'].copy()

        # Add the 10-day RSI indicator to the xlp_df DataFrame
        xlp_df['xlp_rsi10'] = RSIIndicator(close=xlp_df['Close'], window=10).rsi()

        xlp_df['xlp_rsi10_wilder'] = rsi_advanced(xlp_df['Close'], 10, 'wilder')
        # xlp_df['xlp_rsi10_ema'] = rsi_advanced(xlp_df['Close'], 10, 'ema')
        xlp_df['xlp_rsi10_adaptive'] = rsi_advanced(xlp_df['Close'], 10, 'adaptive')
        # xlp_df['xlp_rsi10_average'] = rsi_advanced(xlp_df['Close'], 10, 'average')
        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['XLP'] = xlp_df[['xlp_rsi10','xlp_rsi10_wilder' ,'xlp_rsi10_adaptive' ]]
    else:
        logging.warning("Ticker 'XLP' not found in stock data. Skipping symbol-specific indicators.")

    #XLI#
    if 'XLI' in stock_data:
        # Create a copy of the XLI data from stock_data
        xli_df = stock_data['XLI'].copy()

        # Add RSI indicators with different windows to the xli_df DataFrame
        xli_df['xli_rsi2'] = RSIIndicator(close=xli_df['Close'], window=2).rsi()
        xli_df['xli_rsi7'] = RSIIndicator(close=xli_df['Close'], window=7).rsi()
        xli_df['xli_rsi10'] = RSIIndicator(close=xli_df['Close'], window=10).rsi()

        # Add SMA indicators with different windows
        xli_df['xli_sma_20'] = SMAIndicator(close=xli_df['Close'], window=20).sma_indicator()
        xli_df['xli_sma_50'] = SMAIndicator(close=xli_df['Close'], window=50).sma_indicator()
        xli_df['xli_sma_100'] = SMAIndicator(close=xli_df['Close'], window=100).sma_indicator()
        xli_df['xli_sma_200'] = SMAIndicator(close=xli_df['Close'], window=200).sma_indicator()

        # Add KAMA indicators with different windows
        xli_df['xli_kama_20'] = KAMAIndicator(close=xli_df['Close'], window=20, pow1=2, pow2=14).kama()
        xli_df['xli_kama_50'] = KAMAIndicator(close=xli_df['Close'], window=50, pow1=2, pow2=14).kama()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['XLI'] = xli_df[
            [
                'xli_rsi2', 'xli_rsi7', 'xli_rsi10', 
                'xli_sma_20', 'xli_sma_50', 'xli_sma_100', 'xli_sma_200', 
                'xli_kama_20', 'xli_kama_50'
            ]
        ]
    else:
        logging.warning("Ticker 'XLI' not found in stock data. Skipping symbol-specific indicators.")


    #XLK#
    if 'XLK' in stock_data:
        # Create a copy of the XLK data from stock_data
        xlk_df = stock_data['XLK'].copy()

        # Add the 10-day RSI indicator to the xlk_df DataFrame
        xlk_df['xlk_rsi10'] = RSIIndicator(close=xlk_df['Close'], window=10).rsi()
        xlk_df['xlk_rsi10_wilder'] = rsi_advanced(xlk_df['Close'], 10, 'wilder')
        # xlk_df['xlk_rsi10_ema'] = rsi_advanced(xlk_df['Close'], 10, 'ema')
        xlk_df['xlk_rsi10_adaptive'] = rsi_advanced(xlk_df['Close'], 10, 'adaptive')
        # xlk_df['xlk_rsi10_average'] = rsi_advanced(xlk_df['Close'], 10, 'average')
        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['XLK'] = xlk_df[['xlk_rsi10','xlk_rsi10_wilder' ,'xlk_rsi10_adaptive' ]]
    else:
        logging.warning("Ticker 'XLK' not found in stock data. Skipping symbol-specific indicators.")


    #XLY#
    if 'XLY' in stock_data:
        # Create a copy of the XLY data from stock_data
        xly_df = stock_data['XLY'].copy()

        # Add the 10-day RSI indicator to the xly_df DataFrame
        xly_df['xly_rsi10'] = RSIIndicator(close=xly_df['Close'], window=10).rsi()

        # Store the indicator in the symbol_indicators dictionary
        symbol_indicators['XLY'] = xly_df[['xly_rsi10']]
    else:
        logging.warning("Ticker 'XLY' not found in stock data. Skipping symbol-specific indicators.")

    # XLU
    if 'XLU' in stock_data:
        # Create a copy of the XLU data from stock_data
        xlu_df = stock_data['XLU'].copy()

        # Add the 20-day and 200-day RSI indicators to the xlu_df DataFrame
        xlu_df['xlu_rsi10'] = RSIIndicator(close=xlu_df['Close'], window=10).rsi()
        xlu_df['xlu_rsi20'] = RSIIndicator(close=xlu_df['Close'], window=20).rsi()
        xlu_df['xlu_rsi200'] = RSIIndicator(close=xlu_df['Close'], window=200).rsi()

        # Create shifted versions for the 20-day RSI indicator
        for i in range(1, 5):  # Creates shifts from 1 to 4
            xlu_df[f'xlu_rsi20_shift{i}'] = xlu_df['xlu_rsi20'].shift(i)

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['XLU'] = xlu_df[
            ['xlu_rsi10','xlu_rsi20', 'xlu_rsi200'] +
            [f'xlu_rsi20_shift{i}' for i in range(1, 5)]
        ]
    else:
        logging.warning("Ticker 'XLU' not found in stock data. Skipping symbol-specific indicators.")

    #XTN#
    if 'XTN' in stock_data:
        # Create a copy of the XTN data from stock_data
        xtn_df = stock_data['XTN'].copy()

        # Add RSI indicators with different windows to the xtn_df DataFrame
        xtn_df['xtn_rsi2'] = RSIIndicator(close=xtn_df['Close'], window=2).rsi()
        xtn_df['xtn_rsi7'] = RSIIndicator(close=xtn_df['Close'], window=7).rsi()
        xtn_df['xtn_rsi10'] = RSIIndicator(close=xtn_df['Close'], window=10).rsi()

        # Add SMA indicators with different windows to the xtn_df DataFrame
        xtn_df['xtn_sma_20'] = SMAIndicator(close=xtn_df['Close'], window=20).sma_indicator()
        xtn_df['xtn_sma_50'] = SMAIndicator(close=xtn_df['Close'], window=50).sma_indicator()
        xtn_df['xtn_sma_100'] = SMAIndicator(close=xtn_df['Close'], window=100).sma_indicator()
        xtn_df['xtn_sma_200'] = SMAIndicator(close=xtn_df['Close'], window=200).sma_indicator()

        # Add KAMA indicators with different windows to the xtn_df DataFrame
        xtn_df['xtn_kama_20'] = KAMAIndicator(close=xtn_df['Close'], window=20, pow1=2, pow2=14).kama()
        xtn_df['xtn_kama_50'] = KAMAIndicator(close=xtn_df['Close'], window=50, pow1=2, pow2=14).kama()

        # Store the indicators in the symbol_indicators dictionary
        symbol_indicators['XTN'] = xtn_df[
            [
                'xtn_rsi2', 'xtn_rsi7', 'xtn_rsi10', 
                'xtn_sma_20', 'xtn_sma_50', 'xtn_sma_100', 'xtn_sma_200', 
                'xtn_kama_20', 'xtn_kama_50'
            ]
        ]
    else:
        logging.warning("Ticker 'XTN' not found in stock data. Skipping symbol-specific indicators.")

    return symbol_indicators
def combine_all_symbol_indicators(symbol_indicators):
    """
    Merges every symbol-specific indicator DataFrame into a single wide DataFrame
    that has an index of ALL dates from all tickers, and columns for every ticker's 
    indicator, e.g. 'bdry_rsi2', 'bnd_roc7', etc.
    """
    import pandas as pd

    # Collect all date indexes from every ticker
    all_dates = pd.DatetimeIndex([])
    for df_ind in symbol_indicators.values():
        all_dates = all_dates.union(df_ind.index)

    all_dates = all_dates.sort_values()

    # Start an empty DataFrame with the union of all dates
    combined = pd.DataFrame(index=all_dates)

    # Join every symbol’s DataFrame into combined
    for ticker, df_ind in symbol_indicators.items():
        # Reindex the symbol’s DataFrame to have the same full date range
        df_ind = df_ind.reindex(all_dates)
        # Join on columns
        combined = combined.join(df_ind, how="left")

    # Optionally remove duplicates if any slipped in
    combined = combined.loc[:, ~combined.columns.duplicated()].copy()

    return combined

def calculate_indicators_for_all(stock_data):
    # -------------------------------
    # (1) Symbol-specific indicators
    # -------------------------------
    symbol_ind = build_symbol_specific_indicators(stock_data)
    
    # -------------------------------
    # (2) Combine into one wide DataFrame
    # -------------------------------
    all_symbol_indicators = combine_all_symbol_indicators(symbol_ind)
    # Now all_symbol_indicators has columns for BDRY, BIL, BND, etc. 
    # e.g. "bdry_rsi2", "bnd_roc7", "fdn_rsi10" ... 
    # for the union of all date indexes.

    # Collect union of all dates for the main tickers
    all_dates = pd.DatetimeIndex([])
    for df in stock_data.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.union(all_symbol_indicators.index)
    all_dates = all_dates.sort_values()

    # -----------------------------------
    # (3) Attach those columns to each ticker
    # -----------------------------------
    for ticker, df in stock_data.items():
        # Reindex the ticker’s DataFrame to the union date range
        df = df.reindex(all_dates)

        # Join all symbol indicators
        df = df.join(all_symbol_indicators, how="left")

        new_cols = {}


        #Chaikin Money Flow (CMF) indicator
        new_cols['cmf_2'] = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=2).chaikin_money_flow()

        # IBS indicator calculations
        # Internal Bar Strength (IBS)
        new_cols['ibs'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        new_cols['ibs_shift1'] = new_cols['ibs'].shift(1)  # Shift by 1 period


        df = compute_atr_and_natr_multi(df, periods=[2, 14, 18])

        # Rate of Change (ROC)
        new_cols['roc_2'] = ROCIndicator(close=df['Close'], window=2).roc()
        # new_cols['roc_2_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['roc_2'].shift(1) > new_cols['roc_2'])
        # new_cols['roc_2_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['roc_2'].shift(1) < new_cols['roc_2'])
        # new_cols['roc_2_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['roc_2'].shift(1) < new_cols['roc_2'])
        # new_cols['roc_2_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['roc_2'].shift(1) > new_cols['roc_2'])
        # new_cols['roc_3'] = ROCIndicator(close=df['Close'], window=3).roc()
        # new_cols['roc_3_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['roc_3'].shift(1) > new_cols['roc_3'])
        # new_cols['roc_3_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['roc_3'].shift(1) < new_cols['roc_3'])
        # new_cols['roc_3_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['roc_3'].shift(1) < new_cols['roc_3'])
        # new_cols['roc_3_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['roc_3'].shift(1) > new_cols['roc_3'])
 
 
 
        # new_cols['roc_7'] = ROCIndicator(close=df['Close'], window=7).roc()
        # new_cols['roc_7_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['roc_7'].shift(1) > new_cols['roc_7'])
        # new_cols['roc_7_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['roc_7'].shift(1) < new_cols['roc_7'])
        # new_cols['roc_7_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['roc_7'].shift(1) < new_cols['roc_7'])
        # new_cols['roc_7_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['roc_7'].shift(1) > new_cols['roc_7'])

        # RSI indicators and shifted versions
        new_cols['rsi_2'] = RSIIndicator(close=df['Close'], window=2).rsi()
        new_cols['rsi_2_shift1'] = new_cols['rsi_2'].shift(1)  # Shift by 1 period
        new_cols['rsi_2_shift2'] = new_cols['rsi_2'].shift(2)  # Shift by 2 periods
        new_cols['rsi_2_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['rsi_2'].shift(1) > new_cols['rsi_2'])
        # new_cols['rsi_2_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['rsi_2'].shift(1) < new_cols['rsi_2'])
        new_cols['rsi_2_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['rsi_2'].shift(1) < new_cols['rsi_2'])
        # new_cols['rsi_2_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['rsi_2'].shift(1) > new_cols['rsi_2'])
        new_cols['rsi_3'] = RSIIndicator(close=df['Close'], window=3).rsi()
        new_cols['rsi_3_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['rsi_3'].shift(1) > new_cols['rsi_3'])
        # new_cols['rsi_3_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['rsi_3'].shift(1) < new_cols['rsi_3'])
        new_cols['rsi_3_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['rsi_3'].shift(1) < new_cols['rsi_3'])
        new_cols['rsi_3_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['rsi_3'].shift(1) > new_cols['rsi_3'])
        new_cols['rsi_4'] = RSIIndicator(close=df['Close'], window=4).rsi()
        # new_cols['rsi_4_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['rsi_4'].shift(1) > new_cols['rsi_4'])
        # new_cols['rsi_4_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['rsi_4'].shift(1) < new_cols['rsi_4'])
        # new_cols['rsi_4_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['rsi_4'].shift(1) < new_cols['rsi_4'])
        # new_cols['rsi_4_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['rsi_4'].shift(1) > new_cols['rsi_4'])
        new_cols['rsi_5'] = RSIIndicator(close=df['Close'], window=5).rsi()
        # new_cols['rsi_5_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['rsi_5'].shift(1) > new_cols['rsi_5'])
        new_cols['rsi_5_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['rsi_5'].shift(1) < new_cols['rsi_5'])
        new_cols['rsi_5_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['rsi_5'].shift(1) < new_cols['rsi_5'])
        new_cols['rsi_5_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['rsi_5'].shift(1) > new_cols['rsi_5'])
        new_cols['rsi_6'] = RSIIndicator(close=df['Close'], window=6).rsi()


        # # RSI indicators and shifted versions
        # new_cols['rsi_3_shift1'] = new_cols['rsi_3'].shift(1)  # Shift RSI(5) by 1 period
        # new_cols['rsi_4_shift1'] = new_cols['rsi_4'].shift(1)  # Shift RSI(5) by 1 period
        # new_cols['rsi_5_shift1'] = new_cols['rsi_5'].shift(1)  # Shift RSI(5) by 1 period
        new_cols['rsi_6_shift1'] = new_cols['rsi_6'].shift(1)  # Shift RSI(5) by 1 period

        # new_cols['rsi_7'] = RSIIndicator(close=df['Close'], window=7).rsi()
        new_cols['rsi_10'] = RSIIndicator(close=df['Close'], window=10).rsi()

        # RSI-10 Divergence Signals
        # new_cols['rsi_10_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['rsi_10'].shift(1) > new_cols['rsi_10'])
        # new_cols['rsi_10_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['rsi_10'].shift(1) < new_cols['rsi_10'])
        # new_cols['rsi_10_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['rsi_10'].shift(1) < new_cols['rsi_10'])
        # new_cols['rsi_10_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['rsi_10'].shift(1) > new_cols['rsi_10'])


        # #Money FLow Indicator
        # new_cols['mfi_5'] = MFIIndicator(high=df['High'],low=df['Low'],close=df['Close'],volume=df['Volume'],window=5,fillna=False).money_flow_index()
        
        # # Exponential Moving Averages (EMA)
        # new_cols['ema_9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
        # new_cols['ema_9_shift1'] = new_cols['ema_9'].shift(1)  # Shift by 1 period
        # new_cols['ema_15'] = EMAIndicator(close=df['Close'], window=15).ema_indicator()
        # new_cols['ema_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        # new_cols['ema_50_shift1'] = new_cols['ema_50'].shift(1)  # Shift by 1 period 
        

        # # Calculate Awesome Oscillator (AO)
        # new_cols['ao_5_25'] = AwesomeOscillatorIndicator(high=df['High'],low=df['Low'],window1=5,window2=25).awesome_oscillator()
        # # Bullish Divergence for AO
        # new_cols['ao_5_25_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['ao_5_25'].shift(1) > new_cols['ao_5_25'])
        # new_cols['ao_5_25_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['ao_5_25'].shift(1) < new_cols['ao_5_25'])
        # new_cols['ao_5_25_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['ao_5_25'].shift(1) < new_cols['ao_5_25'])
        # new_cols['ao_5_25_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['ao_5_25'].shift(1) > new_cols['ao_5_25'])

         # Simple Moving Averages (SMA)
        # new_cols['sma_5'] = SMAIndicator(close=df['Close'], window=5).sma_indicator()
        new_cols['sma_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        # new_cols['sma_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        # new_cols['sma_50_shift1'] = new_cols['sma_50'].shift(1)  # Shift by 1 period
        new_cols['sma_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
        new_cols['sma_200_shift1'] = new_cols['sma_200'].shift(1)  # Shift by 1 period

        # Shifted SMA(20) values
        new_cols['sma20_shift1'] = new_cols['sma_20'].shift(1)  # Shift by 1 period
        new_cols['sma20_shift2'] = new_cols['sma_20'].shift(2)  # Shift by 2 periods

        # Ultimate Oscillator with different parameter sets
        new_cols['uo_1,3,5'] = UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], window1=1, window2=3, window3=5, weight1=4.0, weight2=2.0, weight3=1.0).ultimate_oscillator()
        new_cols['uo_1,3,5_shift1'] = new_cols['uo_1,3,5'].shift(1)  # Shift by 1 period
        # new_cols['uo_135'] = UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], window1=1, window2=3, window3=5, weight1=4.0, weight2=2.0, weight3=1.0).ultimate_oscillator()
        # new_cols['uo_135_shift1'] = new_cols['uo_135'].shift(1)  # Shift by 1 period
        new_cols['uo_2,4,6'] = UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], window1=2, window2=4, window3=6, weight1=4.0, weight2=2.0, weight3=1.0).ultimate_oscillator()

        # Williams %R with different lookback periods and shifted versions
        new_cols['wr_2'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=2).williams_r()
        new_cols['wr_2_shift1'] = new_cols['wr_2'].shift(1)  # Shift by 1 period
        new_cols['wr_2_shift2'] = new_cols['wr_2'].shift(2)  # Shift by 2 periods
        # new_cols['wr_4'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=4).williams_r()
        # new_cols['wr_4_bullish_divergence'] = (df['Low'].shift(1) < df['Low']) & (new_cols['wr_4'].shift(1) > new_cols['wr_4'])
        # new_cols['wr_4_bearish_divergence'] = (df['High'].shift(1) > df['High']) & (new_cols['wr_4'].shift(1) < new_cols['wr_4'])
        # new_cols['wr_4_hidden_bullish_divergence'] = (df['Low'].shift(1) > df['Low']) & (new_cols['wr_4'].shift(1) < new_cols['wr_4'])
        # new_cols['wr_4_hidden_bearish_divergence'] = (df['High'].shift(1) < df['High']) & (new_cols['wr_4'].shift(1) > new_cols['wr_4'])

        # new_cols['wr_5'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=5).williams_r()

        # # Shifted Williams %R (5) and other shifted values
        # new_cols['wr_5_shift1'] = new_cols['wr_5'].shift(1)  # Shift wr_5 by 1 period

     # ── Range Expansion Index (REI 2 … 10) ───────────────────────
        for p in range(2, 3):
            new_cols[f'rei_{p}'] = rei(df, period=p)

        # Shifted High and Close values
        new_cols['High_shift1'] = df['High'].shift(1)  # Shift High by 1 period
        new_cols['Close_shift1'] = df['Close'].shift(1)  # Shift Close by 1 period
        new_cols['Close_shift2'] = df['Close'].shift(2)  # Shift Close by 2 periods
        new_cols['Close_shift3'] = df['Close'].shift(3)  # Shift Close by 3 periods
        new_cols['Close_shift5'] = df['Close'].shift(5)  # Shift Close by 5 periods


        # Rolling maximum values for different windows
        new_cols['rolling_max_2'] = df['Close'].rolling(window=2).max()
        new_cols['rolling_max_5'] = df['Close'].rolling(window=5).max()
        new_cols['rolling_max_14'] = df['Close'].rolling(window=14).max()
        new_cols['rolling_max_20'] = df['Close'].rolling(window=20).max()

        # Drawdown calculations based on rolling maximums
        new_cols['drawdown_2'] = (df['Close'] - new_cols['rolling_max_2']) / new_cols['rolling_max_2']
        new_cols['drawdown_5'] = (df['Close'] - new_cols['rolling_max_5']) / new_cols['rolling_max_5']
        new_cols['drawdown_14'] = (df['Close'] - new_cols['rolling_max_14']) / new_cols['rolling_max_14']
        new_cols['drawdown_20'] = (df['Close'] - new_cols['rolling_max_20']) / new_cols['rolling_max_20']

        # # Calculate the difference between Close and SMA
        # new_cols['close_minus_sma_20'] = df['Close'] - new_cols['sma_20']
        # new_cols['close_minus_sma_5'] = df['Close'] - new_cols['sma_5']

        # Calculate the RSI(2) for the difference
        # new_cols['rsi_2_closeminussma_5'] = RSIIndicator(close=new_cols['close_minus_sma_5'], window=2).rsi()
        # new_cols['rsi_2_closeminussma_20'] = RSIIndicator(close=new_cols['close_minus_sma_20'], window=2).rsi()
        # new_cols['rsi_2_cmf2'] = RSIIndicator(close=new_cols['cmf_2'], window=2).rsi()
        new_cols['rsi_2_wr_2'] = RSIIndicator(close=new_cols['wr_2'], window=2).rsi()
        # new_cols['rsi_2_wr_4'] = RSIIndicator(close=new_cols['wr_4'], window=2).rsi()
        # new_cols['rsi_2_wr_5'] = RSIIndicator(close=new_cols['wr_5'], window=2).rsi()
        # new_cols['rsi_2_ao_5_25'] = RSIIndicator(close=new_cols['ao_5_25'], window=2).rsi()
        # new_cols['rsi_2_roc_2'] = RSIIndicator(close=new_cols['roc_2'], window=2).rsi()
        # new_cols['rsi_2_roc_7'] = RSIIndicator(close=new_cols['roc_7'], window=2).rsi()


    # Add rolling high and close calculations
        for window in [5, 10, 20, 25, 50, 100, 200]:  # Example rolling windows
            # Rolling max and min calculations
            new_cols[f'High_rolling_max_{window}'] = df['High'].rolling(window=window, min_periods=1).max()
            new_cols[f'Close_rolling_max_{window}'] = df['Close'].rolling(window=window, min_periods=1).max()
            new_cols[f'Low_rolling_min_{window}'] = df['Low'].rolling(window=window, min_periods=1).min()
            new_cols[f'Close_rolling_min_{window}'] = df['Close'].rolling(window=window, min_periods=1).min()
            
            # Yesterday's shifted versions
            new_cols[f'High_rolling_max_{window}_yesterday'] = new_cols[f'High_rolling_max_{window}'].shift(1)
            new_cols[f'Close_rolling_max_{window}_yesterday'] = new_cols[f'Close_rolling_max_{window}'].shift(1)
            new_cols[f'Low_rolling_min_{window}_yesterday'] = new_cols[f'Low_rolling_min_{window}'].shift(1)
            new_cols[f'Close_rolling_min_{window}_yesterday'] = new_cols[f'Close_rolling_min_{window}'].shift(1)

                        # =============================================================================
        # NEW ADVANCED TECHNICAL INDICATORS (from paste-2.txt)
        # =============================================================================
        
        # Advanced RSI methods using global RSI_MODE setting
        for period in [2, 3, 4, 5, 6, 10, 14]:
            rsi_col = f'rsi_{period}_{RSI_MODE}'
            new_cols[rsi_col] = rsi_advanced(df['Close'], period, RSI_MODE)
            
            # Add all RSI methods for comparison
            new_cols[f'rsi_{period}_wilder'] = rsi_advanced(df['Close'], period, 'wilder')
            # new_cols[f'rsi_{period}_ema'] = rsi_advanced(df['Close'], period, 'ema')
            new_cols[f'rsi_{period}_adaptive'] = rsi_advanced(df['Close'], period, 'adaptive')
            # new_cols[f'rsi_{period}_average'] = rsi_advanced(df['Close'], period, 'average')

        # SuperTrend indicator (12,4) - key for entry conditions
        new_cols['supertrend_12_4'] = supertrend_advanced(df['High'], df['Low'], df['Close'], 12, 4)
        # new_cols['supertrend_12_3'] = supertrend_advanced(df['High'], df['Low'], df['Close'], 12, 3)
        
        # Check if price is Above/Below SuperTrend
        new_cols['price_above_supertrend_12_4'] = df['Close'] > new_cols['supertrend_12_4']
        # new_cols['price_below_supertrend_12_4'] = df['Close'] < new_cols['supertrend_12_4']

        # Advanced MACD (2,6,3) - for macdh conditions
        macd_line, signal_line, histogram = macd_advanced(df['Close'], 2, 6, 3)
        new_cols['macd_2_6_3_line'] = macd_line
        new_cols['macd_2_6_3_signal'] = signal_line
        new_cols['macd_2_6_3_histogram'] = histogram
        
        # # Standard MACD (12,26,9)
        # macd_std_line, macd_std_signal, macd_std_hist = macd_advanced(df['Close'], 12, 26, 9)
        # new_cols['macd_12_26_9_line'] = macd_std_line
        # new_cols['macd_12_26_9_signal'] = macd_std_signal
        # new_cols['macd_12_26_9_histogram'] = macd_std_hist

        # Balance of Power (BOP) - for divergence conditions
        new_cols['bop_14'] = bop_advanced(df['Open'], df['High'], df['Low'], df['Close'], 14)
        new_cols['bop_10'] = bop_advanced(df['Open'], df['High'], df['Low'], df['Close'], 10)

        # Ultimate Oscillator (2,4,6) - for divergence conditions
        new_cols['uo_2_4_6'] = ultimate_oscillator_advanced(df['High'], df['Low'], df['Close'], 2, 4, 6)
        # new_cols['uo_7_14_28'] = ultimate_oscillator_advanced(df['High'], df['Low'], df['Close'], 7, 14, 28)

        # Advanced IBS with different periods
        new_cols['ibs_advanced'] = ibs_advanced(df['High'], df['Low'], df['Close'])
        new_cols['ibs_10'] = new_cols['ibs_advanced'].rolling(window=10).mean()  # Smoothed IBS

        # Higher Highs detection (5 periods) - for entry conditions
        new_cols['higher_highs_5'] = higher_highs_advanced(df['Close'], 5)
        new_cols['higher_highs_3'] = higher_highs_advanced(df['Close'], 3)

        # # Additional SMAs for ratio calculations
        # new_cols['sma_10'] = sma(df['Close'], 10)
        # new_cols['sma_50'] = sma(df['Close'], 50)
        # new_cols['sma_100'] = sma(df['Close'], 100)
        # new_cols['sma_200'] = sma(df['Close'], 200)



        # Advanced ATR and NATR
        new_cols['atr_10'] = atr_advanced(df['High'], df['Low'], df['Close'], 10)
        new_cols['atr_20'] = atr_advanced(df['High'], df['Low'], df['Close'], 20)
        new_cols['natr_10'] = (new_cols['atr_10'] / df['Close']) * 100
        new_cols['natr_20'] = (new_cols['atr_20'] / df['Close']) * 100

        # # Cross Above/Below indicators for key levels
        # new_cols['ibs_ca_0_65'] = cross_above(new_cols['ibs_advanced'], pd.Series(0.65, index=df.index))
        # new_cols['ibs_cb_0_65'] = cross_below(new_cols['ibs_advanced'], pd.Series(0.65, index=df.index))


        # =============================================================================
        # DIVERGENCE CALCULATION PARAMETERS
        # =============================================================================
        
        # Configure divergence detection windows here
        SLOPE_WINDOW = 10    # 10
        PIVOT_WINDOW = 10    # 10
        
        # =============================================================================
        # DIVERGENCE CALCULATIONS (Slope-based and Pivot-based)
        # =============================================================================
        
        # IBS Divergences (both slope and pivot methods)
        ibs_div_slope = detect_divergence_slope(df['Close'], new_cols['ibs_advanced'], SLOPE_WINDOW)
        ibs_div_pivot = detect_divergence_pivot(df['Close'], new_cols['ibs_advanced'], PIVOT_WINDOW)
        
        new_cols['ibs_div_bull_slope'] = ibs_div_slope['div_bull']
        # new_cols['ibs_div_bear_slope'] = ibs_div_slope['div_bear']
        new_cols['ibs_hdiv_bull_slope'] = ibs_div_slope['hdiv_bull']
        # new_cols['ibs_hdiv_bear_slope'] = ibs_div_slope['hdiv_bear']
        
        new_cols['ibs_div_bull_pivot'] = ibs_div_pivot['div_bull']
        # new_cols['ibs_div_bear_pivot'] = ibs_div_pivot['div_bear']
        new_cols['ibs_hdiv_bull_pivot'] = ibs_div_pivot['hdiv_bull']
        # new_cols['ibs_hdiv_bear_pivot'] = ibs_div_pivot['hdiv_bear']

        # IBS_10 Divergences
        ibs10_div_slope = detect_divergence_slope(df['Close'], new_cols['ibs_10'], SLOPE_WINDOW)
        ibs10_div_pivot = detect_divergence_pivot(df['Close'], new_cols['ibs_10'], PIVOT_WINDOW)
        
        new_cols['ibs_10_div_bull_slope'] = ibs10_div_slope['div_bull']
        # new_cols['ibs_10_div_bear_slope'] = ibs10_div_slope['div_bear']
        new_cols['ibs_10_hdiv_bull_slope'] = ibs10_div_slope['hdiv_bull']
        # new_cols['ibs_10_hdiv_bear_slope'] = ibs10_div_slope['hdiv_bear']
        
        new_cols['ibs_10_div_bull_pivot'] = ibs10_div_pivot['div_bull']
        # new_cols['ibs_10_div_bear_pivot'] = ibs10_div_pivot['div_bear']
        new_cols['ibs_10_hdiv_bull_pivot'] = ibs10_div_pivot['hdiv_bull']
        # new_cols['ibs_10_hdiv_bear_pivot'] = ibs10_div_pivot['hdiv_bear']

        # MACD Histogram Divergences
        macdh_div_slope = detect_divergence_slope(df['Close'], new_cols['macd_2_6_3_histogram'], SLOPE_WINDOW)
        macdh_div_pivot = detect_divergence_pivot(df['Close'], new_cols['macd_2_6_3_histogram'], PIVOT_WINDOW)
        
        new_cols['macdh_div_bull_slope'] = macdh_div_slope['div_bull']
        # new_cols['macdh_div_bear_slope'] = macdh_div_slope['div_bear']
        new_cols['macdh_hdiv_bull_slope'] = macdh_div_slope['hdiv_bull']
        # new_cols['macdh_hdiv_bear_slope'] = macdh_div_slope['hdiv_bear']
        
        new_cols['macdh_div_bull_pivot'] = macdh_div_pivot['div_bull']
        # new_cols['macdh_div_bear_pivot'] = macdh_div_pivot['div_bear']
        new_cols['macdh_hdiv_bull_pivot'] = macdh_div_pivot['hdiv_bull']
        # new_cols['macdh_hdiv_bear_pivot'] = macdh_div_pivot['hdiv_bear']

        # BOP Divergences
        bop_div_slope = detect_divergence_slope(df['Close'], new_cols['bop_14'], SLOPE_WINDOW)
        bop_div_pivot = detect_divergence_pivot(df['Close'], new_cols['bop_14'], PIVOT_WINDOW)
        
        new_cols['bop_div_bull_slope'] = bop_div_slope['div_bull']
        # new_cols['bop_div_bear_slope'] = bop_div_slope['div_bear']
        new_cols['bop_hdiv_bull_slope'] = bop_div_slope['hdiv_bull']
        # new_cols['bop_hdiv_bear_slope'] = bop_div_slope['hdiv_bear']
        
        new_cols['bop_div_bull_pivot'] = bop_div_pivot['div_bull']
        # new_cols['bop_div_bear_pivot'] = bop_div_pivot['div_bear']
        new_cols['bop_hdiv_bull_pivot'] = bop_div_pivot['hdiv_bull']
        # new_cols['bop_hdiv_bear_pivot'] = bop_div_pivot['hdiv_bear']

        # Ultimate Oscillator Divergences
        uo_div_slope = detect_divergence_slope(df['Close'], new_cols['uo_2_4_6'], SLOPE_WINDOW)
        uo_div_pivot = detect_divergence_pivot(df['Close'], new_cols['uo_2_4_6'], PIVOT_WINDOW)
        
        new_cols['uo_div_bull_slope'] = uo_div_slope['div_bull']
        # new_cols['uo_div_bear_slope'] = uo_div_slope['div_bear']
        new_cols['uo_hdiv_bull_slope'] = uo_div_slope['hdiv_bull']
        # new_cols['uo_hdiv_bear_slope'] = uo_div_slope['hdiv_bear']
        
        new_cols['uo_div_bull_pivot'] = uo_div_pivot['div_bull']
        # new_cols['uo_div_bear_pivot'] = uo_div_pivot['div_bear']
        new_cols['uo_hdiv_bull_pivot'] = uo_div_pivot['hdiv_bull']
        # new_cols['uo_hdiv_bear_pivot'] = uo_div_pivot['hdiv_bear']

        # RSI(4) Divergences (key for entry conditions)
        rsi4_div_slope = detect_divergence_slope(df['Close'], new_cols['rsi_4'], SLOPE_WINDOW)
        rsi4_div_pivot = detect_divergence_pivot(df['Close'], new_cols['rsi_4'], PIVOT_WINDOW)
        
        new_cols['rsi4_div_bull_slope'] = rsi4_div_slope['div_bull']
        # new_cols['rsi4_div_bear_slope'] = rsi4_div_slope['div_bear']
        new_cols['rsi4_hdiv_bull_slope'] = rsi4_div_slope['hdiv_bull']
        # new_cols['rsi4_hdiv_bear_slope'] = rsi4_div_slope['hdiv_bear']
        
        new_cols['rsi4_div_bull_pivot'] = rsi4_div_pivot['div_bull']
        # new_cols['rsi4_div_bear_pivot'] = rsi4_div_pivot['div_bear']
        new_cols['rsi4_hdiv_bull_pivot'] = rsi4_div_pivot['hdiv_bull']
        # new_cols['rsi4_hdiv_bear_pivot'] = rsi4_div_pivot['hdiv_bear']

      
        # RSI crossovers for exit conditions
        for period in [2, 3, 10]:
            rsi_col = f'rsi_{period}'
            if rsi_col in new_cols:
                new_cols[f'rsi{period}_ca_70'] = cross_above(new_cols[rsi_col], pd.Series(70, index=df.index))
                new_cols[f'rsi{period}_cb_30'] = cross_below(new_cols[rsi_col], pd.Series(30, index=df.index))
                new_cols[f'rsi{period}_ca_80'] = cross_above(new_cols[rsi_col], pd.Series(80, index=df.index))
                new_cols[f'rsi{period}_cb_20'] = cross_below(new_cols[rsi_col], pd.Series(20, index=df.index))

        # MACD crossovers
        new_cols['macd_line_ca_signal'] = cross_above(new_cols['macd_2_6_3_line'], new_cols['macd_2_6_3_signal'])
        new_cols['macd_line_cb_signal'] = cross_below(new_cols['macd_2_6_3_line'], new_cols['macd_2_6_3_signal'])

         
         # Concatenate new columns to the DataFrame

        new_cols_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_cols_df], axis=1)

        stock_data[ticker] = df

    return stock_data

