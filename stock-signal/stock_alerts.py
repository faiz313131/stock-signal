import numpy as np
import pandas as pd
import yfinance as yf
import csv
import matplotlib.pyplot as plt
import os

# -----------------
# Constants (with type hints for Pylance)
# -----------------
SIGNAL_LOG_FILE: str = "signals_log.csv"
BACKTEST_LOG_FILE: str = "backtest_log.csv"

# -----------------
# Dummy WhatsApp Function (disabled)
# -----------------
def send_whatsapp(msg):
    # WhatsApp removed for deployment
    print("[INFO] WhatsApp alert skipped:", msg[:50], "...")

# -----------------
# Parameters
# -----------------
EMA_FAST = 50
EMA_SLOW = 200
VOLUME_MULTIPLIER = 1.2
LOOKBACK_SWING = 3
BACKTEST_SL_PCT = 0.015
BACKTEST_TP_PCT = 0.03
COMMISSION = 0.001
SLIPPAGE = 0.001

SIGNAL_LOG_FILE = "signals_log.csv"
BACKTEST_LOG_FILE = "backtest_log.csv"

# Ensure CSV files exist
if not os.path.exists(SIGNAL_LOG_FILE):
    with open(SIGNAL_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ticker", "Date", "EntryPrice", "SignalType"])

if not os.path.exists(BACKTEST_LOG_FILE):
    with open(BACKTEST_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ticker", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "Result"])

# -----------------
# Helper Functions
# -----------------
def fetch_data(ticker, period='1y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def compute_indicators(df):
    df['EMA50'] = df['Close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['AvgVol20'] = df['Volume'].rolling(window=20, min_periods=5).mean()
    return df

def safe_scalar(x):
    if isinstance(x, pd.Series):
        return x.item() if len(x) == 1 else np.nan
    return x

# -----------------
# Swing Detection
# -----------------
def detect_swings(df):
    highs, lows = [], []
    for i in range(LOOKBACK_SWING, len(df) - LOOKBACK_SWING):
        high = safe_scalar(df['High'].iloc[i])
        low = safe_scalar(df['Low'].iloc[i])
        start = i - LOOKBACK_SWING
        end = i + LOOKBACK_SWING + 1
        window_high = df['High'].iloc[start:end].max()
        window_low = df['Low'].iloc[start:end].min()
        if pd.notna(high) and high == window_high:
            highs.append((i, high))
        if pd.notna(low) and low == window_low:
            lows.append((i, low))
    return highs, lows

# -----------------
# Candlestick Patterns
# -----------------
def is_bullish_engulfing(prev, curr):
    prev_close, prev_open = safe_scalar(prev['Close']), safe_scalar(prev['Open'])
    curr_close, curr_open = safe_scalar(curr['Close']), safe_scalar(curr['Open'])
    if any(pd.isna([prev_close, prev_open, curr_close, curr_open])):
        return False
    return prev_close < prev_open and curr_close > curr_open and curr_close > prev_open and curr_open < prev_close

def is_hammer(row):
    close, open_, low, high = safe_scalar(row['Close']), safe_scalar(row['Open']), safe_scalar(row['Low']), safe_scalar(row['High'])
    if any(pd.isna([close, open_, low, high])):
        return False
    body = abs(close - open_)
    lower_shadow = open_ - low if close > open_ else close - low
    upper_shadow = high - max(close, open_)
    return lower_shadow > 2 * body and upper_shadow < body

# -----------------
# Signal Detection
# -----------------
def detect_signals(df, ticker):
    df['Signal'] = 0
    swings_high, swings_low = detect_swings(df)
    sr_high = max([h[1] for h in swings_high[-5:]], default=np.nan)
    sr_low = min([l[1] for l in swings_low[-5:]], default=np.nan)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        ema50, ema200 = safe_scalar(row['EMA50']), safe_scalar(row['EMA200'])
        close, volume, avgvol20 = safe_scalar(row['Close']), safe_scalar(row['Volume']), safe_scalar(row['AvgVol20'])
        if any(pd.isna([ema50, ema200, close, volume, avgvol20])):
            continue

        trend_ok = ema50 > ema200
        breakout = not pd.isna(sr_high) and close > sr_high
        vol_ok = avgvol20 > 0 and volume > VOLUME_MULTIPLIER * avgvol20
        candle_ok = is_bullish_engulfing(prev, row) or is_hammer(row)
        support_ok = not pd.isna(sr_low) and close > sr_low

        if all([trend_ok, breakout, vol_ok, candle_ok, support_ok]):
            df.at[df.index[i], 'Signal'] = 1
            # Log signal
            with open(SIGNAL_LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ticker, df.index[i], close, "BUY"])
    return df

# -----------------
# Backtest
# -----------------
def backtest(df, ticker):
    trades = []
    in_trade = False
    entry_price = stop = take = None
    for i in range(len(df)):
        row = df.iloc[i]
        signal = safe_scalar(row['Signal'])
        if pd.isna(signal):
            continue
        if not in_trade and signal == 1:
            if i + 1 < len(df):
                next_open = safe_scalar(df['Open'].iloc[i+1])
                if pd.isna(next_open):
                    continue
                entry_price = next_open * (1 + SLIPPAGE)
                stop = entry_price * (1 - BACKTEST_SL_PCT)
                take = entry_price * (1 + BACKTEST_TP_PCT)
                entry_date = df.index[i+1]
                in_trade = True
        elif in_trade:
            low = safe_scalar(row['Low'])
            high = safe_scalar(row['High'])
            if pd.isna(low) or pd.isna(high):
                continue
            if low <= stop:
                exit_price = stop * (1 - COMMISSION)
                trades.append((entry_date, df.index[i], entry_price, exit_price, 'SL'))
                in_trade = False
            elif high >= take:
                exit_price = take * (1 - COMMISSION)
                trades.append((entry_date, df.index[i], entry_price, exit_price, 'TP'))
                in_trade = False
    # Log trades
    for tr in trades:
        with open(BACKTEST_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ticker, tr[0], tr[1], tr[2], tr[3], tr[4]])
    return trades

# -----------------
# Plot Trades
# -----------------
def plot_trades(df, trades, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close Price')
    for tr in trades:
        plt.scatter(tr[0], tr[2], color='green', marker='^', s=100)
        plt.scatter(tr[1], tr[3], color='red', marker='v', s=100)
    plt.title(f"{ticker} - Trades")
    plt.legend()
    plt.show()

# -----------------
# Main
# -----------------
if __name__ == "__main__":
    watchlist = ['CROMPTON.NS', 'ESSENTIA.NS', 'NAVNETEDUL.NS', 'SUZLON.NS']
    print(f"[{pd.Timestamp.now()}] Starting scan for {len(watchlist)} tickers...")
    for ticker in watchlist:
        print(f"[Scan] Ticker: {ticker}")
        try:
            df = fetch_data(ticker)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            continue
        if df.empty:
            print(f"No data for {ticker}, skipping...")
            continue
        df = compute_indicators(df)
        df = detect_signals(df, ticker)
        last_signal = safe_scalar(df['Signal'].iloc[-1])
        if not pd.isna(last_signal) and last_signal == 1:
            entry_price = safe_scalar(df['Close'].iloc[-1])
            msg = (f"BUY SIGNAL: {ticker}\n"
                   f"Entry: {entry_price:.2f}\n"
                   f"Stop Loss: {entry_price*(1-BACKTEST_SL_PCT):.2f}\n"
                   f"Target: {entry_price*(1+BACKTEST_TP_PCT):.2f}\n"
                   f"Timeframe: 3-30 Days\nTrend: Bullish\nConfirmation: Swing breakout + High Volume + Candle pattern")
            send_whatsapp(msg)
        trades = backtest(df, ticker)
        if trades:
            winrate = sum(1 for tr in trades if tr[3] > tr[2]) / len(trades)
            print(f"Backtest {ticker}: {len(trades)} trades, winrate {winrate:.2f}")
            plot_trades(df, trades, ticker)
        else:
            print(f"No trades for {ticker}")
