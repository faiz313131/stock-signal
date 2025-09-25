from flask import Flask, render_template
import pandas as pd
from datetime import datetime
import os
from stock_alerts import fetch_data, compute_indicators, detect_signals, backtest, safe_scalar

app = Flask(__name__)
DATA_FILE = "monthly_calls.csv"

# ----------------- Initialize monthly CSV -----------------
def init_month_file():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=["Ticker","Date","EntryPrice","SL","TP","Status"])
        df.to_csv(DATA_FILE,index=False)

# ----------------- Load current month calls -----------------
def get_current_calls():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["Ticker","Date","EntryPrice","SL","TP","Status"])
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    month = datetime.now().month
    return df[df['Date'].dt.month == month]

# ----------------- Update signals -----------------
def update_signals():
    watchlist = ['CROMPTON.NS', 'ESSENTIA.NS', 'NAVNETEDUL.NS', 'SUZLON.NS',
        'VASCONEQ.NS', 'VBL.NS', 'TRIDENT.NS', 'WIPRO.NS', 'MIDHANI.NS', 'DOLLAR.NS',
        'ICIL.NS', 'SRF.NS', 'RAJOOENG.NS', 'VOLTAS.NS', 'LTFOODS.NS', 'ASTRAL.NS', 'SYRMA.NS',
        'RELIANCE.NS', 'TTKPRESTIG.NS', 'CHAMBLFERT.NS', 'SIEMENS.NS', 'CAMLINFINE.NS', 
        'VINNY.NS', 'SANSTAR.NS', 'MSUMI.NS', 'SSDL.NS', 'SEPC.NS', 'SALASAR.NS', 'RELAXO.NS',
        'INFIBEAM.NS', 'RVNL.NS', 'TITAGARH.NS', 'APOLLOTYRE.NS']

    monthly_calls = get_current_calls()

    for ticker in watchlist:
        try:
            df = fetch_data(ticker)
            if df.empty:
                continue
            df = compute_indicators(df)
            df = detect_signals(df)
            last_signal = safe_scalar(df['Signal'].iloc[-1])
            entry_price = safe_scalar(df['Close'].iloc[-1])
            SL = entry_price * 0.985   # BACKTEST_SL_PCT
            TP = entry_price * 1.03    # BACKTEST_TP_PCT

            # If new signal and not already in CSV, add it
            if last_signal == 1 and not ((monthly_calls['Ticker']==ticker) & (monthly_calls['EntryPrice']==entry_price)).any():
                new_row = pd.DataFrame([{
                    "Ticker": ticker,
                    "Date": datetime.now(),
                    "EntryPrice": round(entry_price,2),
                    "SL": round(SL,2),
                    "TP": round(TP,2),
                    "Status": "Open"
                }])
                monthly_calls = pd.concat([monthly_calls, new_row], ignore_index=True)
                monthly_calls.to_csv(DATA_FILE,index=False)

            # Update status based on last price
            low = safe_scalar(df['Low'].iloc[-1])
            high = safe_scalar(df['High'].iloc[-1])
            if not monthly_calls.empty and ticker in monthly_calls['Ticker'].values:
                idx = monthly_calls[(monthly_calls['Ticker']==ticker) & (monthly_calls['Status']=="Open")].index
                for i in idx:
                    if low <= monthly_calls.at[i,'SL']:
                        monthly_calls.at[i,'Status'] = "Loss"
                    elif high >= monthly_calls.at[i,'TP']:
                        monthly_calls.at[i,'Status'] = "Win"

            # Save CSV
            monthly_calls.to_csv(DATA_FILE,index=False)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return monthly_calls

# ----------------- Flask Route -----------------
@app.route("/")
def dashboard():
    calls = update_signals()
    total = len(calls)
    if total > 0:
        wins = len(calls[calls['Status']=="Win"])
        losses = len(calls[calls['Status']=="Loss"])
        win_pct = round(wins/total*100,2)
        loss_pct = round(losses/total*100,2)
    else:
        win_pct = loss_pct = 0
    return render_template("dashboard.html", calls=calls.to_dict(orient="records"), win_pct=win_pct, loss_pct=loss_pct)

# ----------------- Run App -----------------
if __name__=="__main__":
    init_month_file()
    app.run(debug=True)
