import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os

def load_and_process_data(ticker, start_date, end_date, save_path="data/processed"):
    """
    Downloads stock data, adds technical indicators, and saves to CSV.
    """
    print(f"⬇️  Downloading data for {ticker}...")

    # 1. Download Data
    df = yf.download(ticker, start=start_date, end=end_date)

    # Fix for MultiIndex columns (common yfinance issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Feature Engineering (The "Eyes" of the AI)
    # RSI (Momentum): Helps AI know if bought too much
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # SMA (Trend): Helps AI see the long-term trend
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)

    # MACD (Trend Reversal): Helps AI detect changes
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_SIGNAL'] = macd['MACDs_12_26_9']

    # 3. Cleaning
    # Indicators create NaN values (empty rows) at the start. Drop them.
    df.dropna(inplace=True)

    # 4. Save to CSV
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{save_path}/{ticker}_processed.csv"
    df.to_csv(file_name)

    print(f"✅ Data processed and saved to: {file_name}")
    print(f"📊 Shape: {df.shape} (Rows, Columns)")
    return df

# Test the function immediately if this script is run
if __name__ == "__main__":
    # Let's test with Apple stock
    data = load_and_process_data("AAPL", "2015-01-01", "2024-01-01")
    print(data.head())