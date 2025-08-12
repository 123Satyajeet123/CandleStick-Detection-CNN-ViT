from __future__ import annotations
import os
from typing import List
import pandas as pd
import yfinance as yf

def fetch_and_save(
    tickers: List[str],
    start_date: str,
    end_date: str,
    raw_dir: str,
) -> None:
    """
    Fetch historical OHLCV data for given tickers and save as parquet files.
    
    Args:
        tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format  
        raw_dir: Directory to save the parquet files
    """
    os.makedirs(raw_dir, exist_ok=True)
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        
        # Download OHLCV data from Yahoo Finance
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,  # Keep original OHLC values
            progress=False,      # Disable progress bar
            group_by="ticker",
            interval="1d",       # Daily data
        )
        
        if df is None or df.empty:
            print(f"Warning: No data found for {ticker}")
            continue
        
        # Handle MultiIndex columns if present (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten the MultiIndex by taking the second level (price data)
            df.columns = df.columns.get_level_values(1)
        
        # Normalize column names to lowercase for consistency
        df = df.rename(columns={
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Close": "close", 
            "Volume": "volume"
        })
        
        # Set index name for clarity
        df.index.name = "date"
        
        # Add ticker column for easy identification
        df["ticker"] = ticker
        
        # Save as parquet (efficient, compressed format)
        out_path = os.path.join(raw_dir, f"{ticker}.parquet")
        df.to_parquet(out_path, index=True)
        
        print(f"Saved {len(df)} rows for {ticker} to {out_path}")
    
    print(f"Data fetching complete. Files saved to {raw_dir}")
