from __future__ import annotations
import os
import yaml
from typing import Tuple
import pandas as pd

from src.detect_candlestick.rendering.renderer import render_window_to_png

def load_yaml(path: str):
    """Load and return YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_images_and_manifest(
    interim_dir: str,
    images_dir: str,
    dataset_yaml: str,
) -> None:
    """
    Build candlestick images and manifest from labeled data.
    
    This function:
    1. Reads the labels manifest (dates where patterns were detected)
    2. For each labeled date, takes a window of OHLCV data ending at that date
    3. Renders a clean candlestick image
    4. Creates a manifest linking images to their labels
    
    Args:
        interim_dir: Directory containing labeled data and manifests
        images_dir: Directory to save rendered images
        dataset_yaml: Path to dataset configuration file
    """
    # Load configuration
    cfg = load_yaml(dataset_yaml)
    win = int(cfg["render"]["window_size"])           # Number of bars in context window
    img_size = int(cfg["render"]["image_size"])       # Final image size in pixels
    dpi = int(cfg["render"]["dpi"])                   # DPI for rendering
    up_color = cfg["render"]["up_color"]              # Bullish candle color
    down_color = cfg["render"]["down_color"]          # Bearish candle color
    facecolor = cfg["render"]["facecolor"]            # Background color
    edgecolor = cfg["render"]["edgecolor"]            # Edge color

    # Check if labels manifest exists
    labels_manifest = os.path.join(interim_dir, "labels_manifest.csv")
    if not os.path.exists(labels_manifest):
        raise FileNotFoundError(
            f"Labels manifest not found at {labels_manifest}. "
            "Run the labeling step first."
        )

    # Load the labels manifest (contains all detected patterns)
    lm = pd.read_csv(labels_manifest, parse_dates=["date"])
    print(f"Found {len(lm)} pattern detections to process")
    
    # For fast lookup, cache all OHLCV data by ticker
    cache: dict[str, pd.DataFrame] = {}
    for fname in os.listdir(interim_dir):
        if fname.endswith("_labeled.parquet"):
            ticker = fname.split("_labeled.parquet")[0]
            df = pd.read_parquet(os.path.join(interim_dir, fname))
            # Keep only OHLCV columns and sort by date
            df = df[["open", "high", "low", "close", "volume"]].sort_index()
            cache[ticker] = df
            print(f"Cached {len(df)} bars for {ticker}")

    # Process each pattern detection
    out_rows = []
    processed = 0
    
    for _, row in lm.iterrows():
        ticker = row["ticker"]
        label = row["label"]
        date = row["date"]
        
        # Get the full OHLCV data for this ticker
        df = cache[ticker]
        
        # Check if the labeled date exists in our data
        if date not in df.index:
            print(f"Warning: Date {date} not found for {ticker}")
            continue
            
        # Find the position of this date in the DataFrame
        end_loc = df.index.get_loc(date)
        if isinstance(end_loc, slice):
            # This shouldn't happen with unique DateIndex, but handle it
            continue
            
        # Calculate the start position for our window
        start_loc = end_loc - (win - 1)
        if start_loc < 0:
            # Not enough historical data for this window
            print(f"Warning: Insufficient data for {ticker} at {date} (need {win} bars)")
            continue
            
        # Extract the window of data ending at the labeled date
        window = df.iloc[start_loc:end_loc + 1].copy()
        window.index.name = "Date"  # mplfinance expects Date-like index
        
        # Create the image filename and path
        save_rel = os.path.join(label, f"{ticker}_{pd.to_datetime(date).date()}.png")
        save_abs = os.path.join(images_dir, save_rel)
        
        # Render the candlestick image
        render_window_to_png(
            df_window=window,
            save_path=save_abs,
            up_color=up_color,
            down_color=down_color,
            facecolor=facecolor,
            edgecolor=edgecolor,
            image_size=img_size,
            dpi=dpi,
        )
        
        # Add to our output manifest
        out_rows.append({
            "image_path": save_rel,
            "label": label,
            "ticker": ticker,
            "date": str(pd.to_datetime(date).date()),
        })
        
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} images...")

    # Save the final images manifest
    if out_rows:
        man = pd.DataFrame(out_rows)
        os.makedirs(os.path.join(interim_dir), exist_ok=True)
        manifest_path = os.path.join(interim_dir, "images_manifest.csv")
        man.to_csv(manifest_path, index=False)
        print(f"Created {len(out_rows)} images")
        print(f"Images manifest saved to {manifest_path}")
        
        # Show class distribution
        print("\nClass distribution:")
        print(man["label"].value_counts())
    else:
        print("No images were created!")
