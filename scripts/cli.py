from __future__ import annotations
import os
import typer
import yaml

from src.detect_candlestick.data_fetch import fetch_and_save
from src.detect_candlestick.label_patterns import label_and_save
from src.detect_candlestick.build_dataset import build_images_and_manifest
from src.detect_candlestick.split import split_by_date

app = typer.Typer(help="Candlestick dataset builder (Phase 1)")

def load_yaml(path: str):
    """Load and return YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

@app.command()
def fetch(
    tickers_file: str = typer.Option("configs/tickers.txt", help="File containing ticker symbols"),
    dataset_yaml: str = typer.Option("configs/dataset.yaml", help="Dataset configuration file"),
    raw_dir: str = typer.Option("data/raw", help="Directory to save raw OHLCV data"),
):
    """Fetch historical OHLCV data for configured tickers."""
    cfg = load_yaml(dataset_yaml)
    with open(tickers_file, "r") as f:
        tickers = [t.strip() for t in f if t.strip()]
    
    print(f"Fetching data for {len(tickers)} tickers...")
    fetch_and_save(tickers, cfg["timeframe"]["start"], cfg["timeframe"]["end"], raw_dir)
    typer.echo(f"✅ Fetched {len(tickers)} tickers into {raw_dir}")

@app.command()
def label(
    raw_dir: str = typer.Option("data/raw", help="Directory containing raw OHLCV data"),
    interim_dir: str = typer.Option("data/interim", help="Directory to save labeled data"),
    patterns_yaml: str = typer.Option("configs/patterns.yaml", help="Patterns configuration file"),
):
    """Label OHLCV data with candlestick patterns using TA-Lib."""
    print("Labeling data with candlestick patterns...")
    label_and_save(raw_dir, interim_dir, patterns_yaml)
    typer.echo(f"✅ Labeled data saved to {interim_dir}")

@app.command(name="build-images")
def build_images(
    interim_dir: str = typer.Option("data/interim", help="Directory containing labeled data"),
    images_dir: str = typer.Option("data/images", help="Directory to save rendered images"),
    dataset_yaml: str = typer.Option("configs/dataset.yaml", help="Dataset configuration file"),
):
    """Build candlestick images from labeled data."""
    print("Building candlestick images...")
    build_images_and_manifest(interim_dir, images_dir, dataset_yaml)
    typer.echo(f"✅ Images and manifest saved to {interim_dir} and {images_dir}")

@app.command()
def split(
    interim_dir: str = typer.Option("data/interim", help="Directory containing images manifest"),
    dataset_yaml: str = typer.Option("configs/dataset.yaml", help="Dataset configuration file"),
):
    """Split dataset into train/val/test sets by date."""
    print("Splitting dataset by date...")
    split_by_date(interim_dir, dataset_yaml)
    typer.echo(f"✅ Created train/val/test CSVs in {interim_dir}")

@app.command(name="run-all")
def run_all(
    tickers_file: str = typer.Option("configs/tickers.txt"),
    dataset_yaml: str = typer.Option("configs/dataset.yaml"),
    patterns_yaml: str = typer.Option("configs/patterns.yaml"),
    raw_dir: str = typer.Option("data/raw"),
    interim_dir: str = typer.Option("data/interim"),
    images_dir: str = typer.Option("data/images"),
):
    """Run the complete Phase 1 pipeline: fetch → label → build-images → split."""
    print("🚀 Running complete Phase 1 pipeline...")
    
    # Step 1: Fetch data
    print("\n📥 Step 1: Fetching OHLCV data...")
    cfg = load_yaml(dataset_yaml)
    with open(tickers_file, "r") as f:
        tickers = [t.strip() for t in f if t.strip()]
    fetch_and_save(tickers, cfg["timeframe"]["start"], cfg["timeframe"]["end"], raw_dir)
    
    # Step 2: Label patterns
    print("\n🏷️ Step 2: Labeling candlestick patterns...")
    label_and_save(raw_dir, interim_dir, patterns_yaml)
    
    # Step 3: Build images
    print("\n🖼️ Step 3: Building candlestick images...")
    build_images_and_manifest(interim_dir, images_dir, dataset_yaml)
    
    # Step 4: Split dataset
    print("\n✂️ Step 4: Splitting dataset...")
    split_by_date(interim_dir, dataset_yaml)
    
    print("\n✅ Phase 1 pipeline complete!")
    print(f"📊 Check results in: {interim_dir}")
    

if __name__ == "__main__":
    app()
