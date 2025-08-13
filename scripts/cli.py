from __future__ import annotations
import os
import typer
import yaml
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detect_candlestick.data_fetch import fetch_and_save
from detect_candlestick.label_patterns import label_and_save
from detect_candlestick.build_dataset import build_images_and_manifest
from detect_candlestick.split import split_by_date

app = typer.Typer(help="Candlestick dataset builder")

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
    typer.echo(f"Fetching data for {len(tickers)} tickers...")
    fetch_and_save(tickers, cfg["timeframe"]["start"], cfg["timeframe"]["end"], raw_dir)
    typer.echo(f"Fetched {len(tickers)} tickers into {raw_dir}")

@app.command()
def label(
    raw_dir: str = typer.Option("data/raw", help="Directory containing raw OHLCV data"),
    interim_dir: str = typer.Option("data/interim", help="Directory to save labeled data"),
    patterns_yaml: str = typer.Option("configs/patterns.yaml", help="Patterns configuration file"),
):
    """Label OHLCV data with candlestick patterns using TA-Lib."""
    typer.echo("Labeling data with candlestick patterns...")
    label_and_save(raw_dir, interim_dir, patterns_yaml)
    typer.echo(f"Labeled data saved to {interim_dir}")

@app.command(name="build-images")
def build_images(
    interim_dir: str = typer.Option("data/interim", help="Directory containing labeled data"),
    images_dir: str = typer.Option("data/images", help="Directory to save rendered images"),
    dataset_yaml: str = typer.Option("configs/dataset.yaml", help="Dataset configuration file"),
):
    """Build candlestick images from labeled data."""
    typer.echo("Building candlestick images...")
    build_images_and_manifest(interim_dir, images_dir, dataset_yaml)
    typer.echo(f"Images and manifest saved to {interim_dir} and {images_dir}")

@app.command()
def split(
    interim_dir: str = typer.Option("data/interim", help="Directory containing images manifest"),
    dataset_yaml: str = typer.Option("configs/dataset.yaml", help="Dataset configuration file"),
):
    """Split dataset into train/val/test sets by date."""
    typer.echo("Splitting dataset by date...")
    split_by_date(interim_dir, dataset_yaml)
    typer.echo(f"Created train/val/test CSVs in {interim_dir}")

@app.command(name="run-all")
def run_all(
    tickers_file: str = typer.Option("configs/tickers.txt"),
    dataset_yaml: str = typer.Option("configs/dataset.yaml"),
    patterns_yaml: str = typer.Option("configs/patterns.yaml"),
    raw_dir: str = typer.Option("data/raw"),
    interim_dir: str = typer.Option("data/interim"),
    images_dir: str = typer.Option("data/images"),
):
    """Run the data generation pipeline: fetch → label → build-images → split."""

    cfg = load_yaml(dataset_yaml)
    with open(tickers_file, "r") as f:
        tickers = [t.strip() for t in f if t.strip()]
    fetch_and_save(tickers, cfg["timeframe"]["start"], cfg["timeframe"]["end"], raw_dir)

    label_and_save(raw_dir, interim_dir, patterns_yaml)

    build_images_and_manifest(interim_dir, images_dir, dataset_yaml)
    
    split_by_date(interim_dir, dataset_yaml)

    print(f"Check results in: {interim_dir}")
    

if __name__ == "__main__":
    app()
