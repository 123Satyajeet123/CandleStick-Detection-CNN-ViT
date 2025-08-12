from __future__ import annotations
import os
import yaml
import pandas as pd

def load_yaml(path: str):
    """Load and return YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def split_by_date(
    interim_dir: str,
    dataset_yaml: str,
) -> None:
    """
    Split the images manifest into train/val/test sets by date.
    
    This ensures no data leakage by strictly separating time periods.
    
    Args:
        interim_dir: Directory containing the images manifest
        dataset_yaml: Path to dataset configuration file
    """
    cfg = load_yaml(dataset_yaml)
    train_end = pd.to_datetime(cfg["split"]["train_end"])
    val_end = pd.to_datetime(cfg["split"]["val_end"])
    test_end = pd.to_datetime(cfg["split"]["test_end"])
    
    # Load the images manifest
    images_manifest = os.path.join(interim_dir, "images_manifest.csv")
    if not os.path.exists(images_manifest):
        raise FileNotFoundError(
            f"Images manifest not found at {images_manifest}. "
            "Run the build-images step first."
        )
    
    df = pd.read_csv(images_manifest, parse_dates=["date"])
    print(f"Loaded {len(df)} images for splitting")
    
    # Split by date ranges
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[(df["date"] > val_end) & (df["date"] <= test_end)]
    
    print(f"Split results:")
    print(f"  Train: {len(train)} images ({train['date'].min()} to {train['date'].max()})")
    print(f"  Val:   {len(val)} images ({val['date'].min()} to {val['date'].max()})")
    print(f"  Test:  {len(test)} images ({test['date'].min()} to {test['date'].max()})")
    
    # Save splits
    out_dir = os.path.join(interim_dir)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    
    # Show class distribution for each split
    print("\nClass distribution:")
    print("Train:")
    print(train["label"].value_counts())
    print("\nVal:")
    print(val["label"].value_counts())
    print("\nTest:")
    print(test["label"].value_counts())
