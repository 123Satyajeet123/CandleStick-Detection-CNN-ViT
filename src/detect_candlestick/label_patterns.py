from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import talib  # requires TA-Lib to be installed (C lib + Python wrapper)

@dataclass(frozen=True)
class PatternSpec:
    talib_fn: str
    label: str
    priority: int

def load_pattern_specs(path: str) -> tuple[List[PatternSpec], Dict[str, Any]]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    specs = [PatternSpec(**p) for p in cfg["patterns"]]
    return specs, cfg

def apply_patterns(df: pd.DataFrame, specs: List[PatternSpec], min_abs_value: int, allow_negative: bool) -> pd.DataFrame:
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    results: Dict[str, np.ndarray] = {}
    
    # Special handling for engulfing patterns
    engulfing_results = talib.CDLENGULFING(o, h, l, c)
    
    for spec in specs:
        if spec.talib_fn == "CDLENGULFING":
            # Handle engulfing specially - split into bullish/bearish
            if spec.label == "BullishEngulfing":
                results[spec.label] = np.where(engulfing_results > 0, engulfing_results, 0)
            elif spec.label == "BearishEngulfing":
                results[spec.label] = np.where(engulfing_results < 0, engulfing_results, 0)
        else:
            # Regular pattern
            try:
                fn = getattr(talib, spec.talib_fn)
                results[spec.label] = fn(o, h, l, c)
            except AttributeError:
                print(f"Warning: TA-Lib function {spec.talib_fn} not available, skipping {spec.label}")
                results[spec.label] = np.zeros(len(o))  # Fill with zeros
    
    out = df.copy()
    for label, arr in results.items():
        out[label] = arr

    # Single-label resolution using priority
    by_priority = sorted(specs, key=lambda s: s.priority)
    chosen_labels: List[str] = []
    chosen_scores: List[int] = []

    for idx in range(len(out)):
        winner_label = "None"
        winner_score = 0
        for spec in by_priority:
            val = int(out.iloc[idx][spec.label])
            if abs(val) >= min_abs_value and (allow_negative or val > 0):
                winner_label = spec.label
                winner_score = val
                break
        chosen_labels.append(winner_label)
        chosen_scores.append(winner_score)

    out["label"] = chosen_labels
    out["label_score"] = chosen_scores
    return out

def label_and_save(raw_dir: str, interim_dir: str, patterns_yaml: str) -> None:
    os.makedirs(interim_dir, exist_ok=True)
    specs, cfg = load_pattern_specs(patterns_yaml)
    min_abs_value = int(cfg.get("min_abs_value", 100))
    allow_negative = bool(cfg.get("allow_negative", True))

    manifests = []
    for fname in os.listdir(raw_dir):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(raw_dir, fname)
        df = pd.read_parquet(path)
        df = df.sort_index()
        labeled = apply_patterns(df, specs, min_abs_value, allow_negative)
        ticker = labeled["ticker"].iloc[0]
        outp = os.path.join(interim_dir, f"{ticker}_labeled.parquet")
        labeled.to_parquet(outp)
        # Build per-ticker manifest rows where label != None
        subset = labeled[labeled["label"] != "None"].reset_index().rename(columns={"index":"date"})
        subset["ticker"] = ticker
        manifests.append(subset[["date", "ticker", "label", "label_score"]])

    if manifests:
        man = pd.concat(manifests, ignore_index=True)
        man = man.sort_values(["ticker", "date"])
        man.to_csv(os.path.join(interim_dir, "labels_manifest.csv"), index=False)
