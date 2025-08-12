from __future__ import annotations
import os
from typing import Tuple
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import mplfinance as mpf

# Set matplotlib settings for better quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['patch.antialiased'] = True

def render_window_to_png(
    df_window: pd.DataFrame,
    save_path: str,
    up_color: str,
    down_color: str,
    facecolor: str,
    edgecolor: str,
    image_size: int,
    dpi: int,
) -> None:
    """
    Render a window of OHLCV data as a clean candlestick chart image.
    
    Args:
        df_window: DataFrame with OHLCV data (must have Date index)
        save_path: Path to save the PNG image
        up_color: Color for bullish candles (e.g., "#16a34a" for green)
        down_color: Color for bearish candles (e.g., "#dc2626" for red)
        facecolor: Background color of the plot
        edgecolor: Edge color of the plot
        image_size: Final image size in pixels (width=height)
        dpi: Dots per inch for rendering
    """
    # Configure market colors for candlesticks
    mc = mpf.make_marketcolors(
        up=up_color,      # Bullish candle color
        down=down_color,   # Bearish candle color
        edge="inherit",    # Use same colors for edges
        wick="inherit",    # Use same colors for wicks
        volume="inherit"   # Use same colors for volume (if shown)
    )
    
    # Create a clean style with no grid, consistent colors
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle="",           # No grid lines
        facecolor=facecolor,    # Plot background
        edgecolor=edgecolor,    # Plot edge
        figcolor=facecolor,     # Figure background
    )
    
    # Calculate figure size in inches (image_size pixels / dpi)
    fig_inches = image_size / dpi
    
    # Create the candlestick plot with improved settings
    fig, axlist = mpf.plot(
        df_window,
        type="candle",          # Candlestick chart type
        style=style,
        returnfig=True,         # Return figure and axes for customization
        axisoff=True,           # Hide axes (clean image for ML)
        figsize=(fig_inches, fig_inches),  # Square aspect ratio
        xrotation=0,            # No x-axis rotation
        volume=False,           # Disable volume for cleaner look
        tight_layout=True,      # Better spacing
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the image with tight layout (no padding)
    fig.savefig(
        save_path, 
        dpi=dpi, 
        bbox_inches="tight",    # Remove extra whitespace
        pad_inches=0            # No padding
    )
    
    # Close the figure to free memory
    plt.close(fig)
