# src/utilities/plotting.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_barh_frequency(freq_df: pd.DataFrame, *, title: str, out_path: Path) -> None:
    """
    Save a horizontal bar plot for a frequency table with columns ["item", "frequency"].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if freq_df is None or freq_df.empty:
        fig = plt.figure()
        plt.title(title)
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    labels = freq_df["item"].astype(str).tolist()
    values = freq_df["frequency"].tolist()

    fig = plt.figure(figsize=(10, max(3, 0.4 * len(labels))))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Item")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)