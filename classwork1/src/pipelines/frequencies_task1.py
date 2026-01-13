from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from traccia import Trail, step

from src.utilities.fs import ensure_dir
from src.utilities.frequencies import freq_table, top_bottom_k
from src.utilities.plotting import plot_barh_frequency

from src.domain.footprint import FirstClassworkFootprint

@step("task1_compute_frequencies")
def task1_compute_frequencies(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Compute frequency tables for each merchandising level (liv1..liv4).

    Stores results in fp.freq_tables with keys:
      - "liv1", "liv2", "liv3", "liv4"
    """
    cfg = fp.config
    if fp.clean_df is None:
        raise ValueError("fp.clean_df is None. Run cleaning pipeline first.")

    df = fp.clean_df

    levels = {
        "liv1": cfg.col_liv1,
        "liv2": cfg.col_liv2,
        "liv3": cfg.col_liv3,
        "liv4": cfg.col_liv4,
    }

    out: Dict[str, pd.DataFrame] = {}
    for level_name, col in levels.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in clean_df (needed for {level_name}).")

        out[level_name] = freq_table(
            df,
            item_col=col,
            weight_col=cfg.col_quantity,
            use_quantity=cfg.frequency_use_quantity,
        )

    fp.freq_tables.update(out)
    fp.get_metadata().add_extra("task1_levels_computed", list(out.keys()))
    return fp


@step("task1_plot_top_bottom")
def task1_plot_top_bottom(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    For each level frequency table, create top-k and bottom-k barplots.
    Saves them under output_dir/task1/
    and appends saved paths to fp.plots.
    """
    cfg = fp.config
    if not fp.freq_tables:
        raise ValueError("fp.freq_tables is empty. Run task1_compute_frequencies first.")

    out_dir = ensure_dir(Path(cfg.output_dir) / "task1")
    saved: List[str] = []

    for level_name, freq_df in fp.freq_tables.items():
        # Only plot the task1 tables (liv1..liv4). Ignore any other tables added later.
        if level_name not in {"liv1", "liv2", "liv3", "liv4"}:
            continue

        top, bottom = top_bottom_k(freq_df, cfg.top_k)

        top_path = out_dir / f"{level_name}_top{cfg.top_k}.png"
        bottom_path = out_dir / f"{level_name}_bottom{cfg.top_k}.png"

        plot_barh_frequency(top, title=f"Task 1 - {level_name} Top {cfg.top_k}", out_path=top_path)
        plot_barh_frequency(bottom, title=f"Task 1 - {level_name} Bottom {cfg.top_k}", out_path=bottom_path)

        saved.extend([str(top_path), str(bottom_path)])

    fp.plots.extend(saved)
    fp.get_metadata().add_extra("task1_plots_saved", saved)
    return fp


# -----------------------------------------------------------------------------
# Trail
# -----------------------------------------------------------------------------
TASK1_FREQUENCIES_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(task1_compute_frequencies)
    .then(task1_plot_top_bottom)
)