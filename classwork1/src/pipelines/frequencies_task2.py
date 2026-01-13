from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from traccia import Trail, step

from src.domain.footprint import FirstClassworkFootprint
from src.utilities.fs import ensure_dir
from src.utilities.frequencies import freq_table, top_bottom_k
from src.utilities.plotting import plot_barh_frequency


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _levels_map(fp: FirstClassworkFootprint) -> Dict[str, str]:
    cfg = fp.config
    return {
        "liv1": cfg.col_liv1,
        "liv2": cfg.col_liv2,
        "liv3": cfg.col_liv3,
        "liv4": cfg.col_liv4,
    }


def _compute_and_store_stratified(
    fp: FirstClassworkFootprint,
    *,
    stratum_col: str,
    out_subdir: str,
    key_prefix: str,
) -> FirstClassworkFootprint:
    """
    Generic routine for Task2:
    - for each stratum value in df[stratum_col]
    - for each merchandising level (liv1..liv4)
      compute frequency table, store in fp.freq_tables, and save top/bottom plots.

    Parameters:
      stratum_col: "month_range" or "time_slot"
      out_subdir:  "month_range" or "time_slot"
      key_prefix:  "task2:month_range" or "task2:time_slot"
    """
    cfg = fp.config
    if fp.clean_df is None:
        raise ValueError("fp.clean_df is None. Run cleaning pipeline first.")

    df = fp.clean_df

    if stratum_col not in df.columns:
        raise KeyError(
            f"Column '{stratum_col}' not found in clean_df. "
            "Make sure Phase B adds it (add_time_bins step)."
        )

    levels = _levels_map(fp)

    # We only consider non-null strata; if you want OUTSIDE included for time_slot, keep it.
    strata_values = (
        df[stratum_col]
        .astype("string")
        .dropna()
        .unique()
        .tolist()
    )
    strata_values = sorted([s for s in strata_values if s != ""])

    base_out = ensure_dir(Path(cfg.output_dir) / "task2" / out_subdir)

    saved_paths: List[str] = []
    tables_written = 0

    for sv in strata_values:
        sdf = df[df[stratum_col].astype("string") == sv]
        # Create output folder per stratum value
        stratum_out = ensure_dir(base_out / str(sv))

        for level_name, col in levels.items():
            if col not in sdf.columns:
                raise KeyError(f"Column '{col}' not found in clean_df (needed for {level_name}).")

            ft = freq_table(
                sdf,
                item_col=col,
                weight_col=cfg.col_quantity,
                use_quantity=cfg.frequency_use_quantity,
            )

            # Store table in footprint
            key = f"{key_prefix}:{sv}:{level_name}"
            fp.freq_tables[key] = ft
            tables_written += 1

            top, bottom = top_bottom_k(ft, cfg.top_k)

            top_path = stratum_out / f"{level_name}_top{cfg.top_k}.png"
            bottom_path = stratum_out / f"{level_name}_bottom{cfg.top_k}.png"

            plot_barh_frequency(
                top,
                title=f"Task 2 ({stratum_col}={sv}) - {level_name} Top {cfg.top_k}",
                out_path=top_path,
            )
            plot_barh_frequency(
                bottom,
                title=f"Task 2 ({stratum_col}={sv}) - {level_name} Bottom {cfg.top_k}",
                out_path=bottom_path,
            )

            saved_paths.extend([str(top_path), str(bottom_path)])

    fp.plots.extend(saved_paths)
    fp.get_metadata().add_extra(f"{key_prefix}_strata_values", strata_values)
    fp.get_metadata().add_extra(f"{key_prefix}_tables_written", int(tables_written))
    fp.get_metadata().add_extra(f"{key_prefix}_plots_saved_count", int(len(saved_paths)))
    return fp


# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------
@step("task2_month_range_frequencies_and_plots")
def task2_month_range_frequencies_and_plots(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Task2 part A: stratify by month_range.
    Output directory: outputs/task2/month_range/<range>/
    """
    return _compute_and_store_stratified(
        fp,
        stratum_col="month_range",
        out_subdir="month_range",
        key_prefix="task2:month_range",
    )


@step("task2_time_slot_frequencies_and_plots")
def task2_time_slot_frequencies_and_plots(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Task2 part B: stratify by time_slot.
    Output directory: outputs/task2/time_slot/<slot>/
    """
    return _compute_and_store_stratified(
        fp,
        stratum_col="time_slot",
        out_subdir="time_slot",
        key_prefix="task2:time_slot",
    )


# -----------------------------------------------------------------------------
# Trails
# -----------------------------------------------------------------------------
TASK2_MONTH_RANGE_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(task2_month_range_frequencies_and_plots)
)

TASK2_TIME_SLOT_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(task2_time_slot_frequencies_and_plots)
)

TASK2_FULL_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(task2_month_range_frequencies_and_plots)
    .then(task2_time_slot_frequencies_and_plots)
)
