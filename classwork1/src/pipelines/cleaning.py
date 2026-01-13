from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time
from pathlib import Path
from typing import Iterable, Optional, Tuple
import re

import numpy as np
import pandas as pd

from traccia import Trail, step

from src.domain.config import Config
from src.domain.footprint import FirstClassworkFootprint


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _infer_reader(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet"}:
        return "parquet"
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".tsv"}:
        return "tsv"
    raise ValueError(f"Unsupported dataset format: '{suffix}'. Use CSV/TSV/Parquet.")


def _safe_str_series(s: pd.Series) -> pd.Series:
    # Avoid issues with mixed types / NaN
    return s.astype("string").fillna("")


def _contains_any_keyword(text_series: pd.Series, keywords: Tuple[str, ...]) -> pd.Series:
    """
    Case-insensitive keyword check over a Series. Returns a boolean mask.
    """
    if not keywords:
        return pd.Series(False, index=text_series.index)

    s = _safe_str_series(text_series).str.upper()
    pattern = "|".join([re.escape(k.upper()) for k in keywords if k])
    if not pattern:
        return pd.Series(False, index=text_series.index)
    return s.str.contains(pattern, regex=True, na=False)


def _parse_date_series(date_s: pd.Series, *, date_format: Optional[str], dayfirst: bool) -> pd.Series:
    """
    Parse dates robustly. If a format is provided, parsing is deterministic and warning-free.
    """
    if date_format:
        return pd.to_datetime(date_s, errors="coerce", format=date_format)
    return pd.to_datetime(date_s, errors="coerce", dayfirst=dayfirst)


def _parse_time_series(time_s: pd.Series, *, time_format: Optional[str]) -> pd.Series:
    """
    Parse time robustly. If a format is provided, parsing is deterministic and faster.
    Returns pandas datetime64[ns] timestamps anchored to 1970-01-01.
    """
    if np.issubdtype(time_s.dtype, np.datetime64): # type: ignore
        return pd.to_datetime(time_s, errors="coerce")

    if time_s.apply(lambda x: isinstance(x, time)).any():
        tmp = time_s.apply(lambda t: t.strftime("%H:%M:%S") if isinstance(t, time) else t)
        time_s = tmp

    if time_format:
        return pd.to_datetime(time_s, errors="coerce", format=time_format)

    # Fallback: try the two most common formats first to reduce warnings and cost
    parsed = pd.to_datetime(time_s, errors="coerce", format="%H:%M:%S")
    missing = parsed.isna()
    if missing.any():
        parsed2 = pd.to_datetime(time_s[missing], errors="coerce", format="%H:%M")
        parsed.loc[missing] = parsed2
    return parsed


def _bin_month_range(d: pd.Timestamp, *, mid_day: int) -> Optional[str]:
    """
    Map a date to one of the three assignment-required month ranges:
      - Jan → mid May
      - mid May → Sep
      - Oct → Dec

    Convention:
      - "mid May" = day <= mid_day belongs to the first range, day > mid_day belongs to the second range.
    """
    if pd.isna(d):
        return None

    m = int(d.month)
    if m < 5:
        return "Jan-midMay"
    if m > 5 and m <= 9:
        return "midMay-Sep"
    if m >= 10:
        return "Oct-Dec"

    # m == 5 (May): split by mid_day
    day = int(d.day)
    return "Jan-midMay" if day <= mid_day else "midMay-Sep"


def _bin_time_slot(t: pd.Timestamp) -> Optional[str]:
    """
    Map a time to one of the three assignment-required slots:
      - 08:30 → 12:30
      - 12:30 → 16:30
      - 16:30 → 20:30

    Convention:
      - start is inclusive, end is exclusive for the first two slots
      - last slot end is inclusive (to catch exactly 20:30 if present)

    Note: t is a pandas Timestamp (anchored date). We use only hour/minute.
    """
    if pd.isna(t):
        return None

    hh = int(t.hour)
    mm = int(t.minute)
    minutes = hh * 60 + mm

    s1 = 8 * 60 + 30
    e1 = 12 * 60 + 30
    s2 = 12 * 60 + 30
    e2 = 16 * 60 + 30
    s3 = 16 * 60 + 30
    e3 = 20 * 60 + 30

    if s1 <= minutes < e1:
        return "08:30-12:30"
    if s2 <= minutes < e2:
        return "12:30-16:30"
    if s3 <= minutes <= e3:
        return "16:30-20:30"
    return "OUTSIDE"


# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------
@step("log_config")
def log_config(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Store config snapshot in metadata for reproducibility.
    """
    fp.get_metadata().add_extra("config", asdict(fp.config))
    return fp


@step("load_dataset")
def load_dataset(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Load the dataset from fp.config.data_path into fp.raw_df.

    Supports:
      - CSV/TSV (using pandas.read_csv)
      - Parquet (using pandas.read_parquet)

    Notes:
      - For huge datasets, you can extend this to chunked reads later.
    """
    cfg = fp.config
    reader = _infer_reader(cfg.data_path)

    if reader == "parquet":
        df = pd.read_parquet(cfg.data_path)
    else:
        # Default separators: CSV -> ',', TSV -> '\t'
        sep = "\t" if reader == "tsv" else ","
        df = pd.read_csv(cfg.data_path, sep=sep, low_memory=False)

    fp.raw_df = df
    fp.get_metadata().add_extra("rows_loaded", int(len(df)))
    return fp


@step("normalize_schema")
def normalize_schema(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Normalize the dataset schema to the canonical column names stored in Config.

    This step:
      - checks that the required columns exist
      - creates fp.clean_df as a working copy

    If your dataset uses different column names, you can add a rename mapping
    here (or in Config) later.
    """
    cfg = fp.config
    df = fp.raw_df.copy() # type: ignore

    required_cols = [
        cfg.col_receipt_id,
        cfg.col_product_code,
        cfg.col_product_desc,
        cfg.col_liv1,
        cfg.col_liv2,
        cfg.col_liv3,
        cfg.col_liv4,
        cfg.col_date,
        cfg.col_time,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Dataset is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)[:50]}"
        )

    # Optional column checks
    if cfg.col_card_id and cfg.col_card_id not in df.columns:
        # Not fatal (some datasets might have empty card column)
        fp.get_metadata().add_extra("warning_missing_card_col", cfg.col_card_id)

    if cfg.col_quantity and cfg.col_quantity not in df.columns:
        fp.get_metadata().add_extra("warning_missing_quantity_col", cfg.col_quantity)

    fp.clean_df = df
    fp.get_metadata().add_extra("rows_after_schema", int(len(df)))
    return fp


@step("parse_datetime")
def parse_datetime(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Parse date + time into a single datetime column `receipt_datetime`,
    also store parsed `parsed_date` and `parsed_time` for later binning.
    """
    cfg = fp.config
    df = fp.clean_df

    parsed_date = _parse_date_series(
        df[cfg.col_date], # type: ignore
        date_format=cfg.date_format,
        dayfirst=cfg.dayfirst,
    )
    parsed_time = _parse_time_series(
        df[cfg.col_time], # type: ignore
        time_format=cfg.time_format,
    )


    # Combine into datetime: anchor times to the parsed date
    # We keep separate columns too, because task-2 needs month/time bins.
    receipt_dt = pd.to_datetime(
        parsed_date.dt.strftime("%Y-%m-%d") + " " + parsed_time.dt.strftime("%H:%M:%S"), # type: ignore
        errors="coerce",
    )

    df = df.copy() # type: ignore
    df["parsed_date"] = parsed_date
    df["parsed_time"] = parsed_time
    df["receipt_datetime"] = receipt_dt

    fp.clean_df = df
    fp.get_metadata().add_extra("rows_after_datetime_parse", int(len(df)))
    fp.get_metadata().add_extra(
        "null_receipt_datetime", int(df["receipt_datetime"].isna().sum())
    )
    return fp


@step("drop_invalid_rows")
def drop_invalid_rows(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Drop rows with missing product or missing date/time depending on config.
    """
    cfg = fp.config
    df = fp.clean_df.copy() # type: ignore

    before = len(df)

    if cfg.drop_null_products:
        df = df.dropna(subset=[cfg.col_product_code, cfg.col_product_desc])

    if cfg.drop_null_dates:
        df = df.dropna(subset=["receipt_datetime"])

    after = len(df)
    fp.clean_df = df
    fp.get_metadata().add_extra("rows_dropped_invalid", int(before - after))
    fp.get_metadata().add_extra("rows_after_drop_invalid", int(after))
    return fp


@step("exclude_shoppers")
def exclude_shoppers(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Exclude 'shoppers' lines as required by the assignment.

    Policy:
      - if Config.exclude_shoppers is True, remove rows where product description
        contains any of Config.shoppers_keywords (case-insensitive).
      - this is configurable because the dataset may encode shoppers differently.
    """
    cfg = fp.config
    df = fp.clean_df.copy() # type: ignore

    if not cfg.exclude_shoppers:
        fp.get_metadata().add_extra("exclude_shoppers_applied", False)
        return fp

    before = len(df)
    mask = _contains_any_keyword(df[cfg.col_product_desc], cfg.shoppers_keywords)
    df = df.loc[~mask].copy()

    after = len(df)
    fp.clean_df = df
    fp.get_metadata().add_extra("exclude_shoppers_applied", True)
    fp.get_metadata().add_extra("rows_excluded_shoppers", int(before - after))
    fp.get_metadata().add_extra("rows_after_exclude_shoppers", int(after))
    return fp


@step("add_time_bins")
def add_time_bins(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Add stratification columns used by Task 2:
      - month_range: 3 ranges split at mid-May
      - time_slot: 3 slots (plus OUTSIDE)

    This step is safe to run always; it only creates extra columns.
    """
    cfg = fp.config
    df = fp.clean_df.copy() # type: ignore

    df["month_range"] = df["parsed_date"].apply(lambda d: _bin_month_range(d, mid_day=cfg.mid_month_day))
    df["time_slot"] = df["parsed_time"].apply(_bin_time_slot)

    fp.clean_df = df
    fp.get_metadata().add_extra("time_bins_added", True)
    fp.get_metadata().add_extra("null_month_range", int(df["month_range"].isna().sum()))
    fp.get_metadata().add_extra("null_time_slot", int(df["time_slot"].isna().sum()))
    return fp


@step("finalize_cleaning")
def finalize_cleaning(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Final sanity normalization:
      - enforce string dtype for key categorical columns
    """
    cfg = fp.config
    df = fp.clean_df.copy() # type: ignore

    # Force stable dtypes for downstream groupby/pivot/rules
    for col in [
        cfg.col_receipt_id,
        cfg.col_card_id if cfg.col_card_id in df.columns else None,
        cfg.col_product_code,
        cfg.col_product_desc,
        cfg.col_liv1,
        cfg.col_liv2,
        cfg.col_liv3,
        cfg.col_liv4,
        "month_range",
        "time_slot",
    ]:
        if col and col in df.columns:
            df[col] = df[col].astype("string")

    fp.clean_df = df
    fp.get_metadata().add_extra("rows_final_clean", int(len(df)))
    return fp

CLEANING_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(log_config)
    .then(load_dataset)
    .then(normalize_schema)
    .then(parse_datetime)
    .then(drop_invalid_rows)
    .then(exclude_shoppers)
    .then(add_time_bins)
    .then(finalize_cleaning)
)