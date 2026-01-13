# src/utilities/frequencies.py
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def freq_table(
    df: pd.DataFrame,
    item_col: str,
    *,
    weight_col: Optional[str] = None,
    use_quantity: bool = False,
) -> pd.DataFrame:
    """
    Compute frequency table for a categorical column.

    If use_quantity=True and weight_col exists, frequency is sum(weight_col).
    Otherwise frequency is count of rows.

    Returns a DataFrame with columns: ["item", "frequency"] sorted by frequency desc.
    """
    if use_quantity and weight_col and weight_col in df.columns:
        qty = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
        out = (
            df.assign(_qty=qty)
            .groupby(item_col, dropna=False)["_qty"]
            .sum()
            .reset_index()
            .rename(columns={item_col: "item", "_qty": "frequency"})
        )
    else:
        out = (
            df.groupby(item_col, dropna=False)
            .size()
            .reset_index(name="frequency")
            .rename(columns={item_col: "item"})
        )

    out["item"] = out["item"].astype("string").fillna("")
    out = out.sort_values(["frequency", "item"], ascending=[False, True]).reset_index(drop=True)
    return out


def top_bottom_k(freq_df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return top-k and bottom-k rows from a frequency table.
    If fewer than k items exist, returns as many as possible.
    """
    if freq_df.empty:
        return freq_df.copy(), freq_df.copy()

    k = max(int(k), 1)
    top = freq_df.head(k).copy()

    bottom = (
        freq_df.sort_values(["frequency", "item"], ascending=[True, True])
        .head(k)
        .copy()
        .reset_index(drop=True)
    )
    return top, bottom