from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

from traccia import Trail, step

from src.domain.footprint import FirstClassworkFootprint
from src.utilities.fs import ensure_dir


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _build_transactions_lvl4(
    df: pd.DataFrame,
    *,
    receipt_col: str,
    item_col: str,
    min_support: float,
    max_items: int,
) -> pd.DataFrame:
    """
    Build a one-hot encoded transaction matrix at level 4, with item filtering.

    Strategy:
    1) compute item support (share of receipts containing the item)
    2) keep only items with support >= min_support
    3) if still too many, keep only top `max_items` by support
    4) build basket (receipts x items) as boolean
    """
    # Unique item per receipt
    pairs = df[[receipt_col, item_col]].dropna().drop_duplicates()

    # Support = receipts containing item / total receipts
    total_receipts = pairs[receipt_col].nunique()
    item_counts = pairs.groupby(item_col)[receipt_col].nunique().sort_values(ascending=False)
    item_support = item_counts / max(total_receipts, 1)

    # Keep items by min_support
    keep = item_support[item_support >= min_support]

    # If still too many, keep top max_items
    if max_items is not None and len(keep) > max_items:
        keep = keep.head(max_items)

    keep_items = set(keep.index.astype("string").tolist())

    # Filter pairs
    pairs = pairs[pairs[item_col].astype("string").isin(keep_items)]

    # Build basket
    basket = (
        pairs.assign(value=True)
        .pivot_table(index=receipt_col, columns=item_col, values="value", fill_value=False)
    )

    # Ensure boolean dtype
    basket = basket.astype(bool)
    return basket


def _save_rules(
    rules: pd.DataFrame,
    *,
    out_path: Path,
) -> None:
    """Save association rules to CSV."""
    rules.to_csv(out_path, index=False)


# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------
@step("build_transactions_lvl4")
def build_transactions_lvl4(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    cfg = fp.config
    if fp.clean_df is None:
        raise ValueError("fp.clean_df is None. Run cleaning pipeline first.")

    basket = _build_transactions_lvl4(
        fp.clean_df,
        receipt_col=cfg.col_receipt_id,
        item_col=cfg.col_liv4,
        min_support=cfg.min_support,
        max_items=cfg.rules_item_max_count,
    )

    fp.transactions_lvl4 = basket # type: ignore
    fp.get_metadata().add_extra("transactions_count", int(basket.shape[0]))
    fp.get_metadata().add_extra("unique_items_lvl4_after_filter", int(basket.shape[1]))
    return fp


@step("apriori_rules")
def apriori_rules(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Mine association rules using the Apriori algorithm.
    """
    cfg = fp.config
    basket = fp.transactions_lvl4
    if basket is None:
        raise ValueError("transactions_lvl4 not found. Run build_transactions_lvl4 first.")

    freq_items = apriori(
        basket,
        min_support=cfg.min_support,
        use_colnames=True,
        low_memory=True,
        max_len=cfg.rules_apriori_max_len,
    )

    rules = association_rules(
        freq_items,
        metric="confidence",
        min_threshold=cfg.min_confidence,
    )

    rules = rules[rules["lift"] >= cfg.min_lift].reset_index(drop=True)

    fp.rules_apriori = rules
    fp.get_metadata().add_extra("apriori_rules_count", int(len(rules)))
    return fp


@step("fpgrowth_rules")
def fpgrowth_rules(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Mine association rules using the FP-Growth algorithm.
    """
    cfg = fp.config
    basket = fp.transactions_lvl4
    if basket is None:
        raise ValueError("transactions_lvl4 not found. Run build_transactions_lvl4 first.")

    freq_items = fpgrowth(
        basket,
        min_support=cfg.min_support,
        use_colnames=True,
        max_len=cfg.max_rule_length,
    )

    rules = association_rules(
        freq_items,
        metric="confidence",
        min_threshold=cfg.min_confidence,
    )

    rules = rules[rules["lift"] >= cfg.min_lift].reset_index(drop=True)

    fp.rules_fpgrowth = rules
    fp.get_metadata().add_extra("fpgrowth_rules_count", int(len(rules)))
    return fp


@step("save_rules")
def save_rules(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Save Apriori and FP-Growth rules to disk.
    """
    cfg = fp.config
    out_dir = ensure_dir(Path(cfg.output_dir) / "task3_4_rules")

    if fp.rules_apriori is not None:
        path = out_dir / "apriori_rules.csv"
        _save_rules(fp.rules_apriori, out_path=path)
        fp.plots.append(str(path))

    if fp.rules_fpgrowth is not None:
        path = out_dir / "fpgrowth_rules.csv"
        _save_rules(fp.rules_fpgrowth, out_path=path)
        fp.plots.append(str(path))

    return fp


# -----------------------------------------------------------------------------
# Trails
# -----------------------------------------------------------------------------
TASK3_APRIORI_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(build_transactions_lvl4)
    .then(apriori_rules)
    .then(save_rules)
)

TASK4_FPGROWTH_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(build_transactions_lvl4)
    .then(fpgrowth_rules)
    .then(save_rules)
)

TASK3_4_FULL_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(build_transactions_lvl4)
    .then(apriori_rules)
    .then(fpgrowth_rules)
    .then(save_rules)
)
