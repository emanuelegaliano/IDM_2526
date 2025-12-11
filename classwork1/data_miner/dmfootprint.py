from dataclasses import dataclass, field
from typing import Any, Hashable
from pandas import DataFrame, Series
from traccia import Footprint


@dataclass
class DataMiningFootprint(Footprint):
    # --- Main tabular dataset ---
    raw_df: DataFrame
    cleaned_df: DataFrame | None = None

    # --- Dataset partitions (e.g., time ranges, segments, categories) ---
    # key = partition name, value = corresponding dataframe
    partitions: dict[str, DataFrame] = field(default_factory=dict)

    # --- Hierarchical structures (levels, categories, groupings, etc.) ---
    # key = hierarchy name or level, value = any associated structure
    hierarchies: dict[str, Any] = field(default_factory=dict)

    # --- Aggregation and descriptive analysis results ---
    # key = metric or grouping name, value = DataFrame with results
    aggregation_results: dict[str, DataFrame] = field(default_factory=dict)

    # --- Market Basket Analysis (MBA) ---
    # list of transactions, each transaction = list of items
    transactions: list[list[Hashable]] | None = None

    # MBA results (rules, support/confidence tables, frequent itemsets, etc.)
    # key = result type, value = DataFrame
    market_basket_results: dict[str, DataFrame] = field(default_factory=dict)

    # --- Entity–Item matrix (e.g., user-item, customer-product, etc.) ---
    entity_item_matrix: DataFrame | None = None

    # --- Dimensionality reduction results (PCA, t-SNE, UMAP, etc.) ---
    reduced_representation: DataFrame | None = None

    # --- Clustering / segmentation labels ---
    cluster_labels: Series | None = None

    # --- Miscellaneous artifacts (plots, metrics, serialized objects, etc.) ---
    artifacts: dict[str, Any] = field(default_factory=dict)
