"""
FirstClassworkFootprint

This module defines the shared state container (Footprint) used by TRACCIA trails.
It stores:
- configuration
- raw and cleaned dataframes
- intermediate artifacts produced by later tasks (frequencies, plots, rules, clustering, ...)

Recommended location:
    src/first_classwork/domain/footprint.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

# TRACCIA core objects
from traccia import Footprint
from traccia import FootprintMetadata

from src.domain.config import Config


@dataclass
class FirstClassworkFootprint(Footprint):
    """
    Project footprint shared across all pipelines.

    Notes:
    - raw_df / clean_df are filled by the cleaning pipeline
    - other artifacts are filled by subsequent task pipelines
    """

    # ---- configuration ----
    config: Config

    # ---- dataframes ----
    raw_df: Optional[pd.DataFrame] = None
    clean_df: Optional[pd.DataFrame] = None

    # ---- task 1 & 2 outputs ----
    freq_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    plots: List[str] = field(default_factory=list)

    # ---- task 3 & 4 outputs (association rules) ----
    transactions_lvl4: Optional[List[List[str]]] = None
    rules_apriori: Optional[pd.DataFrame] = None
    rules_fpgrowth: Optional[pd.DataFrame] = None

    # ---- task 5 outputs (card-product matrix, PCA, clustering) ----
    card_product_matrix: Optional[pd.DataFrame] = None
    pca_embeddings: Optional[Any] = None  # typically np.ndarray
    cluster_labels: Optional[Any] = None  # typically np.ndarray
    metrics: Dict[str, Any] = field(default_factory=dict)

    # ---- internal metadata (TRACCIA) ----
    _metadata: FootprintMetadata = field(default_factory=FootprintMetadata, init=False, repr=False)

    def get_metadata(self) -> FootprintMetadata:
        """
        TRACCIA expects footprints to expose metadata for auditability and debug.
        """
        return self._metadata
