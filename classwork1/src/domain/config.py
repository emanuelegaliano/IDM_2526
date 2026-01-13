from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class Config:
    """
    Global configuration object for the First Classwork project.
    All design choices and hyperparameters must be declared here.
    """

    # -------------------------
    # Paths
    # -------------------------
    data_path: str
    output_dir: str = "outputs"

    # -------------------------
    # Dataset schema
    # -------------------------
    col_receipt_id: str = "scontrino_id"
    col_card_id: str = "tessera"
    col_product_code: str = "cod_prod"
    col_product_desc: str = "descr_prod"

    col_liv1: str = "liv1"
    col_liv2: str = "liv2"
    col_liv3: str = "liv3"
    col_liv4: str = "liv4"

    col_date: str = "data"
    col_time: str = "ora"

    col_quantity: Optional[str] = "r_qta_pezzi"

    # -------------------------
    # Cleaning rules
    # -------------------------
    exclude_shoppers: bool = True
    shoppers_keywords: Tuple[str, ...] = ("SHOPPER", "SHOPPERS")

    drop_null_products: bool = True
    drop_null_dates: bool = True

    # -------------------------
    # Frequency analysis
    # -------------------------
    frequency_use_quantity: bool = False
    top_k: int = 5

    # -------------------------
    # Time stratification
    # -------------------------
    # Month ranges expressed as (start_month, end_month)
    month_ranges: Tuple[Tuple[int, int], ...] = (
        (1, 5),   # Jan - mid May
        (5, 9),   # mid May - Sep
        (10, 12)  # Oct - Dec
    )

    mid_month_day: int = 15

    # Time slots expressed as (start_hour, start_min, end_hour, end_min)
    time_slots: Tuple[Tuple[int, int, int, int], ...] = (
        (8, 30, 12, 30),
        (12, 30, 16, 30),
        (16, 30, 20, 30),
    )

    # -------------------------
    # Association rules
    # -------------------------
    min_support: float = 0.02
    min_confidence: float = 0.3
    min_lift: float = 1.0
    max_rule_length: int = 4
    
    # scalability controls
    rules_item_max_count: int = 2000
    rules_apriori_max_len: int = 2
    rules_use_sparse: bool = True

    # -------------------------
    # Card x Product matrix
    # -------------------------
    min_transactions_per_card: int = 1

    # -------------------------
    # PCA
    # -------------------------
    pca_n_components: Optional[int] = None
    pca_explained_variance: float = 0.8

    # -------------------------
    # Clustering
    # -------------------------
    clustering_method: str = "kmeans"
    n_clusters: int = 5
    random_state: int = 42

    # -------------------------
    # Parsing formats
    # -------------------------
    date_format: Optional[str] = None   # e.g. "%Y-%m-%d" or "%d/%m/%Y"
    time_format: Optional[str] = None   # e.g. "%H:%M:%S" or "%H:%M"
    dayfirst: bool = True
