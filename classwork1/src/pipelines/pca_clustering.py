from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from traccia import Trail, step

from src.domain.footprint import FirstClassworkFootprint
from src.utilities.fs import ensure_dir


# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------
@step("filter_valid_cards")
def filter_valid_cards(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Keep only transactions with a non-empty loyalty card.
    """
    cfg = fp.config
    df = fp.clean_df
    if df is None:
        raise ValueError("fp.clean_df is None. Run cleaning first.")

    card_col = cfg.col_card_id
    if card_col not in df.columns:
        raise KeyError(f"Card column '{card_col}' not found in dataset.")

    mask = df[card_col].notna() & (df[card_col].astype(str) != "")
    filtered = df.loc[mask].copy()

    fp.clean_df = filtered
    fp.get_metadata().add_extra("task5_cards_rows", int(len(filtered)))
    fp.get_metadata().add_extra("task5_unique_cards", int(filtered[card_col].nunique()))
    return fp


@step("build_card_product_matrix")
def build_card_product_matrix(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Build card x product frequency matrix.
    Values = purchase frequency.
    """
    cfg = fp.config
    df = fp.clean_df
    if df is None:
        raise ValueError("fp.clean_df is None.")

    card_col = cfg.col_card_id
    prod_col = cfg.col_product_code

    matrix = (
        df.groupby([card_col, prod_col])
        .size()
        .unstack(fill_value=0)
        .astype(float)
    )

    fp.card_product_matrix = matrix
    fp.get_metadata().add_extra("task5_matrix_shape", matrix.shape)
    return fp


@step("scale_matrix")
def scale_matrix(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Standardize the card-product matrix.
    """
    X = fp.card_product_matrix
    if X is None:
        raise ValueError("card_product_matrix is None.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fp.card_product_matrix = pd.DataFrame(
        X_scaled,
        index=X.index,
        columns=X.columns,
    )
    return fp


@step("apply_pca")
def apply_pca(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Apply PCA to reduce dimensionality.
    """
    cfg = fp.config
    X = fp.card_product_matrix
    if X is None:
        raise ValueError("card_product_matrix is None.")

    pca = PCA(n_components=cfg.pca_explained_variance, random_state=cfg.random_state)
    X_pca = pca.fit_transform(X)

    fp.pca_embeddings = X_pca
    fp.get_metadata().add_extra("task5_pca_components", int(X_pca.shape[1]))
    fp.get_metadata().add_extra(
        "task5_pca_explained_variance",
        float(np.sum(pca.explained_variance_ratio_)),
    )
    return fp


@step("cluster_cards")
def cluster_cards(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Cluster cards using KMeans in PCA space.
    """
    cfg = fp.config
    X = fp.pca_embeddings
    if X is None:
        raise ValueError("pca_embeddings is None.")

    kmeans = KMeans(
        n_clusters=cfg.n_clusters,
        random_state=cfg.random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(X)

    fp.cluster_labels = labels

    # Quality metric
    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
        fp.metrics["task5_silhouette"] = float(sil)

    return fp


@step("save_task5_outputs")
def save_task5_outputs(fp: FirstClassworkFootprint) -> FirstClassworkFootprint:
    """
    Save clustering results to disk.
    """
    cfg = fp.config
    out_dir = ensure_dir(Path(cfg.output_dir) / "task5_clustering")

    # Save embeddings + cluster labels
    df_out = pd.DataFrame(
        fp.pca_embeddings,
        index=fp.card_product_matrix.index, # type: ignore
    )
    df_out["cluster"] = fp.cluster_labels
    df_out.to_csv(out_dir / "card_clusters.csv")

    fp.plots.append(str(out_dir / "card_clusters.csv"))
    return fp


# -----------------------------------------------------------------------------
# Trail
# -----------------------------------------------------------------------------
TASK5_PCA_CLUSTERING_TRAIL: Trail[FirstClassworkFootprint] = (
    Trail()
    .then(filter_valid_cards)
    .then(build_card_product_matrix)
    .then(scale_matrix)
    .then(apply_pca)
    .then(cluster_cards)
    .then(save_task5_outputs)
)
