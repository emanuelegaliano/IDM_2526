# src/phase1_pca2d.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from traccia import Trail, step, FootprintMetadata


@dataclass(slots=True)
class Phase1Config:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    input_clean_csv: str = "phase0_clean.csv"
    output_png: str = "phase1_pca2d.png"
    target_col: str = "class"
    patient_id_col: str = "patient_id"
    scale_before_pca: bool = True
    random_state: int = 42


@dataclass(slots=True)
class Phase1Footprint:
    cfg: Phase1Config
    df: Optional[pd.DataFrame] = None
    X: Optional[np.ndarray] = None
    y: Optional[pd.Series] = None
    X_2d: Optional[np.ndarray] = None
    evr: Optional[list[float]] = None
    output_path: Optional[Path] = None

    _meta: FootprintMetadata = field(default_factory=FootprintMetadata)

    def get_metadata(self) -> FootprintMetadata:
        return self._meta


@step("LoadClean")
def load_clean(fp: Phase1Footprint) -> Phase1Footprint:
    in_path = fp.cfg.data_dir / fp.cfg.input_clean_csv
    if not in_path.exists():
        raise FileNotFoundError(f"File clean non trovato: {in_path}")
    fp.df = pd.read_csv(in_path)
    return fp


@step("BuildXY")
def build_xy(fp: Phase1Footprint) -> Phase1Footprint:
    if fp.df is None:
        raise RuntimeError("df mancante.")
    df = fp.df
    if fp.cfg.target_col not in df.columns:
        raise KeyError(f"Colonna target '{fp.cfg.target_col}' non trovata.")
    y = df[fp.cfg.target_col].astype(str)
    drop_cols = [fp.cfg.target_col]
    if fp.cfg.patient_id_col in df.columns:
        drop_cols.append(fp.cfg.patient_id_col)
    X_df = df.drop(columns=drop_cols)
    non_numeric = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
    if non_numeric:
        raise TypeError(f"Feature non numeriche: {non_numeric}")
    fp.X = X_df.to_numpy(dtype=float)
    fp.y = y
    return fp


@step("ComputePCA2D")
def compute_pca_2d(fp: Phase1Footprint) -> Phase1Footprint:
    if fp.X is None or fp.y is None:
        raise RuntimeError("X/y mancanti.")
    X = fp.X
    if fp.cfg.scale_before_pca:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=fp.cfg.random_state)
    fp.X_2d = pca.fit_transform(X)
    fp.evr = pca.explained_variance_ratio_.tolist()
    return fp


@step("PlotAndSavePNG")
def plot_and_save_png(fp: Phase1Footprint) -> Phase1Footprint:
    if fp.X_2d is None or fp.y is None:
        raise RuntimeError("X_2d/y mancanti.")
    out_path = fp.cfg.reports_dir / fp.cfg.output_png
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X2 = fp.X_2d
    y = fp.y.astype(str)
    classes = sorted(y.unique().tolist())

    plt.figure()
    for cls in classes:
        mask = (y == cls).to_numpy()
        plt.scatter(X2[mask, 0], X2[mask, 1], label=cls)

    if fp.evr and len(fp.evr) == 2:
        plt.title(f"PCA 2D (EVR: {fp.evr[0]:.2f}, {fp.evr[1]:.2f})")
    else:
        plt.title("PCA 2D")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    fp.output_path = out_path
    return fp


def build_phase1_trail(cfg: Phase1Config, *, trace: bool = False) -> Trail[Phase1Footprint]:
    return (
        Trail[Phase1Footprint](name="phase1_pca2d")
        .then(load_clean, build_xy, compute_pca_2d, plot_and_save_png)
        .with_tag("phase", "1")
        .trace(trace)
    )


def run_phase1(cfg: Optional[Phase1Config] = None, *, trace: bool = False) -> Path:
    cfg = cfg or Phase1Config()
    fp = Phase1Footprint(cfg=cfg)
    fp = build_phase1_trail(cfg, trace=trace).run(fp)
    assert fp.output_path is not None
    return fp.output_path
