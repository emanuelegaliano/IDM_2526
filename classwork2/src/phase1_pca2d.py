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


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Phase1Config:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    # input (da phase0)
    input_clean_csv: str = "phase0_clean.csv"

    # output (UNICO FILE)
    output_png: str = "phase1_pca2d.png"

    # colonne speciali
    target_col: str = "class"
    patient_id_col: str = "patient_id"  # se non esiste, ignorata

    # PCA
    scale_before_pca: bool = True
    random_state: int = 42


# ---------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Phase1Footprint:
    cfg: Phase1Config
    df: Optional[pd.DataFrame] = None
    X: Optional[np.ndarray] = None
    y: Optional[pd.Series] = None
    X_2d: Optional[np.ndarray] = None
    evr: Optional[list[float]] = None  # explained variance ratio [pc1, pc2]
    output_path: Optional[Path] = None

    _meta: FootprintMetadata = field(default_factory=FootprintMetadata)

    def get_metadata(self) -> FootprintMetadata:
        return self._meta


# ---------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------
@step("LoadClean")
def load_clean(fp: Phase1Footprint) -> Phase1Footprint:
    cfg = fp.cfg
    in_path = cfg.data_dir / cfg.input_clean_csv
    if not in_path.exists():
        raise FileNotFoundError(f"File clean non trovato: {in_path}")

    df = pd.read_csv(in_path)
    fp.df = df

    # metadata: solo tag/extra via metodi (no accesso diretto ad attributi)
    fp.get_metadata().add_tag("input_file", str(in_path))
    fp.get_metadata().add_extra("rows", int(len(df)))
    fp.get_metadata().add_extra("cols", int(df.shape[1]))
    return fp


@step("BuildXY")
def build_xy(fp: Phase1Footprint) -> Phase1Footprint:
    cfg = fp.cfg
    if fp.df is None:
        raise RuntimeError("df mancante. Eseguire LoadClean prima.")

    df = fp.df

    if cfg.target_col not in df.columns:
        raise KeyError(f"Colonna target '{cfg.target_col}' non trovata nel dataset clean.")

    y = df[cfg.target_col].astype(str)

    drop_cols = [cfg.target_col]
    if cfg.patient_id_col in df.columns:
        drop_cols.append(cfg.patient_id_col)

    X_df = df.drop(columns=drop_cols)

    non_numeric = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
    if non_numeric:
        raise TypeError(
            "Trovate colonne non numeriche nelle feature. "
            f"Controlla phase0 oppure encoda/rimuovi: {non_numeric}"
        )

    fp.X = X_df.to_numpy(dtype=float)
    fp.y = y

    fp.get_metadata().add_extra("n_features", int(fp.X.shape[1]))
    fp.get_metadata().add_extra("classes", sorted(y.unique().tolist()))
    return fp


@step("ComputePCA2D")
def compute_pca_2d(fp: Phase1Footprint) -> Phase1Footprint:
    cfg = fp.cfg
    if fp.X is None or fp.y is None:
        raise RuntimeError("X/y mancanti. Eseguire BuildXY prima.")

    X = fp.X
    if cfg.scale_before_pca:
        X = StandardScaler().fit_transform(X)
        fp.get_metadata().add_extra("scaled", True)
    else:
        fp.get_metadata().add_extra("scaled", False)

    pca = PCA(n_components=2, random_state=cfg.random_state)
    fp.X_2d = pca.fit_transform(X)
    fp.evr = pca.explained_variance_ratio_.tolist()

    fp.get_metadata().add_extra("explained_variance_ratio", fp.evr)
    return fp


@step("PlotAndSavePNG")
def plot_and_save_png(fp: Phase1Footprint) -> Phase1Footprint:
    cfg = fp.cfg
    if fp.X_2d is None or fp.y is None:
        raise RuntimeError("X_2d/y mancanti. Eseguire ComputePCA2D prima.")

    out_path = cfg.reports_dir / cfg.output_png
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X2 = fp.X_2d
    y = fp.y.astype(str)
    classes = sorted(y.unique().tolist())

    plt.figure()
    for cls in classes:
        mask = (y == cls).to_numpy()
        plt.scatter(X2[mask, 0], X2[mask, 1], label=cls)

    if fp.evr and len(fp.evr) == 2:
        title = f"PCA 2D (EVR: {fp.evr[0]:.2f}, {fp.evr[1]:.2f})"
    else:
        title = "PCA 2D"

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    fp.output_path = out_path
    fp.get_metadata().add_tag("phase1_output", str(out_path))
    return fp


# ---------------------------------------------------------------------
# Trail builder + runner
# ---------------------------------------------------------------------
def build_phase1_trail(cfg: Phase1Config, *, trace: bool = False) -> Trail[Phase1Footprint]:
    return (
        Trail[Phase1Footprint](name="phase1_pca2d")
        .then(load_clean, build_xy, compute_pca_2d, plot_and_save_png)
        .with_tag("phase", "1")
        .trace(trace)
    )


def run_phase1(cfg: Optional[Phase1Config] = None, *, trace: bool = False) -> Path:
    """
    Ritorna il Path dell'UNICO file prodotto: reports/phase1_pca2d.png
    """
    cfg = cfg or Phase1Config()
    fp = Phase1Footprint(cfg=cfg)
    trail = build_phase1_trail(cfg, trace=trace)
    fp = trail.run(fp)
    assert fp.output_path is not None
    return fp.output_path