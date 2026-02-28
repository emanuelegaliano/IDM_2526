# src/phase3_ensembles.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

from traccia import Trail, step, FootprintMetadata


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Phase3Config:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    # input
    input_clean_csv: str = "phase0_clean.csv"

    # output (UNICO FILE)
    output_json: str = "phase3_ensembles.json"

    # colonne speciali
    target_col: str = "class"
    patient_id_col: str = "patient_id"  # se non esiste, ignorata

    # split
    test_size: float = 0.2
    random_state: int = 42

    # Bagging params
    bag_n_estimators: int = 300
    bag_max_samples: float = 0.8
    bag_max_features: float = 1.0
    bag_base_max_depth: Optional[int] = None  # es. 5 per regolarizzare

    # Boosting params (AdaBoost)
    ada_n_estimators: int = 300
    ada_learning_rate: float = 0.5
    ada_base_max_depth: int = 1  # stump (weak learner)


# ---------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Phase3Footprint:
    cfg: Phase3Config
    df: Optional[pd.DataFrame] = None
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    results: dict[str, Any] = field(default_factory=dict)
    output_path: Optional[Path] = None

    _meta: FootprintMetadata = field(default_factory=FootprintMetadata)

    def get_metadata(self) -> FootprintMetadata:
        return self._meta


# ---------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------
@step("LoadClean")
def load_clean(fp: Phase3Footprint) -> Phase3Footprint:
    cfg = fp.cfg
    in_path = cfg.data_dir / cfg.input_clean_csv
    if not in_path.exists():
        raise FileNotFoundError(f"File clean non trovato: {in_path}")

    fp.df = pd.read_csv(in_path)
    fp.get_metadata().add_tag("input_file", str(in_path))
    fp.get_metadata().add_extra("rows", int(len(fp.df)))
    fp.get_metadata().add_extra("cols", int(fp.df.shape[1]))
    return fp


@step("BuildXY")
def build_xy(fp: Phase3Footprint) -> Phase3Footprint:
    cfg = fp.cfg
    if fp.df is None:
        raise RuntimeError("df mancante. Eseguire LoadClean prima.")

    df = fp.df
    if cfg.target_col not in df.columns:
        raise KeyError(f"Colonna target '{cfg.target_col}' non trovata nel dataset clean.")

    y = df[cfg.target_col].astype(str).to_numpy()

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
    fp.get_metadata().add_extra("classes", sorted(pd.unique(y).tolist()))
    return fp


@step("SplitTrainTest")
def split_train_test(fp: Phase3Footprint) -> Phase3Footprint:
    cfg = fp.cfg
    if fp.X is None or fp.y is None:
        raise RuntimeError("X/y mancanti. Eseguire BuildXY prima.")

    X_train, X_test, y_train, y_test = train_test_split(
        fp.X,
        fp.y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=fp.y,
    )

    fp.X_train, fp.X_test, fp.y_train, fp.y_test = X_train, X_test, y_train, y_test
    fp.get_metadata().add_extra("test_size", cfg.test_size)
    fp.get_metadata().add_extra("random_state", cfg.random_state)
    return fp


def _evaluate(model_name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    labels = sorted(pd.unique(y_test).tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    rep = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy_test": acc,
        "f1_macro_test": f1m,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
    }


@step("TrainBagging")
def train_bagging(fp: Phase3Footprint) -> Phase3Footprint:
    cfg = fp.cfg
    if fp.X_train is None or fp.y_train is None or fp.X_test is None or fp.y_test is None:
        raise RuntimeError("Split mancante. Eseguire SplitTrainTest prima.")

    base_dt = DecisionTreeClassifier(
        random_state=cfg.random_state,
        max_depth=cfg.bag_base_max_depth,
    )

    bag = BaggingClassifier(
        estimator=base_dt,
        n_estimators=cfg.bag_n_estimators,
        max_samples=cfg.bag_max_samples,
        max_features=cfg.bag_max_features,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    bag.fit(fp.X_train, fp.y_train)

    res = _evaluate("Bagging(DecisionTree)", bag, fp.X_test, fp.y_test)
    res["params"] = {
        "n_estimators": cfg.bag_n_estimators,
        "max_samples": cfg.bag_max_samples,
        "max_features": cfg.bag_max_features,
        "base_max_depth": cfg.bag_base_max_depth,
        "random_state": cfg.random_state,
    }

    fp.results["bagging"] = res
    return fp


@step("TrainBoosting")
def train_boosting(fp: Phase3Footprint) -> Phase3Footprint:
    cfg = fp.cfg
    if fp.X_train is None or fp.y_train is None or fp.X_test is None or fp.y_test is None:
        raise RuntimeError("Split mancante. Eseguire SplitTrainTest prima.")

    base_dt = DecisionTreeClassifier(
        random_state=cfg.random_state,
        max_depth=cfg.ada_base_max_depth,
    )

    ada = AdaBoostClassifier(
        estimator=base_dt,
        n_estimators=cfg.ada_n_estimators,
        learning_rate=cfg.ada_learning_rate,
        random_state=cfg.random_state,
    )

    ada.fit(fp.X_train, fp.y_train)

    res = _evaluate("AdaBoost(DecisionTree)", ada, fp.X_test, fp.y_test)
    res["params"] = {
        "n_estimators": cfg.ada_n_estimators,
        "learning_rate": cfg.ada_learning_rate,
        "base_max_depth": cfg.ada_base_max_depth,
        "random_state": cfg.random_state,
    }

    fp.results["boosting"] = res
    return fp


@step("ExportJSON")
def export_json(fp: Phase3Footprint) -> Phase3Footprint:
    cfg = fp.cfg
    out_path = cfg.reports_dir / cfg.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": 3,
        "input_clean": str(cfg.data_dir / cfg.input_clean_csv),
        "split": {"test_size": cfg.test_size, "random_state": cfg.random_state},
        "results": fp.results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fp.output_path = out_path
    fp.get_metadata().add_tag("phase3_output", str(out_path))
    return fp


# ---------------------------------------------------------------------
# Trail builder + runner
# ---------------------------------------------------------------------
def build_phase3_trail(cfg: Phase3Config, *, trace: bool = False) -> Trail[Phase3Footprint]:
    return (
        Trail[Phase3Footprint](name="phase3_ensembles")
        .then(
            load_clean,
            build_xy,
            split_train_test,
            train_bagging,
            train_boosting,
            export_json,
        )
        .with_tag("phase", "3")
        .trace(trace)
    )


def run_phase3(cfg: Optional[Phase3Config] = None, *, trace: bool = False) -> Path:
    cfg = cfg or Phase3Config()
    fp = Phase3Footprint(cfg=cfg)
    trail = build_phase3_trail(cfg, trace=trace)
    fp = trail.run(fp)
    assert fp.output_path is not None
    return fp.output_path