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


@dataclass(slots=True)
class Phase3Config:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    input_clean_csv: str = "phase0_clean.csv"
    output_json: str = "phase3_ensembles.json"
    target_col: str = "class"
    patient_id_col: str = "patient_id"
    test_size: float = 0.2
    random_state: int = 42
    bag_n_estimators: int = 300
    bag_max_samples: float = 0.8
    bag_max_features: float = 1.0
    bag_base_max_depth: Optional[int] = None
    ada_n_estimators: int = 300
    ada_learning_rate: float = 0.5
    ada_base_max_depth: int = 1


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


@step("LoadClean")
def load_clean(fp: Phase3Footprint) -> Phase3Footprint:
    in_path = fp.cfg.data_dir / fp.cfg.input_clean_csv
    if not in_path.exists():
        raise FileNotFoundError(f"File clean non trovato: {in_path}")
    fp.df = pd.read_csv(in_path)
    return fp


@step("BuildXY")
def build_xy(fp: Phase3Footprint) -> Phase3Footprint:
    if fp.df is None:
        raise RuntimeError("df mancante.")
    df = fp.df
    if fp.cfg.target_col not in df.columns:
        raise KeyError(f"Colonna target '{fp.cfg.target_col}' non trovata.")
    fp.y = df[fp.cfg.target_col].astype(str).to_numpy()
    drop_cols = [fp.cfg.target_col]
    if fp.cfg.patient_id_col in df.columns:
        drop_cols.append(fp.cfg.patient_id_col)
    X_df = df.drop(columns=drop_cols)
    non_numeric = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
    if non_numeric:
        raise TypeError(f"Feature non numeriche: {non_numeric}")
    fp.X = X_df.to_numpy(dtype=float)
    return fp


@step("SplitTrainTest")
def split_train_test(fp: Phase3Footprint) -> Phase3Footprint:
    if fp.X is None or fp.y is None:
        raise RuntimeError("X/y mancanti.")
    fp.X_train, fp.X_test, fp.y_train, fp.y_test = train_test_split(
        fp.X,
        fp.y,
        test_size=fp.cfg.test_size,
        random_state=fp.cfg.random_state,
        stratify=fp.y,
    )
    return fp


def _evaluate(model_name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    labels = sorted(pd.unique(y_test).tolist())
    return {
        "model_name": model_name,
        "accuracy_test": float(accuracy_score(y_test, y_pred)),
        "f1_macro_test": float(f1_score(y_test, y_pred, average="macro")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


@step("TrainBagging")
def train_bagging(fp: Phase3Footprint) -> Phase3Footprint:
    cfg = fp.cfg
    if fp.X_train is None or fp.y_train is None or fp.X_test is None or fp.y_test is None:
        raise RuntimeError("Split mancante.")

    base_dt = DecisionTreeClassifier(random_state=cfg.random_state, max_depth=cfg.bag_base_max_depth)
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
        raise RuntimeError("Split mancante.")

    base_dt = DecisionTreeClassifier(random_state=cfg.random_state, max_depth=cfg.ada_base_max_depth)
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
    out_path = fp.cfg.reports_dir / fp.cfg.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": 3,
        "input_clean": str(fp.cfg.data_dir / fp.cfg.input_clean_csv),
        "split": {"test_size": fp.cfg.test_size, "random_state": fp.cfg.random_state},
        "results": fp.results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fp.output_path = out_path
    return fp


def build_phase3_trail(cfg: Phase3Config, *, trace: bool = False) -> Trail[Phase3Footprint]:
    return (
        Trail[Phase3Footprint](name="phase3_ensembles")
        .then(load_clean, build_xy, split_train_test, train_bagging, train_boosting, export_json)
        .with_tag("phase", "3")
        .trace(trace)
    )


def run_phase3(cfg: Optional[Phase3Config] = None, *, trace: bool = False) -> Path:
    cfg = cfg or Phase3Config()
    fp = Phase3Footprint(cfg=cfg)
    fp = build_phase3_trail(cfg, trace=trace).run(fp)
    assert fp.output_path is not None
    return fp.output_path
