# src/phase2_gridsearch.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from traccia import Trail, step, FootprintMetadata


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Phase2Config:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    input_clean_csv: str = "phase0_clean.csv"
    output_json: str = "phase2_gridsearch.json"

    target_col: str = "class"
    patient_id_col: str = "patient_id"

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    scoring: str = "f1_macro"


# ---------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Phase2Footprint:
    cfg: Phase2Config
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
def load_clean(fp: Phase2Footprint) -> Phase2Footprint:
    cfg = fp.cfg
    in_path = cfg.data_dir / cfg.input_clean_csv
    if not in_path.exists():
        raise FileNotFoundError(f"File clean non trovato: {in_path}")

    fp.df = pd.read_csv(in_path)
    return fp


@step("BuildXY")
def build_xy(fp: Phase2Footprint) -> Phase2Footprint:
    cfg = fp.cfg
    df = fp.df

    if df is None:
        raise RuntimeError("df mancante.")

    y = df[cfg.target_col].astype(str).to_numpy()

    drop_cols = [cfg.target_col]
    if cfg.patient_id_col in df.columns:
        drop_cols.append(cfg.patient_id_col)

    X_df = df.drop(columns=drop_cols)
    fp.X = X_df.to_numpy(dtype=float)
    fp.y = y
    return fp


@step("SplitTrainTest")
def split_train_test(fp: Phase2Footprint) -> Phase2Footprint:
    cfg = fp.cfg

    X_train, X_test, y_train, y_test = train_test_split(
        fp.X,
        fp.y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=fp.y,
    )

    fp.X_train, fp.X_test = X_train, X_test
    fp.y_train, fp.y_test = y_train, y_test
    return fp


def _evaluate(name: str, model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "model_name": name,
        "best_params": model.best_params_,
        "cv_best_score": float(model.best_score_),
        "accuracy_test": float(accuracy_score(y_test, y_pred)),
        "f1_macro_test": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }


@step("GridSearchAllModels")
def gridsearch_all(fp: Phase2Footprint) -> Phase2Footprint:
    cfg = fp.cfg

    models = {}

    # 1️⃣ Decision Tree
    dt = DecisionTreeClassifier(random_state=cfg.random_state)
    dt_params = {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
    }
    models["DecisionTree"] = (dt, dt_params)

    # 2️⃣ Random Forest
    rf = RandomForestClassifier(random_state=cfg.random_state)
    rf_params = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
    }
    models["RandomForest"] = (rf, rf_params)

    # 3️⃣ SVC (con scaling)
    svc_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC())
    ])
    svc_params = {
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["linear", "rbf"],
    }
    models["SVC"] = (svc_pipe, svc_params)

    # 4️⃣ KNN (con scaling)
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])
    knn_params = {
        "knn__n_neighbors": [3, 5, 7],
        "knn__weights": ["uniform", "distance"],
    }
    models["KNN"] = (knn_pipe, knn_params)

    for name, (model, params) in models.items():
        gs = GridSearchCV(
            model,
            params,
            cv=cfg.cv_folds,
            scoring=cfg.scoring,
            n_jobs=-1,
        )
        gs.fit(fp.X_train, fp.y_train) # type: ignore

        fp.results[name] = _evaluate(
            name,
            gs,
            fp.X_test,
            fp.y_test,
        )

    return fp


@step("ExportJSON")
def export_json(fp: Phase2Footprint) -> Phase2Footprint:
    cfg = fp.cfg
    out_path = cfg.reports_dir / cfg.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": 2,
        "split": {
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "cv_folds": cfg.cv_folds,
            "scoring": cfg.scoring,
        },
        "results": fp.results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fp.output_path = out_path
    return fp


# ---------------------------------------------------------------------
# Trail builder + runner
# ---------------------------------------------------------------------
def build_phase2_trail(cfg: Phase2Config, *, trace: bool = False) -> Trail[Phase2Footprint]:
    return (
        Trail[Phase2Footprint](name="phase2_gridsearch")
        .then(
            load_clean,
            build_xy,
            split_train_test,
            gridsearch_all,
            export_json,
        )
        .with_tag("phase", "2")
        .trace(trace)
    )


def run_phase2(cfg: Optional[Phase2Config] = None, *, trace: bool = False) -> Path:
    cfg = cfg or Phase2Config()
    fp = Phase2Footprint(cfg=cfg)
    trail = build_phase2_trail(cfg, trace=trace)
    fp = trail.run(fp)
    assert fp.output_path is not None
    return fp.output_path