# src/phase0_clean.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from traccia import Trail, step, FootprintMetadata


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Phase0Config:
    data_dir: Path = Path("data")

    # input (UNICO)
    input_xlsx: str = "dataset_raw.xlsx"

    # output (UNICO FILE)
    output_clean_csv: str = "phase0_clean.csv"

    # regole assignment
    min_age_equivalent_months: int = 12

    # gestione missing: "drop_rows" oppure "impute_median"
    missing_strategy: Literal["drop_rows", "impute_median"] = "drop_rows"

    # numeric coercion aggressiva (virgole -> punti, trim, ecc.)
    aggressive_numeric_coerce: bool = True

    # se True, rinomina "Pazienti" in "patient_id" (e la lascia come stringa)
    keep_patient_id: bool = True


# ---------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------
@dataclass(slots=True)
class Phase0Footprint:
    cfg: Phase0Config
    raw_df: Optional[pd.DataFrame] = None
    clean_df: Optional[pd.DataFrame] = None
    output_path: Optional[Path] = None

    _meta: FootprintMetadata = field(default_factory=FootprintMetadata)

    def get_metadata(self) -> FootprintMetadata:
        # CONTRATTO TRACCIA: deve ritornare sempre la stessa istanza
        return self._meta


# ---------------------------------------------------------------------
# Utilities (tutte interne al file)
# ---------------------------------------------------------------------
def _normalize_colname(c: str) -> str:
    c = str(c).strip()
    c = " ".join(c.split())
    if c.lower().startswith("unnamed:"):
        return ""
    return c


def _coerce_numeric_series(s: pd.Series, aggressive: bool) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s
    if aggressive:
        s2 = s.astype(str).str.strip()
        s2 = s2.str.replace(",", ".", regex=False)
        s2 = s2.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        return pd.to_numeric(s2, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def _find_col(df: pd.DataFrame, name: str) -> Optional[str]:
    target = name.casefold()
    for c in df.columns:
        if str(c).casefold() == target:
            return c
    return None


def _rename_duplicate_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rinomina colonne duplicate "semantiche" tipo:
      B10, B10.1 -> B10_y1, B10_y2
      D4, D4.1, D4.2 -> D4_y1, D4_y2, D4_y3

    Lascia inalterate colonne non in formato lettera+numero (es. 'Sesso', 'Età equivalente', 'class').
    """
    cols = df.columns.tolist()
    new_cols: list[str] = []

    # conta occorrenze per base (B10, D4, ...)
    # base = parte prima di eventuale ".<n>"
    def base_name(c: str) -> str:
        c = str(c)
        if "." in c:
            left, right = c.rsplit(".", 1)
            if right.isdigit():
                return left
        return c

    bases = [base_name(c) for c in cols]

    # occorrenze progressiva per base
    seen: dict[str, int] = {}
    total: dict[str, int] = {}
    for b in bases:
        total[b] = total.get(b, 0) + 1

    for c, b in zip(cols, bases):
        b_str = str(b)
        c_str = str(c)

        # rinomina solo colonne feature in stile lettera+numero (B10, D4, ecc.)
        is_code_like = (
            len(b_str) >= 2
            and b_str[0].isalpha()
            and any(ch.isdigit() for ch in b_str)
            and b_str.replace("_", "").replace("-", "").isalnum()
        )

        if total[b_str] > 1 and is_code_like:
            seen[b_str] = seen.get(b_str, 0) + 1
            new_cols.append(f"{b_str}_y{seen[b_str]}")
        else:
            new_cols.append(c_str)

    # garantisci univocità assoluta in caso di edge cases
    final_cols: list[str] = []
    used: dict[str, int] = {}
    for c in new_cols:
        if c not in used:
            used[c] = 1
            final_cols.append(c)
        else:
            used[c] += 1
            final_cols.append(f"{c}__{used[c]}")

    df = df.copy()
    df.columns = final_cols
    return df


# ---------------------------------------------------------------------
# Steps (tutti qui dentro)
# ---------------------------------------------------------------------
@step("LoadXLSX")
def load_xlsx(fp: Phase0Footprint) -> Phase0Footprint:
    cfg = fp.cfg
    xlsx_path = cfg.data_dir / cfg.input_xlsx
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX non trovato: {xlsx_path}")

    # Nel tuo file la riga 0 contiene intestazioni raggruppate (Scala A/B/D...),
    # la riga 1 contiene gli header reali.
    header_row = 1

    sheet_map = {
        "ASD": "ASD",
        "GDD": "GDD",
        "Controlli": "Controls",  # normalizzo per coerenza
    }

    frames: list[pd.DataFrame] = []
    for sheet_name, class_name in sheet_map.items():
        df_sh = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row)

        # normalizza nomi colonne e rimuovi unnamed vuote
        df_sh.columns = [_normalize_colname(c) for c in df_sh.columns.tolist()]
        df_sh = df_sh.loc[:, [c for c in df_sh.columns if c != ""]]

        df_sh["class"] = class_name
        frames.append(df_sh)

    df = pd.concat(frames, ignore_index=True)

    fp.raw_df = df
    fp.get_metadata().add_tag("input_format", "xlsx")
    fp.get_metadata().add_tag("input_file", str(xlsx_path))
    fp.get_metadata().add_extra("sheets", list(sheet_map.keys()))
    return fp


@step("NormalizeAndRename")
def normalize_and_rename(fp: Phase0Footprint) -> Phase0Footprint:
    if fp.raw_df is None:
        raise RuntimeError("raw_df mancante. Eseguire LoadXLSX prima.")

    df = fp.raw_df.copy()

    # rinomina duplicati tipo B10, B10.1 -> B10_y1, B10_y2
    df = _rename_duplicate_feature_columns(df)

    # opzionale: Pazienti -> patient_id (lasciata stringa)
    paz_col = _find_col(df, "Pazienti")
    if paz_col and fp.cfg.keep_patient_id:
        df = df.rename(columns={paz_col: "patient_id"})

    fp.raw_df = df
    return fp


@step("DropAssignmentColumns")
def drop_assignment_columns(fp: Phase0Footprint) -> Phase0Footprint:
    if fp.raw_df is None:
        raise RuntimeError("raw_df mancante.")

    df = fp.raw_df.copy()

    # colonne da rimuovere (come da consegna)
    to_drop = [
        "Età cronologica (mesi)",
        "Scala B",
        "Scala D",
        "TOT.",
        "Score di rischio",
    ]

    present: list[str] = []
    for name in to_drop:
        col = _find_col(df, name)
        if col is not None:
            present.append(col)

    df = df.drop(columns=present, errors="ignore")
    fp.raw_df = df
    fp.get_metadata().add_extra("dropped_columns", present)
    return fp


@step("FilterAgeEquivalent")
def filter_age_equivalent(fp: Phase0Footprint) -> Phase0Footprint:
    cfg = fp.cfg
    if fp.raw_df is None:
        raise RuntimeError("raw_df mancante.")

    df = fp.raw_df.copy()

    age_col = _find_col(df, "Età equivalente")
    if age_col is None:
        raise KeyError("Colonna 'Età equivalente' non trovata nel dataset.")

    df[age_col] = _coerce_numeric_series(df[age_col], cfg.aggressive_numeric_coerce)

    before = len(df)
    df = df[df[age_col] >= cfg.min_age_equivalent_months].copy()
    after = len(df)

    fp.raw_df = df
    fp.get_metadata().add_extra("filtered_age_equivalent_removed_rows", before - after)
    return fp


@step("EncodeSex")
def encode_sex(fp: Phase0Footprint) -> Phase0Footprint:
    if fp.raw_df is None:
        raise RuntimeError("raw_df mancante.")

    df = fp.raw_df.copy()

    sex_col = _find_col(df, "Sesso")
    if sex_col is None:
        fp.raw_df = df
        fp.get_metadata().add_extra("sex_encoding", "skipped_missing_column")
        return fp

    mapping = {"f": 0, "m": 1}
    df[sex_col] = (
        df[sex_col]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map(mapping)
    )

    fp.raw_df = df
    fp.get_metadata().add_extra("sex_encoding", "F->0, M->1")
    return fp


@step("CoerceNumericAndMissing")
def coerce_numeric_and_missing(fp: Phase0Footprint) -> Phase0Footprint:
    cfg = fp.cfg
    if fp.raw_df is None:
        raise RuntimeError("raw_df mancante.")

    df = fp.raw_df.copy()

    # non coerco a numerico le colonne "patient_id" e "class"
    skip = {"class"}
    if cfg.keep_patient_id and "patient_id" in df.columns:
        skip.add("patient_id")

    for c in df.columns:
        if c in skip:
            continue
        df[c] = _coerce_numeric_series(df[c], cfg.aggressive_numeric_coerce)

    if cfg.missing_strategy == "drop_rows":
        before = len(df)
        df = df.dropna(axis=0).copy()
        fp.get_metadata().add_extra("missing_drop_rows_removed", before - len(df))
    else:
        num_cols = [c for c in df.columns if c not in skip]
        medians = df[num_cols].median(numeric_only=True)
        df[num_cols] = df[num_cols].fillna(medians)
        fp.get_metadata().add_extra("missing_impute", "median")

    fp.clean_df = df
    return fp


@step("ExportCleanCSV")
def export_clean_csv(fp: Phase0Footprint) -> Phase0Footprint:
    cfg = fp.cfg
    if fp.clean_df is None:
        raise RuntimeError("clean_df mancante. Eseguire gli step di pulizia prima.")

    out_path = cfg.data_dir / cfg.output_clean_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fp.clean_df.to_csv(out_path, index=False, encoding="utf-8")
    fp.output_path = out_path

    fp.get_metadata().add_tag("phase0_output", str(out_path))
    fp.get_metadata().add_extra("rows", int(len(fp.clean_df)))
    fp.get_metadata().add_extra("cols", int(fp.clean_df.shape[1]))
    return fp


# ---------------------------------------------------------------------
# Trail builder + runner (per notebook)
# ---------------------------------------------------------------------
def build_phase0_trail(cfg: Phase0Config, *, trace: bool = False) -> Trail[Phase0Footprint]:
    return (
        Trail[Phase0Footprint](name="phase0_clean")
        .then(
            load_xlsx,
            normalize_and_rename,
            drop_assignment_columns,
            filter_age_equivalent,
            encode_sex,
            coerce_numeric_and_missing,
            export_clean_csv,
        )
        .with_tag("phase", "0")
        .trace(trace)
    )


def run_phase0(cfg: Optional[Phase0Config] = None, *, trace: bool = False) -> Path:
    """
    Helper comodo per notebook.
    Ritorna il Path dell'UNICO file prodotto: data/phase0_clean.csv
    """
    cfg = cfg or Phase0Config()
    fp = Phase0Footprint(cfg=cfg)
    trail = build_phase0_trail(cfg, trace=trace)
    fp = trail.run(fp)
    assert fp.output_path is not None
    return fp.output_path