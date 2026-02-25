from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging
from logging.handlers import RotatingFileHandler
from shutil import rmtree


# ==========================================================
# Logging
# ==========================================================
LOG_LEVEL = logging.INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = False       # True = log su file + console; False = solo console

_THIS_DIR = Path(__file__).resolve().parent          # .../src/case2
SRC_DIR = _THIS_DIR.parent                           # .../src
PROJECT_ROOT = SRC_DIR.parent                         # .../project

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "project.log"


def get_logger(name: str = "project") -> logging.Logger:
    """
    Returns a configured logger (console + optional rotating file).
    Safe to call multiple times: handlers are added only once.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    if getattr(logger, "_configured", False):
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if LOG_TO_FILE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    return logger


log = get_logger("cfg")


# ==========================================================
# Base directories (independent from execution location)
# ==========================================================
BASE_DIR = _THIS_DIR                   # .../src/case2
DATASETS_DIR = SRC_DIR / "datasets"    # .../src/datasets

# Output directories (outside datasets)
OUT_DIR = PROJECT_ROOT / "out"
GENERATED_DIR = OUT_DIR / "generated"
FINAL_DIR = OUT_DIR / "final"


# ==========================================================
# Core node files
# ==========================================================
DISEASES_PATH: Path = DATASETS_DIR / "diseases.tsv"
PATIENTS_PATH: Path = DATASETS_DIR / "patients.tsv"


# ==========================================================
# Edge files (non chromosome-specific)
# ==========================================================
EDGE_PATIENT_DISEASE_PATH: Path = DATASETS_DIR / "patients_diseases.tsv"


# ==========================================================
# Chromosome-based files (generalized)
# ==========================================================
# You can list multiple chromosomes here (e.g., ["22","17","X"]).
CHROMOSOMES: List[str] = ["22"]

MUTATION_FILES: List[Path] = [DATASETS_DIR / f"chr_{chr_id}.tsv" for chr_id in CHROMOSOMES]
EDGE_PATIENT_MUTATION_FILES: List[Path] = [
    DATASETS_DIR / f"chr_{chr_id}_patients_mutations_edges.tsv" for chr_id in CHROMOSOMES
]


# ==========================================================
# Gene expression (Xena STAR-FPKM)
# ==========================================================
EXPRESSION_MATRIX_FILES: List[Path] = [
    DATASETS_DIR / "TCGA-OV.star_fpkm.tsv",
]


# ==========================================================
# Schema configuration (column names)
# ==========================================================
# Mutations nodes file schema (chr_*.tsv)
MUTATION_ID_COL = "unique_id"
MUTATION_GENE_COL = "Gene.refGene"
MUTATION_CHR_COL = "Chr"

# If MUTATION_GENE_COL contains multiple genes separated by ';'
MUTATION_GENE_SPLIT_CHAR = ";"
MUTATION_GENE_TAKE_FIRST = True

# Expression matrix schema
EXPR_GENE_COL: Optional[str] = None  # None = first column


# ==========================================================
# Optional column overrides (set to None to autodetect)
# ==========================================================
PATIENT_ID_COL: Optional[str] = None
DISEASE_ID_COL: Optional[str] = "disease_id"

EDGE_PD_PATIENT_COL: Optional[str] = None
EDGE_PD_DISEASE_COL: Optional[str] = None

EDGE_PM_PATIENT_COL: Optional[str] = None
EDGE_PM_MUTATION_COL: Optional[str] = "mutation_id"

# Gene ID mapping (Ensembl -> gene symbol)
ENSEMBL_TO_SYMBOL_PATH = DATASETS_DIR / "ensembl_to_symbol.tsv"
ENSEMBL_COL = "ensembl_id"
SYMBOL_COL = "gene_symbol"


# ==========================================================
# Validation utilities
# ==========================================================
def validate_paths() -> None:
    """
    Ensures all configured input paths exist.
    Logs every path checked.
    Raises FileNotFoundError if any are missing.
    """
    paths_to_check: List[Path] = [
        DISEASES_PATH,
        PATIENTS_PATH,
        EDGE_PATIENT_DISEASE_PATH,
        *MUTATION_FILES,
        *EDGE_PATIENT_MUTATION_FILES,
        *EXPRESSION_MATRIX_FILES,
    ]

    log.info("Validating configured input paths (%d total)...", len(paths_to_check))
    missing: List[Path] = []

    for p in paths_to_check:
        if p.exists():
            log.info("OK: %s", p)
        else:
            log.error("MISSING: %s", p)
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "Some configured input paths do not exist:\n" + "\n".join(str(m) for m in missing)
        )


def discover_chromosomes() -> List[str]:
    """
    Automatically detect available chromosome mutation files inside DATASETS_DIR.
    Detects files like: chr_1.tsv, chr_22.tsv, chr_X.tsv
    Returns sorted chromosome ids (strings).
    """
    chromosomes: List[str] = []
    for file in DATASETS_DIR.glob("chr_*.tsv"):
        chr_id = file.stem.replace("chr_", "")
        chromosomes.append(chr_id)
    return sorted(chromosomes)


# =============================
# Cleaning utilities
# =============================

CLEAN_DIRECTORIES: bool = True
CLEAN_GENERATED: bool = True
CLEAN_FINAL: bool = True
CLEAN_LOGS: bool = True

def clean_output(*, generated: bool = CLEAN_GENERATED, final: bool = CLEAN_FINAL, logs: bool = CLEAN_LOGS) -> None:
    """
    Pulisce le cartelle di output prima di rigenerare i file.
    - generated: rimuove out/generated
    - final: rimuove out/final
    - logs: opzionale, rimuove logs/
    """
    log = get_logger("cfg")

    targets = []
    if generated:
        targets.append(GENERATED_DIR)
    if final:
        targets.append(FINAL_DIR)
    if logs:
        targets.append(LOG_DIR)

    for d in targets:
        d = Path(d)
        if d.exists():
            log.info("Cleaning output directory: %s", d)
            rmtree(d)
        else:
            log.info("Output directory not present (skip): %s", d)