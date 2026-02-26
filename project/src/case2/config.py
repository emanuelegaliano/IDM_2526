"""
case2/config.py

Configurazione centrale per il package case2.
Gestisce:
- Path dataset
- Parametri performance
- Strategia gene_id
- Logging (console e file separati)
- Validazione finale
"""

from dataclasses import dataclass, field
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


@dataclass
class Case2Config:
    # -----------------------
    # Path principali
    # -----------------------
    datasets_root: Path = Path("datasets")
    output_dir: Path = Path("output")
    tmp_dir: Path = Path("output/tmp")

    # -----------------------
    # Directory pattern
    # -----------------------
    mutations_dir: str = "mutations"
    edges_dir: str = "edges"
    tumors_dir: str = "tumors"

    # -----------------------
    # Performance
    # -----------------------
    chunk_size_rows: int = 10_000
    workers: int = 1

    # -----------------------
    # Gene ID strategy
    # -----------------------
    gene_id_mode: str = "symbol"

    # -----------------------
    # Validation
    # -----------------------
    validate: bool = False

    # -----------------------
    # Logging
    # -----------------------
    verbose: bool = True          # console logging
    log_to_file: bool = False     # file logging
    log_level: str = "INFO"
    log_file_path: Optional[Path] = None
    log_file_max_bytes: int = 5_000_000
    log_file_backup_count: int = 3

    # Runtime
    logger: logging.Logger = field(init=False, repr=False)

    # ============================================================
    # INIT
    # ============================================================

    def __post_init__(self):
        self.datasets_root = Path(self.datasets_root)
        self.output_dir = Path(self.output_dir)
        self.tmp_dir = Path(self.tmp_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        if self.log_to_file and self.log_file_path is None:
            self.log_file_path = self.output_dir / "case2.log"

        self.logger = self._setup_logger()

    # ============================================================
    # LOGGER SETUP
    # ============================================================

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("case2")
        logger.setLevel(self._get_log_level())
        logger.propagate = False

        # Rimuove eventuali handler precedenti
        if logger.handlers:
            logger.handlers.clear()

        # Se entrambi disabilitati → logger completamente silenzioso
        if not self.verbose and not self.log_to_file:
            logger.disabled = True
            return logger

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self._get_log_level())
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if self.log_to_file:
            file_handler = RotatingFileHandler(
                filename=self.log_file_path,
                maxBytes=self.log_file_max_bytes,
                backupCount=self.log_file_backup_count,
            )
            file_handler.setLevel(self._get_log_level())
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _get_log_level(self):
        return getattr(logging, self.log_level.upper(), logging.INFO)

    # ============================================================
    # PATH HELPERS
    # ============================================================

    @property
    def mutations_path(self) -> Path:
        return self.datasets_root / self.mutations_dir

    @property
    def tumors_path(self) -> Path:
        return self.datasets_root / self.tumors_dir

    @property
    def genes_output_path(self) -> Path:
        return self.output_dir / "genes.tsv"

    @property
    def genes_mutations_edges_path(self) -> Path:
        return self.output_dir / "genes_mutations_edges.tsv"

    @property
    def patients_genes_expression_edges_path(self) -> Path:
        return self.output_dir / "patients_genes_expression_edges.tsv"