# src/case2/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging


@dataclass
class Case2Config:
    """
    Config Case2.

    Struttura dataset attesa (sotto datasets_root):
      - diseases/diseases.tsv
      - patients/patients.tsv
      - edges/patients_diseases/patients_diseases.tsv
      - mutations/chr_*.tsv
      - tumors/*.tsv (es. TCGA-OV.star_fpkm.tsv)
      - mappings/symbol_to_ensg.tsv (BioMart)  <-- per modalità offline

    Output:
      - output_dir/genes.tsv
      - output_dir/genes_mutations_edges.tsv
      - output_dir/patients_genes_expression_edges.tsv
    """

    # input
    datasets_root: Path

    # output
    output_dir: Path
    tmp_dir: Path

    # runtime
    workers: int = 4

    # logging
    verbose: bool = True          # se False e log_to_file=False, logging completamente disabilitato
    log_to_file: bool = False     # se True, scrive anche su file
    log_file_path: Optional[Path] = None

    # gene-id resolution
    # per questa versione supportiamo SOLO offline mapping
    gene_id_mode: str = "offline_mapping"
    mapping_tsv: Optional[Path] = None  # default: datasets_root/mappings/symbol_to_ensg.tsv

    # validation
    validate: bool = True

    def resolve_paths(self) -> None:
        """Normalizza e completa i path (default sensati)."""
        self.datasets_root = Path(self.datasets_root).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        self.tmp_dir = Path(self.tmp_dir).resolve()

        if self.mapping_tsv is None:
            self.mapping_tsv = (self.datasets_root / "mappings" / "symbol_to_ensg.tsv").resolve()
        else:
            self.mapping_tsv = Path(self.mapping_tsv).resolve()

        if self.log_file_path is None:
            # mettiamo i log nella tmp_dir così non “sporchiamo” output finale
            self.log_file_path = (self.tmp_dir / "case2.log").resolve()
        else:
            self.log_file_path = Path(self.log_file_path).resolve()


def setup_logger(
    name: str,
    *,
    verbose: bool,
    log_to_file: bool,
    log_file_path: Path,
) -> logging.Logger:
    """
    Crea un logger:

    - verbose=True  -> log su stdout
    - log_to_file=True -> log anche su file (log_file_path)
    - verbose=False e log_to_file=False -> logger disabilitato (nessun output)
    """
    logger = logging.getLogger(name)

    # reset handlers per evitare duplicati (es. in test o run multiple)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # IMPORTANT: se disabilitato, non deve stampare nulla
    if not verbose and not log_to_file:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.disabled = True
        return logger

    logger.disabled = False
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if verbose:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_to_file:
        log_file_path = Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger