# src/case2/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
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
      - tumors/*.tsv (più tumori supportati)
      - mappings/symbol_to_ensg.tsv (BioMart)  <-- per modalità offline

    Output:
      - output_dir/genes.tsv
      - output_dir/genes_mutations_edges.tsv
      - output_dir/patients_genes_expression_edges.tsv   (multi-tumor con colonna tumor_id)
    """

    # input
    datasets_root: Path

    # output
    output_dir: Path
    tmp_dir: Path

    # runtime
    workers: int = 4

    # logging
    verbose: bool = True
    log_to_file: bool = False
    log_file_path: Optional[Path] = None

    # gene-id resolution
    gene_id_mode: str = "offline_mapping"
    mapping_tsv: Optional[Path] = None  # default: datasets_root/mappings/symbol_to_ensg.tsv

    # validation
    validate: bool = True

    # ----------------------------
    # MULTI-TUMOR (NUOVO)
    # ----------------------------
    # Se tumor_files=None => usa tutti i file matching tumors_glob in datasets_root/tumors/
    tumor_files: Optional[List[str]] = None
    tumors_glob: str = "*.tsv"

    # tumor_id derivato dal filename: "TCGA-OV.star_fpkm.tsv" -> "TCGA-OV"
    tumor_id_split_on: str = "."

    # Per evitare collisioni patient_id tra tumori (consigliato True)
    prefix_patient_id_with_tumor: bool = True

    # resolved
    tumor_matrix_paths: List[Path] = None  # type: ignore

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
            self.log_file_path = (self.tmp_dir / "case2.log").resolve()
        else:
            self.log_file_path = Path(self.log_file_path).resolve()

        tumors_dir = (self.datasets_root / "tumors").resolve()
        if not tumors_dir.exists():
            raise FileNotFoundError(f"Directory tumors non trovata: {tumors_dir}")

        paths: List[Path] = []
        if self.tumor_files:
            for name in self.tumor_files:
                p = (tumors_dir / name).resolve()
                if not p.exists():
                    raise FileNotFoundError(f"Tumor file non trovato: {p}")
                paths.append(p)
        else:
            paths = sorted(tumors_dir.glob(self.tumors_glob))

        if not paths:
            raise FileNotFoundError(f"Nessun file tumore trovato in {tumors_dir} con glob={self.tumors_glob}")

        self.tumor_matrix_paths = paths

    def tumor_id_from_path(self, p: Path) -> str:
        name = p.name
        if self.tumor_id_split_on and self.tumor_id_split_on in name:
            return name.split(self.tumor_id_split_on, 1)[0]
        return p.stem


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

    for h in list(logger.handlers):
        logger.removeHandler(h)

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