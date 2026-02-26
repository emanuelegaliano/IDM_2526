"""
case2/validate_outputs.py

Validazione finale per output del caso 2.
Controlla struttura e coerenza dei file generati:

Richiesti (assignment caso 2):
- genes.tsv:              gene_id (+ chromosome consigliato)
- patients_genes_expression_edges.tsv: patient_id, gene_id, expression_value
- genes_mutations_edges.tsv: gene_id, mutation_id

In più: controlli di consistenza e sanity check sui valori.

Uso:
  python -m case2.validate_outputs
oppure (da main) import e chiamata validate_case2_outputs(config)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .config import Case2Config


# ----------------------------
# Helpers
# ----------------------------

def _read_header(path: Path) -> List[str]:
    # Gestisce sia LF che CRLF e ripulisce eventuali \r residui sui campi
    with open(path, "r", newline="") as f:
        line = f.readline()
    if not line:
        return []
    # rimuove CR/LF e poi pulisce ogni colonna
    return [c.strip() for c in line.strip("\r\n").split("\t")]


def _iter_tsv_rows(path: Path) -> Iterable[List[str]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            yield row


def _count_rows(path: Path) -> int:
    # Conta righe (escluso header) senza caricare tutto in RAM
    n = 0
    with open(path, "r", newline="") as f:
        next(f, None)  # header
        for _ in f:
            n += 1
    return n


def _sample_is_float(values: List[str]) -> Tuple[int, int]:
    ok = 0
    bad = 0
    for v in values:
        try:
            x = float(v)
            if math.isfinite(x):
                ok += 1
            else:
                bad += 1
        except Exception:
            bad += 1
    return ok, bad


@dataclass
class ValidationReport:
    ok: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, int]


# ----------------------------
# Main validator
# ----------------------------

def validate_case2_outputs(config: Case2Config) -> ValidationReport:
    logger = config.logger

    genes_path = config.genes_output_path
    gme_path = config.genes_mutations_edges_path
    pge_path = config.patients_genes_expression_edges_path

    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, int] = {}

    logger.info("=== VALIDAZIONE OUTPUT CASE2 ===")

    # 1) Esistenza file
    for p in [genes_path, gme_path, pge_path]:
        if not p.exists():
            errors.append(f"File mancante: {p}")
    if errors:
        for e in errors:
            logger.error(e)
        return ValidationReport(ok=False, errors=errors, warnings=warnings, stats=stats)

    # 2) Header richiesti
    h_genes = _read_header(genes_path)
    h_gme = _read_header(gme_path)
    h_pge = _read_header(pge_path)

    logger.info(f"Header genes.tsv: {h_genes}")
    logger.info(f"Header genes_mutations_edges.tsv: {h_gme}")
    logger.info(f"Header patients_genes_expression_edges.tsv: {h_pge}")

    # Assignment caso 2: gene_id (e chromosome consigliato)
    if "gene_id" not in h_genes:
        errors.append("genes.tsv deve contenere colonna 'gene_id'")
    if "chromosome" not in h_genes:
        warnings.append("genes.tsv non contiene 'chromosome' (consigliato per assignment)")

    # genes_mutations_edges.tsv: gene_id, mutation_id
    if h_gme != ["gene_id", "mutation_id"]:
        errors.append("genes_mutations_edges.tsv header atteso: gene_id, mutation_id")

    # patients_genes_expression_edges.tsv: patient_id, gene_id, expression_value
    if h_pge != ["patient_id", "gene_id", "expression_value"]:
        errors.append("patients_genes_expression_edges.tsv header atteso: patient_id, gene_id, expression_value")

    if errors:
        for e in errors:
            logger.error(e)
        for w in warnings:
            logger.warning(w)
        return ValidationReport(ok=False, errors=errors, warnings=warnings, stats=stats)

    # 3) Conteggi righe
    stats["genes_rows"] = _count_rows(genes_path)
    stats["genes_mutations_edges_rows"] = _count_rows(gme_path)
    stats["patients_genes_expression_edges_rows"] = _count_rows(pge_path)

    logger.info(f"Righe genes.tsv: {stats['genes_rows']}")
    logger.info(f"Righe genes_mutations_edges.tsv: {stats['genes_mutations_edges_rows']}")
    logger.info(f"Righe patients_genes_expression_edges.tsv: {stats['patients_genes_expression_edges_rows']}")

    if stats["genes_rows"] == 0:
        errors.append("genes.tsv è vuoto (0 geni).")
    if stats["genes_mutations_edges_rows"] == 0:
        errors.append("genes_mutations_edges.tsv è vuoto (0 archi).")
    if stats["patients_genes_expression_edges_rows"] == 0:
        errors.append("patients_genes_expression_edges.tsv è vuoto (0 archi).")

    # 4) Coerenza gene_id: edges devono riferirsi a geni presenti nei nodes
    logger.info("Caricamento set geni da genes.tsv (unico set in RAM)...")
    genes_set: Set[str] = set()
    gene_id_col = h_genes.index("gene_id")

    for row in _iter_tsv_rows(genes_path):
        if len(row) <= gene_id_col:
            continue
        genes_set.add(row[gene_id_col])

    stats["genes_unique"] = len(genes_set)
    logger.info(f"Geni unici in genes.tsv: {stats['genes_unique']}")

    if stats["genes_unique"] == 0:
        errors.append("genes.tsv non contiene gene_id validi.")

    # Check genes_mutations_edges gene_id
    bad_gene_refs_gme = 0
    bad_rows_gme = 0
    for row in _iter_tsv_rows(gme_path):
        if len(row) != 2:
            bad_rows_gme += 1
            continue
        gene_id = row[0]
        if gene_id not in genes_set:
            bad_gene_refs_gme += 1

    if bad_rows_gme > 0:
        warnings.append(f"genes_mutations_edges.tsv contiene {bad_rows_gme} righe malformate.")
    if bad_gene_refs_gme > 0:
        errors.append(
            f"genes_mutations_edges.tsv contiene {bad_gene_refs_gme} riferimenti a gene_id non presenti in genes.tsv."
        )

    # Check patients_genes_expression_edges gene_id + expression_value numeric (campione)
    bad_gene_refs_pge = 0
    bad_rows_pge = 0
    sample_values: List[str] = []
    sample_limit = 5000  # campione numerico

    for i, row in enumerate(_iter_tsv_rows(pge_path), start=1):
        if len(row) != 3:
            bad_rows_pge += 1
            continue
        gene_id = row[1]
        if gene_id not in genes_set:
            bad_gene_refs_pge += 1
        if len(sample_values) < sample_limit:
            sample_values.append(row[2])
        if i >= 200_000 and len(sample_values) >= sample_limit:
            # basta per non scorrere tutto se è enorme
            break

    if bad_rows_pge > 0:
        warnings.append(f"patients_genes_expression_edges.tsv contiene {bad_rows_pge} righe malformate.")
    if bad_gene_refs_pge > 0:
        errors.append(
            f"patients_genes_expression_edges.tsv contiene {bad_gene_refs_pge} riferimenti a gene_id non presenti in genes.tsv."
        )

    ok_num, bad_num = _sample_is_float(sample_values)
    stats["expression_value_sample_ok"] = ok_num
    stats["expression_value_sample_bad"] = bad_num
    logger.info(f"Campione expression_value: ok={ok_num} bad={bad_num} (su {len(sample_values)})")

    if bad_num > 0:
        warnings.append(
            f"Ci sono {bad_num} valori non numerici/non finiti nel campione di expression_value (controlla parsing)."
        )

    # 5) Coerenza mutation_id: deve esistere nei file mutations (unique_id)
    #    Costruiamo set mutation_id da mutations/* (può essere grande, ma di solito gestibile;
    #    se diventa enorme, possiamo fare check campionato invece).
    logger.info("Caricamento mutation_id (unique_id) dai file datasets/mutations/chr_*.tsv ...")
    mutation_ids: Set[str] = set()

    mutation_files = sorted(config.mutations_path.glob("chr_*.tsv"))
    if not mutation_files:
        warnings.append("Nessun file mutations chr_*.tsv trovato: skip controllo mutation_id.")
    else:
        for mp in mutation_files:
            with open(mp, "r", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if "unique_id" not in (reader.fieldnames or []):
                    warnings.append(f"{mp.name}: manca colonna unique_id (skip file).")
                    continue
                for row in reader:
                    mid = row.get("unique_id")
                    if mid:
                        mutation_ids.add(mid)

        stats["mutation_ids_unique"] = len(mutation_ids)
        logger.info(f"mutation_id unici caricati: {stats['mutation_ids_unique']}")

        missing_mutation_refs = 0
        checked = 0
        for row in _iter_tsv_rows(gme_path):
            if len(row) != 2:
                continue
            checked += 1
            mid = row[1]
            if mid not in mutation_ids:
                missing_mutation_refs += 1

            # se enormi, basta un limite ragionevole per decidere
            if checked >= 1_000_000:
                break

        if missing_mutation_refs > 0:
            errors.append(
                f"genes_mutations_edges.tsv contiene {missing_mutation_refs} mutation_id non presenti in mutations/*.tsv (unique_id)."
            )

    # Final report
    if errors:
        logger.error("VALIDAZIONE FALLITA.")
        for e in errors:
            logger.error(e)
    else:
        logger.info("VALIDAZIONE OK.")

    for w in warnings:
        logger.warning(w)

    return ValidationReport(ok=len(errors) == 0, errors=errors, warnings=warnings, stats=stats)


# ----------------------------
# CLI entry (opzionale)
# ----------------------------

def main():
    # Usa i default della config (ma nel tuo progetto conviene passare la stessa config del main)
    cfg = Case2Config(
        datasets_root=Path("datasets"),
        output_dir=Path("datasets/case2_generated"),
        tmp_dir=Path("datasets/case2_generated/tmp"),
        log_to_file=True,
        log_level="INFO",
        gene_id_mode="symbol",
        use_sqlite_index=True,
    )
    report = validate_case2_outputs(cfg)
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()