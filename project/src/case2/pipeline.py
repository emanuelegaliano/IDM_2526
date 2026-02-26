"""
case2/pipeline.py

Pipeline principale per il caso 2.
Genera:
- genes.tsv
- genes_mutations_edges.tsv
- patients_genes_expression_edges.tsv

Se config.validate=True, esegue anche validate_case2_outputs(config) a fine pipeline.
"""

from __future__ import annotations

import csv
import sys
import sqlite3
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

from .config import Case2Config
from .gene_resolver import EnsemblRestCachedClient


# ============================================================
# CSV FIELD LIMIT FIX
# ============================================================

def _raise_csv_field_limit():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int = int(max_int / 10)

_raise_csv_field_limit()


# ============================================================
# PUBLIC ENTRYPOINT
# ============================================================

def run_case2(config: Case2Config):
    logger = config.logger
    logger.info("=== START CASE2 PIPELINE ===")

    mutation_files = discover_chromosomes(config)
    logger.info(f"Trovati {len(mutation_files)} cromosomi.")

    gene_set_or_none = index_mutations(config=config, mutation_files=mutation_files)

    build_genes_nodes(config=config, gene_set=gene_set_or_none)

    build_expression_edges(config=config, gene_set=gene_set_or_none)

    # Validation (opzionale)
    if config.validate:
        logger.info("Validazione abilitata (config.validate=True). Avvio validazione output...")
        from .validate_outputs import validate_case2_outputs  # import lazy
        report = validate_case2_outputs(config)
        if not report.ok:
            logger.error("Validazione fallita. Interrompo con exit code 1.")
            raise SystemExit(1)
        logger.info("Validazione completata con successo.")

    logger.info("=== CASE2 PIPELINE COMPLETATA ===")


# ============================================================
# DISCOVERY
# ============================================================

def discover_chromosomes(config: Case2Config) -> List[Path]:
    logger = config.logger
    mutations_dir = config.mutations_path

    if not mutations_dir.exists():
        logger.error(f"Directory mutazioni non trovata: {mutations_dir}")
        raise FileNotFoundError(mutations_dir)

    files = sorted(mutations_dir.glob("chr_*.tsv"))
    logger.info(f"Discovery completata: {len(files)} file mutazioni trovati.")
    return files


# ============================================================
# STEP 1 — INDEX MUTATIONS
# ============================================================

def index_mutations(*, config: Case2Config, mutation_files: List[Path]) -> Optional[Set[str]]:
    logger = config.logger
    logger.info("Indicizzazione mutazioni...")

    gene_set: Set[str] = set()

    conn = None
    cur = None
    if config.use_sqlite_index:
        db_path = config.tmp_dir / "genes_index.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS genes (gene_id TEXT, chr TEXT)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_genes_gene ON genes(gene_id)")
        conn.commit()
        logger.debug(f"Indice SQLite pronto: {db_path}")

    with open(config.genes_mutations_edges_path, "w", newline="") as edge_out:
        writer = csv.writer(edge_out, delimiter="\t")
        writer.writerow(["gene_id", "mutation_id"])

        for file_path in mutation_files:
            chr_name = file_path.stem.replace("chr_", "")
            logger.info(f"Processing {file_path.name}")

            rows_seen = 0
            edges_written = 0
            sqlite_batch: List[Tuple[str, str]] = []

            with open(file_path, "r", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")

                if "unique_id" not in (reader.fieldnames or []) or "Gene.refGene" not in (reader.fieldnames or []):
                    logger.error(
                        f"Header non valido in {file_path.name}. Richiesti: unique_id, Gene.refGene"
                    )
                    raise ValueError(f"Invalid header in {file_path}")

                for row in reader:
                    rows_seen += 1
                    mutation_id = row.get("unique_id")
                    gene_field = row.get("Gene.refGene")

                    if not mutation_id or not gene_field:
                        continue

                    for gene_id in parse_gene_field(gene_field):
                        writer.writerow([gene_id, mutation_id])
                        edges_written += 1

                        if config.use_sqlite_index:
                            sqlite_batch.append((gene_id, chr_name))
                            if len(sqlite_batch) >= config.chunk_size_rows:
                                cur.executemany("INSERT INTO genes VALUES (?, ?)", sqlite_batch)
                                conn.commit()
                                sqlite_batch.clear()
                        else:
                            gene_set.add(gene_id)

                    if rows_seen % (config.chunk_size_rows * 5) == 0:
                        logger.info(
                            f"{file_path.name} | righe lette: {rows_seen} | edges scritti: {edges_written}"
                        )

            if config.use_sqlite_index and sqlite_batch:
                cur.executemany("INSERT INTO genes VALUES (?, ?)", sqlite_batch)
                conn.commit()
                sqlite_batch.clear()

            logger.info(f"{file_path.name} completato | righe: {rows_seen} | edges: {edges_written}")

    if config.use_sqlite_index:
        logger.info("Indicizzazione completata (SQLite).")
        conn.close()
        return None

    logger.info(f"Indicizzazione completata ({len(gene_set)} geni in RAM).")
    return gene_set


# ============================================================
# STEP 2 — BUILD GENE NODES
# ============================================================

def build_genes_nodes(*, config: Case2Config, gene_set: Optional[Set[str]]):
    logger = config.logger
    logger.info("Costruzione genes.tsv...")

    with open(config.genes_output_path, "w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["gene_id", "chromosome"])

        if config.use_sqlite_index:
            db_path = config.tmp_dir / "genes_index.db"
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            query = """
                SELECT gene_id, GROUP_CONCAT(DISTINCT chr)
                FROM genes
                GROUP BY gene_id
            """
            count = 0
            for gene_id, chr_list in cur.execute(query):
                writer.writerow([gene_id, chr_list])
                count += 1
            conn.close()
            logger.info(f"genes.tsv creato (SQLite) | geni: {count}")
        else:
            count = 0
            for gene_id in sorted(gene_set or []):
                writer.writerow([gene_id, "NA"])
                count += 1
            logger.info(f"genes.tsv creato (RAM) | geni: {count}")


# ============================================================
# STEP 3 — BUILD PATIENT->GENE EXPRESSION EDGES
# ============================================================

def build_expression_edges(*, config: Case2Config, gene_set: Optional[Set[str]]):
    logger = config.logger
    logger.info("Costruzione patients_genes_expression_edges.tsv...")

    tumor_files = sorted(config.tumors_path.glob("*.tsv"))
    if not tumor_files:
        logger.error(f"Nessuna matrice espressione trovata in: {config.tumors_path}")
        raise FileNotFoundError("No tumor expression file found.")

    tumor_file = tumor_files[0]
    logger.info(f"Usando matrice espressione: {tumor_file.name}")

    if config.use_sqlite_index:
        db_path = config.tmp_dir / "genes_index.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        whitelist_symbols = [r[0] for r in cur.execute("SELECT DISTINCT gene_id FROM genes")]
        conn.close()
    else:
        whitelist_symbols = sorted(gene_set or [])

    logger.info(f"Geni in whitelist (symbol da mutazioni): {len(whitelist_symbols)}")

    mode = (config.gene_id_mode or "symbol").lower()
    ensg_to_symbol: Dict[str, str] = {}

    if mode == "ensembl_api_cache":
        client = EnsemblRestCachedClient(
            logger=logger,
            cache_db_path=(config.tmp_dir / "ensembl_cache.db"),
        )

        resolved = 0
        missing = 0
        logger.info("Risoluzione whitelist SYMBOL -> ENSG via Ensembl REST (cache sqlite)...")

        for i, sym in enumerate(whitelist_symbols, start=1):
            ensg = client.symbol_to_ensembl_gene_id(sym)
            if ensg:
                ensg_to_symbol[ensg] = sym
                resolved += 1
            else:
                missing += 1

            if i % 100 == 0:
                logger.info(
                    f"Whitelist mapping progress: {i}/{len(whitelist_symbols)} | resolved={resolved} | missing={missing}"
                )

        logger.info(f"Whitelist mapping completato | resolved={resolved} | missing={missing}")

        if resolved == 0:
            logger.error(
                "Nessun gene symbol della whitelist è stato risolto in ENSG. "
                "Controlla internet / species / o validità dei simboli."
            )

    whitelist_set = set(whitelist_symbols)  # per mode symbol/ensembl

    total_rows = 0
    matched_rows = 0
    written_edges = 0

    with open(config.patients_genes_expression_edges_path, "w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["patient_id", "gene_id", "expression_value"])

        with open(tumor_file, "r", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header or len(header) < 2:
                raise ValueError(f"Header matrice espressione non valido: {tumor_file}")

            patient_ids = header[1:]
            logger.info(f"Pazienti trovati nella matrice: {len(patient_ids)}")

            for row in reader:
                total_rows += 1
                raw_key = (row[0] or "").strip()
                if not raw_key:
                    continue

                if mode == "symbol":
                    gene_id = raw_key
                    if gene_id not in whitelist_set:
                        continue

                elif mode == "ensembl":
                    gene_id = raw_key.split(".", 1)[0]
                    if gene_id not in whitelist_set:
                        continue

                elif mode == "ensembl_api_cache":
                    ensg = raw_key.split(".", 1)[0]
                    sym = ensg_to_symbol.get(ensg)
                    if not sym:
                        continue
                    gene_id = sym

                else:
                    raise ValueError(f"gene_id_mode non supportato: {config.gene_id_mode}")

                matched_rows += 1
                values = row[1:]
                for patient_id, value in zip(patient_ids, values):
                    writer.writerow([patient_id, gene_id, value])
                    written_edges += 1

                if total_rows % (config.chunk_size_rows * 5) == 0:
                    logger.info(
                        f"Matrice | righe lette: {total_rows} | righe matchate: {matched_rows} | edges scritti: {written_edges}"
                    )

    logger.info("patients_genes_expression_edges.tsv creato.")
    logger.info(
        f"Matrice | righe lette: {total_rows} | righe matchate: {matched_rows} | edges scritti: {written_edges}"
    )

    if matched_rows == 0:
        logger.error(
            "Nessuna riga della matrice ha matchato la whitelist. "
            "Con mutazioni=SYMBOL e matrice=ENSG, usa gene_id_mode='ensembl_api_cache'."
        )


# ============================================================
# UTILITIES
# ============================================================

def parse_gene_field(gene_field: str) -> List[str]:
    genes: List[str] = []
    for g in (gene_field or "").split(";"):
        g = g.strip()
        if g and g != ".":
            genes.append(g)
    return genes