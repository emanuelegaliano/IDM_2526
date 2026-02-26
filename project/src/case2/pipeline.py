"""
case2/pipeline.py

Pipeline Caso 2 (parallela):
- Step 1 (mutations): parallelizza per cromosoma con ProcessPoolExecutor.
  Ogni worker scrive part files:
    - genes_mutations_edges.<chr>.tsv
    - genes.<chr>.tsv  (gene_id, chromosome)
  Poi merge nel main:
    - genes_mutations_edges.tsv
    - genes.tsv

- Step 3 (expression): streaming della matrice, ma la risoluzione whitelist SYMBOL->ENSG
  è parallela con ThreadPoolExecutor (I/O bound).

Se config.validate=True, esegue validate_outputs a fine pipeline.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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

    parts_dir = config.tmp_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: parallel processing mutations
    logger.info(f"STEP 1 (parallel) — indicizzazione mutazioni con workers={config.workers}")
    part_edges_files, part_genes_files = parallel_index_mutations(
        config=config,
        mutation_files=mutation_files,
        parts_dir=parts_dir,
    )

    # Merge parts -> final edges + genes nodes
    logger.info("STEP 1b — merge part files in output finali")
    merge_edges_parts(
        part_files=part_edges_files,
        out_path=config.genes_mutations_edges_path,
        header=["gene_id", "mutation_id"],
    )

    genes_map = merge_genes_parts(part_genes_files)
    write_genes_tsv(
        out_path=config.genes_output_path,
        genes_map=genes_map,
    )
    logger.info(f"genes.tsv creato | geni: {len(genes_map)}")

    # STEP 3: expression edges (streaming) + whitelist mapping (parallel threads)
    build_expression_edges(
        config=config,
        genes_map=genes_map,
    )

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
# STEP 1 — PARALLEL MUTATIONS INDEXING
# ============================================================

def parallel_index_mutations(
    *,
    config: Case2Config,
    mutation_files: List[Path],
    parts_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """
    Esegue in parallelo il parsing dei file chr_*.tsv.
    Ogni worker produce:
      - genes_mutations_edges.<chr>.tsv
      - genes.<chr>.tsv
    Ritorna la lista dei file part prodotti.
    """
    logger = config.logger
    workers = max(1, int(config.workers))

    tasks: List[Tuple[str, Path]] = []
    for fp in mutation_files:
        chr_name = fp.stem.replace("chr_", "")
        tasks.append((chr_name, fp))

    part_edges_files: List[Path] = []
    part_genes_files: List[Path] = []

    # Se 1 worker, evita overhead multiprocessing
    if workers == 1:
        for chr_name, fp in tasks:
            pe, pg, stats = _process_mutation_file_worker(
                chr_name=chr_name,
                file_path=fp,
                parts_dir=parts_dir,
            )
            part_edges_files.append(pe)
            part_genes_files.append(pg)
            logger.info(
                f"{fp.name} completato | righe={stats['rows']} edges={stats['edges']} geni={stats['genes']}"
            )
        return part_edges_files, part_genes_files

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _process_mutation_file_worker,
                chr_name,
                fp,
                parts_dir,
            )
            for chr_name, fp in tasks
        ]

        for fut in as_completed(futures):
            pe, pg, stats = fut.result()
            part_edges_files.append(pe)
            part_genes_files.append(pg)
            logger.info(
                f"Worker done | {stats['chr']} | righe={stats['rows']} edges={stats['edges']} geni={stats['genes']}"
            )

    # deterministic order
    part_edges_files = sorted(part_edges_files)
    part_genes_files = sorted(part_genes_files)
    return part_edges_files, part_genes_files


def _process_mutation_file_worker(
    chr_name: str,
    file_path: Path,
    parts_dir: Path,
) -> Tuple[Path, Path, Dict[str, int]]:
    """
    Worker process-safe: non usa logger.
    Scrive due file part:
      - genes_mutations_edges.<chr>.tsv
      - genes.<chr>.tsv
    """
    out_edges = parts_dir / f"genes_mutations_edges.{chr_name}.tsv"
    out_genes = parts_dir / f"genes.{chr_name}.tsv"

    rows_seen = 0
    edges_written = 0
    genes_seen: Set[str] = set()

    with open(out_edges, "w", newline="") as fe:
        w_edges = csv.writer(fe, delimiter="\t")
        w_edges.writerow(["gene_id", "mutation_id"])

        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            # required columns
            if not reader.fieldnames or "unique_id" not in reader.fieldnames or "Gene.refGene" not in reader.fieldnames:
                raise ValueError(f"Invalid header in {file_path}")

            for row in reader:
                rows_seen += 1
                mutation_id = row.get("unique_id")
                gene_field = row.get("Gene.refGene")
                if not mutation_id or not gene_field:
                    continue

                for gene_id in parse_gene_field(gene_field):
                    w_edges.writerow([gene_id, mutation_id])
                    edges_written += 1
                    genes_seen.add(gene_id)

    with open(out_genes, "w", newline="") as fg:
        w_genes = csv.writer(fg, delimiter="\t")
        w_genes.writerow(["gene_id", "chromosome"])
        for g in genes_seen:
            w_genes.writerow([g, chr_name])

    stats = {"chr": int(chr_name) if chr_name.isdigit() else chr_name, "rows": rows_seen, "edges": edges_written, "genes": len(genes_seen)}
    return out_edges, out_genes, stats


# ============================================================
# MERGE PARTS
# ============================================================

def merge_edges_parts(*, part_files: List[Path], out_path: Path, header: List[str]) -> None:
    """
    Concatena part_files (ognuno con header) in un unico file con un solo header.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(header)
        for pf in part_files:
            with open(pf, "r", newline="") as f:
                r = csv.reader(f, delimiter="\t")
                next(r, None)  # skip header
                for row in r:
                    if row:
                        w.writerow(row)


def merge_genes_parts(part_genes_files: List[Path]) -> Dict[str, Set[str]]:
    """
    Legge i files genes.<chr>.tsv e aggrega gene_id -> set(chromosome)
    """
    genes_map: Dict[str, Set[str]] = {}
    for pf in part_genes_files:
        with open(pf, "r", newline="") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                if len(row) != 2:
                    continue
                gene_id, chr_name = row[0], row[1]
                if not gene_id:
                    continue
                genes_map.setdefault(gene_id, set()).add(chr_name)
    return genes_map


def write_genes_tsv(*, out_path: Path, genes_map: Dict[str, Set[str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["gene_id", "chromosome"])
        for gene_id in sorted(genes_map.keys()):
            chr_list = ",".join(sorted(genes_map[gene_id], key=lambda x: (len(x), x)))
            w.writerow([gene_id, chr_list])


# ============================================================
# STEP 3 — EXPRESSION EDGES
# ============================================================

def build_expression_edges(*, config: Case2Config, genes_map: Dict[str, Set[str]]) -> None:
    logger = config.logger
    logger.info("Costruzione patients_genes_expression_edges.tsv...")

    tumor_files = sorted(config.tumors_path.glob("*.tsv"))
    if not tumor_files:
        logger.error(f"Nessuna matrice espressione trovata in: {config.tumors_path}")
        raise FileNotFoundError("No tumor expression file found.")

    tumor_file = tumor_files[0]
    logger.info(f"Usando matrice espressione: {tumor_file.name}")

    whitelist_symbols = list(genes_map.keys())
    logger.info(f"Geni in whitelist (symbol da mutazioni): {len(whitelist_symbols)}")

    mode = (config.gene_id_mode or "symbol").lower()

    ensg_to_symbol: Dict[str, str] = {}

    if mode == "ensembl_api_cache":
        client = EnsemblRestCachedClient(
            logger=logger,
            cache_db_path=(config.tmp_dir / "ensembl_cache.db"),
        )

        logger.info("Risoluzione whitelist SYMBOL -> ENSG via Ensembl REST (cache sqlite), in parallelo (threads)...")

        resolved = 0
        missing = 0

        # ThreadPool perché I/O bound
        max_threads = min(8, max(4, config.workers))
        with ThreadPoolExecutor(max_workers=max_threads) as ex:
            futs = {ex.submit(client.symbol_to_ensembl_gene_id, sym): sym for sym in whitelist_symbols}
            done = 0
            for fut in as_completed(futs):
                sym = futs[fut]
                ensg = fut.result()
                done += 1
                if ensg:
                    ensg_to_symbol[ensg] = sym
                    resolved += 1
                else:
                    missing += 1

                if done % 100 == 0:
                    logger.info(
                        f"Whitelist mapping progress: {done}/{len(whitelist_symbols)} | resolved={resolved} | missing={missing}"
                    )

        logger.info(f"Whitelist mapping completato | resolved={resolved} | missing={missing}")
        if resolved == 0:
            logger.error("Nessun gene risolto in ENSG. Controlla internet/species/symbol.")

    whitelist_set = set(whitelist_symbols)

    total_rows = 0
    matched_rows = 0
    written_edges = 0

    with open(config.patients_genes_expression_edges_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["patient_id", "gene_id", "expression_value"])

        with open(tumor_file, "r", newline="") as f:
            r = csv.reader(f, delimiter="\t")
            header = next(r, None)
            if not header or len(header) < 2:
                raise ValueError(f"Header matrice espressione non valido: {tumor_file}")

            patient_ids = header[1:]
            logger.info(f"Pazienti trovati nella matrice: {len(patient_ids)}")

            for row in r:
                total_rows += 1
                raw_key = (row[0] or "").strip()
                if not raw_key:
                    continue

                if mode == "symbol":
                    gene_id = raw_key
                    if gene_id not in whitelist_set:
                        continue

                elif mode == "ensembl":
                    # NOTA: questo non matcha mutazioni=SYMBOL, ma lo lasciamo per completezza
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
                for pid, val in zip(patient_ids, values):
                    w.writerow([pid, gene_id, val])
                    written_edges += 1

                if total_rows % (config.chunk_size_rows * 5) == 0:
                    logger.info(
                        f"Matrice | righe lette: {total_rows} | righe matchate: {matched_rows} | edges scritti: {written_edges}"
                    )

    logger.info("patients_genes_expression_edges.tsv creato.")
    logger.info(f"Matrice | righe lette: {total_rows} | righe matchate: {matched_rows} | edges scritti: {written_edges}")

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