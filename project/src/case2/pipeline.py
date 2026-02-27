from __future__ import annotations

import csv
import os
import re
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from case2.config import Case2Config
from case2.gene_resolver import OfflineGeneResolver


# ------------------------------------------------------------
# CSV safety (alcune righe possono avere campi lunghi)
# ------------------------------------------------------------
def _set_csv_limits() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


# ------------------------------------------------------------
# Discovery
# ------------------------------------------------------------
@dataclass(frozen=True)
class ChromosomeFile:
    chromosome: str  # es "22"
    path: Path       # file chr_22.tsv


def discover_chromosomes(config: Case2Config, logger) -> List[ChromosomeFile]:
    mutations_dir = config.datasets_root / "mutations"
    if not mutations_dir.exists():
        logger.error(f"Directory mutazioni non trovata: {mutations_dir}")
        raise FileNotFoundError(mutations_dir)

    files = sorted(mutations_dir.glob("chr_*.tsv"))
    logger.info(f"Discovery completata: {len(files)} file mutazioni trovati.")
    chromosomes: List[ChromosomeFile] = []

    rgx = re.compile(r"chr_(\d+|X|Y|MT)\.tsv$", re.IGNORECASE)
    for fp in files:
        m = rgx.search(fp.name)
        if not m:
            continue
        chrom = m.group(1).upper()
        # normalizziamo MT in "MT" ma nel tuo genes.tsv hai numeri: ok, lasciamo stringa
        chromosomes.append(ChromosomeFile(chromosome=str(chrom), path=fp))

    logger.info(f"Trovati {len(chromosomes)} cromosomi.")
    return chromosomes


# ------------------------------------------------------------
# Step 1 worker: indicizza un cromosoma -> part edges + part genes
# ------------------------------------------------------------
@dataclass
class WorkerResult:
    chromosome: str
    rows: int
    edges: int
    genes: int
    part_edges_path: Path
    part_genes_path: Path


def _parse_gene_list(gene_field: str) -> List[str]:
    """
    Gene.refGene può essere:
    - '.' oppure vuoto
    - singolo gene
    - lista separata da ';'
    """
    if not gene_field:
        return []
    gene_field = gene_field.strip()
    if not gene_field or gene_field == ".":
        return []
    # split su ';' e pulizia
    out = []
    for g in gene_field.split(";"):
        g = g.strip()
        if not g or g == ".":
            continue
        out.append(g)
    return out


def _index_single_chromosome(args) -> WorkerResult:
    """
    Funzione eseguita in processi separati.
    Non usa logger (o usa print) per non complicare multiprocess.
    """
    chr_file: ChromosomeFile
    tmp_dir: Path
    chr_file, tmp_dir = args

    _set_csv_limits()

    in_path = chr_file.path
    chrom = chr_file.chromosome

    part_edges = tmp_dir / f"genes_mutations_edges.chr_{chrom}.part.tsv"
    part_genes = tmp_dir / f"genes.chr_{chrom}.part.tsv"

    rows = 0
    edges = 0
    genes_set: Set[str] = set()

    with in_path.open("r", encoding="utf-8", newline="") as f_in, \
         part_edges.open("w", encoding="utf-8", newline="") as f_edges:

        reader = csv.DictReader(f_in, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Header mancante in {in_path}")

        if "unique_id" not in reader.fieldnames or "Gene.refGene" not in reader.fieldnames:
            raise ValueError(
                f"{in_path.name} deve contenere colonne 'unique_id' e 'Gene.refGene' "
                f"(trovate: {reader.fieldnames})"
            )

        w_edges = csv.writer(f_edges, delimiter="\t", lineterminator="\n")
        # no header nei part file (lo aggiunge il merge)

        for row in reader:
            rows += 1
            mut_id = (row.get("unique_id") or "").strip()
            gene_field = row.get("Gene.refGene") or ""
            if not mut_id:
                continue

            genes = _parse_gene_list(gene_field)
            if not genes:
                continue

            for g in genes:
                genes_set.add(g)
                w_edges.writerow([g, mut_id])
                edges += 1

    # scrivi lista geni del cromosoma (uno per riga)
    with part_genes.open("w", encoding="utf-8", newline="") as f_g:
        w = csv.writer(f_g, delimiter="\t", lineterminator="\n")
        for g in sorted(genes_set):
            w.writerow([g, chrom])

    return WorkerResult(
        chromosome=chrom,
        rows=rows,
        edges=edges,
        genes=len(genes_set),
        part_edges_path=part_edges,
        part_genes_path=part_genes,
    )


# ------------------------------------------------------------
# Step 1 merge: unisci parts -> output finali
# ------------------------------------------------------------
def _merge_parts_to_outputs(
    config: Case2Config,
    logger,
    results: List[WorkerResult],
) -> Tuple[Path, Path, Set[str]]:
    """
    Ritorna:
    - genes_tsv_path
    - genes_mut_edges_path
    - whitelist_symbols (set di gene symbols)
    """
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    genes_tsv = out_dir / "genes.tsv"
    genes_mut_edges = out_dir / "genes_mutations_edges.tsv"

    # 1) merge genes_mutations_edges.tsv
    with genes_mut_edges.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "mutation_id"])

        for r in sorted(results, key=lambda x: x.chromosome):
            with r.part_edges_path.open("r", encoding="utf-8", newline="") as f_in:
                for line in f_in:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    # line già in TSV: gene \t mut
                    f_out.write(line + "\n")

    # 2) merge genes.tsv
    # gene -> set cromosomi
    gene_to_chroms: Dict[str, Set[str]] = {}
    for r in results:
        with r.part_genes_path.open("r", encoding="utf-8", newline="") as f_in:
            reader = csv.reader(f_in, delimiter="\t")
            for row in reader:
                if not row or len(row) < 2:
                    continue
                g = (row[0] or "").strip()
                c = (row[1] or "").strip()
                if not g or not c:
                    continue
                gene_to_chroms.setdefault(g, set()).add(c)

    whitelist_symbols = set(gene_to_chroms.keys())

    with genes_tsv.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "chromosome"])
        for g in sorted(gene_to_chroms.keys()):
            chroms = ",".join(sorted(gene_to_chroms[g], key=lambda x: (len(x), x)))
            w.writerow([g, chroms])

    logger.info(f"genes.tsv creato | geni: {len(whitelist_symbols)}")
    return genes_tsv, genes_mut_edges, whitelist_symbols


# ------------------------------------------------------------
# Step 3: patients_genes_expression_edges.tsv (OFFLINE mapping)
# ------------------------------------------------------------
def build_expression_edges_offline(
    config: Case2Config,
    logger,
    whitelist_symbols: Set[str],
) -> Path:
    tumors_dir = config.datasets_root / "tumors"
    # scegli il primo .tsv (nel tuo caso TCGA-OV.star_fpkm.tsv)
    tumor_files = sorted(tumors_dir.glob("*.tsv"))
    if not tumor_files:
        raise FileNotFoundError(f"Nessun file tumore trovato in {tumors_dir}")
    tumor_matrix = tumor_files[0]
    logger.info(f"Usando matrice espressione: {tumor_matrix.name}")

    # resolver offline
    if config.gene_id_mode != "offline_mapping":
        raise ValueError(
            f"Questa versione della pipeline supporta solo gene_id_mode='offline_mapping' "
            f"(attuale: {config.gene_id_mode})"
        )

    mapping_db = config.tmp_dir / "gene_mapping.db"
    resolver = OfflineGeneResolver(
        logger=logger,
        mapping_tsv=config.mapping_tsv,
        cache_db=mapping_db,
    )

    # converti whitelist SYMBOL -> ENSG (serve per matchare velocemente sulla matrice)
    logger.info(f"Geni in whitelist (symbol da mutazioni): {len(whitelist_symbols)}")
    logger.info("Risoluzione whitelist SYMBOL -> ENSG via mapping OFFLINE (TSV+SQLite)...")

    sym_to_ensg, missing = resolver.resolve_whitelist_symbols_to_ensg(sorted(whitelist_symbols))
    whitelist_ensg = set(sym_to_ensg.values())

    logger.info(f"Offline mapping completato | resolved={len(sym_to_ensg)} | missing={len(missing)}")

    # output
    out_path = config.output_dir / "patients_genes_expression_edges.tsv"

    _set_csv_limits()

    # Leggiamo la matrice in streaming:
    # - header: Ensembl_ID \t patient1 \t patient2 ...
    # - righe: ENSG....(.version) \t val1 \t val2 ...
    rows_read = 0
    rows_matched = 0
    edges_written = 0

    with tumor_matrix.open("r", encoding="utf-8", newline="") as f_in, \
         out_path.open("w", encoding="utf-8", newline="") as f_out:

        reader = csv.reader(f_in, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"Header non valido in {tumor_matrix}")

        # pazienti sono le colonne dopo la prima
        patient_ids = header[1:]
        logger.info(f"Pazienti trovati nella matrice: {len(patient_ids)}")

        w = csv.writer(f_out, delimiter="\t", lineterminator="\n")
        w.writerow(["patient_id", "gene_id", "expression_value"])

        for row in reader:
            rows_read += 1
            if not row:
                continue
            ensg_raw = (row[0] or "").strip()
            if not ensg_raw:
                continue

            # strip version
            ensg = ensg_raw.split(".", 1)[0]
            if ensg not in whitelist_ensg:
                continue

            # gene_id nell'output deve essere SYMBOL (come hai già ora)
            symbol = resolver.ensg_to_symbol(ensg)
            if not symbol:
                continue
            if symbol not in whitelist_symbols:
                continue

            rows_matched += 1

            # scrivi edges per ciascun paziente
            # Nota: non filtriamo zeri; scriviamo tutti i valori numerici presenti.
            vals = row[1:]
            # se riga corta, pad
            if len(vals) < len(patient_ids):
                vals = vals + [""] * (len(patient_ids) - len(vals))

            for pid, v in zip(patient_ids, vals):
                v = (v or "").strip()
                if not v:
                    continue
                # valida numerico leggero
                try:
                    float(v)
                except Exception:
                    continue
                w.writerow([pid, symbol, v])
                edges_written += 1

    logger.info("patients_genes_expression_edges.tsv creato.")
    logger.info(
        f"Matrice | righe lette: {rows_read} | righe matchate: {rows_matched} | edges scritti: {edges_written}"
    )
    return out_path


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def _read_tsv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        line = f.readline()
        return [x.strip() for x in line.rstrip("\n").split("\t") if x.strip()]


def _count_lines(path: Path) -> int:
    # conta righe excl header
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        next(f, None)
        for _ in f:
            n += 1
    return n


def validate_outputs(config: Case2Config, logger) -> None:
    logger.info("=== VALIDAZIONE OUTPUT CASE2 ===")

    genes_tsv = config.output_dir / "genes.tsv"
    genes_mut_edges = config.output_dir / "genes_mutations_edges.tsv"
    expr_edges = config.output_dir / "patients_genes_expression_edges.tsv"

    ok = True

    # headers
    h1 = _read_tsv_header(genes_tsv)
    h2 = _read_tsv_header(genes_mut_edges)
    h3 = _read_tsv_header(expr_edges)

    logger.info(f"Header genes.tsv: {h1}")
    logger.info(f"Header genes_mutations_edges.tsv: {h2}")
    logger.info(f"Header patients_genes_expression_edges.tsv: {h3}")

    if h1 != ["gene_id", "chromosome"]:
        logger.error("genes.tsv header atteso: gene_id, chromosome")
        ok = False
    if h2 != ["gene_id", "mutation_id"]:
        logger.error("genes_mutations_edges.tsv header atteso: gene_id, mutation_id")
        ok = False
    if h3 != ["patient_id", "gene_id", "expression_value"]:
        logger.error("patients_genes_expression_edges.tsv header atteso: patient_id, gene_id, expression_value")
        ok = False

    # counts
    n_genes = _count_lines(genes_tsv)
    n_edges = _count_lines(genes_mut_edges)
    n_expr = _count_lines(expr_edges)

    logger.info(f"Righe genes.tsv: {n_genes}")
    logger.info(f"Righe genes_mutations_edges.tsv: {n_edges}")
    logger.info(f"Righe patients_genes_expression_edges.tsv: {n_expr}")

    # load gene set in RAM (piccolo)
    logger.info("Caricamento set geni da genes.tsv (unico set in RAM)...")
    gene_set: Set[str] = set()
    with genes_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            g = (row.get("gene_id") or "").strip()
            if g:
                gene_set.add(g)
    logger.info(f"Geni unici in genes.tsv: {len(gene_set)}")
    if len(gene_set) != n_genes:
        logger.warning("genes.tsv potrebbe contenere duplicati (non critico, ma inatteso).")

    # sample expression values numeric
    ok_num = 0
    bad_num = 0
    sample_limit = 5000
    with expr_edges.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= sample_limit:
                break
            v = (row.get("expression_value") or "").strip()
            try:
                float(v)
                ok_num += 1
            except Exception:
                bad_num += 1
    logger.info(f"Campione expression_value: ok={ok_num} bad={bad_num} (su {min(sample_limit, n_expr)})")
    if bad_num > 0:
        logger.error("Trovati expression_value non numerici.")
        ok = False

    # check mutation_id set membership (costoso ma fattibile se fai set grande)
    # Nota: per multi-cromosoma può essere grande; lo facciamo comunque perché serve.
    logger.info("Caricamento mutation_id (unique_id) dai file datasets/mutations/chr_*.tsv ...")
    mutations_dir = config.datasets_root / "mutations"
    mut_ids: Set[str] = set()
    _set_csv_limits()
    for fp in sorted(mutations_dir.glob("chr_*.tsv")):
        with fp.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames or "unique_id" not in reader.fieldnames:
                continue
            for row in reader:
                mid = (row.get("unique_id") or "").strip()
                if mid:
                    mut_ids.add(mid)
    logger.info(f"mutation_id unici caricati: {len(mut_ids)}")

    # controlla un campione di edges
    sample_limit = 20000
    missing = 0
    with genes_mut_edges.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= sample_limit:
                break
            mid = (row.get("mutation_id") or "").strip()
            if mid and mid not in mut_ids:
                missing += 1
    if missing > 0:
        logger.error(f"Trovati mutation_id non presenti nei chr_*.tsv (sample): {missing}")
        ok = False

    if ok:
        logger.info("VALIDAZIONE OK.")
    else:
        logger.error("Validazione fallita. Interrompo con exit code 1.")
        raise SystemExit(1)


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run_case2(config: Case2Config, logger) -> None:
    config.resolve_paths()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== START CASE2 PIPELINE ===")

    chromosomes = discover_chromosomes(config, logger)
    if not chromosomes:
        logger.error("Nessun cromosoma trovato in datasets/mutations (atteso chr_*.tsv).")
        raise SystemExit(1)

    # tmp per part files
    parts_dir = config.tmp_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"STEP 1 (parallel) — indicizzazione mutazioni con workers={config.workers}")

    results: List[WorkerResult] = []
    if config.workers <= 1:
        for cf in chromosomes:
            r = _index_single_chromosome((cf, parts_dir))
            logger.info(f"{cf.path.name} completato | righe={r.rows} edges={r.edges} geni={r.genes}")
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as ex:
            futures = [ex.submit(_index_single_chromosome, (cf, parts_dir)) for cf in chromosomes]
            for fut in as_completed(futures):
                r = fut.result()
                logger.info(f"Worker done | {r.chromosome} | righe={r.rows} edges={r.edges} geni={r.genes}")
                results.append(r)

    logger.info("STEP 1b — merge part files in output finali")
    genes_tsv, genes_mut_edges, whitelist_symbols = _merge_parts_to_outputs(config, logger, results)

    logger.info("Costruzione patients_genes_expression_edges.tsv...")
    _ = build_expression_edges_offline(config, logger, whitelist_symbols)

    if config.validate:
        logger.info("Validazione abilitata (config.validate=True). Avvio validazione output...")
        validate_outputs(config, logger)
        logger.info("Validazione completata con successo.")

    logger.info("=== CASE2 PIPELINE COMPLETATA ===")