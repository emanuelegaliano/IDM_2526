import csv
import hashlib
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor 
from .gene_resolver import EnsemblAPIResolver

csv.field_size_limit(2147483647)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gene_chrom_map(m_file):
    """Estrae la mappatura Gene -> Cromosoma usando la colonna corretta."""
    local_map = {}
    chrom = m_file.stem.replace("chr_", "")
    with open(m_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_symbols = row.get("Gene.refGene")
            if raw_symbols:
                # Se ci sono più geni (TUBB8;ZMYND11), li prendo tutti
                symbols = raw_symbols.split(';')
                for s in symbols:
                    local_map[s.strip()] = chrom
    return local_map

def run_case1(base_dir: Path, workers: int = 4):
    logger.info("=== AVVIO PIPELINE CASO 1 ===")
    
    datasets_root = base_dir / "datasets"
    output_dir = base_dir / "output_case1"
    tmp_dir = output_dir / "tmp"
    output_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    # Resolver
    mapping_file = datasets_root / "mappings" / "symbol_to_ensg.tsv"
    resolver = EnsemblAPIResolver(str(tmp_dir / "ensembl_cache_case1.db"), mapping_file)

    # --- FASE 1: MULTITHREADING PER CROMOSOMI ---
    gene_to_chromosome = {}
    mutations_files = list((datasets_root / "mutations").glob("chr_*.tsv"))
    
    logger.info(f"Avvio estrazione cromosomi con {workers} thread...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(get_gene_chrom_map, mutations_files))
        for res in results:
            gene_to_chromosome.update(res)

    # --- FASE 2: STREAMING MATRICE ---
    tumor_files = list((datasets_root / "tumors").glob("*.tsv"))
    nodes_file = output_dir / "expression_profiling_nodes.tsv"
    edges_file = output_dir / "patient_expression_edges.tsv"

    with open(nodes_file, "w", newline="") as f_n, open(edges_file, "w", newline="") as f_e:
        node_writer = csv.writer(f_n, delimiter="\t")
        edge_writer = csv.writer(f_e, delimiter="\t")
        node_writer.writerow(["gene_exp_id", "gene_name", "expression_value", "chromosome"])
        edge_writer.writerow(["patient_id", "gene_exp_id"])

        for t_file in tumor_files:
            logger.info(f"Streaming matrice: {t_file.name}")
            with open(t_file, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                patient_ids = [p[:12] for p in header[1:]] # TCGA-XX-XXXX

                for row in reader:
                    symbol = resolver.get_symbol(row[0])
                    if not symbol: continue
                    
                    chrom = gene_to_chromosome.get(symbol, "Unknown")
                    for i, val in enumerate(row[1:]):
                        if float(val) > 0:
                            p_id = patient_ids[i]
                            md5_id = hashlib.md5(f"{symbol}_{p_id}".encode()).hexdigest()
                            node_writer.writerow([md5_id, symbol, val, chrom])
                            edge_writer.writerow([p_id, md5_id])

    logger.info("Pipeline conclusa con successo.")
    