import csv
import hashlib
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ottimizzazione CSV
csv.field_size_limit(2147483647)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def get_gene_data_worker(m_file: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Worker per caricare mutazioni e cromosomi in parallelo."""
    local_chrom_map = {}
    local_mut_map = {}
    chrom = m_file.stem.replace("chr_", "")
    
    with open(m_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_symbols = row.get("Gene.refGene")
            m_id = row.get("unique_id")
            if raw_symbols:
                symbols = [s.strip() for s in raw_symbols.split(';') if s.strip()]
                for s in symbols:
                    local_chrom_map[s] = chrom
                    if m_id:
                        local_mut_map.setdefault(s, []).append(m_id)
    return local_chrom_map, local_mut_map

def process_tumor_file(args):
    """Worker per processare un singolo file tumore (Fase 2)."""
    t_file, gene_to_chromosome, gene_to_mutations, output_dir, resolver_db_path, mapping_tsv = args
    
    # Re-import locale necessario per multiprocessing
    import hashlib
    import csv
    from case1.gene_resolver import OfflineGeneResolver
    
    # Ogni processo apre i propri file di output (partizionati per evitare conflitti)
    part_name = t_file.stem
    nodes_path = output_dir / f"nodes_{part_name}.part.tsv"
    pat_exp_path = output_dir / f"pat_exp_{part_name}.part.tsv"
    exp_mut_path = output_dir / f"exp_mut_{part_name}.part.tsv"
    
    # Inizializza resolver locale al processo
    resolver = OfflineGeneResolver(resolver_db_path, mapping_tsv)
    
    nodes_buffer, pat_buffer, mut_buffer = [], [], []
    CHUNK_SIZE = 99999 # Scrittura a blocchi
    
    with open(t_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        patient_ids = [p[:12] for p in header[1:]]

        for row in reader:
            symbol = resolver.get_symbol(row[0])
            if not symbol: continue
            
            chrom = gene_to_chromosome.get(symbol, "Unknown")
            muts = gene_to_mutations.get(symbol, [])
            
            for i, val in enumerate(row[1:]):
                if float(val) > 0:
                    p_id = patient_ids[i]
                    # Ottimizzazione: codifica stringa fissa una volta
                    base_str = f"{symbol}_{p_id}".encode()
                    md5_id = hashlib.md5(base_str).hexdigest()
                    
                    nodes_buffer.append([md5_id, symbol, val, chrom])
                    pat_buffer.append([p_id, md5_id])
                    
                    for m_id in muts:
                        mut_buffer.append([md5_id, m_id])
            
            # Flush dei buffer su disco
            if len(nodes_buffer) > CHUNK_SIZE:
                _flush_to_disk(nodes_path, nodes_buffer, pat_exp_path, pat_buffer, exp_mut_path, mut_buffer)
                nodes_buffer, pat_buffer, mut_buffer = [], [], []

    # Ultimo flush
    _flush_to_disk(nodes_path, nodes_buffer, pat_exp_path, pat_buffer, exp_mut_path, mut_buffer)
    return True

def _flush_to_disk(n_p, n_b, p_p, p_b, m_p, m_b):
    for path, buffer in [(n_p, n_b), (p_p, p_b), (m_p, m_b)]:
        if buffer:
            with open(path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f, delimiter="\t").writerows(buffer)

def validate_case1_outputs(output_dir: Path, logger: logging.Logger) -> bool:
    logger.info("=== INIZIO VALIDAZIONE OUTPUT CASO 1 ===")
    
    nodes_file = output_dir / "expression_profiling_nodes.tsv"
    pat_exp_file = output_dir / "patient_expression_edges.tsv"
    exp_mut_file = output_dir / "expression_mutation_edges.tsv"

    ok = True

    # 1. Controllo esistenza file
    for f in [nodes_file, pat_exp_file, exp_mut_file]:
        if not f.exists():
            logger.error(f"File critico mancante: {f.name}")
            return False

    # 2. Controllo header (Integrità strutturale)
    def check_header(file_path, expected):
        nonlocal ok
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                header = next(csv.reader(f, delimiter="\t"))
                if header != expected:
                    logger.error(f"Header errato in {file_path.name}. \nAtteso: {expected}\nTrovato: {header}")
                    ok = False
        except Exception as e:
            logger.error(f"Errore durante la lettura di {file_path.name}: {e}")
            ok = False

    check_header(nodes_file, ["gene_exp_id", "gene_name", "expression_value", "chromosome"])
    check_header(pat_exp_file, ["patient_id", "gene_exp_id"])
    check_header(exp_mut_file, ["gene_exp_id", "mutation_id"])

    # 3. Validazione dati e caricamento ID per integrità referenziale
    # Nota: Usiamo un set per gene_exp_ids. Con 105GB di input, assicurati di avere RAM a sufficienza.
    gene_exp_ids = set()
    bad_num = 0
    logger.info("Verifica valori numerici e indicizzazione gene_exp_id...")
    
    with open(nodes_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene_exp_ids.add(row["gene_exp_id"])
            try:
                float(row["expression_value"])
            except ValueError:
                bad_num += 1

    if bad_num > 0:
        logger.error(f"Trovati {bad_num} valori di espressione non validi (non numerici) in {nodes_file.name}.")
        ok = False

    # 4. Controllo Integrità Referenziale (Archi -> Nodi)
    def check_referential_integrity(file_path):
        nonlocal ok
        missing_ids = 0
        logger.info(f"Controllo coerenza archi in {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["gene_exp_id"] not in gene_exp_ids:
                    missing_ids += 1
        
        if missing_ids > 0:
            logger.error(f"Incoerenza dati: {missing_ids} archi in {file_path.name} puntano a un gene_exp_id inesistente.")
            ok = False

    check_referential_integrity(pat_exp_file)
    check_referential_integrity(exp_mut_file)

    if ok:
        logger.info("VALIDAZIONE COMPLETATA CON SUCCESSO: Tutti i test sono passati.")
    else:
        logger.error("VALIDAZIONE FALLITA: Controllare i log sopra per i dettagli degli errori.")
        # Se vuoi bloccare l'esecuzione in caso di errore:
        # raise SystemExit(1)
    
    return ok

def run_case1(base_dir: Path, workers: int = 4, validate: bool = True):
    logger.info("=== AVVIO PIPELINE OTTIMIZZATA CASO 1 ===")
    
    datasets_root = base_dir / "datasets"
    output_dir = base_dir / "output_case1"
    tmp_dir = output_dir / "tmp"
    output_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    mapping_tsv = datasets_root / "mappings" / "symbol_to_ensg.tsv"
    resolver_db = str(tmp_dir / "ensembl_cache_case1.db")

    # --- FASE 1: MULTIPROCESSING MUTAZIONI ---
    gene_to_chromosome = {}
    gene_to_mutations = {}
    mutations_files = list((datasets_root / "mutations").glob("chr_*.tsv"))
    
    logger.info(f"Fase 1: Caricamento mutazioni con {workers} processi...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(get_gene_data_worker, f): f for f in mutations_files}
        for future in as_completed(futures):
            l_chroms, l_muts = future.result()
            gene_to_chromosome.update(l_chroms)
            for g, m in l_muts.items():
                gene_to_mutations.setdefault(g, []).extend(m)
            logger.info(f"Caricato: {futures[future].name}")

    # --- FASE 2: MULTIPROCESSING MATRICI ---
    tumor_files = list((datasets_root / "tumors").glob("*.tsv"))
    logger.info(f"Fase 2: Processing di {len(tumor_files)} matrici...")
    
    args = [(f, gene_to_chromosome, gene_to_mutations, tmp_dir, resolver_db, mapping_tsv) for f in tumor_files]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_tumor_file, args))

    # --- FASE 3: MERGE DEI FILE PARTIAL ---
    logger.info("Fase 3: Merge dei file parziali...")
    _merge_outputs(tmp_dir, output_dir)

    if validate:
        # Chiama la funzione direttamente senza ri-importarla
        validate_case1_outputs(output_dir, logger)
        logger.info("Validazione completata con successo.")

    logger.info("=== CASE 1 PIPELINE COMPLETATA ===")

def _merge_outputs(tmp_dir, output_dir):
    for final_name, prefix in [
        ("expression_profiling_nodes.tsv", "nodes_"),
        ("patient_expression_edges.tsv", "pat_exp_"),
        ("expression_mutation_edges.tsv", "exp_mut_")
    ]:
        final_path = output_dir / final_name
        with open(final_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            # Scrittura Header
            if "nodes" in prefix: 
                writer.writerow(["gene_exp_id", "gene_name", "expression_value", "chromosome"])
            elif "pat_exp" in prefix: 
                writer.writerow(["patient_id", "gene_exp_id"])
            else: 
                writer.writerow(["gene_exp_id", "mutation_id"])
            
            # Unione dei file part
            for part in sorted(tmp_dir.glob(f"{prefix}*.part.tsv")):
                with open(part, "r", encoding="utf-8") as f_in:
                    # Usiamo shutil.copyfileobj o un ciclo per non caricare tutto in RAM
                    import shutil
                    shutil.copyfileobj(f_in, f_out)
                part.unlink() # Rimuove il file temporaneo dopo il merge