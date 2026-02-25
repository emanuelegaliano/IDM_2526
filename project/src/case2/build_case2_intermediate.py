from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Iterable

import pandas as pd
import requests

from . import cfg

log = cfg.get_logger("build_case2_intermediate")

# ----------------------------
# Helpers
# ----------------------------
def md5_id(obj: dict) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    sep = "\t" if ext == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    return df


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def strip_ensembl_version(x: str) -> str:
    x = str(x).strip()
    if "." in x:
        return x.split(".", 1)[0]
    return x


def tcga_case3(barcode: str) -> str:
    """
    Normalize TCGA barcode to first 3 blocks (case-level):
      TCGA-06-0881-10A-01W-0424-08 -> TCGA-06-0881
      TCGA-24-1104-01A -> TCGA-24-1104
    """
    s = str(barcode).strip()
    if not s:
        return s
    parts = s.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return s


def normalize_mutation_gene(g: str) -> str:
    if g is None:
        return g
    g = str(g).strip()
    if not g:
        return g

    take_first = getattr(cfg, "MUTATION_GENE_TAKE_FIRST", False)
    split_char = getattr(cfg, "MUTATION_GENE_SPLIT_CHAR", None)
    if take_first and split_char and split_char in g:
        g = g.split(split_char, 1)[0].strip()
    return g


def detect_patient_id_column(patients_df: pd.DataFrame) -> str:
    override = getattr(cfg, "PATIENT_ID_COL", None)
    if override:
        if override not in patients_df.columns:
            raise ValueError(f"Configured PATIENT_ID_COL='{override}' not found. Columns={list(patients_df.columns)}")
        return override

    for c in ["patient_id", "case_id", "sample_id", "id"]:
        if c in patients_df.columns:
            return c

    fallback = [c for c in patients_df.columns if "patient" in c.lower() and "id" in c.lower()]
    if fallback:
        return fallback[0]

    raise ValueError(f"Cannot detect patient id column. Columns={list(patients_df.columns)}")


# ----------------------------
# Ensembl mapping auto-generation (Ensembl REST)
# ----------------------------
ENSEMBL_LOOKUP_POST = "https://rest.ensembl.org/lookup/id"
ENSEMBL_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "IDM-case2/1.0",
}


def _chunked(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_ensembl_ids_from_expression(expr_path: Path) -> List[str]:
    df = pd.read_csv(expr_path, sep="\t", usecols=[0], dtype=str, low_memory=False)
    gene_col = df.columns[0]
    ids = df[gene_col].dropna().astype(str).map(strip_ensembl_version)
    ids = ids[ids != ""]
    ids = ids[ids.str.startswith("ENSG")]
    return ids.drop_duplicates().tolist()


def generate_ensembl_to_symbol_tsv(expr_path: Path, out_path: Path, batch_size: int = 2000) -> None:
    """
    Generate mapping TSV by querying Ensembl REST in batches.
    batch_size increased to reduce number of requests.
    Writes columns: ensembl_id, gene_symbol
    """
    log.info("Generating Ensembl->Symbol mapping via Ensembl REST...")
    ensg_ids = extract_ensembl_ids_from_expression(expr_path)
    log.info("Unique ENSG ids found in expression: %s", f"{len(ensg_ids):,}")

    rows: List[tuple[str, str]] = []

    for b, batch in enumerate(_chunked(ensg_ids, batch_size), start=1):
        payload = {"ids": batch}
        r = requests.post(
            ENSEMBL_LOOKUP_POST,
            headers=ENSEMBL_HEADERS,
            data=json.dumps(payload),
            timeout=120,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Ensembl API error {r.status_code}: {r.text[:300]}")

        data: Dict[str, Any] = r.json()
        for ensembl_id, obj in data.items():
            if not obj:
                continue
            symbol = obj.get("display_name")
            if symbol:
                rows.append((ensembl_id, str(symbol).strip()))

        # soft rate limit
        time.sleep(0.1)

        if b % 5 == 0:
            log.info("Mapping batches processed: %s", b)

    out = pd.DataFrame(rows, columns=["ensembl_id", "gene_symbol"]).drop_duplicates(subset=["ensembl_id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)

    log.info("Wrote mapping file: %s (%s rows)", out_path, f"{len(out):,}")


def load_or_create_mapping() -> pd.DataFrame:
    mapping_path = getattr(cfg, "ENSEMBL_TO_SYMBOL_PATH", None)
    if mapping_path is None:
        mapping_path = cfg.DATASETS_DIR / "ensembl_to_symbol.tsv"
    mapping_path = Path(mapping_path)

    ensembl_col = getattr(cfg, "ENSEMBL_COL", "ensembl_id")
    symbol_col = getattr(cfg, "SYMBOL_COL", "gene_symbol")

    if not mapping_path.exists():
        if not cfg.EXPRESSION_MATRIX_FILES:
            raise ValueError("No expression files configured in cfg.EXPRESSION_MATRIX_FILES.")
        expr_path = Path(cfg.EXPRESSION_MATRIX_FILES[0])
        generate_ensembl_to_symbol_tsv(expr_path=expr_path, out_path=mapping_path)

    df = read_table(mapping_path)
    if ensembl_col not in df.columns or symbol_col not in df.columns:
        raise ValueError(
            f"Mapping file must contain columns '{ensembl_col}' and '{symbol_col}'. "
            f"Found columns={list(df.columns)}"
        )

    out = df[[ensembl_col, symbol_col]].copy()
    out = out.rename(columns={ensembl_col: "ensembl_id", symbol_col: "gene_symbol"})
    out["ensembl_id"] = out["ensembl_id"].astype(str).str.strip().map(strip_ensembl_version)
    out["gene_symbol"] = out["gene_symbol"].astype(str).str.strip()
    out = out.dropna()
    out = out[(out["ensembl_id"] != "") & (out["gene_symbol"] != "")]
    out = out.drop_duplicates(subset=["ensembl_id"])
    log.info("Loaded Ensembl->Symbol mapping: %s rows", f"{len(out):,}")
    return out


# ----------------------------
# Mutations -> Genes -> Edges
# ----------------------------
def load_mutations_with_chr() -> pd.DataFrame:
    dfs = []

    for idx, path in enumerate(cfg.MUTATION_FILES):
        path = Path(path)
        chr_id = cfg.CHROMOSOMES[idx] if idx < len(cfg.CHROMOSOMES) else "NA"
        df = read_table(path)

        mut_id_col = getattr(cfg, "MUTATION_ID_COL", None)
        gene_col = getattr(cfg, "MUTATION_GENE_COL", None)
        chr_col = getattr(cfg, "MUTATION_CHR_COL", None)

        if not mut_id_col or not gene_col:
            raise ValueError("cfg must define MUTATION_ID_COL and MUTATION_GENE_COL")

        missing = [c for c in [mut_id_col, gene_col] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required mutation columns in {path.name}: {missing}. Columns={list(df.columns)}"
            )

        out = df[[mut_id_col, gene_col]].copy()
        out = out.rename(columns={mut_id_col: "mutation_id", gene_col: "gene_name"})

        out["mutation_id"] = out["mutation_id"].astype(str).str.strip()
        out["gene_name"] = out["gene_name"].map(normalize_mutation_gene)

        if chr_col and chr_col in df.columns:
            out["chromosome"] = df[chr_col].astype(str).str.strip()
        else:
            out["chromosome"] = str(chr_id)

        out = out.dropna()
        out = out[(out["mutation_id"] != "") & (out["gene_name"] != "")]
        out = out.drop_duplicates(subset=["mutation_id"])

        dfs.append(out)
        log.info("Loaded mutations file: %s | rows=%s", path.name, f"{len(out):,}")

    if not dfs:
        return pd.DataFrame(columns=["mutation_id", "gene_name", "chromosome"])
    return pd.concat(dfs, ignore_index=True)


def build_gene_table(mutations_df: pd.DataFrame) -> pd.DataFrame:
    if mutations_df.empty:
        raise ValueError("mutations_df is empty")

    counts = (
        mutations_df.groupby(["gene_name", "chromosome"])
        .size()
        .reset_index(name="n")
        .sort_values(["gene_name", "n"], ascending=[True, False])
    )
    top_chr = counts.drop_duplicates(subset=["gene_name"], keep="first")[["gene_name", "chromosome"]]

    genes = top_chr.copy()
    genes["gene_id"] = genes.apply(
        lambda r: md5_id({"gene_name": r["gene_name"], "chromosome": r["chromosome"]}),
        axis=1,
    )
    genes = genes[["gene_id", "gene_name", "chromosome"]].drop_duplicates(subset=["gene_id"])

    log.info("Built genes table: %s genes", f"{len(genes):,}")
    return genes


def build_gene_mutation_edges(genes_df: pd.DataFrame, mutations_df: pd.DataFrame) -> pd.DataFrame:
    merged = mutations_df.merge(genes_df, on=["gene_name", "chromosome"], how="inner")
    edges = merged[["gene_id", "mutation_id"]].drop_duplicates()
    log.info("Built Gene->Mutation edges: %s", f"{len(edges):,}")
    return edges


# ----------------------------
# Expression -> Patient->Gene edges
# ----------------------------
def build_patient_gene_expression_edges(
    genes_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    expression_files: List[Path],
) -> pd.DataFrame:
    patient_id_col = detect_patient_id_column(patients_df)

    # Normalize patient ids to 4-block TCGA sample barcode
    patient_ids = set(
        patients_df[patient_id_col].dropna().astype(str).str.strip().map(tcga_case3)
    )

    gene_symbols_needed = set(genes_df["gene_name"].dropna().astype(str).str.strip())
    mapping = load_or_create_mapping()
    all_edges = []

    for expr_path in expression_files:
        expr_path = Path(expr_path)
        log.info("Reading expression matrix: %s", expr_path.name)

        header = pd.read_csv(expr_path, sep="\t", nrows=0).columns.tolist()
        if not header:
            raise ValueError(f"Empty header in expression file: {expr_path}")

        gene_col = cfg.EXPR_GENE_COL if getattr(cfg, "EXPR_GENE_COL", None) else header[0]
        if gene_col not in header:
            raise ValueError(f"Configured EXPR_GENE_COL='{gene_col}' not found in header for {expr_path.name}")

        sample_cols = [c for c in header if c != gene_col]

        # Keep expression columns whose normalized sample id exists in patients
        keep_samples = [c for c in sample_cols if tcga_case3(c) in patient_ids]

        log.info(
            "Expression columns: %s | matched (after TCGA 4-block normalization): %s",
            f"{len(sample_cols):,}",
            f"{len(keep_samples):,}",
        )

        if not keep_samples:
            log.warning(
                "No overlapping patient/sample columns found in %s even after normalization. Skipping.",
                expr_path.name,
            )
            continue

        usecols = [gene_col] + keep_samples
        expr = pd.read_csv(expr_path, sep="\t", usecols=usecols, dtype=str, low_memory=False)

        expr[gene_col] = expr[gene_col].astype(str).str.strip().map(strip_ensembl_version)
        expr = expr.rename(columns={gene_col: "ensembl_id"})

        # Map ENSG -> symbol
        expr = expr.merge(mapping, on="ensembl_id", how="inner")
        expr = expr.rename(columns={"gene_symbol": "gene_name"})

        # Keep only genes present in graph
        expr = expr[expr["gene_name"].isin(gene_symbols_needed)]
        if expr.empty:
            log.warning(
                "No genes left after mapping+filtering for %s. Check mapping symbols vs Gene.refGene symbols.",
                expr_path.name,
            )
            continue

        # wide -> long
        long = expr.drop(columns=["ensembl_id"]).melt(
            id_vars=["gene_name"],
            var_name="patient_id",
            value_name="expression_value",
        )

        # Normalize patient_id from expression to 4-block TCGA key
        long["patient_id"] = long["patient_id"].astype(str).str.strip().map(tcga_case3)
        long["expression_value"] = long["expression_value"].astype(str).str.strip()

        # Join to gene_id
        long = long.merge(genes_df[["gene_id", "gene_name"]], on="gene_name", how="inner")
        long = long[["patient_id", "gene_id", "expression_value"]].dropna()
        long = long[long["expression_value"] != ""]
        long = long.drop_duplicates()

        log.info("Patient->Gene edges from %s: %s", expr_path.name, f"{len(long):,}")
        all_edges.append(long)

    edges = (
        pd.concat(all_edges, ignore_index=True)
        if all_edges
        else pd.DataFrame(columns=["patient_id", "gene_id", "expression_value"])
    )
    log.info("Total Patient->Gene edges: %s", f"{len(edges):,}")
    return edges


# ----------------------------
# Pipeline
# ----------------------------
def run() -> None:
    cfg.validate_paths()
    ensure_dir(Path(cfg.GENERATED_DIR))

    log.info("Building Case2 intermediate files in: %s", cfg.GENERATED_DIR)

    patients_df = read_table(cfg.PATIENTS_PATH)
    mutations_df = load_mutations_with_chr()

    genes_df = build_gene_table(mutations_df)
    gene_mut_edges = build_gene_mutation_edges(genes_df, mutations_df)

    patient_gene_edges = build_patient_gene_expression_edges(
        genes_df=genes_df,
        patients_df=patients_df,
        expression_files=[Path(p) for p in cfg.EXPRESSION_MATRIX_FILES],
    )

    genes_out = Path(cfg.GENERATED_DIR) / "genes.tsv"
    gm_out = Path(cfg.GENERATED_DIR) / "gene_mutation_edges.tsv"
    pg_out = Path(cfg.GENERATED_DIR) / "patient_gene_expression_edges.tsv"

    write_tsv(genes_df, genes_out)
    write_tsv(gene_mut_edges, gm_out)
    write_tsv(patient_gene_edges, pg_out)

    log.info("Wrote: %s", genes_out)
    log.info("Wrote: %s", gm_out)
    log.info("Wrote: %s", pg_out)

    if patient_gene_edges.empty:
        log.warning(
            "patient_gene_expression_edges.tsv is empty. "
            "If mapping exists, check patient ID normalization and mapping symbol compatibility."
        )

    log.info("Done.")


def main() -> None:
    try:
        run()
    except Exception as e:
        log.exception("Error: %s", e)


if __name__ == "__main__":
    main()