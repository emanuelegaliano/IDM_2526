import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from dotenv import load_dotenv


def md5_id(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def load_env() -> dict:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        script_dir / ".env",
        script_dir.parent / ".env",
        script_dir.parent.parent / ".env",
    ]
    loaded_from = None
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=True)
            loaded_from = p
            break
    if loaded_from is None:
        raise FileNotFoundError("No .env found")

    cfg = {
        "CASE2_STAGE_DIR": os.getenv("CASE2_STAGE_DIR", "./case2_stage"),
        "CASE2_OUT_DIR": os.getenv("CASE2_OUT_DIR", "./case2_out"),
    }
    print(f"Loaded .env from: {loaded_from}")
    return cfg


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def normalize_tcga_id(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip()
    if not x:
        return ""
    parts = x.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return x


def main():
    cfg = load_env()
    stage_dir = Path(cfg["CASE2_STAGE_DIR"])
    out_dir = Path(cfg["CASE2_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Input stage files
    patients_ids_path = stage_dir / "case2_patients_ids.tsv"
    gene_list_path = stage_dir / "case2_gene_list.tsv"
    mut_gene_map_path = stage_dir / "case2_chr22_mutations_gene_map.tsv"
    expr_submatrix_path = stage_dir / "case2_expression_submatrix.tsv"

    for p in [patients_ids_path, gene_list_path, mut_gene_map_path, expr_submatrix_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing stage file: {p}")

    print("Loading stage inputs...")
    patients_ids = read_tsv(patients_ids_path)
    genes_list = read_tsv(gene_list_path)
    mut_gene_map = read_tsv(mut_gene_map_path)
    expr = read_tsv(expr_submatrix_path)

    # --- Validate columns
    if "patient_id_norm" not in patients_ids.columns:
        raise ValueError(f"{patients_ids_path} must have column patient_id_norm")
    if "gene_name" not in genes_list.columns:
        raise ValueError(f"{gene_list_path} must have column gene_name")
    if not {"unique_id", "gene_name"}.issubset(mut_gene_map.columns):
        raise ValueError(f"{mut_gene_map_path} must have columns unique_id, gene_name")

    # Expression: first column is row id (gene symbol), remaining columns are samples
    if expr.shape[1] < 2:
        raise ValueError("Expression submatrix invalid (needs at least 2 columns)")

    # First column = gene symbol column (even if named "sample")
    gene_col = expr.columns[0]

    # Rename it explicitly to avoid collision with melt()
    expr = expr.rename(columns={gene_col: "gene_name"})
    gene_col = "gene_name"

    sample_cols = list(expr.columns[1:])

    # --- Build allowed sets
    patient_norm_set: Set[str] = set(patients_ids["patient_id_norm"].dropna().astype(str).tolist())
    gene_set: Set[str] = set(genes_list["gene_name"].dropna().astype(str).str.strip().tolist())

    # --- Create Gene nodes (chromosome optional, we only have symbol here)
    gene_records: List[dict] = []
    gene_id_by_name: Dict[str, str] = {}

    for g in sorted(set(expr[gene_col].astype(str).str.strip().tolist())):
        if not g or g == ".":
            continue
        if g not in gene_set:
            continue
        gid = md5_id({"gene_name": g})
        gene_id_by_name[g] = gid
        gene_records.append({"gene_id": gid, "gene_name": g})

    # --- Patient-Gene expression edges
    # Convert expression submatrix -> long format (patient_id_norm, gene_id, expression_value)
    print("Building patient-gene expression edges...")

    gene_col = "gene_name"  # dopo il rename della prima colonna

    expr_long = expr.melt(
        id_vars=[gene_col],
        value_vars=sample_cols,
        var_name="sample",
        value_name="expression_value",
    )

    # pulizia
    expr_long[gene_col] = expr_long[gene_col].astype(str).str.strip()
    expr_long["patient_id_norm"] = expr_long["sample"].astype(str).map(normalize_tcga_id)
    expr_long["expression_value"] = expr_long["expression_value"].astype(str).str.strip()

    # keep only valid
    expr_long = expr_long[expr_long["patient_id_norm"].isin(patient_norm_set)]
    expr_long = expr_long[expr_long[gene_col].isin(gene_id_by_name.keys())]
    expr_long = expr_long[(expr_long["expression_value"] != "") & (expr_long["expression_value"].str.lower() != "nan")]

    # add gene_id
    expr_long["gene_id"] = expr_long[gene_col].map(gene_id_by_name)

    patient_gene_edges = expr_long[["patient_id_norm", "gene_id", "expression_value"]].rename(
        columns={"patient_id_norm": "patient_id"}
    )

    # --- Gene-Mutation edges (from stage mapping)
    print("Building gene-mutation edges...")
    mut_gene_map["gene_name"] = mut_gene_map["gene_name"].astype(str).str.strip()
    mut_gene_map["unique_id"] = mut_gene_map["unique_id"].astype(str).str.strip()

    # Keep only genes that exist in our final gene list (393 genes in expr_submatrix)
    mg = mut_gene_map[mut_gene_map["gene_name"].isin(gene_id_by_name.keys())].copy()
    mg["gene_id"] = mg["gene_name"].map(gene_id_by_name)
    gene_mut_edges = mg[["gene_id", "unique_id"]].rename(columns={"unique_id": "mutation_id"}).drop_duplicates()

    # --- Write outputs
    genes_out = out_dir / "case2_genes.tsv"
    pg_out = out_dir / "case2_patient_gene_expression_edges.tsv"
    gm_out = out_dir / "case2_gene_mutation_edges.tsv"
    summary_out = out_dir / "case2_summary.txt"

    pd.DataFrame(gene_records).to_csv(genes_out, sep="\t", index=False)
    patient_gene_edges.to_csv(pg_out, sep="\t", index=False)
    gene_mut_edges.to_csv(gm_out, sep="\t", index=False)

    summary_lines = [
        f"Stage dir: {stage_dir}",
        f"Out dir: {out_dir}",
        "",
        f"Genes (nodes): {len(gene_records)}",
        f"Patient-Gene expression edges: {patient_gene_edges.shape[0]}",
        f"Gene-Mutation edges: {gene_mut_edges.shape[0]}",
        "",
        f"Expression submatrix shape: {expr.shape}",
        f"Unique patients in expr edges: {patient_gene_edges['patient_id'].nunique()}",
        f"Unique genes in expr edges: {patient_gene_edges['gene_id'].nunique()}",
        f"Unique mutations in gene-mutation edges: {gene_mut_edges['mutation_id'].nunique()}",
    ]
    summary_out.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Done.")
    print(f"Wrote: {genes_out}")
    print(f"Wrote: {pg_out}")
    print(f"Wrote: {gm_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()