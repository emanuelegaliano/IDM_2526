import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from dotenv import load_dotenv


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


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


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
        "MUTATIONS_PATH": os.getenv("MUTATIONS_PATH"),
        "PATIENTS_PATH": os.getenv("PATIENTS_PATH"),
        "EDGE_PATIENT_MUTATION_PATH": os.getenv("EDGE_PATIENT_MUTATION_PATH"),
        "EXPRESSION_MATRIX_PATH": os.getenv("EXPRESSION_MATRIX_PATH"),
        "CASE2_STAGE_DIR": os.getenv("CASE2_STAGE_DIR", "./case2_stage"),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise ValueError(f"Missing env vars: {missing}")
    print(f"Loaded .env from: {loaded_from}")
    return cfg


def extract_genes_from_gene_refgene(series: pd.Series) -> Set[str]:
    """
    Gene.refGene può essere:
      - "DUXAP8;CCT8L2"
      - "."
      - vuoto
    """
    genes: Set[str] = set()
    s = series.dropna().astype(str)
    for v in s:
        v = v.strip()
        if not v or v == ".":
            continue
        # split multi-gene
        for g in v.split(";"):
            g = g.strip()
            if g and g != ".":
                genes.add(g)
    return genes


def main():
    cfg = load_env()
    stage_dir = Path(cfg["CASE2_STAGE_DIR"])
    stage_dir.mkdir(parents=True, exist_ok=True)

    report_lines: List[str] = []

    # --- Patients
    patients = read_tsv(cfg["PATIENTS_PATH"])
    if "patient_id" not in patients.columns:
        raise ValueError(f"patients.tsv must have patient_id. Found: {list(patients.columns)}")
    patients["patient_id_norm"] = patients["patient_id"].astype(str).map(normalize_tcga_id)
    patient_norm_set = set(patients["patient_id_norm"].dropna().unique().tolist())

    report_lines.append(f"Patients raw: {patients.shape[0]}")
    report_lines.append(f"Patients normalized unique: {len(patient_norm_set)}")

    pd.DataFrame({"patient_id_norm": sorted(patient_norm_set)}).to_csv(
        stage_dir / "case2_patients_ids.tsv", sep="\t", index=False
    )

    # --- Mutations: gene list from chr22 using Gene.refGene
    mut = read_tsv(cfg["MUTATIONS_PATH"])
    if "unique_id" not in mut.columns:
        raise ValueError("chr_22.tsv must have unique_id column")
    if "Gene.refGene" not in mut.columns:
        raise ValueError("chr_22.tsv must have Gene.refGene column (you have it)")

    genes_from_mut = extract_genes_from_gene_refgene(mut["Gene.refGene"])
    report_lines.append(f"Genes from chr22 Gene.refGene: {len(genes_from_mut)}")

    pd.DataFrame({"gene_name": sorted(genes_from_mut)}).to_csv(
        stage_dir / "case2_gene_list.tsv", sep="\t", index=False
    )

    # --- Build mutation->gene map (explode Gene.refGene)
    mm = mut[["unique_id", "Gene.refGene"]].dropna().copy()
    mm["Gene.refGene"] = mm["Gene.refGene"].astype(str).str.strip()
    mm = mm[mm["Gene.refGene"].ne("") & mm["Gene.refGene"].ne(".")]

    mm["gene_name"] = mm["Gene.refGene"].str.split(";")
    mm = mm.explode("gene_name")
    mm["gene_name"] = mm["gene_name"].astype(str).str.strip()
    mm = mm[mm["gene_name"].ne("") & mm["gene_name"].ne(".")]

    mm = mm[["unique_id", "gene_name"]].drop_duplicates()
    mm.to_csv(stage_dir / "case2_chr22_mutations_gene_map.tsv", sep="\t", index=False)
    report_lines.append(f"Mutation->Gene rows (dedup): {mm.shape[0]}")

    # --- Expression matrix: detect overlap samples + filter columns
    expr_path = cfg["EXPRESSION_MATRIX_PATH"]
    header = pd.read_csv(expr_path, sep="\t", nrows=0).columns.tolist()
    if len(header) < 2:
        raise ValueError("Expression matrix invalid (expected sample + at least 1 column)")

    row_id_col = header[0]
    sample_cols = header[1:]

    # normalize sample ids and keep those that match patients
    kept_samples: List[str] = []
    for s in sample_cols:
        if normalize_tcga_id(s) in patient_norm_set:
            kept_samples.append(s)

    report_lines.append(f"Expression samples total: {len(sample_cols)}")
    report_lines.append(f"Expression samples kept (overlap with patients): {len(kept_samples)}")

    if not kept_samples:
        raise ValueError("No overlapping samples between expression matrix and patients.tsv after normalization")

    # filter rows to only genes of interest (gene-level: row_id should be gene symbol here)
    # stream chunks to avoid loading whole file
    genes_wanted = genes_from_mut
    usecols = [row_id_col] + kept_samples

    out_expr = stage_dir / "case2_expression_submatrix.tsv"
    wrote_header = False
    kept_rows = 0

    for chunk in pd.read_csv(expr_path, sep="\t", dtype=str, usecols=usecols, chunksize=5000, low_memory=False):
        chunk[row_id_col] = chunk[row_id_col].astype(str).str.strip()
        sub = chunk[chunk[row_id_col].isin(genes_wanted)]
        if sub.empty:
            continue
        sub.to_csv(out_expr, sep="\t", index=False, mode="a", header=not wrote_header)
        wrote_header = True
        kept_rows += sub.shape[0]

    report_lines.append(f"Expression rows kept (genes in chr22 gene list): {kept_rows}")

    if kept_rows == 0:
        report_lines.append("WARNING: 0 expression rows kept. Likely row ids in expression matrix are not gene symbols.")
        report_lines.append("Check first column values (head of expression file) and compare with gene list.")
    else:
        report_lines.append(f"Wrote expression submatrix: {out_expr}")

    # write report
    (stage_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"Stage outputs written to: {stage_dir}")


if __name__ == "__main__":
    main()