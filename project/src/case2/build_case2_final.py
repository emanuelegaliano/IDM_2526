from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from . import cfg

log = cfg.get_logger("build_case2_final")


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    return df


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def autodetect_col(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    low = label.lower()
    fallback = [c for c in df.columns if low in c.lower() and "id" in c.lower()]
    if fallback:
        return fallback[0]
    generic = [c for c in df.columns if "id" in c.lower()]
    if generic:
        return generic[0]
    raise ValueError(f"Cannot detect column for {label}. Candidates={candidates}. Columns={list(df.columns)}")


def resolve_or_detect(df: pd.DataFrame, override: Optional[str], candidates: List[str], label: str) -> str:
    if override:
        if override in df.columns:
            return override
        log.warning(
            "Configured column '%s' not found for %s. Falling back to autodetect. Columns=%s",
            override, label, list(df.columns)
        )
    return autodetect_col(df, candidates, label)


def tcga_case3(barcode: str) -> str:
    s = str(barcode).strip()
    if not s:
        return s
    parts = s.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return s


# ----------------------------
# Load base nodes/edges (small)
# ----------------------------
def load_patients() -> pd.DataFrame:
    df = read_table(cfg.PATIENTS_PATH)
    patient_id_col = resolve_or_detect(
        df,
        getattr(cfg, "PATIENT_ID_COL", None),
        candidates=["patient_id", "case_id", "sample_id", "id"],
        label="patient",
    )
    if patient_id_col != "patient_id":
        df = df.rename(columns={patient_id_col: "patient_id"})

    df["patient_id"] = df["patient_id"].astype(str).str.strip().map(tcga_case3)
    df = df.dropna(subset=["patient_id"])
    df = df[df["patient_id"] != ""]
    df = df.drop_duplicates(subset=["patient_id"])
    return df


def load_diseases() -> pd.DataFrame:
    df = read_table(cfg.DISEASES_PATH)
    disease_id_col = resolve_or_detect(
        df,
        getattr(cfg, "DISEASE_ID_COL", None),
        candidates=["disease_id", "id"],
        label="disease",
    )
    if disease_id_col != "disease_id":
        df = df.rename(columns={disease_id_col: "disease_id"})

    df["disease_id"] = df["disease_id"].astype(str).str.strip()
    df = df.dropna(subset=["disease_id"])
    df = df[df["disease_id"] != ""]
    df = df.drop_duplicates(subset=["disease_id"])
    return df


def load_edges_patient_disease() -> pd.DataFrame:
    df = read_table(cfg.EDGE_PATIENT_DISEASE_PATH)

    pcol = resolve_or_detect(
        df,
        getattr(cfg, "EDGE_PD_PATIENT_COL", None),
        candidates=["patient_id", "case_id", "sample_id", "id"],
        label="patient",
    )
    dcol = resolve_or_detect(
        df,
        getattr(cfg, "EDGE_PD_DISEASE_COL", None),
        candidates=["disease_id", "id"],
        label="disease",
    )

    if pcol != "patient_id":
        df = df.rename(columns={pcol: "patient_id"})
    if dcol != "disease_id":
        df = df.rename(columns={dcol: "disease_id"})

    df["patient_id"] = df["patient_id"].astype(str).str.strip().map(tcga_case3)
    df["disease_id"] = df["disease_id"].astype(str).str.strip()
    df = df[["patient_id", "disease_id"]].dropna()
    df = df[(df["patient_id"] != "") & (df["disease_id"] != "")]
    df = df.drop_duplicates()
    return df


def load_mutations_all_chr() -> pd.DataFrame:
    dfs = []
    for p in cfg.MUTATION_FILES:
        df = read_table(p)

        mut_id_col = getattr(cfg, "MUTATION_ID_COL", None) or autodetect_col(
            df, ["mutation_id", "unique_id", "id", "variant_id"], "mutation"
        )
        if mut_id_col != "mutation_id":
            df = df.rename(columns={mut_id_col: "mutation_id"})

        df["mutation_id"] = df["mutation_id"].astype(str).str.strip()
        df = df.dropna(subset=["mutation_id"])
        df = df[df["mutation_id"] != ""]
        df = df.drop_duplicates(subset=["mutation_id"])

        dfs.append(df)
        log.info("Mutations loaded: %s | rows=%s", Path(p).name, f"{len(df):,}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ----------------------------
# Load intermediate outputs (small)
# ----------------------------
def load_intermediate_genes() -> pd.DataFrame:
    p = Path(cfg.GENERATED_DIR) / "genes.tsv"
    df = read_table(p)
    needed = ["gene_id", "gene_name"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Intermediate genes.tsv missing '{c}'. Columns={list(df.columns)}")

    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    df["gene_name"] = df["gene_name"].astype(str).str.strip()
    df = df.dropna(subset=["gene_id"])
    df = df[df["gene_id"] != ""]
    df = df.drop_duplicates(subset=["gene_id"])
    return df[["gene_id", "gene_name"]]


def load_intermediate_patient_gene_edges() -> pd.DataFrame:
    p = Path(cfg.GENERATED_DIR) / "patient_gene_expression_edges.tsv"
    df = read_table(p)
    needed = ["patient_id", "gene_id", "expression_value"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Intermediate patient_gene_expression_edges.tsv missing '{c}'. Columns={list(df.columns)}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip().map(tcga_case3)
    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    df["expression_value"] = df["expression_value"].astype(str).str.strip()

    df = df.dropna(subset=["patient_id", "gene_id", "expression_value"])
    df = df[(df["patient_id"] != "") & (df["gene_id"] != "") & (df["expression_value"] != "")]
    df = df.drop_duplicates()
    return df[["patient_id", "gene_id", "expression_value"]]


def load_intermediate_gene_mut_edges() -> pd.DataFrame:
    p = Path(cfg.GENERATED_DIR) / "gene_mutation_edges.tsv"
    df = read_table(p)
    needed = ["gene_id", "mutation_id"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Intermediate gene_mutation_edges.tsv missing '{c}'. Columns={list(df.columns)}")

    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    df["mutation_id"] = df["mutation_id"].astype(str).str.strip()
    df = df.dropna(subset=["gene_id", "mutation_id"])
    df = df[(df["gene_id"] != "") & (df["mutation_id"] != "")]
    df = df.drop_duplicates()
    return df[["gene_id", "mutation_id"]]


# ----------------------------
# STREAMING: Patient-Mutation edges (huge)
# ----------------------------
def stream_write_patient_mutation_edges(
    in_paths: List[Path],
    out_path: Path,
    valid_patients: Set[str],
    valid_mutations: Optional[Set[str]] = None,
    chunksize: int = 500_000,
) -> int:
    """
    Stream huge patient-mutation edges without keeping them in RAM.
    Always writes output with columns in the order: patient_id, mutation_id
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always start from scratch (avoid duplicated header)
    if out_path.exists():
        out_path.unlink()

    wrote_header = False
    total_written = 0

    for p in in_paths:
        p = Path(p)
        log.info("Streaming patient-mutation edges from %s ...", p.name)

        head = pd.read_csv(p, sep="\t", nrows=0)
        cols = [c.strip() for c in head.columns]
        df0 = pd.DataFrame(columns=cols)

        pcol = resolve_or_detect(
            df0,
            getattr(cfg, "EDGE_PM_PATIENT_COL", None),
            candidates=["patient_id", "case_id", "sample_id", "id"],
            label="patient",
        )
        mcol = resolve_or_detect(
            df0,
            getattr(cfg, "EDGE_PM_MUTATION_COL", None),
            candidates=["mutation_id", "unique_id", "variant_id", "id"],
            label="mutation",
        )

        reader = pd.read_csv(
            p,
            sep="\t",
            usecols=[pcol, mcol],
            dtype=str,
            low_memory=False,
            chunksize=chunksize,
        )

        for i, chunk in enumerate(reader, start=1):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk.rename(columns={pcol: "patient_id", mcol: "mutation_id"})

            chunk["patient_id"] = chunk["patient_id"].astype(str).str.strip().map(tcga_case3)
            chunk["mutation_id"] = chunk["mutation_id"].astype(str).str.strip()

            chunk = chunk.dropna(subset=["patient_id", "mutation_id"])
            chunk = chunk[(chunk["patient_id"] != "") & (chunk["mutation_id"] != "")]

            chunk = chunk[chunk["patient_id"].isin(valid_patients)]
            if valid_mutations is not None:
                chunk = chunk[chunk["mutation_id"].isin(valid_mutations)]

            chunk = chunk.drop_duplicates(subset=["patient_id", "mutation_id"])

            if not chunk.empty:
                chunk = chunk[["patient_id", "mutation_id"]]  # force column order
                chunk.to_csv(out_path, sep="\t", index=False, mode="a", header=not wrote_header)
                wrote_header = True
                total_written += len(chunk)

            if i % 5 == 0:
                log.info("... chunks processed: %s | total edges written so far: %s", i, f"{total_written:,}")

    return total_written


# ----------------------------
# Final build
# ----------------------------
def run() -> None:
    cfg.validate_paths()
    ensure_dir(Path(cfg.FINAL_DIR))

    final_dir = Path(cfg.FINAL_DIR)
    log.info("Building final Case2 export in: %s", final_dir)

    patients_df = load_patients()
    diseases_df = load_diseases()
    edge_pd_df = load_edges_patient_disease()

    mutations_df = load_mutations_all_chr()

    genes_df = load_intermediate_genes()
    edge_pg_df = load_intermediate_patient_gene_edges()
    edge_gm_df = load_intermediate_gene_mut_edges()

    # enforce edge column order
    edge_pd_df = edge_pd_df[["patient_id", "disease_id"]]
    edge_pg_df = edge_pg_df[["patient_id", "gene_id", "expression_value"]]
    edge_gm_df = edge_gm_df[["gene_id", "mutation_id"]]

    valid_patients = set(patients_df["patient_id"].astype(str))
    valid_mutations = set(mutations_df["mutation_id"].astype(str)) if not mutations_df.empty else None

    out_edge_pm = final_dir / "edges_patient_mutation.tsv"
    pm_written = stream_write_patient_mutation_edges(
        in_paths=[Path(p) for p in cfg.EDGE_PATIENT_MUTATION_FILES],
        out_path=out_edge_pm,
        valid_patients=valid_patients,
        valid_mutations=valid_mutations,
        chunksize=500_000,
    )
    log.info("Patient-Mutation edges written (streaming): %s", f"{pm_written:,}")

    # Load only the two columns back (still big, but limited)
    edge_pm_df = (
        pd.read_csv(out_edge_pm, sep="\t", dtype=str, usecols=["patient_id", "mutation_id"], low_memory=False)
        if out_edge_pm.exists()
        else pd.DataFrame(columns=["patient_id", "mutation_id"])
    )

    referenced_pat = (
        set(edge_pd_df["patient_id"].astype(str))
        | set(edge_pm_df["patient_id"].astype(str))
        | set(edge_pg_df["patient_id"].astype(str))
    )
    patients_df = patients_df[patients_df["patient_id"].astype(str).isin(referenced_pat)].copy()
    log.info("Filtered patients by referenced edges -> %s", f"{len(patients_df):,}")

    referenced_mut = set(edge_pm_df["mutation_id"].astype(str)) | set(edge_gm_df["mutation_id"].astype(str))
    if not mutations_df.empty:
        mutations_df = mutations_df[mutations_df["mutation_id"].astype(str).isin(referenced_mut)].copy()
        log.info("Filtered mutations by referenced edges -> %s", f"{len(mutations_df):,}")

    # Output paths
    out_patients = final_dir / "patients_nodes.tsv"
    out_diseases = final_dir / "diseases_nodes.tsv"
    out_mutations = final_dir / "mutations_nodes.tsv"
    out_genes = final_dir / "genes_nodes.tsv"

    out_edge_pd = final_dir / "edges_patient_disease.tsv"
    out_edge_pg = final_dir / "edges_patient_gene_expression.tsv"
    out_edge_gm = final_dir / "edges_gene_mutation.tsv"

    # Write nodes
    write_tsv(patients_df.drop_duplicates(subset=["patient_id"]), out_patients)
    write_tsv(diseases_df.drop_duplicates(subset=["disease_id"]), out_diseases)
    write_tsv(mutations_df.drop_duplicates(subset=["mutation_id"]), out_mutations)
    write_tsv(genes_df.drop_duplicates(subset=["gene_id"]), out_genes)

    # Write edges
    write_tsv(edge_pd_df.drop_duplicates(), out_edge_pd)
    write_tsv(edge_pg_df.drop_duplicates(), out_edge_pg)
    write_tsv(edge_gm_df.drop_duplicates(), out_edge_gm)

    log.info("Wrote nodes: %s", out_patients)
    log.info("Wrote nodes: %s", out_diseases)
    log.info("Wrote nodes: %s", out_mutations)
    log.info("Wrote nodes: %s", out_genes)

    log.info("Wrote edges: %s", out_edge_pd)
    log.info("Wrote edges: %s", out_edge_pm)
    log.info("Wrote edges: %s", out_edge_pg)
    log.info("Wrote edges: %s", out_edge_gm)

    log.info("Done.")


def main() -> None:
    try:
        run()
    except Exception as e:
        log.exception("Error: %s", e)


if __name__ == "__main__":
    main()