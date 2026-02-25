from pathlib import Path
import pandas as pd
from . import cfg

log = cfg.get_logger("check_dataset")


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".tsv":
        sep = "\t"
        df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    elif ext == ".csv":
        sep = ","
        df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    else:
        # fallback: try TSV then CSV
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        except Exception:
            df = pd.read_csv(path, sep=",", dtype=str, low_memory=False)

    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    return df


def detect_id_column(df, entity_name):
    candidates = [
        f"{entity_name}_id",
        "id",
        f"{entity_name}ID",
        f"{entity_name}Id",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    fallback = [c for c in df.columns if entity_name in c.lower() and "id" in c.lower()]
    if fallback:
        return fallback[0]

    generic = [c for c in df.columns if "id" in c.lower()]
    if generic:
        return generic[0]

    raise ValueError(
        f"Unable to detect ID column for '{entity_name}'. Available columns: {list(df.columns)}"
    )


def build_id_set(df, col_name):
    values = df[col_name].dropna().astype(str).str.strip()
    values = values[values != ""]
    return set(values.tolist())


def check_foreign_key(edge_df, col_from, valid_from, col_to, valid_to, edge_name):
    log.info("Checking %s", edge_name)
    log.info("Rows: %s", f"{len(edge_df):,}")

    from_values = edge_df[col_from].dropna().astype(str).str.strip()
    from_values = from_values[from_values != ""]
    to_values = edge_df[col_to].dropna().astype(str).str.strip()
    to_values = to_values[to_values != ""]

    missing_from = sorted(set(from_values) - valid_from)
    missing_to = sorted(set(to_values) - valid_to)

    if missing_from:
        log.warning("Invalid references in column '%s': %s", col_from, f"{len(missing_from):,}")
        log.warning("Examples (up to 20): %s", missing_from[:20])
    else:
        log.info("All '%s' values are valid.", col_from)

    if missing_to:
        log.warning("Invalid references in column '%s': %s", col_to, f"{len(missing_to):,}")
        log.warning("Examples (up to 20): %s", missing_to[:20])
    else:
        log.info("All '%s' values are valid.", col_to)

    duplicates = edge_df.duplicated(subset=[col_from, col_to], keep=False)
    dup_count = int(duplicates.sum())
    if dup_count > 0:
        log.warning("Duplicate edges found (same pair %s,%s): %s", col_from, col_to, f"{dup_count:,}")
    else:
        log.info("No duplicate edges found (same pair).")


def run():
    log.info("Loading datasets...")

    diseases_df = read_table(cfg.DISEASES_PATH)
    patients_df = read_table(cfg.PATIENTS_PATH)
    edge_pd_df = read_table(cfg.EDGE_PATIENT_DISEASE_PATH)

    log.info("Loading mutation node files: %s", len(cfg.MUTATION_FILES))
    mutations_dfs = [read_table(p) for p in cfg.MUTATION_FILES]
    mutations_df = pd.concat(mutations_dfs, ignore_index=True) if mutations_dfs else pd.DataFrame()

    log.info("Loading patient-mutation edge files: %s", len(cfg.EDGE_PATIENT_MUTATION_FILES))
    edge_pm_dfs = [read_table(p) for p in cfg.EDGE_PATIENT_MUTATION_FILES]
    edge_pm_df = pd.concat(edge_pm_dfs, ignore_index=True) if edge_pm_dfs else pd.DataFrame()

    log.info("Datasets loaded (rows, cols):")
    log.info("Diseases: %s", diseases_df.shape)
    log.info("Patients: %s", patients_df.shape)
    log.info("Patient-Disease edges: %s", edge_pd_df.shape)
    log.info("Mutations (all selected chr): %s", mutations_df.shape)
    log.info("Patient-Mutation edges (all selected chr): %s", edge_pm_df.shape)

    if mutations_df.empty:
        raise ValueError("No mutation files loaded. Check cfg.MUTATION_FILES and CHROMOSOMES.")
    if edge_pm_df.empty:
        raise ValueError("No patient-mutation edge files loaded. Check cfg.EDGE_PATIENT_MUTATION_FILES and CHROMOSOMES.")

    # Detect ID columns
    patient_id_col = detect_id_column(patients_df, "patient")
    disease_id_col = detect_id_column(diseases_df, "disease")
    mutation_id_col = detect_id_column(mutations_df, "mutation")

    edge_pd_patient_col = detect_id_column(edge_pd_df, "patient")
    edge_pd_disease_col = detect_id_column(edge_pd_df, "disease")

    edge_pm_patient_col = detect_id_column(edge_pm_df, "patient")
    edge_pm_mutation_col = detect_id_column(edge_pm_df, "mutation")

    log.info("Detected ID columns:")
    log.info("Patient ID: %s", patient_id_col)
    log.info("Disease ID: %s", disease_id_col)
    log.info("Mutation ID: %s", mutation_id_col)
    log.info("Edge Patient-Disease: %s -> %s", edge_pd_patient_col, edge_pd_disease_col)
    log.info("Edge Patient-Mutation: %s -> %s", edge_pm_patient_col, edge_pm_mutation_col)

    # Build ID sets
    patient_ids = build_id_set(patients_df, patient_id_col)
    disease_ids = build_id_set(diseases_df, disease_id_col)
    mutation_ids = build_id_set(mutations_df, mutation_id_col)

    log.info("Unique ID counts:")
    log.info("Patients: %s", f"{len(patient_ids):,}")
    log.info("Diseases: %s", f"{len(disease_ids):,}")
    log.info("Mutations: %s", f"{len(mutation_ids):,}")

    # FK checks
    check_foreign_key(
        edge_pd_df,
        edge_pd_patient_col,
        patient_ids,
        edge_pd_disease_col,
        disease_ids,
        "Patient-Disease edges"
    )

    check_foreign_key(
        edge_pm_df,
        edge_pm_patient_col,
        patient_ids,
        edge_pm_mutation_col,
        mutation_ids,
        "Patient-Mutation edges"
    )

    # Extra: mutations present but never referenced by any selected-chr edges
    edge_mut_ids = set(edge_pm_df[edge_pm_mutation_col].dropna().astype(str).str.strip())
    orphan_mut = sorted(mutation_ids - edge_mut_ids)
    log.info("Extra check: mutations not referenced by any patient-mutation edge: %s", f"{len(orphan_mut):,}")
    if orphan_mut:
        log.info("Examples (up to 20): %s", orphan_mut[:20])

    log.info("Integrity check completed.")


def main():
    try:
        cfg.validate_paths()
        run()
    except Exception as e:
        log.exception("Error during integrity check: %s", e)


if __name__ == "__main__":
    main()