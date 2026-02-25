import os
import pandas as pd
from dotenv import load_dotenv


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

MUTATIONS_PATH = os.getenv("MUTATIONS_PATH")
DISEASES_PATH = os.getenv("DISEASES_PATH")
PATIENTS_PATH = os.getenv("PATIENTS_PATH")
EDGE_PATIENT_DISEASE_PATH = os.getenv("EDGE_PATIENT_DISEASE_PATH")
EDGE_PATIENT_MUTATION_PATH = os.getenv("EDGE_PATIENT_MUTATION_PATH")

REQUIRED_VARS = {
    "MUTATIONS_PATH": MUTATIONS_PATH,
    "DISEASES_PATH": DISEASES_PATH,
    "PATIENTS_PATH": PATIENTS_PATH,
    "EDGE_PATIENT_DISEASE_PATH": EDGE_PATIENT_DISEASE_PATH,
    "EDGE_PATIENT_MUTATION_PATH": EDGE_PATIENT_MUTATION_PATH,
}

for var_name, value in REQUIRED_VARS.items():
    if not value:
        raise ValueError(f"Missing environment variable: {var_name}")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def read_csv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
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
    print(f"\nChecking {edge_name}")
    print(f"Rows: {len(edge_df)}")

    from_values = edge_df[col_from].dropna().astype(str).str.strip()
    to_values = edge_df[col_to].dropna().astype(str).str.strip()

    missing_from = sorted(set(from_values) - valid_from)
    missing_to = sorted(set(to_values) - valid_to)

    if missing_from:
        print(f"Invalid references in column '{col_from}': {len(missing_from)}")
        print(f"Examples: {missing_from[:20]}")
    else:
        print(f"All '{col_from}' values are valid.")

    if missing_to:
        print(f"Invalid references in column '{col_to}': {len(missing_to)}")
        print(f"Examples: {missing_to[:20]}")
    else:
        print(f"All '{col_to}' values are valid.")

    duplicates = edge_df.duplicated(subset=[col_from, col_to], keep=False)
    if duplicates.sum() > 0:
        print(f"Duplicate edges found: {duplicates.sum()}")
    else:
        print("No duplicate edges found.")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading datasets...")

    mutations_df = read_csv(MUTATIONS_PATH)
    diseases_df = read_csv(DISEASES_PATH)
    patients_df = read_csv(PATIENTS_PATH)
    edge_pd_df = read_csv(EDGE_PATIENT_DISEASE_PATH)
    edge_pm_df = read_csv(EDGE_PATIENT_MUTATION_PATH)

    print("Datasets loaded:")
    print(f"Mutations: {mutations_df.shape}")
    print(f"Diseases: {diseases_df.shape}")
    print(f"Patients: {patients_df.shape}")
    print(f"Patient-Disease edges: {edge_pd_df.shape}")
    print(f"Patient-Mutation edges: {edge_pm_df.shape}")

    # Detect ID columns
    patient_id_col = detect_id_column(patients_df, "patient")
    disease_id_col = detect_id_column(diseases_df, "disease")
    mutation_id_col = detect_id_column(mutations_df, "mutation")

    edge_pd_patient_col = detect_id_column(edge_pd_df, "patient")
    edge_pd_disease_col = detect_id_column(edge_pd_df, "disease")

    edge_pm_patient_col = detect_id_column(edge_pm_df, "patient")
    edge_pm_mutation_col = detect_id_column(edge_pm_df, "mutation")

    print("\nDetected ID columns:")
    print(f"Patient ID: {patient_id_col}")
    print(f"Disease ID: {disease_id_col}")
    print(f"Mutation ID: {mutation_id_col}")

    # Build ID sets
    patient_ids = build_id_set(patients_df, patient_id_col)
    disease_ids = build_id_set(diseases_df, disease_id_col)
    mutation_ids = build_id_set(mutations_df, mutation_id_col)

    print("\nUnique ID counts:")
    print(f"Patients: {len(patient_ids)}")
    print(f"Diseases: {len(disease_ids)}")
    print(f"Mutations: {len(mutation_ids)}")

    # Foreign key checks
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

    print("\nIntegrity check completed.")


if __name__ == "__main__":
    main()