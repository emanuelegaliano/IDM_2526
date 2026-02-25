#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "src" / "datasets"

EXPR_PATH = DATASETS_DIR / "TCGA-OV.star_fpkm.tsv"
OUT_PATH = DATASETS_DIR / "ensembl_to_symbol.tsv"

ENSEMBL_LOOKUP_POST = "https://rest.ensembl.org/lookup/id"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "IDM-project/1.0"
}


def strip_version(x: str) -> str:
    x = str(x).strip()
    if "." in x:
        return x.split(".", 1)[0]
    return x


def read_ensembl_ids(expr_path: Path) -> List[str]:
    # Read only first column (gene ids)
    df = pd.read_csv(expr_path, sep="\t", usecols=[0], dtype=str, low_memory=False)
    gene_col = df.columns[0]
    ids = df[gene_col].dropna().astype(str).map(strip_version)
    ids = ids[ids != ""]
    # keep only ENSG*
    ids = ids[ids.str.startswith("ENSG")]
    return ids.drop_duplicates().tolist()


def chunks(lst: List[str], n: int) -> List[List[str]]: # type: ignore
    for i in range(0, len(lst), n):
        yield lst[i:i + n] # type: ignore


def lookup_batch(ids: List[str], sleep_s: float = 0.2) -> Dict[str, Any]:
    payload = {"ids": ids}
    r = requests.post(ENSEMBL_LOOKUP_POST, headers=HEADERS, data=json.dumps(payload), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Ensembl API error {r.status_code}: {r.text[:200]}")
    time.sleep(sleep_s)
    return r.json()


def main() -> None:
    if not EXPR_PATH.exists():
        raise FileNotFoundError(f"Expression file not found: {EXPR_PATH}")

    ensg = read_ensembl_ids(EXPR_PATH)
    print(f"Found {len(ensg):,} unique ENSG IDs in expression matrix.")

    rows = []
    batch_size = 1000  # Ensembl REST bulk lookup supports batches; 1000 is a safe size
    for b, ids in enumerate(chunks(ensg, batch_size), start=1):
        data = lookup_batch(ids)
        for ensembl_id, obj in data.items():
            if not obj:
                continue
            symbol = obj.get("display_name")
            if symbol:
                rows.append((ensembl_id, symbol))

        if b % 5 == 0:
            print(f"Processed batches: {b}")

    out = pd.DataFrame(rows, columns=["ensembl_id", "gene_symbol"]).drop_duplicates(subset=["ensembl_id"])
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote mapping: {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()