import sqlite3
import csv
from pathlib import Path
from typing import Optional

class EnsemblAPIResolver:
    def __init__(self, db_path: str, mapping_tsv: Optional[Path] = None):
        self.db_path = db_path
        self._init_db()
        if mapping_tsv and mapping_tsv.exists():
            self._load_from_tsv(mapping_tsv)

    def _init_db(self):
        query = "CREATE TABLE IF NOT EXISTS gene_mapping (ensembl_id TEXT PRIMARY KEY, gene_symbol TEXT);"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query)

    def _load_from_tsv(self, tsv_path: Path):
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            with sqlite3.connect(self.db_path) as conn:
                query = "INSERT OR IGNORE INTO gene_mapping (ensembl_id, gene_symbol) VALUES (?, ?);"
                data = []
                for row in reader:
                    # Adattamento nomi colonne file 
                    eid = row.get('ensembl_id', '').split('.')[0]
                    sym = row.get('gene_symbol', '')
                    if eid and sym:
                        data.append((eid, sym))
                conn.executemany(query, data)

    def get_symbol(self, ensembl_id: str) -> Optional[str]:
        clean_id = ensembl_id.split('.')[0]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT gene_symbol FROM gene_mapping WHERE ensembl_id = ?;", (clean_id,))
            row = cursor.fetchone()
            return row[0] if row else None