"""
case2/gene_resolver.py

Resolver OFFLINE basato su mapping TSV (BioMart) + cache SQLite.

Scopo:
- Eliminare completamente le chiamate a Ensembl REST (niente 429/500/timeout)
- Risolvere:
  - SYMBOL -> ENSG
  - ENSG -> SYMBOL

Il TSV di input (BioMart) atteso:
  Gene stable ID <TAB> Gene name
  ENSG00000...   <TAB> TP53

NOTE:
- Il TSV può essere grande: lo importiamo in SQLite una sola volta (o quando cambia)
- SQLite è usato come indice per lookup O(1) senza caricare tutto in RAM
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Tuple
import csv
import hashlib
import logging
import os
import sqlite3
import time


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class OfflineGeneResolver:
    """
    Resolver offline basato su TSV BioMart + cache sqlite.

    - mapping_tsv: file TSV scaricato da BioMart (Gene stable ID, Gene name)
    - cache_db: DB sqlite creato in tmp_dir (es: output/tmp/gene_map.db)
    """
    logger: logging.Logger
    mapping_tsv: Path
    cache_db: Path

    def __post_init__(self):
        self.mapping_tsv = Path(self.mapping_tsv)
        self.cache_db = Path(self.cache_db)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)

        if not self.mapping_tsv.exists():
            raise FileNotFoundError(f"Mapping TSV non trovato: {self.mapping_tsv}")

        self._init_db()
        self._ensure_index_built()

    # ---------------- DB init / build ----------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta ("
                "k TEXT PRIMARY KEY, "
                "v TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS map ("
                "ensg TEXT PRIMARY KEY, "
                "symbol TEXT)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_map_symbol ON map(symbol)")
            conn.commit()

    def _get_meta(self, conn: sqlite3.Connection, key: str) -> Optional[str]:
        cur = conn.execute("SELECT v FROM meta WHERE k=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES (?,?)", (key, value))

    def _ensure_index_built(self) -> None:
        """
        Ricostruisce l’indice sqlite se:
        - DB vuoto
        - oppure il TSV è cambiato (sha256 differente)
        """
        tsv_sha = _sha256_file(self.mapping_tsv)

        with sqlite3.connect(self.cache_db) as conn:
            prev_sha = self._get_meta(conn, "mapping_sha256")
            count = conn.execute("SELECT COUNT(*) FROM map").fetchone()[0]

            if prev_sha == tsv_sha and count > 0:
                self.logger.info(f"Gene resolver offline: cache sqlite già pronta ({count} righe).")
                return

            self.logger.info("Gene resolver offline: costruzione/aggiornamento indice SQLite dal TSV (streaming)...")
            t0 = time.time()
            conn.execute("DELETE FROM map")

            inserted = 0
            with self.mapping_tsv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader, None)

                # header previsto: ["Gene stable ID", "Gene name"]
                # tolleriamo differenze minime
                if not header or len(header) < 2:
                    raise ValueError(f"Header TSV non valido in {self.mapping_tsv}")

                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    ensg = (row[0] or "").strip()
                    symbol = (row[1] or "").strip()

                    if not ensg or not ensg.startswith("ENSG"):
                        continue
                    if not symbol or symbol == ".":
                        continue

                    conn.execute("INSERT OR REPLACE INTO map(ensg, symbol) VALUES (?,?)", (ensg, symbol))
                    inserted += 1

                    if inserted % 200000 == 0:
                        conn.commit()
                        self.logger.info(f"Gene resolver offline: indicizzati {inserted} record...")

            self._set_meta(conn, "mapping_sha256", tsv_sha)
            conn.commit()
            dt = time.time() - t0
            self.logger.info(f"Gene resolver offline: indicizzazione completata | righe={inserted} | {dt:.1f}s")

    # ---------------- Public API ----------------

    @staticmethod
    def _strip_ensg_version(ensg: str) -> str:
        # ENSG00000123456.17 -> ENSG00000123456
        return ensg.split(".", 1)[0] if ensg else ensg

    def ensg_to_symbol(self, ensg: str) -> Optional[str]:
        ensg = self._strip_ensg_version((ensg or "").strip())
        if not ensg:
            return None
        with sqlite3.connect(self.cache_db) as conn:
            cur = conn.execute("SELECT symbol FROM map WHERE ensg=?", (ensg,))
            row = cur.fetchone()
            return row[0] if row else None

    def symbol_to_ensg(self, symbol: str) -> Optional[str]:
        symbol = (symbol or "").strip()
        if not symbol or symbol == ".":
            return None
        with sqlite3.connect(self.cache_db) as conn:
            cur = conn.execute("SELECT ensg FROM map WHERE symbol=? LIMIT 1", (symbol,))
            row = cur.fetchone()
            return row[0] if row else None

    def resolve_whitelist_symbols_to_ensg(self, symbols: Iterable[str]) -> Tuple[dict[str, str], set[str]]:
        """
        Ritorna:
        - mapping: SYMBOL -> ENSG (solo per quelli risolti)
        - missing: set di SYMBOL non trovati
        """
        mapping: dict[str, str] = {}
        missing: set[str] = set()

        # query per-symbol; con sqlite è veloce e non serve RAM
        i = 0
        for sym in symbols:
            i += 1
            ensg = self.symbol_to_ensg(sym)
            if ensg:
                mapping[sym] = ensg
            else:
                missing.add(sym)

            if i % 200 == 0:
                self.logger.info(f"Offline mapping progress: {i} | resolved={len(mapping)} | missing={len(missing)}")

        return mapping, missing