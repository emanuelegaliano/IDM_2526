"""
case2/gene_resolver.py

Client Ensembl REST con cache sqlite per risolvere:
- gene SYMBOL -> Ensembl Gene ID (ENSG...)

Fix:
- chiusura esplicita di HTTPError per evitare warning su finalizzazione HTTPResponse
- gestione Retry-After su HTTP 429
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import sqlite3
import json
import time
import urllib.parse
import urllib.request
import urllib.error


@dataclass
class EnsemblRestCachedClient:
    logger: logging.Logger
    cache_db_path: Path
    species: str = "homo_sapiens"
    timeout_sec: int = 15
    max_retries: int = 5
    backoff_base_sec: float = 0.75

    def __post_init__(self):
        self.cache_db_path = Path(self.cache_db_path)
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sym2ens ("
                "symbol TEXT PRIMARY KEY, "
                "ensembl_id TEXT, "
                "payload TEXT)"
            )
            conn.commit()

    # -----------------------------
    # Public API
    # -----------------------------

    def symbol_to_ensembl_gene_id(self, symbol: str) -> Optional[str]:
        symbol = (symbol or "").strip()
        if not symbol or symbol == ".":
            return None

        cached = self._get_cached_sym2ens(symbol)
        if cached is not None:
            return cached  # può essere None

        payload = self._fetch_xrefs_symbol(symbol)
        ensg = self._extract_first_ensg(payload)

        self._set_cached_sym2ens(symbol, ensg, payload)
        return ensg

    # -----------------------------
    # Cache helpers
    # -----------------------------

    def _get_cached_sym2ens(self, symbol: str) -> Optional[Optional[str]]:
        with sqlite3.connect(self.cache_db_path) as conn:
            cur = conn.execute("SELECT ensembl_id FROM sym2ens WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]  # può essere None

    def _set_cached_sym2ens(self, symbol: str, ensembl_id: Optional[str], payload: object):
        payload_str = json.dumps(payload) if payload is not None else None
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sym2ens(symbol, ensembl_id, payload) VALUES (?, ?, ?)",
                (symbol, ensembl_id, payload_str),
            )
            conn.commit()

    # -----------------------------
    # Ensembl REST calls
    # -----------------------------

    def _fetch_xrefs_symbol(self, symbol: str) -> object:
        """
        GET /xrefs/symbol/{species}/{symbol}?content-type=application/json
        Ritorna JSON (lista di xrefs).
        """
        sym_enc = urllib.parse.quote(symbol, safe="")
        url = f"https://rest.ensembl.org/xrefs/symbol/{self.species}/{sym_enc}?content-type=application/json"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "case2-pipeline/1.0",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    data = resp.read().decode("utf-8")
                    return json.loads(data)

            except urllib.error.HTTPError as e:
                # Importante: HTTPError è anche un file-like -> chiudere sempre.
                try:
                    code = e.code
                    retry_after = e.headers.get("Retry-After") if getattr(e, "headers", None) else None

                    if code in (429, 500, 502, 503, 504):
                        if retry_after:
                            try:
                                wait = float(retry_after)
                            except Exception:
                                wait = self.backoff_base_sec * (2 ** (attempt - 1))
                        else:
                            wait = self.backoff_base_sec * (2 ** (attempt - 1))

                        self.logger.warning(
                            f"Ensembl REST HTTP {code} su symbol={symbol} (attempt {attempt}/{self.max_retries}) "
                            f"-> retry tra {wait:.2f}s"
                        )
                        last_err = e
                        time.sleep(wait)
                        continue

                    # 400/404 ecc: non retryare
                    self.logger.warning(f"Ensembl REST HTTP {code} per symbol={symbol} (no-retry)")
                    return []

                finally:
                    # chiusura esplicita per evitare warning in GC/finalizer
                    try:
                        e.close()
                    except Exception:
                        pass

            except urllib.error.URLError as e:
                wait = self.backoff_base_sec * (2 ** (attempt - 1))
                self.logger.warning(
                    f"Ensembl REST URLError su symbol={symbol} (attempt {attempt}/{self.max_retries}) "
                    f"-> retry tra {wait:.2f}s | {e}"
                )
                time.sleep(wait)
                last_err = e
                continue

            except Exception as e:
                wait = self.backoff_base_sec * (2 ** (attempt - 1))
                self.logger.warning(
                    f"Ensembl REST error su symbol={symbol} (attempt {attempt}/{self.max_retries}) "
                    f"-> retry tra {wait:.2f}s | {e}"
                )
                time.sleep(wait)
                last_err = e
                continue

        self.logger.error(f"Ensembl REST fallito su symbol={symbol} dopo {self.max_retries} tentativi: {last_err}")
        return []

    @staticmethod
    def _extract_first_ensg(payload: object) -> Optional[str]:
        if not isinstance(payload, list):
            return None

        for item in payload:
            if not isinstance(item, dict):
                continue
            _id = item.get("id")
            _type = item.get("type")
            if _type == "gene" and isinstance(_id, str) and _id.startswith("ENSG"):
                return _id

        for item in payload:
            if isinstance(item, dict):
                _id = item.get("id")
                if isinstance(_id, str) and _id.startswith("ENSG"):
                    return _id

        return None