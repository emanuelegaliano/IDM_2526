# src/utilities/fs.py
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create directory if missing and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p