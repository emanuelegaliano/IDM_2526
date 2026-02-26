"""
case2 package

Public API:
- Case2Config: configurazione della pipeline
- run_case2: esegue la pipeline del caso 2

Opzionali:
- validate_case2_outputs: validazione standalone (la pipeline la chiama se config.validate=True)
"""

from .config import Case2Config
from .pipeline import run_case2
from .validate_outputs import validate_case2_outputs

__all__ = [
    "Case2Config",
    "run_case2",
    "validate_case2_outputs",
]

__version__ = "0.1.0"