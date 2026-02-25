#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Valore non valido per --verbose. Usa true/false.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline progetto (Case 1 / Case 2)")
    p.add_argument("--case", required=True, choices=["1", "2"], help="Seleziona il case: 1 oppure 2")
    p.add_argument(
        "--verbose",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Abilita logging verboso (true/false). Esempi: --verbose true oppure --verbose",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Rendi importabile src/ (che contiene i package case1/case2)
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))

    if args.case == "1":
        # Placeholder: quando avrai src/case1 con __init__.py e run_pipeline, sarà identico a case2.
        print("Case 1 non è ancora implementato.")
        return 3
    elif args.case == "2":
        try:
            import case2
        except Exception as e:
            print(f"Errore importando case2: {e}", file=sys.stderr)
            return 2

        try:
            case2.run_pipeline(verbose=args.verbose)
            return 0
        except Exception as e:
            print(f"Errore durante l'esecuzione della pipeline Case 2: {e}", file=sys.stderr)
            return 1

    print(f"Case sconosciuto: {args.case}", file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())