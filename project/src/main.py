from __future__ import annotations

import argparse
from pathlib import Path

from case2 import Case2Config, run_case2


def run_case2_from_cli(*, base_dir: Path, verbose: bool, log_file: bool, workers: int) -> None:
    """
    Esegue il caso 2 con output in: base_dir / "output"
    """
    datasets_root = base_dir / "datasets"
    output_dir = base_dir / "output"
    tmp_dir = output_dir / "tmp"

    cfg = Case2Config(
        datasets_root=datasets_root,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        workers=workers,
        verbose=verbose,          # <--- nuovo flag: console on/off
        log_to_file=log_file,     # <--- file on/off
        log_level="INFO",
        gene_id_mode="ensembl_api_cache",
        chunk_size_rows=10_000,
        validate=False,
    )

    cfg.logger.info("Avvio Caso 2 (CLI)")
    cfg.logger.info(f"datasets_root={cfg.datasets_root}")
    cfg.logger.info(f"output_dir={cfg.output_dir}")
    cfg.logger.info(f"tmp_dir={cfg.tmp_dir}")
    cfg.logger.info(f"workers={cfg.workers}")
    cfg.logger.info(f"verbose={cfg.verbose} log_to_file={cfg.log_to_file}")

    run_case2(cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Progetto Data Mining - runner")
    p.add_argument("--case", type=int, required=True, choices=[1, 2], help="Caso da eseguire (1 o 2)")
    p.add_argument(
        "--verbose",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Se true, log a console (default: true)",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Se true, log anche su file (default: false)",
    )
    p.add_argument("--workers", type=int, default=1, help="Numero di workers (default: 1)")
    return p


def main() -> None:
    base_dir = Path(__file__).resolve().parent  # .../project/src
    parser = build_parser()
    args = parser.parse_args()

    verbose = args.verbose.lower() == "true"
    log_file = args.log_file.lower() == "true"

    if args.case == 2:
        run_case2_from_cli(
            base_dir=base_dir,
            verbose=verbose,
            log_file=log_file,
            workers=args.workers,
        )
        return

    raise SystemExit("Caso 1 non implementato in questo main.py (usa --case 2).")


if __name__ == "__main__":
    main()