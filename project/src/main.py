from __future__ import annotations

import argparse
from pathlib import Path

from case1.pipeline import run_case1

from case2 import Case2Config, run_case2
from case2.config import setup_logger


def _parse_csv_list(s: str | None) -> list[str] | None:
    """
    Converte "a.tsv,b.tsv, c.tsv" -> ["a.tsv","b.tsv","c.tsv"]
    Se s è None o vuota -> None
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


def run_case2_from_cli(
    *,
    base_dir: Path,
    verbose: bool,
    log_file: bool,
    workers: int,
    validate: bool,
    tumors_glob: str,
    tumor_files: list[str] | None,
    prefix_patient_id_with_tumor: bool,
    tumor_id_split_on: str,
) -> None:
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
        verbose=verbose,
        log_to_file=log_file,
        gene_id_mode="offline_mapping",
        validate=validate,

        # MULTI-TUMOR (nuovi campi)
        tumors_glob=tumors_glob,
        tumor_files=tumor_files,
        prefix_patient_id_with_tumor=prefix_patient_id_with_tumor,
        tumor_id_split_on=tumor_id_split_on,
    )

    cfg.resolve_paths()

    logger = setup_logger(
        "case2",
        verbose=cfg.verbose,
        log_to_file=cfg.log_to_file,
        log_file_path=cfg.log_file_path,
    )

    logger.info("Avvio Caso 2 (CLI)")
    logger.info(f"datasets_root={cfg.datasets_root}")
    logger.info(f"output_dir={cfg.output_dir}")
    logger.info(f"tmp_dir={cfg.tmp_dir}")
    logger.info(f"workers={cfg.workers}")
    logger.info(f"verbose={cfg.verbose} log_to_file={cfg.log_to_file}")
    logger.info(f"gene_id_mode={cfg.gene_id_mode}")
    logger.info(f"mapping_tsv={cfg.mapping_tsv}")
    logger.info(f"validate={cfg.validate}")

    # MULTI-TUMOR logging
    logger.info(f"tumors_glob={cfg.tumors_glob}")
    logger.info(f"tumor_files={cfg.tumor_files}")
    logger.info(f"prefix_patient_id_with_tumor={cfg.prefix_patient_id_with_tumor}")
    logger.info(f"tumor_id_split_on={cfg.tumor_id_split_on}")
    logger.info(f"tumors_resolved={len(cfg.tumor_matrix_paths)}")
    if len(cfg.tumor_matrix_paths) <= 20:
        for p in cfg.tumor_matrix_paths:
            logger.info(f"tumor: {p.name}")

    run_case2(cfg, logger)


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
    p.add_argument(
        "--validate",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Abilita la validazione degli output (default: true)",
    )
    p.add_argument("--workers", type=int, default=1, help="Numero di workers (default: 1)")

    # -------------------------
    # Case2: MULTI-TUMOR args
    # -------------------------
    p.add_argument(
        "--tumors_glob",
        type=str,
        default="*.tsv",
        help="Glob per selezionare i tumori in datasets/tumors/ (default: *.tsv)",
    )
    p.add_argument(
        "--tumor_files",
        type=str,
        default="",
        help="Lista esplicita di tumor file (separati da virgola). Se vuoto, usa tumors_glob.",
    )
    p.add_argument(
        "--prefix_patient_id",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Se true, prefissa patient_id con tumor_id:: per evitare collisioni (default: true)",
    )
    p.add_argument(
        "--tumor_id_split_on",
        type=str,
        default=".",
        help="Come ricavare tumor_id dal filename: prende la parte prima del primo separatore (default: '.')",
    )

    return p


def run_case1_from_cli(*, base_dir: Path, verbose: bool, log_file: bool, workers: int, validate: bool) -> None:
    datasets_root = base_dir / "datasets"
    output_dir = base_dir / "output_case1"
    tmp_dir = output_dir / "tmp"

    log_file_path = tmp_dir / "case1.log"
    logger = setup_logger(
        "case1",
        verbose=verbose,
        log_to_file=log_file,
        log_file_path=log_file_path
    )

    logger.info("Avvio Caso 1 (CLI) ")
    logger.info(f"datasets_root={datasets_root} ")
    logger.info(f"output_dir={output_dir} ")
    logger.info(f"tmp_dir={tmp_dir} ")
    logger.info(f"workers={workers} ")
    logger.info(f"verbose={verbose} log_to_file={log_file} ")
    logger.info(f"validate={validate} ")

    run_case1(base_dir=base_dir, workers=workers, validate=validate)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = build_parser()
    args = parser.parse_args()

    verbose = args.verbose.lower() == "true"
    log_file = args.log_file.lower() == "true"
    validate = args.validate.lower() == "true"

    if args.case == 1:
        run_case1_from_cli(
            base_dir=base_dir,
            verbose=verbose,
            log_file=log_file,
            workers=args.workers,
            validate=validate,
        )
        return

    elif args.case == 2:
        tumor_files = _parse_csv_list(args.tumor_files)
        prefix_patient_id = args.prefix_patient_id.lower() == "true"

        run_case2_from_cli(
            base_dir=base_dir,
            verbose=verbose,
            log_file=log_file,
            workers=args.workers,
            validate=validate,
            tumors_glob=args.tumors_glob,
            tumor_files=tumor_files,
            prefix_patient_id_with_tumor=prefix_patient_id,
            tumor_id_split_on=args.tumor_id_split_on,
        )
        return

    raise SystemExit("Caso non implementato.")


if __name__ == "__main__":
    main()