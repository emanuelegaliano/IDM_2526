# src/main.py

from pathlib import Path

from case2.config import Case2Config
from case2.pipeline import run_case2


def main() -> None:
    # Directory di questo file: .../project/src
    base_dir = Path(__file__).resolve().parent

    # Datasets: .../project/src/datasets
    datasets_root = base_dir / "datasets"

    config = Case2Config(
        datasets_root=datasets_root,
        output_dir=datasets_root / "case2_generated",
        tmp_dir=datasets_root / "case2_generated" / "tmp",
        log_to_file=True,
        log_level="INFO",
        gene_id_mode="ensembl_api_cache",
        use_sqlite_index=True,
        chunk_size_rows=10_000,
        validate=True,
        workers=4,
    )

    config.logger.info("Avvio main.py")
    config.logger.info(f"datasets_root={config.datasets_root}")
    config.logger.info(f"output_dir={config.output_dir}")
    config.logger.info(f"gene_id_mode={config.gene_id_mode}")

    run_case2(config)

    config.logger.info("Fine main.py")


if __name__ == "__main__":
    main()