from __future__ import annotations

import logging


def _set_global_verbosity(verbose: bool) -> None:
    """
    Set logging level globally (root + all existing loggers).
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)

    # All already-created loggers
    for obj in logging.Logger.manager.loggerDict.values():
        if isinstance(obj, logging.Logger):
            obj.setLevel(level)
            for handler in obj.handlers:
                handler.setLevel(level)


def run_pipeline(verbose: bool = False) -> None:
    """
    Case 2 Pipeline (ordine richiesto dall'assignment):

    1) validate_paths
    2) check_dataset
    3) build intermediate
    4) build final
    """

    from . import cfg
    from .check_dataset import run as check_dataset_run
    from .build_case2_intermediate import run as build_case2_intermediate_run
    from .build_case2_final import run as build_case2_final_run

    # Set verbosity before executing steps
    _set_global_verbosity(verbose)

    log = cfg.get_logger("case2")
    log.info("Starting Case 2 pipeline (verbose=%s)", verbose)

    # Step 0
    if cfg.CLEAN_DIRECTORIES:
        cfg.clean_output()

    # Step 1
    cfg.validate_paths()

    # Step 2
    check_dataset_run()

    # Step 3
    build_case2_intermediate_run()

    # Step 4
    build_case2_final_run()

    log.info("Case 2 pipeline completed successfully.")