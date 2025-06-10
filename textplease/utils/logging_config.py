import logging


def configure_logging(level: int = logging.INFO, log_format: str = None) -> None:
    """Configure logging"""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler()],
        force=True,
    )
