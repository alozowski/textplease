import logging

import torch


logger = logging.getLogger(__name__)


def detect_device(preferred: str = "cpu") -> str:
    """Select the most suitable device for computation"""
    logger.debug(f"Preferred device: {preferred}")

    if preferred == "cuda":
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"
        logger.warning("CUDA requested but not available.")

    elif preferred == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using Apple MPS device")
            return "mps"
        logger.warning("MPS requested but not available.")

    if torch.cuda.is_available():
        logger.info("No preferred device available. Falling back to CUDA.")
        return "cuda"

    if torch.backends.mps.is_available():
        logger.info("No preferred device available. Falling back to MPS.")
        return "mps"

    logger.info("Defaulting to CPU")
    return "cpu"
