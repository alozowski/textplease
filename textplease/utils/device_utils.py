import logging

import torch


logger = logging.getLogger(__name__)


def detect_device(preferred: str = "cpu") -> str:
    """Select the most suitable device for computation.

    Args:
        preferred: Preferred device ('cpu', 'cuda', 'mps')

    Returns:
        The selected device string ('cpu', 'cuda', or 'mps')

    Raises:
        ValueError: If preferred device string is invalid

    """
    if not isinstance(preferred, str):
        raise ValueError(f"preferred must be a string, got {type(preferred)}")

    preferred_lower = preferred.lower().strip()
    valid_devices = {"cpu", "cuda", "mps"}

    if preferred_lower not in valid_devices:
        logger.warning(f"Invalid device '{preferred}', valid options: {valid_devices}. Defaulting to 'cpu'")
        preferred_lower = "cpu"

    logger.debug(f"Preferred device: {preferred_lower}")

    try:
        if preferred_lower == "cuda":
            if torch.cuda.is_available():
                logger.info("Using CUDA device")
                return "cuda"
            logger.warning("CUDA requested but not available.")

        elif preferred_lower == "mps":
            try:
                if torch.backends.mps.is_available():
                    logger.info("Using Apple MPS device")
                    return "mps"
            except Exception as e:
                logger.warning(f"MPS check failed: {e}")
            logger.warning("MPS requested but not available.")

        # Fallback logic
        if torch.cuda.is_available():
            logger.info("No preferred device available. Falling back to CUDA.")
            return "cuda"

        try:
            if torch.backends.mps.is_available():
                logger.info("No preferred device available. Falling back to MPS.")
                return "mps"
        except Exception as e:
            logger.debug(f"MPS fallback check failed: {e}")

        logger.info("Defaulting to CPU")
        return "cpu"

    except Exception as e:
        logger.error(f"Error during device detection: {e}. Falling back to CPU")
        return "cpu"
