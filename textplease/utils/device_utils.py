import logging

import torch


logger = logging.getLogger(__name__)


def detect_device(preferred: str = "cpu") -> str:
    """Resolve an automatic or explicit PyTorch device selection."""
    preferred_lower = preferred.lower().strip()
    if preferred_lower not in {"auto", "cpu", "cuda", "mps"}:
        logger.warning(f"Invalid device '{preferred}'. Defaulting to 'cpu'.")
        return "cpu"

    if preferred_lower == "cpu":
        return "cpu"
    if preferred_lower == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred_lower == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred_lower != "auto":
        logger.warning(f"Device '{preferred_lower}' is unavailable. Falling back to the best available device.")

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
