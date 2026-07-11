import logging

import torch


logger = logging.getLogger(__name__)


def detect_device(preferred: str = "cpu") -> str:
    """Select the best available device ('cpu', 'cuda', or 'mps'), falling back from the preferred choice."""
    preferred_lower = preferred.lower().strip()
    if preferred_lower not in {"cpu", "cuda", "mps"}:
        logger.warning(f"Invalid device '{preferred}'. Defaulting to 'cpu'.")
        preferred_lower = "cpu"

    if preferred_lower == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred_lower == "mps" and torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
