import logging
from typing import Any, Callable

from textplease.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    model_name: str,
    device: str = "cpu",
    chunk_duration_minutes: int = 10,
    pause_threshold: float = 2.0,
    batch_size: int = 1,
) -> Any:
    """Transcribe an audio file using a specified ASR model.

    Args:
        audio_path: Path to the .wav audio file.
        model_name: ASR model name (in a Hugging Face format).
        device: Hardware device to run inference on ("cpu", "cuda", "mps").
        chunk_duration_minutes: Duration to split long audios into smaller chunks.
        pause_threshold: Duration of silence (in seconds) that defines segment boundaries.
        batch_size: Number of audio chunks processed in a batch.

    Returns:
        List of dictionaries with keys such as 'text', 'start_time', 'end_time'.
    """
    logger.info(
        f"Transcribing '{audio_path}' using '{model_name}' on '{device}' | Chunk: {chunk_duration_minutes}min | Pause: {pause_threshold}s"
    )

    transcriber = get_backend_transcriber(model_name)
    return transcriber(
        audio_path,
        model_name=model_name,
        device=device,
        chunk_duration_minutes=chunk_duration_minutes,
        pause_threshold=pause_threshold,
        batch_size=batch_size,
    )


def get_backend_transcriber(model_name: str) -> Callable:
    """Return a transcription function based on the model name.

    Args:
        model_name: Name of the ASR model.

    Returns:
        A callable that performs audio transcription.

    Raises:
        ValueError: If no backend is matched to the model name.
    """
    model_name_lower = model_name.lower()

    if "parakeet" in model_name_lower or "nvidia" in model_name_lower:
        from textplease.backends.nemo import transcribe

        logger.debug(f"Selected Nvidia backend for model '{model_name}'")
    else:
        from textplease.backends.transformers_pipeline import transcribe

        logger.debug(f"Selected HuggingFace Transformers backend for model '{model_name}'")

    return transcribe
