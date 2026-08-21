import logging
from pathlib import Path

from textplease.utils.audio_utils import normalize_audio
from textplease.backends.transformers_pipeline import transcribe as whisper_transcribe


logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    model_name: str,
    device: str,
    pause_threshold: float = 2.0,
    language: str | None = None,
    batch_size: int = 1,
    *,
    temporary_directory: str | Path,
) -> list[dict]:
    """Normalize local media for the built-in Whisper runtime and transcribe it."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")

    logger.info(f"Transcribing with model: {model_name}")
    normalized_audio_path = normalize_audio(audio_path, temporary_directory)
    return whisper_transcribe(
        audio_path=normalized_audio_path,
        model_name=model_name,
        device=device,
        pause_threshold=pause_threshold,
        language=language or "en",
        batch_size=batch_size,
    )
