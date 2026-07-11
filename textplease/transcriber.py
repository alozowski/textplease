import logging
from pathlib import Path

from textplease.backends.transformers_pipeline import transcribe as whisper_transcribe


logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    model_name: str,
    device: str,
    pause_threshold: float = 2.0,
    language: str | None = None,
    batch_size: int = 1,
) -> list[dict]:
    """Transcribe audio with a Whisper model."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")

    logger.info(f"Transcribing with model: {model_name}")
    return whisper_transcribe(
        audio_path=audio_path,
        model_name=model_name,
        device=device,
        pause_threshold=pause_threshold,
        language=language or "en",
        batch_size=batch_size,
    )
