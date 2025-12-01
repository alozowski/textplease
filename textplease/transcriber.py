import logging
from pathlib import Path

from textplease.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    model_name: str,
    device: str,
    chunk_duration_minutes: int = 10,
    pause_threshold: float = 2.0,
    batch_size: int = 1,
    language: str | None = None,
    max_segment_words: int = 100,
    min_segment_words: int = 3,
    min_segment_chars: int = 15,
) -> list[dict]:
    """Unified entrypoint for transcribing audio using different ASR backends.

    Routes to appropriate backend based on model name.

    Args:
        audio_path: Path to the audio file to transcribe
        model_name: Name/identifier of the ASR model to use
        device: Device for inference ('cpu', 'cuda', 'mps')
        chunk_duration_minutes: Maximum chunk duration in minutes
        pause_threshold: Pause threshold in seconds for segmentation
        batch_size: Batch size for processing
        language: Language code (e.g., 'en', 'es') for Whisper models
        max_segment_words: Maximum words per segment
        min_segment_words: Minimum words per segment
        min_segment_chars: Minimum characters per segment

    Returns:
        List of transcription segments with timestamps and text

    Raises:
        ValueError: If the model name doesn't match any supported backend
        ImportError: If required backend dependencies are not installed
        FileNotFoundError: If audio file doesn't exist
        Exception: If transcription fails for any other reason

    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"Invalid model_name: {model_name}")

    if not audio_path or not isinstance(audio_path, str):
        raise ValueError(f"Invalid audio_path: {audio_path}")

    # Verify audio file exists before attempting transcription
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")

    model_name_lower = model_name.lower()

    try:
        if "whisper" in model_name_lower:
            try:
                from textplease.backends.transformers_pipeline import transcribe as whisper_transcribe
            except ImportError as e:
                logger.error(f"Failed to import Whisper backend: {e}")
                raise ImportError(
                    "Whisper backend dependencies not installed. "
                    "Install with: pip install transformers torch torchaudio"
                ) from e

            logger.info(f"Using Whisper backend with model: {model_name}")

            # Default to English if no language specified
            effective_language = language if language else "en"

            return whisper_transcribe(
                audio_path=audio_path,
                model_name=model_name,
                device=device,
                chunk_duration_minutes=chunk_duration_minutes,
                pause_threshold=pause_threshold,
                batch_size=batch_size,
                language=effective_language,
            )

        elif "nemo" in model_name_lower or "parakeet" in model_name_lower:
            try:
                from textplease.backends.nemo import transcribe as nemo_transcribe
            except ImportError as e:
                logger.error(f"Failed to import NeMo backend: {e}")
                raise ImportError(
                    "NeMo backend dependencies not installed. "
                    "Install with: pip install nemo_toolkit[asr]"
                ) from e

            logger.info(f"Using NeMo backend with model: {model_name}")

            return nemo_transcribe(
                audio_path=audio_path,
                model_name=model_name,
                device=device,
                chunk_duration_minutes=chunk_duration_minutes,
                pause_threshold=pause_threshold,
                batch_size=batch_size,
                max_segment_words=max_segment_words,
                min_segment_words=min_segment_words,
                min_segment_chars=min_segment_chars,
            )

        else:
            error_msg = (
                f"Unsupported model name or backend: {model_name}. "
                f"Supported backends: 'whisper' (e.g., openai/whisper-*), "
                f"'nemo'/'parakeet' (e.g., nvidia/parakeet-*)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    except (ValueError, ImportError, FileNotFoundError):
        # Re-raise validation, import, and file errors as-is
        raise
    except Exception as e:
        logger.error(f"Transcription failed for model '{model_name}': {e}", exc_info=True)
        raise
