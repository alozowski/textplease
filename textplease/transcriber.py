import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    model_name: str,
    device: str,
    chunk_duration_minutes: int = 10,
    pause_threshold: float = 2.0,
    batch_size: int = 1,
    language: str | None = None,
) -> list[dict]:
    """Transcribe audio, routing to the ASR backend selected by model name."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")

    model_name_lower = model_name.lower()

    if "whisper" in model_name_lower:
        try:
            from textplease.backends.transformers_pipeline import transcribe as whisper_transcribe
        except ImportError as e:
            raise ImportError(
                "Whisper backend dependencies not installed. Install with: pip install transformers torch torchaudio"
            ) from e

        logger.info(f"Using Whisper backend with model: {model_name}")
        return whisper_transcribe(
            audio_path=audio_path,
            model_name=model_name,
            device=device,
            chunk_duration_minutes=chunk_duration_minutes,
            pause_threshold=pause_threshold,
            batch_size=batch_size,
            language=language or "en",
        )

    if "nemo" in model_name_lower or "parakeet" in model_name_lower:
        try:
            from textplease.backends.nemo import transcribe as nemo_transcribe
        except ImportError as e:
            raise ImportError(
                "NeMo backend dependencies not installed. Install with: pip install nemo_toolkit[asr]"
            ) from e

        logger.info(f"Using NeMo backend with model: {model_name}")
        return nemo_transcribe(
            audio_path=audio_path,
            model_name=model_name,
            device=device,
            chunk_duration_minutes=chunk_duration_minutes,
            pause_threshold=pause_threshold,
            batch_size=batch_size,
        )

    raise ValueError(
        f"Unsupported model name or backend: {model_name}. "
        f"Supported backends: 'whisper' (e.g. openai/whisper-*), 'nemo'/'parakeet' (e.g. nvidia/parakeet-*)"
    )
