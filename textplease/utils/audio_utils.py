import os
import logging
from pathlib import Path

import ffmpeg


logger = logging.getLogger(__name__)

# Target audio format for ASR models
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def extract_audio(input_path: str) -> str:
    """Convert audio or video to mono 16kHz WAV, returning the compliant file's path."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    base, ext = os.path.splitext(input_path)
    if ext.lower() == ".wav":
        try:
            info = ffmpeg.probe(input_path)
            audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
            if not audio_stream:
                raise ValueError(f"No audio stream found in file: {input_path}")

            channels = int(audio_stream.get("channels", 0))
            sample_rate = int(audio_stream.get("sample_rate", 0))
            if channels == TARGET_CHANNELS and sample_rate == TARGET_SAMPLE_RATE:
                logger.info("Input WAV is already mono 16kHz. Skipping re-encoding.")
                return input_path

            logger.warning(
                f"WAV file '{input_path}' has incompatible format "
                f"(channels={channels}, sample_rate={sample_rate}). Re-encoding to mono 16kHz WAV."
            )
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.warning(f"FFmpeg probe failed: {error_msg}. Will attempt conversion.")
    else:
        logger.info(f"Input file '{input_path}' is not a WAV. Converting to mono 16kHz WAV.")

    return _convert_to_mono_wav(input_path, f"{base}_processed.wav")


def _convert_to_mono_wav(input_path: str, output_path: str) -> str:
    """Convert audio to mono 16kHz WAV using ffmpeg."""
    logger.info(f"Converting audio: {input_path} -> {output_path}")

    try:
        ffmpeg.input(input_path).output(
            output_path, acodec="pcm_s16le", ac=TARGET_CHANNELS, ar=str(TARGET_SAMPLE_RATE)
        ).overwrite_output().run(quiet=True, capture_stderr=True)
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Audio conversion failed: {error_msg}") from e

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Conversion produced no usable output: {output_path}")

    logger.info(f"Audio successfully converted: {output_path}")
    return output_path


def cleanup_temp(original_path: str, extracted_path: str) -> bool:
    """Return True if the extracted audio is a newly created file safe to delete."""
    return os.path.abspath(original_path) != os.path.abspath(extracted_path)
