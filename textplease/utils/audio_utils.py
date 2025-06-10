import os
import logging
from typing import Optional

import ffmpeg


logger = logging.getLogger(__name__)

# Constants for conversion
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def extract_audio(input_path: str) -> str:
    """Convert audio or video to mono 16kHz WAV if not already compliant.

    Args:
        input_path: Path to input file (.wav, .mp3, .mp4, etc.).

    Returns:
        Path to compliant mono 16kHz WAV file.
    """
    base, ext = os.path.splitext(input_path)
    ext = ext.lower()
    output_path: Optional[str] = ""

    if ext == ".wav":
        try:
            info = ffmpeg.probe(input_path)
            audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
            if not audio_stream:
                raise ValueError("No audio stream found in file.")

            channels = int(audio_stream.get("channels", 0))
            sample_rate = int(audio_stream.get("sample_rate", 0))

            if channels == TARGET_CHANNELS and sample_rate == TARGET_SAMPLE_RATE:
                logger.info("Input WAV is already mono 16kHz. Skipping re-encoding.")
                return input_path

            logger.info(f"Re-encoding WAV due to format mismatch (channels={channels}, sample_rate={sample_rate})")
            output_path = f"{base}_processed.wav"

        except ffmpeg.Error as e:
            logger.warning(f"FFmpeg probe failed: {e.stderr.decode() if e.stderr else str(e)}")
            output_path = f"{base}_processed.wav"

    else:
        logger.info(f"Converting non-WAV file '{input_path}' to mono 16kHz WAV format")
        output_path = f"{base}.wav"

    return _convert_to_mono_wav(input_path, output_path)


def _convert_to_mono_wav(input_path: str, output_path: str) -> str:
    """Convert audio to mono 16kHz WAV format using ffmpeg.

    Args:
        input_path: Path to source file.
        output_path: Destination path for processed WAV file.

    Returns:
        Path to the converted file.

    Raises:
        RuntimeError: If conversion fails or file is missing.
    """
    try:
        ffmpeg.input(input_path).output(
            output_path, acodec="pcm_s16le", ac=TARGET_CHANNELS, ar=str(TARGET_SAMPLE_RATE)
        ).overwrite_output().run(quiet=True)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not found: {output_path}")

        logger.info(f"Audio successfully converted: {output_path}")
        return output_path

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg conversion failed: {error_msg}")
        raise RuntimeError("Audio extraction failed.")


def cleanup_temp(original_path: str, extracted_path: str) -> bool:
    """Check whether the extracted audio is a new file.

    Args:
        original_path: Path to original input file.
        extracted_path: Path to possibly re-encoded audio.

    Returns:
        True if the audio was newly created and should be deleted later.
    """
    return original_path != extracted_path
