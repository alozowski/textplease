import os
import logging
from pathlib import Path

import ffmpeg  # type: ignore


logger = logging.getLogger(__name__)

# Target audio format for ASR models
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def extract_audio(input_path: str) -> str:
    """Convert audio or video to mono 16kHz WAV if not already compliant.

    Args:
        input_path: Path to input file (.wav, .mp3, .mp4, etc.).

    Returns:
        Path to compliant mono 16kHz WAV file.

    Raises:
        ValueError: If input_path is invalid or file doesn't exist
        RuntimeError: If audio conversion fails

    """
    if not input_path or not isinstance(input_path, str):
        raise ValueError(f"Invalid input_path: {input_path}")

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    try:
        base, ext = os.path.splitext(input_path)
        ext = ext.lower()
        output_path: str | None = ""

        if ext == ".wav":
            try:
                info = ffmpeg.probe(input_path)
                audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
                if not audio_stream:
                    logger.error(f"No audio stream found in file: {input_path}")
                    raise ValueError(f"No audio stream found in file: {input_path}")

                channels = int(audio_stream.get("channels", 0))
                sample_rate = int(audio_stream.get("sample_rate", 0))

                # Skip conversion if already in correct format
                if channels == TARGET_CHANNELS and sample_rate == TARGET_SAMPLE_RATE:
                    logger.info("Input WAV is already mono 16kHz. Skipping re-encoding.")
                    return input_path

                logger.warning(
                    f"WAV file '{input_path}' has incompatible format (channels={channels}, sample_rate={sample_rate}). "
                    "Re-encoding to mono 16kHz WAV."
                )
                output_path = f"{base}_processed.wav"

            except ffmpeg.Error as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                logger.warning(f"FFmpeg probe failed: {error_msg}. Will attempt conversion.")
                output_path = f"{base}_processed.wav"
            except ValueError:
                raise
            except Exception as e:
                logger.error(f"Unexpected error probing WAV file: {e}")
                output_path = f"{base}_processed.wav"

        else:
            # Non-WAV file needs conversion
            logger.info(f"Input file '{input_path}' is not a WAV. Converting to mono 16kHz WAV.")
            output_path = f"{base}.wav"

        return _convert_to_mono_wav(input_path, output_path)

    except (ValueError, RuntimeError, FileNotFoundError):
        raise
    except Exception as e:
        logger.error(f"Audio extraction failed for '{input_path}': {e}", exc_info=True)
        raise RuntimeError(f"Audio extraction failed: {e}") from e


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
        logger.info(f"Converting audio: {input_path} -> {output_path}")

        # Run ffmpeg conversion
        ffmpeg.input(input_path).output(
            output_path,
            acodec="pcm_s16le",
            ac=TARGET_CHANNELS,
            ar=str(TARGET_SAMPLE_RATE)
        ).overwrite_output().run(quiet=True, capture_stderr=True)

        # Verify output file was created
        if not os.path.exists(output_path):
            error_msg = f"Conversion completed but output file not found: {output_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Verify output file is not empty
        if os.path.getsize(output_path) == 0:
            error_msg = f"Conversion produced empty file: {output_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Audio successfully converted: {output_path}")
        return output_path

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg conversion failed: {error_msg}")
        raise RuntimeError(f"Audio conversion failed: {error_msg}") from e
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}", exc_info=True)
        raise RuntimeError(f"Audio conversion failed: {e}") from e


def cleanup_temp(original_path: str, extracted_path: str) -> bool:
    """Check whether the extracted audio is a new file.

    Args:
        original_path: Path to original input file.
        extracted_path: Path to possibly re-encoded audio.

    Returns:
        True if the audio was newly created and should be deleted later.

    Raises:
        ValueError: If paths are invalid

    """
    if not original_path or not isinstance(original_path, str):
        raise ValueError(f"Invalid original_path: {original_path}")

    if not extracted_path or not isinstance(extracted_path, str):
        raise ValueError(f"Invalid extracted_path: {extracted_path}")

    try:
        # Files are different if paths don't match (temp file was created)
        return os.path.abspath(original_path) != os.path.abspath(extracted_path)
    except Exception as e:
        logger.error(f"Error comparing paths: {e}")
        # Conservative default: assume it's a temp file to be safe
        return True
