import wave
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2
FFMPEG_INSTALL_ERROR = (
    "FFmpeg is required to process audio. Install it from https://ffmpeg.org/download.html "
    "and ensure the ffmpeg executable is on PATH."
)


def normalize_audio(input_path: str, temporary_directory: str | Path) -> str:
    """Normalize the first audio stream to a private mono 16 kHz PCM16 WAV."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    ffmpeg_executable = shutil.which("ffmpeg")
    if ffmpeg_executable is None:
        raise RuntimeError(FFMPEG_INSTALL_ERROR)

    work_directory = Path(temporary_directory)
    work_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="audio-", suffix=".wav", dir=work_directory, delete=False) as output_file:
        output_path = Path(output_file.name)
    conversion = subprocess.run(
        [
            ffmpeg_executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_file),
            "-map",
            "0:a:0",
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if conversion.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Audio conversion failed. Verify that the file contains a supported audio stream.")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Audio conversion produced no usable output")

    output_path.chmod(0o600)
    return str(output_path)


def load_pcm_wav(audio_path: str | Path) -> np.ndarray:
    """Load a mono 16 kHz PCM16 WAV as normalized float32 samples."""
    with wave.open(str(audio_path), "rb") as audio_file:
        channels = audio_file.getnchannels()
        sample_rate = audio_file.getframerate()
        sample_width = audio_file.getsampwidth()
        compression = audio_file.getcomptype()
        frame_count = audio_file.getnframes()
        frames = audio_file.readframes(frame_count)

    if (
        channels != TARGET_CHANNELS
        or sample_rate != TARGET_SAMPLE_RATE
        or sample_width != TARGET_SAMPLE_WIDTH
        or compression != "NONE"
    ):
        raise ValueError(
            f"Expected mono 16 kHz PCM16 WAV, got channels={channels}, sample_rate={sample_rate}, "
            f"sample_width={sample_width}, compression={compression}"
        )
    if frame_count == 0:
        raise ValueError("Audio contains no frames")

    expected_size = frame_count * channels * sample_width
    if len(frames) != expected_size:
        raise ValueError(f"Audio WAV is truncated: {audio_path}")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    del frames
    samples /= 32768.0
    return samples
