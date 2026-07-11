import wave
import shutil
import subprocess
from pathlib import Path

import pytest

from textplease.utils.audio_utils import cleanup_temp, extract_audio


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary not available")


def _make_audio(path: Path, channels: int = 2, rate: int = 44100, seconds: int = 1) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ac",
            str(channels),
            "-ar",
            str(rate),
            str(path),
            "-y",
        ],
        check=True,
    )
    return path


def _wav_props(path: str) -> tuple[int, int]:
    with wave.open(path, "rb") as w:
        return w.getnchannels(), w.getframerate()


@pytest.mark.parametrize("ext", [".ogg", ".m4a"])
def test_non_wav_converts_to_mono_16k(tmp_path, ext):
    src = _make_audio(tmp_path / f"clip{ext}")
    out = extract_audio(str(src))
    assert out != str(src)
    assert out.endswith("_processed.wav")
    assert _wav_props(out) == (1, 16000)


def test_conversion_does_not_overwrite_existing_wav(tmp_path):
    existing = _make_audio(tmp_path / "clip.wav", channels=1, rate=16000, seconds=2)
    original_bytes = existing.read_bytes()
    _make_audio(tmp_path / "clip.ogg")

    out = extract_audio(str(tmp_path / "clip.ogg"))

    assert Path(out) != existing
    assert existing.read_bytes() == original_bytes


def test_compliant_wav_returned_unchanged(tmp_path):
    src = _make_audio(tmp_path / "good.wav", channels=1, rate=16000)
    assert extract_audio(str(src)) == str(src)


def test_noncompliant_wav_is_reencoded(tmp_path):
    src = _make_audio(tmp_path / "stereo.wav", channels=2, rate=44100)
    out = extract_audio(str(src))
    assert out != str(src)
    assert _wav_props(out) == (1, 16000)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_audio(str(tmp_path / "nope.ogg"))


def test_cleanup_temp_flags_new_file():
    assert cleanup_temp("a.mp3", "a_processed.wav") is True
    assert cleanup_temp("a.wav", "a.wav") is False
