import wave
import shutil
import socket
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from textplease import transcriber
from textplease.utils import audio_utils
from textplease.backends import transformers_pipeline
from textplease.utils.audio_utils import normalize_audio


requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary not available")


def _make_audio(
    path: Path,
    channels: int = 2,
    rate: int = 44100,
    seconds: int = 1,
    codec: str | None = None,
) -> Path:
    command = [
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
    ]
    if codec is not None:
        command.extend(["-c:a", codec])
    subprocess.run([*command, str(path), "-y"], check=True)
    return path


def _wav_props(path: str) -> tuple[int, int, int]:
    with wave.open(path, "rb") as w:
        return w.getnchannels(), w.getframerate(), w.getsampwidth()


@pytest.mark.parametrize("ext", [".ogg", ".m4a"])
@requires_ffmpeg
def test_non_wav_converts_to_mono_16k(tmp_path, ext):
    src = _make_audio(tmp_path / f"clip{ext}")
    out = normalize_audio(str(src), tmp_path / "temporary")
    assert out != str(src)
    assert Path(out).parent == tmp_path / "temporary"
    assert _wav_props(out) == (1, 16000, 2)


@requires_ffmpeg
def test_conversion_does_not_overwrite_existing_wav(tmp_path):
    existing = _make_audio(tmp_path / "clip_processed.wav", channels=1, rate=16000, seconds=2)
    original_bytes = existing.read_bytes()
    _make_audio(tmp_path / "clip.ogg")

    out = normalize_audio(str(tmp_path / "clip.ogg"), tmp_path / "temporary")

    assert Path(out) != existing
    assert existing.read_bytes() == original_bytes


@requires_ffmpeg
def test_normalization_does_not_overwrite_input_inside_work_directory(tmp_path):
    source = _make_audio(tmp_path / "audio.wav", channels=1, rate=16000)
    original_bytes = source.read_bytes()

    out = normalize_audio(str(source), tmp_path)

    assert Path(out) != source
    assert source.read_bytes() == original_bytes
    assert _wav_props(out) == (1, 16000, 2)


@requires_ffmpeg
def test_compliant_wav_is_normalized_to_private_path(tmp_path):
    src = _make_audio(tmp_path / "good.wav", channels=1, rate=16000)
    out = normalize_audio(str(src), tmp_path / "temporary")
    assert out != str(src)
    assert _wav_props(out) == (1, 16000, 2)


@requires_ffmpeg
def test_mono_16k_float_wav_is_reencoded_as_pcm16(tmp_path):
    src = _make_audio(tmp_path / "float.wav", channels=1, rate=16000, codec="pcm_f32le")

    out = normalize_audio(str(src), tmp_path / "temporary")

    assert out != str(src)
    assert _wav_props(out) == (1, 16000, 2)


@requires_ffmpeg
def test_whisper_adapter_normalizes_media_before_transcription(monkeypatch, tmp_path):
    src = _make_audio(tmp_path / "clip.ogg")

    detector = Mock(return_value=[])
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(transformers_pipeline, "get_speech_timestamps", detector)
    monkeypatch.setattr(
        transformers_pipeline,
        "_load_model_and_processor",
        Mock(side_effect=AssertionError("Whisper must not load for VAD-negative audio")),
    )

    work_directory = tmp_path / "temporary"
    assert (
        transcriber.transcribe_audio(
            str(src),
            "test-model",
            "cpu",
            temporary_directory=work_directory,
        )
        == []
    )

    captured_audio = detector.call_args.args[0].numpy()
    assert captured_audio.dtype == np.float32
    assert captured_audio.ndim == 1
    assert len(captured_audio) == 16000
    assert np.isfinite(captured_audio).all()
    assert np.max(np.abs(captured_audio)) <= 1.0


def test_pcm_decoder_preserves_signed_sample_scale(monkeypatch, tmp_path):
    audio_path = tmp_path / "signed-scale.wav"
    values = np.array([-32768, -16384, 0, 16384, 32767], dtype="<i2")
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(values.tobytes())

    detector = Mock(return_value=[])
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(transformers_pipeline, "get_speech_timestamps", detector)
    monkeypatch.setattr(
        transformers_pipeline,
        "_load_model_and_processor",
        Mock(side_effect=AssertionError("Whisper must not load for VAD-negative audio")),
    )

    assert transformers_pipeline.transcribe(str(audio_path), "test-model", "cpu") == []
    samples = detector.call_args.args[0].numpy()

    np.testing.assert_allclose(samples, [-1.0, -0.5, 0.0, 0.5, 0.9999695])


def test_empty_pcm_is_rejected_before_model_loading(monkeypatch, tmp_path):
    audio_path = tmp_path / "empty.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)

    model_loader = Mock()
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", model_loader)

    with pytest.raises(ValueError, match="no frames"):
        transformers_pipeline.transcribe(str(audio_path), "test-model", "cpu")

    model_loader.assert_not_called()


def test_missing_whisper_model_fails_without_network(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"\0\0")

    connect = Mock(side_effect=AssertionError("Model loading attempted a network connection"))
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 0, "end": 1}],
    )
    transformers_pipeline._load_model_and_processor.cache_clear()
    try:
        with pytest.raises(OSError):
            transformers_pipeline.transcribe(
                str(audio_path),
                "textplease/model-that-is-not-cached",
                "cpu",
            )
    finally:
        transformers_pipeline._load_model_and_processor.cache_clear()

    connect.assert_not_called()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        normalize_audio(str(tmp_path / "nope.ogg"), tmp_path / "temporary")


def test_missing_ffmpeg_has_installation_instruction(monkeypatch, tmp_path):
    input_path = tmp_path / "clip.ogg"
    input_path.touch()
    monkeypatch.setattr(audio_utils.shutil, "which", lambda executable: None)

    with pytest.raises(RuntimeError, match=r"FFmpeg is required.*ffmpeg\.org/download\.html"):
        normalize_audio(str(input_path), tmp_path / "temporary")


@requires_ffmpeg
def test_conversion_error_does_not_expose_input_path(tmp_path):
    input_path = tmp_path / "sensitive-recording-name.ogg"
    input_path.write_text("not audio")

    with pytest.raises(RuntimeError) as error:
        normalize_audio(str(input_path), tmp_path / "temporary")

    assert input_path.name not in str(error.value)
