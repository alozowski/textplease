import os
import csv
import json
import hashlib
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from textplease import pipeline


def test_pipeline_preserves_recognized_segments_exactly(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_path = tmp_path / "output.csv"
    recognized_segments = [
        {
            "start_time": "00:00:00.000",
            "end_time": "00:00:01.000",
            "text": " Thank you for watching. Next sentence.",
        },
        {"start_time": "00:00:01.500", "end_time": "00:00:02.000", "text": "repeat boundary"},
        {"start_time": "00:00:02.500", "end_time": "00:00:03.000", "text": "repeat boundary stays 世界 "},
    ]

    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *args, **kwargs: recognized_segments)

    pipeline.run_transcription_pipeline(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model_name": "test-model",
        }
    )

    with output_path.open(newline="") as output_file:
        rows = list(csv.DictReader(output_file, delimiter="\t"))

    assert rows == recognized_segments


def test_no_speech_writes_header_only_transcript(monkeypatch, tmp_path, caplog):
    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_path = tmp_path / "output.csv"
    output_path.write_text("previous transcript", encoding="utf-8")

    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *args, **kwargs: [])

    with caplog.at_level(logging.INFO):
        pipeline.run_transcription_pipeline(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "model_name": "test-model",
            }
        )

    assert output_path.read_text(encoding="utf-8") == "start_time\tend_time\ttext\n"
    if os.name != "nt":
        assert output_path.stat().st_mode & 0o777 == 0o600
    assert "No speech was transcribed" in caplog.text


def test_pipeline_rejects_input_as_output_before_transcription(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"original audio")
    transcribe_audio = Mock()
    monkeypatch.setattr(pipeline, "transcribe_audio", transcribe_audio)

    with pytest.raises(ValueError, match="different files"):
        pipeline.run_transcription_pipeline(
            {
                "input_path": str(input_path),
                "output_path": str(input_path),
                "model_name": "test-model",
            }
        )

    transcribe_audio.assert_not_called()
    assert input_path.read_bytes() == b"original audio"


def test_pipeline_rejects_removed_text_mutation_settings(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.touch()
    transcribe_audio = Mock()
    monkeypatch.setattr(pipeline, "transcribe_audio", transcribe_audio)

    with pytest.raises(ValueError, match="Unknown config keys.*similarity_threshold"):
        pipeline.run_transcription_pipeline(
            {
                "input_path": str(input_path),
                "output_path": str(tmp_path / "output.csv"),
                "model_name": "test-model",
                "similarity_threshold": 0.75,
            }
        )

    transcribe_audio.assert_not_called()


def test_pipeline_rejects_symlinked_output_to_input(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"original audio")
    output_path = tmp_path / "output.csv"
    try:
        output_path.symlink_to(input_path)
    except OSError as error:
        pytest.skip(f"Symlinks are unavailable: {error}")
    transcribe_audio = Mock()
    monkeypatch.setattr(pipeline, "transcribe_audio", transcribe_audio)

    with pytest.raises(ValueError, match="different files"):
        pipeline.run_transcription_pipeline(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "model_name": "test-model",
            }
        )

    transcribe_audio.assert_not_called()
    assert input_path.read_bytes() == b"original audio"


def test_pipeline_failure_preserves_output_and_removes_temporary_audio(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"original audio")
    output_path = tmp_path / "output.csv"
    output_path.write_text("existing transcript")
    temporary_paths = []

    def fail_transcription(audio_path, *args, temporary_directory, **kwargs):
        assert audio_path == str(input_path)
        temporary_path = Path(temporary_directory)
        temporary_paths.append(temporary_path)
        audio_path = temporary_path / "audio.wav"
        audio_path.write_bytes(b"decoded audio")
        raise RuntimeError("transcription failed")

    monkeypatch.setattr(pipeline, "detect_device", lambda device: "cpu")
    monkeypatch.setattr(pipeline, "transcribe_audio", fail_transcription)

    with pytest.raises(RuntimeError, match="transcription failed"):
        pipeline.run_transcription_pipeline(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "model_name": "test-model",
            }
        )

    assert input_path.read_bytes() == b"original audio"
    assert output_path.read_text() == "existing transcript"
    assert temporary_paths and not temporary_paths[0].exists()


def test_atomic_save_failure_preserves_existing_output(monkeypatch, tmp_path):
    output_path = tmp_path / "output.csv"
    output_path.write_text("existing transcript")
    temporary_directory = tmp_path / "temporary"
    temporary_directory.mkdir()

    def fail_after_partial_write(frame, path, **kwargs):
        Path(path).write_text("partial transcript")
        raise OSError("write failed")

    monkeypatch.setattr(pipeline.pd.DataFrame, "to_csv", fail_after_partial_write)

    with pytest.raises(OSError, match="write failed"):
        pipeline.save_to_csv(
            [{"start_time": "00:00:00.000", "end_time": "00:00:01.000", "text": "Hello"}],
            str(output_path),
            temporary_directory,
        )

    assert output_path.read_text() == "existing transcript"
    assert not (temporary_directory / "transcript.tsv").exists()


def test_debug_mode_keeps_three_columns_and_saves_private_diagnostics(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.touch()
    output = tmp_path / "output.csv"
    segments = [{"start_time": "00:00:01.000", "end_time": "", "text": "unfinished"}]

    def transcribe(*args, diagnostics=None, **kwargs):
        if diagnostics is not None:
            diagnostics.append({"text": "unfinished", "timestamp": [1.0, None], "timing_status": "missing"})
        return segments

    monkeypatch.setattr(pipeline, "transcribe_audio", transcribe)
    config = {"input_path": str(input_path), "output_path": str(output), "model_name": "test"}
    pipeline.run_transcription_pipeline(config)
    normal_bytes = output.read_bytes()
    debug_path = Path(str(output) + ".debug.json")
    assert not debug_path.exists()
    pipeline.run_transcription_pipeline({**config, "log_level": "DEBUG"})
    assert output.read_bytes() == normal_bytes
    assert normal_bytes.startswith(b"start_time\tend_time\ttext\n")
    data = json.loads(debug_path.read_text())
    assert data["transcript_sha256"] == hashlib.sha256(normal_bytes).hexdigest()
    assert data["events"][0]["timing_status"] == "missing"
    if os.name != "nt":
        assert debug_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".textplease-*"))


def test_debug_write_failure_preserves_transcript_and_cleans_temporary_files(monkeypatch, tmp_path):
    output = tmp_path / "output.tsv"
    output.write_text("existing transcript")
    work = tmp_path / "work"
    work.mkdir()

    def fail_replace(source, target):
        raise OSError("debug write failed")

    monkeypatch.setattr(pipeline.os, "replace", fail_replace)
    with pytest.raises(OSError, match="debug write failed"):
        pipeline.save_to_csv(
            [{"start_time": "", "end_time": "", "text": "keep me"}],
            str(output),
            work,
            diagnostics=[{"timing_status": "missing"}],
        )
    assert output.read_text() == "existing transcript"
    assert not list(work.iterdir())


def test_debug_output_cannot_replace_input(monkeypatch, tmp_path):
    output = tmp_path / "output.tsv"
    source = Path(str(output) + ".debug.json")
    source.write_bytes(b"private audio")
    runner = Mock()
    monkeypatch.setattr(pipeline, "transcribe_audio", runner)
    with pytest.raises(ValueError, match="Input and debug output"):
        pipeline.run_transcription_pipeline(
            {
                "input_path": str(source),
                "output_path": str(output),
                "model_name": "test",
                "log_level": "DEBUG",
            }
        )
    runner.assert_not_called()
    assert source.read_bytes() == b"private audio"


def test_transcript_commit_failure_leaves_detectable_debug_mismatch(monkeypatch, tmp_path):
    import hashlib

    output = tmp_path / "output.tsv"
    output.write_text("old transcript")
    work = tmp_path / "work"
    work.mkdir()
    original_replace = pipeline.os.replace

    def fail_transcript(source, target):
        if Path(target) == output:
            raise OSError("transcript commit failed")
        original_replace(source, target)

    monkeypatch.setattr(pipeline.os, "replace", fail_transcript)
    with pytest.raises(OSError, match="transcript commit failed"):
        pipeline.save_to_csv(
            [{"start_time": "", "end_time": "", "text": "new words"}],
            str(output),
            work,
            diagnostics=[{"kind": "span", "text": "new words"}],
        )
    assert output.read_text() == "old transcript"
    document = json.loads(Path(str(output) + ".debug.json").read_text())
    assert document["transcript_sha256"] != hashlib.sha256(output.read_bytes()).hexdigest()
    assert not list(work.iterdir())
