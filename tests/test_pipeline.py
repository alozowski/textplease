import csv
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from textplease import pipeline, segmenter


def test_similarity_threshold_one_skips_embedding_model(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_path = tmp_path / "output.csv"
    sentence_transformer = Mock(side_effect=AssertionError("Embedding model should not load"))

    monkeypatch.setattr(
        pipeline,
        "transcribe_audio",
        lambda *args, **kwargs: [
            {"start_time": "00:00:00.000", "end_time": "00:00:01.000", "text": "Short"},
            {"start_time": "00:00:01.500", "end_time": "00:00:02.000", "text": "fragment"},
        ],
    )
    monkeypatch.setattr(pipeline, "SentenceTransformer", sentence_transformer)
    monkeypatch.setattr(segmenter, "SentenceTransformer", sentence_transformer)

    pipeline.run_transcription_pipeline(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model_name": "test-model",
            "similarity_threshold": 1.0,
        }
    )

    with output_path.open(newline="") as output_file:
        rows = list(csv.DictReader(output_file, delimiter="\t"))

    sentence_transformer.assert_not_called()
    assert [row["text"] for row in rows] == ["Short fragment"]


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


def test_short_segment_log_does_not_include_transcript(caplog):
    private_text = "My private account recovery phrase"

    with caplog.at_level(logging.WARNING):
        segmenter.post_process_segments(
            [{"start_time": "00:00:00.000", "end_time": "00:00:01.000", "text": private_text}],
            min_words=10,
            min_chars=100,
        )

    assert "Keeping a short segment that cannot be merged" in caplog.text
    assert private_text not in caplog.text
