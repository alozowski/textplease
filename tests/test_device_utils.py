from unittest.mock import Mock

from textplease import pipeline
from textplease.utils.device_utils import detect_device


def test_explicit_cpu_is_preserved(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert detect_device("cpu") == "cpu"


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert detect_device("auto") == "cuda"
    assert detect_device("cuda") == "cuda"


def test_auto_and_unavailable_cuda_fall_back_to_mps(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert detect_device("auto") == "mps"
    assert detect_device("cuda") == "mps"


def test_pipeline_uses_resolved_device(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_path = tmp_path / "output.csv"
    embedding_model = object()
    calls = Mock()
    sentence_transformer = Mock(return_value=embedding_model)
    transcribe_audio = Mock(
        return_value=[
            {"start_time": "00:00:00.000", "end_time": "00:00:01.000", "text": "First test segment."},
            {"start_time": "00:00:02.000", "end_time": "00:00:03.000", "text": "Second test segment."},
        ]
    )
    segment_transcript = Mock(side_effect=lambda segments, **kwargs: segments)
    calls.attach_mock(transcribe_audio, "transcription")
    calls.attach_mock(sentence_transformer, "embedding")

    monkeypatch.setattr(pipeline, "detect_device", lambda device: "cuda")
    monkeypatch.setattr(pipeline, "extract_audio", lambda path: path)
    monkeypatch.setattr(pipeline, "SentenceTransformer", sentence_transformer)
    monkeypatch.setattr(pipeline, "transcribe_audio", transcribe_audio)
    monkeypatch.setattr(pipeline, "segment_transcript", segment_transcript)
    monkeypatch.setattr(pipeline, "cleanup_temp", lambda original, extracted: False)

    pipeline.run_transcription_pipeline(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model_name": "test-model",
            "device": "auto",
        }
    )

    assert transcribe_audio.call_args.args[2] == "cuda"
    assert sentence_transformer.call_args.kwargs["device"] == "cuda"
    assert segment_transcript.call_args.kwargs["preferred_device"] == "cuda"
    assert [entry[0] for entry in calls.mock_calls] == ["transcription", "embedding"]
