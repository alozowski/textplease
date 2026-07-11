from types import SimpleNamespace
from unittest.mock import Mock

from textplease import pipeline
from textplease.backends import transformers_pipeline


def test_pipeline_reuses_embedding_model(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_path = tmp_path / "output.csv"
    embedding_model = object()
    sentence_transformer = Mock(return_value=embedding_model)
    segments = [
        {"start_time": "00:00:00.000", "end_time": "00:00:01.000", "text": "First test segment."},
        {"start_time": "00:00:02.000", "end_time": "00:00:03.000", "text": "Second test segment."},
    ]

    monkeypatch.setattr(pipeline, "extract_audio", lambda path: path)
    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *args, **kwargs: segments)
    monkeypatch.setattr(pipeline, "SentenceTransformer", sentence_transformer)
    monkeypatch.setattr(pipeline, "segment_transcript", lambda transcript, **kwargs: transcript)
    monkeypatch.setattr(pipeline, "cleanup_temp", lambda original, extracted: False)

    pipeline._load_embedding_model.cache_clear()
    try:
        config = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model_name": "test-model",
        }
        pipeline.run_transcription_pipeline(config)
        pipeline.run_transcription_pipeline(config)
    finally:
        pipeline._load_embedding_model.cache_clear()

    sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")


def test_transcriber_reuses_whisper_model(monkeypatch):
    class LoadedModel:
        config = SimpleNamespace(max_source_positions=1500)

        def to(self, device):
            return self

        def generate(self, **kwargs):
            raise AssertionError("Generation should be replaced in this cache test")

    processor = object()
    processor_loader = Mock(return_value=processor)
    loaded_model = LoadedModel()
    model_loader = Mock(return_value=loaded_model)

    monkeypatch.setattr(transformers_pipeline.WhisperProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(transformers_pipeline.WhisperForConditionalGeneration, "from_pretrained", model_loader)
    monkeypatch.setattr(transformers_pipeline, "_load_audio", lambda path: object())
    monkeypatch.setattr(transformers_pipeline, "_transcribe_with_fallbacks", lambda *args: [])

    transformers_pipeline._load_model_and_processor.cache_clear()
    try:
        transformers_pipeline.transcribe("input.wav", "test-model", "cpu")
        transformers_pipeline.transcribe("input.wav", "test-model", "cpu")
    finally:
        transformers_pipeline._load_model_and_processor.cache_clear()

    processor_loader.assert_called_once_with("test-model")
    model_loader.assert_called_once()
