from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from textplease import pipeline, segmenter
from textplease.backends import transformers_pipeline
from textplease.utils.audio_utils import TARGET_SAMPLE_RATE


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

    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *args, **kwargs: segments)
    monkeypatch.setattr(pipeline, "SentenceTransformer", sentence_transformer)
    monkeypatch.setattr(pipeline, "segment_transcript", lambda transcript, **kwargs: transcript)

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

    sentence_transformer.assert_called_once_with(
        "all-MiniLM-L6-v2",
        device="cpu",
    )


def test_segmenter_allows_embedding_model_download(monkeypatch):
    embedding_model = Mock()
    embedding_model.encode.return_value = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sentence_transformer = Mock(return_value=embedding_model)
    monkeypatch.setattr(segmenter, "SentenceTransformer", sentence_transformer)

    segmenter.segment_transcript(
        [
            {"start_time": "00:00:00.000", "end_time": "00:00:01.000", "text": "First segment."},
            {"start_time": "00:00:01.000", "end_time": "00:00:02.000", "text": "Second segment."},
        ],
        embedding_model_name="test-embedding-model",
    )

    sentence_transformer.assert_called_once_with(
        "test-embedding-model",
        device="cpu",
    )


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
    monkeypatch.setattr(
        transformers_pipeline,
        "load_pcm_wav",
        lambda path: np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32),
    )
    monkeypatch.setattr(
        transformers_pipeline,
        "_get_speech_segments",
        lambda *args: ([{"start": 0, "end": TARGET_SAMPLE_RATE}], [(0, TARGET_SAMPLE_RATE)]),
    )
    monkeypatch.setattr(
        transformers_pipeline,
        "_transcribe_speech_segments",
        lambda model, processor, audio_chunks, device, language: [[] for _ in audio_chunks],
    )

    transformers_pipeline._load_model_and_processor.cache_clear()
    try:
        transformers_pipeline.transcribe("input.wav", "test-model", "cpu")
        transformers_pipeline.transcribe("input.wav", "test-model", "cpu")
    finally:
        transformers_pipeline._load_model_and_processor.cache_clear()

    processor_loader.assert_called_once_with("test-model")
    assert model_loader.call_args.args == ("test-model",)
    assert "local_files_only" not in model_loader.call_args.kwargs
