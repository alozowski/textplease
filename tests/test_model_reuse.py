from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from textplease.backends import transformers_pipeline
from textplease.utils.audio_utils import TARGET_SAMPLE_RATE


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
