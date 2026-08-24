from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import pytest
from transformers import BatchFeature

from textplease.backends import transformers_pipeline
from textplease.utils.audio_utils import TARGET_SAMPLE_RATE


class FakeTokenizer:
    def __init__(self, include_offsets: bool = True):
        self.include_offsets = include_offsets

    def decode(self, token_ids, **kwargs):
        if not self.include_offsets:
            return {"offsets": []}
        segment_number = int(token_ids[0])
        return {
            "offsets": [
                {
                    "text": f"Segment {segment_number}",
                    "timestamp": (0.0, 0.5),
                }
            ]
        }


class FakeProcessor:
    feature_extractor = SimpleNamespace(chunk_length=30)

    def __init__(self, include_offsets: bool = True):
        self.tokenizer = FakeTokenizer(include_offsets)
        self.audio_batches = []

    def __call__(self, audio, **kwargs):
        chunks = audio if isinstance(audio, list) else [audio]
        self.audio_batches.append([chunk.copy() for chunk in chunks])
        segment_numbers = [float(chunk[0]) for chunk in chunks]
        input_features = torch.tensor(segment_numbers).reshape(-1, 1, 1)
        return BatchFeature(
            {
                "input_features": input_features,
                "attention_mask": torch.ones(len(chunks), 1),
            }
        )


class FakeModel:
    config = SimpleNamespace(max_source_positions=1500)

    def __init__(self, fail_batched: bool = False, error: Exception | None = None):
        self.batch_sizes = []
        self.fail_batched = fail_batched
        self.error = error

    def generate(self, **kwargs):
        input_features = kwargs["input_features"]
        batch_size = len(input_features)
        self.batch_sizes.append(batch_size)
        if self.error is not None:
            raise self.error
        if self.fail_batched and batch_size > 1:
            raise torch.OutOfMemoryError
        return input_features[:, 0, :].to(dtype=torch.long)


def _run_transcription(monkeypatch, batch_size, *, fail_batched=False):
    audio = np.concatenate(
        [np.full(TARGET_SAMPLE_RATE, segment_number, dtype=np.float32) for segment_number in (1, 2, 3)]
    )
    speech_segments = [
        {"start": 0, "end": TARGET_SAMPLE_RATE},
        {"start": TARGET_SAMPLE_RATE, "end": 2 * TARGET_SAMPLE_RATE},
        {"start": 2 * TARGET_SAMPLE_RATE, "end": 3 * TARGET_SAMPLE_RATE},
    ]
    model = FakeModel(fail_batched=fail_batched)
    processor = FakeProcessor()

    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (model, processor))
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "_get_speech_segments", lambda *args: speech_segments)
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    segments = transformers_pipeline.transcribe(
        "input.wav",
        "test-model",
        "cpu",
        batch_size=batch_size,
    )
    return segments, model.batch_sizes


def test_transcribe_batches_without_changing_segments(monkeypatch):
    sequential, sequential_batch_sizes = _run_transcription(monkeypatch, 1)
    batched, batched_batch_sizes = _run_transcription(monkeypatch, 2)

    assert batched == sequential
    assert sequential_batch_sizes == [1, 1, 1]
    assert batched_batch_sizes == [2, 1]


def test_transcribe_retries_batch_after_out_of_memory(monkeypatch):
    segments, batch_sizes = _run_transcription(monkeypatch, 2, fail_batched=True)

    assert [segment["text"] for segment in segments] == ["Segment 1", "Segment 2", "Segment 3"]
    assert batch_sizes == [2, 1, 1, 1]


def test_no_speech_skips_whisper(monkeypatch):
    detector = Mock(return_value=[])
    model_loader = Mock(side_effect=AssertionError("Whisper must not load for VAD-negative audio"))
    monkeypatch.setattr(
        transformers_pipeline,
        "load_pcm_wav",
        lambda path: np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32),
    )
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(transformers_pipeline, "get_speech_timestamps", detector)
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", model_loader)

    assert transformers_pipeline.transcribe("input.wav", "test-model", "cpu") == []
    model_loader.assert_not_called()
    assert detector.call_args.kwargs["min_silence_duration_ms"] == 2000
    assert detector.call_args.kwargs["return_seconds"] is False


@pytest.mark.parametrize("duration_samples", [4000, 6992, 7984])
def test_every_detected_short_interval_reaches_whisper(monkeypatch, duration_samples):
    start = 123
    end = start + duration_samples
    audio = np.linspace(-1.0, 1.0, end + 200, dtype=np.float32)
    detector = Mock(return_value=[{"start": start, "end": end}])
    model = FakeModel()
    processor = FakeProcessor()

    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(transformers_pipeline, "get_speech_timestamps", detector)
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (model, processor))
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    segments = transformers_pipeline.transcribe("input.wav", "test-model", "cpu")

    assert segments
    assert model.batch_sizes == [1]
    assert len(processor.audio_batches) == 1
    np.testing.assert_array_equal(processor.audio_batches[0][0], audio[start:end])
    assert detector.call_args.kwargs["return_seconds"] is False


def test_detected_intervals_are_clamped_to_audio_bounds(monkeypatch):
    audio = np.linspace(1.0, 2.0, TARGET_SAMPLE_RATE, dtype=np.float32)
    detector = Mock(
        return_value=[
            {"start": -100, "end": len(audio) + 100},
            {"start": 500, "end": 500},
            {"start": 700, "end": 600},
        ]
    )
    model = FakeModel()
    processor = FakeProcessor()

    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(transformers_pipeline, "get_speech_timestamps", detector)
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (model, processor))
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    segments = transformers_pipeline.transcribe("input.wav", "test-model", "cpu")

    assert segments
    assert model.batch_sizes == [1]
    np.testing.assert_array_equal(processor.audio_batches[0][0], audio)


def test_empty_decoder_result_does_not_retry_full_audio(monkeypatch):
    audio = np.ones(TARGET_SAMPLE_RATE, dtype=np.float32)
    start = 137
    end = 8137
    model = FakeModel()
    processor = FakeProcessor(include_offsets=False)

    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": start, "end": end}],
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (model, processor))
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    assert transformers_pipeline.transcribe("input.wav", "test-model", "cpu") == []
    assert model.batch_sizes == [1]
    assert len(processor.audio_batches) == 1
    np.testing.assert_array_equal(processor.audio_batches[0][0], audio[start:end])


@pytest.mark.parametrize(
    "decoder_error",
    [ValueError("Unsupported language"), torch.OutOfMemoryError("single segment")],
)
def test_single_segment_decoder_error_propagates(monkeypatch, decoder_error):
    audio = np.ones(TARGET_SAMPLE_RATE, dtype=np.float32)
    model = FakeModel(error=decoder_error)

    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 0, "end": len(audio)}],
    )
    monkeypatch.setattr(
        transformers_pipeline,
        "_load_model_and_processor",
        lambda *args: (model, FakeProcessor()),
    )
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    with pytest.raises(type(decoder_error)) as error:
        transformers_pipeline.transcribe("input.wav", "test-model", "cpu", language="invalid")

    assert error.value is decoder_error
    assert model.batch_sizes == [1]


def test_model_loader_error_propagates(monkeypatch):
    audio = np.ones(TARGET_SAMPLE_RATE, dtype=np.float32)
    expected_error = OSError("checkpoint is unavailable")
    model_loader = Mock(side_effect=expected_error)

    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 0, "end": len(audio)}],
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", model_loader)

    with pytest.raises(OSError) as error:
        transformers_pipeline.transcribe("input.wav", "missing-model", "cpu")

    assert error.value is expected_error
    model_loader.assert_called_once()
