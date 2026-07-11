from types import SimpleNamespace

import numpy as np
import torch
from transformers import BatchFeature

from textplease.backends import transformers_pipeline


class FakeTokenizer:
    def decode(self, token_ids, **kwargs):
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
    tokenizer = FakeTokenizer()

    def __call__(self, audio, **kwargs):
        chunks = audio if isinstance(audio, list) else [audio]
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

    def __init__(self, fail_batched: bool = False):
        self.batch_sizes = []
        self.fail_batched = fail_batched

    def generate(self, **kwargs):
        input_features = kwargs["input_features"]
        batch_size = len(input_features)
        self.batch_sizes.append(batch_size)
        if self.fail_batched and batch_size > 1:
            raise torch.OutOfMemoryError
        return input_features[:, 0, :].to(dtype=torch.long)


def _run_transcription(monkeypatch, batch_size, *, fail_batched=False):
    audio = np.concatenate(
        [np.full(transformers_pipeline.SAMPLE_RATE, segment_number, dtype=np.float32) for segment_number in (1, 2, 3)]
    )
    speech_segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 3.0},
    ]
    model = FakeModel(fail_batched=fail_batched)
    processor = FakeProcessor()

    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (model, processor))
    monkeypatch.setattr(transformers_pipeline, "_load_audio", lambda path: audio)
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
