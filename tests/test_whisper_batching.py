from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import pytest
from transformers import BatchFeature

from textplease.backends import transformers_pipeline
from textplease.utils.audio_utils import TARGET_SAMPLE_RATE


class FakeTokenizer:
    def __init__(
        self,
        include_offsets: bool = True,
        timestamp: tuple[float | None, float | None] = (0.0, 0.5),
    ):
        self.include_offsets = include_offsets
        self.timestamp = timestamp

    def decode(self, token_ids, **kwargs):
        if not self.include_offsets:
            return {"offsets": []}
        segment_number = int(token_ids[0])
        return {
            "offsets": [
                {
                    "text": f"Segment {segment_number}",
                    "timestamp": self.timestamp,
                }
            ]
        }


class FakeProcessor:
    feature_extractor = SimpleNamespace(chunk_length=30)

    def __init__(
        self,
        include_offsets: bool = True,
        timestamp: tuple[float | None, float | None] = (0.0, 0.5),
    ):
        self.tokenizer = FakeTokenizer(include_offsets, timestamp)
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

    def __init__(
        self,
        fail_batched: bool = False,
        error: Exception | None = None,
        *,
        is_multilingual: bool = True,
    ):
        self.batch_sizes = []
        self.fail_batched = fail_batched
        self.error = error
        self.generation_config = SimpleNamespace(is_multilingual=is_multilingual)
        self.generation_calls = []

    def generate(self, **kwargs):
        self.generation_calls.append(kwargs)
        input_features = kwargs["input_features"]
        batch_size = len(input_features)
        self.batch_sizes.append(batch_size)
        if self.error is not None:
            raise self.error
        if self.fail_batched and batch_size > 1:
            raise torch.OutOfMemoryError
        return input_features[:, 0, :].to(dtype=torch.long)


@pytest.fixture
def audio_classifier(monkeypatch):
    feature_extractor = Mock(
        side_effect=lambda audio, **kwargs: BatchFeature(
            {
                "input_values": torch.ones(
                    (len(audio) if isinstance(audio, list) else 1, 1),
                    dtype=torch.float32,
                )
            }
        )
    )
    classifier = Mock(
        side_effect=lambda input_values: SimpleNamespace(
            logits=torch.tensor([[1.0, 1.0]]).repeat(len(input_values), 1)
        )
    )
    classifier.config = SimpleNamespace(label2id={"Speech": 0, "Music": 1})
    classifier.to.return_value = classifier
    classifier.eval.return_value = classifier
    feature_loader = Mock(return_value=feature_extractor)
    model_loader = Mock(return_value=classifier)
    monkeypatch.setattr(transformers_pipeline.ASTFeatureExtractor, "from_pretrained", feature_loader)
    monkeypatch.setattr(transformers_pipeline.ASTForAudioClassification, "from_pretrained", model_loader)
    return SimpleNamespace(
        model=classifier,
        model_loader=model_loader,
        feature_extractor=feature_extractor,
        feature_loader=feature_loader,
    )


def _run_transcription(
    monkeypatch,
    batch_size,
    *,
    fail_batched=False,
    is_multilingual=True,
    language="en",
):
    audio = np.concatenate(
        [np.full(TARGET_SAMPLE_RATE, segment_number, dtype=np.float32) for segment_number in (1, 2, 3)]
    )
    speech_segments = [
        {"start": 0, "end": TARGET_SAMPLE_RATE},
        {"start": TARGET_SAMPLE_RATE, "end": 2 * TARGET_SAMPLE_RATE},
        {"start": 2 * TARGET_SAMPLE_RATE, "end": 3 * TARGET_SAMPLE_RATE},
    ]
    model = FakeModel(fail_batched=fail_batched, is_multilingual=is_multilingual)
    processor = FakeProcessor()

    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (model, processor))
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(
        transformers_pipeline,
        "_get_speech_segments",
        lambda *args: (speech_segments, [(0, len(audio))]),
    )
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    segments = transformers_pipeline.transcribe(
        "input.wav",
        "test-model",
        "cpu",
        batch_size=batch_size,
        language=language,
    )
    return segments, model


def test_transcribe_batches_without_changing_segments(monkeypatch):
    sequential, sequential_model = _run_transcription(monkeypatch, 1)
    batched, batched_model = _run_transcription(monkeypatch, 2)

    assert batched == sequential
    assert sequential_model.batch_sizes == [1, 1, 1]
    assert batched_model.batch_sizes == [2, 1]


def test_transcribe_retries_batch_after_out_of_memory(monkeypatch):
    segments, model = _run_transcription(monkeypatch, 2, fail_batched=True)

    assert [segment["text"] for segment in segments] == ["Segment 1", "Segment 2", "Segment 3"]
    assert model.batch_sizes == [2, 1, 1, 1]


def test_transcribe_clamps_offsets_and_preserves_terminal_text(monkeypatch):
    interval_start = TARGET_SAMPLE_RATE // 4
    interval_end = 3 * TARGET_SAMPLE_RATE // 4
    audio = np.ones(TARGET_SAMPLE_RATE, dtype=np.float32)
    offsets = [
        {"text": " Leading. Still leading.", "timestamp": (-1.0, 0.2)},
        {"text": " overlap", "timestamp": (0.1, 0.3)},
        {"text": " terminal", "timestamp": (0.4, None)},
    ]
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(
        transformers_pipeline,
        "_get_speech_segments",
        lambda *args: (
            [{"start": interval_start, "end": interval_end}],
            [(interval_start, interval_end)],
        ),
    )
    monkeypatch.setattr(
        transformers_pipeline,
        "_load_model_and_processor",
        lambda *args: (FakeModel(), FakeProcessor()),
    )
    monkeypatch.setattr(
        transformers_pipeline,
        "_transcribe_speech_segments",
        lambda *args: [offsets],
    )
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    segments = transformers_pipeline.transcribe(
        "input.wav",
        "test-model",
        "cpu",
    )

    assert segments == [
        {
            "text": " Leading. Still leading.",
            "start_time": "00:00:00.250",
            "end_time": "00:00:00.450",
        },
        {"text": " overlap", "start_time": "00:00:00.450", "end_time": "00:00:00.550"},
        {"text": " terminal", "start_time": "00:00:00.650", "end_time": "00:00:00.750"},
    ]


def test_english_only_model_does_not_request_language_or_task(monkeypatch):
    _, model = _run_transcription(monkeypatch, 1, is_multilingual=False)

    assert model.generation_calls[0]["language"] is None
    assert model.generation_calls[0]["task"] is None


@pytest.mark.parametrize(
    ("language", "expected_language"),
    [("fr", "fr"), (None, None)],
)
def test_multilingual_model_supports_explicit_or_detected_language(monkeypatch, language, expected_language):
    _, model = _run_transcription(monkeypatch, 1, language=language)

    generation_call = model.generation_calls[0]
    assert generation_call["task"] == "transcribe"
    assert generation_call.get("language") == expected_language


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


def test_music_candidate_skips_whisper(monkeypatch, audio_classifier):
    audio_classifier.model.side_effect = None
    audio_classifier.model.return_value = SimpleNamespace(logits=torch.tensor([[-1.0, 1.0]]))
    model_loader = Mock(side_effect=AssertionError("Whisper must not load for music"))
    monkeypatch.setattr(
        transformers_pipeline,
        "load_pcm_wav",
        lambda path: np.ones(TARGET_SAMPLE_RATE, dtype=np.float32),
    )
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 0, "end": TARGET_SAMPLE_RATE}],
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", model_loader)

    assert transformers_pipeline.transcribe("music.wav", "test-model", "cpu") == []
    model_loader.assert_not_called()
    audio_classifier.feature_loader.assert_called_once_with(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        revision="f826b80d28226b62986cc218e5cec390b1096902",
    )
    audio_classifier.model_loader.assert_called_once_with(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        revision="f826b80d28226b62986cc218e5cec390b1096902",
        use_safetensors=True,
    )


def test_classifier_ignores_unclassifiable_tail(monkeypatch, audio_classifier):
    audio = np.ones(10 * TARGET_SAMPLE_RATE + 384, dtype=np.float32)
    model = FakeModel()
    processor = FakeProcessor()
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 0, "end": len(audio)}],
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (model, processor))
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    assert transformers_pipeline.transcribe("input.wav", "test-model", "cpu")

    classifier_windows = audio_classifier.feature_extractor.call_args.args[0]
    assert [len(window) for window in classifier_windows] == [10 * TARGET_SAMPLE_RATE]


def test_classifier_keeps_short_speech_tail(monkeypatch, audio_classifier):
    tail_samples = TARGET_SAMPLE_RATE // 2
    audio = np.concatenate(
        [
            np.zeros(10 * TARGET_SAMPLE_RATE, dtype=np.float32),
            np.ones(tail_samples, dtype=np.float32),
        ]
    )
    audio_classifier.model.side_effect = None
    audio_classifier.model.return_value = SimpleNamespace(
        logits=torch.tensor(
            [
                [-4.0, 2.0],
                [-0.2, 1.4],
            ]
        )
    )
    model = FakeModel()
    processor = FakeProcessor(timestamp=(10.0, 10.5))
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 0, "end": len(audio)}],
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (model, processor))
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    segments = transformers_pipeline.transcribe("mixed.wav", "test-model", "cpu")

    assert [segment["text"] for segment in segments] == ["Segment 0"]
    assert model.batch_sizes == [1]
    classifier_windows = audio_classifier.feature_extractor.call_args.args[0]
    assert [len(window) for window in classifier_windows] == [10 * TARGET_SAMPLE_RATE, tail_samples]


def test_transcribe_drops_text_from_music_only_windows(monkeypatch):
    audio = np.ones(30 * TARGET_SAMPLE_RATE, dtype=np.float32)
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(
        transformers_pipeline,
        "_get_speech_segments",
        lambda *args: (
            [{"start": 0, "end": len(audio)}],
            [(10 * TARGET_SAMPLE_RATE, 20 * TARGET_SAMPLE_RATE)],
        ),
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args: (object(), object()))
    monkeypatch.setattr(
        transformers_pipeline,
        "_transcribe_chunks",
        lambda *args: [
            {"text": "music intro", "timestamp": (1.0, 2.0)},
            {"text": "spoken words", "timestamp": (12.0, 13.0)},
            {"text": "music outro", "timestamp": (21.0, 22.0)},
        ],
    )
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    segments = transformers_pipeline.transcribe("mixed.wav", "test-model", "cpu")

    assert [segment["text"] for segment in segments] == ["spoken words"]


@pytest.mark.parametrize("duration_samples", [4000, 6992, 7984])
def test_every_detected_short_interval_reaches_whisper(monkeypatch, audio_classifier, duration_samples):
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


def test_detected_intervals_are_clamped_to_audio_bounds(monkeypatch, audio_classifier):
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


def test_empty_decoder_result_does_not_retry_full_audio(monkeypatch, audio_classifier):
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
def test_single_segment_decoder_error_propagates(monkeypatch, audio_classifier, decoder_error):
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


def test_model_loader_error_propagates(monkeypatch, audio_classifier):
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
