from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import pytest
from transformers import BatchFeature

from textplease.backends import transformers_pipeline
from textplease.utils.audio_utils import TARGET_SAMPLE_RATE


class FakeTokenizer:
    eos_token_id = 100

    def __init__(self, include_text=True):
        self.include_text = include_text

    def convert_tokens_to_ids(self, token):
        assert token == "<|0.00|>"
        return 1000

    def decode(self, token_ids, **kwargs):
        assert not kwargs.get("output_offsets")
        if not self.include_text:
            return ""
        number = next(int(token) for token in token_ids if int(token) < self.eos_token_id)
        return f"Segment {number}"


class FakeProcessor:
    def __init__(self, include_text: bool = True):
        self.tokenizer = FakeTokenizer(include_text)
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
    def __init__(
        self,
        fail_batched: bool = False,
        error: Exception | None = None,
        *,
        is_multilingual: bool = True,
        timestamp=(0.0, 0.5),
    ):
        self.batch_sizes = []
        self.fail_batched = fail_batched
        self.error = error
        self.generation_config = SimpleNamespace(is_multilingual=is_multilingual)
        self.generation_calls = []
        self.timestamp = timestamp

    def generate(self, **kwargs):
        self.generation_calls.append(kwargs)
        input_features = kwargs["input_features"]
        batch_size = len(input_features)
        self.batch_sizes.append(batch_size)
        if self.error is not None:
            raise self.error
        if self.fail_batched and batch_size > 1:
            raise torch.OutOfMemoryError
        sequences = [torch.tensor([1000, int(item[0, 0]), 1025]) for item in input_features]
        return {
            "sequences": sequences,
            "segments": [
                [{"tokens": tokens, "start": self.timestamp[0], "end": self.timestamp[1]}] for tokens in sequences
            ],
        }


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

    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args, **kwargs: (model, processor))
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(
        transformers_pipeline,
        "_get_speech_segments",
        lambda *args, **kwargs: (speech_segments, [(0, len(audio))]),
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


def test_transcribe_preserves_text_with_uncertain_timing(monkeypatch):
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
        lambda *args, **kwargs: (
            [{"start": interval_start, "end": interval_end}],
            [(interval_start, interval_end)],
        ),
    )
    monkeypatch.setattr(
        transformers_pipeline,
        "_load_model_and_processor",
        lambda *args, **kwargs: (FakeModel(), FakeProcessor()),
    )
    monkeypatch.setattr(
        transformers_pipeline,
        "_transcribe_speech_segments",
        lambda *args, **kwargs: [offsets],
    )
    monkeypatch.setattr(transformers_pipeline.torch.cuda, "is_available", lambda: False)

    segments = transformers_pipeline.transcribe(
        "input.wav",
        "test-model",
        "cpu",
    )

    assert [segment["text"] for segment in segments] == [item["text"] for item in offsets]
    assert all(set(segment) == {"start_time", "end_time", "text"} for segment in segments)
    assert segments[0]["start_time"] == ""
    assert segments[1]["start_time"] == "00:00:00.350"
    assert segments[-1]["end_time"] == ""


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
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args, **kwargs: (model, processor))
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
    model = FakeModel(timestamp=(10.0, 10.5))
    processor = FakeProcessor()
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": 0, "end": len(audio)}],
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args, **kwargs: (model, processor))
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
        lambda *args, **kwargs: (
            [{"start": 0, "end": len(audio)}],
            [(10 * TARGET_SAMPLE_RATE, 20 * TARGET_SAMPLE_RATE)],
        ),
    )
    monkeypatch.setattr(
        transformers_pipeline, "_load_model_and_processor", lambda *args, **kwargs: (object(), object())
    )
    monkeypatch.setattr(
        transformers_pipeline,
        "_transcribe_chunks",
        lambda *args, **kwargs: [
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
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args, **kwargs: (model, processor))
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
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args, **kwargs: (model, processor))
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
    processor = FakeProcessor(include_text=False)

    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: audio)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline,
        "get_speech_timestamps",
        lambda *args, **kwargs: [{"start": start, "end": end}],
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args, **kwargs: (model, processor))
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
        lambda *args, **kwargs: (model, FakeProcessor()),
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


@pytest.mark.parametrize("language", ["ru", "russian"])
def test_english_only_rejects_russian_before_generation(monkeypatch, language):
    with pytest.raises(ValueError, match="English-only"):
        _run_transcription(monkeypatch, 1, is_multilingual=False, language=language)


def test_overlapping_interval_text_is_retained(monkeypatch):
    monkeypatch.setattr(
        transformers_pipeline,
        "_transcribe_speech_segments",
        lambda *args, **kwargs: [
            [
                {"text": "first", "timestamp": (1.0, 10.0)},
                {"text": "second", "timestamp": (5.0, 6.0)},
                {"text": "last", "timestamp": (11.0, None)},
            ]
        ],
    )
    result = transformers_pipeline._transcribe_chunks(None, None, [(0, 480000, np.zeros(480000))], 1, "cpu", "ru")
    assert [item["text"] for item in result] == ["first", "second", "last"]
    assert result[-1]["timestamp"] == (11.0, None)
    assert result[1]["timing_status"] == "overlap"


def test_debug_diagnostics_preserve_timing_without_changing_transcript(monkeypatch):
    offsets = [
        {
            "text": "retained",
            "timestamp": (1.0, None),
            "timing_status": "missing",
            "local_timestamp": (1.0, None),
            "source_timestamp": (0.0, 30.0),
        },
        {"text": "filtered", "timestamp": (20.0, 21.0), "timing_status": "model"},
        {"text": "invalid", "timestamp": (float("nan"), None), "timing_status": "invalid"},
    ]
    monkeypatch.setattr(transformers_pipeline, "load_pcm_wav", lambda path: np.zeros(480000))
    monkeypatch.setattr(
        transformers_pipeline,
        "_get_speech_segments",
        lambda *args, **kwargs: ([{"start": 0, "end": 480000}], [(0, 160000)]),
    )
    monkeypatch.setattr(transformers_pipeline, "_load_model_and_processor", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(transformers_pipeline, "_transcribe_chunks", lambda *args, **kwargs: offsets)
    normal = transformers_pipeline.transcribe("unused", "test", "cpu")
    diagnostics = []
    debug = transformers_pipeline.transcribe("unused", "test", "cpu", diagnostics=diagnostics)
    assert normal == debug
    diagnostics = [record for record in diagnostics if record["kind"] == "span"]
    assert [record["retained"] for record in diagnostics] == [True, False, True]
    assert diagnostics[0]["source_timestamp"] == [0.0, 30.0]
    assert diagnostics[2]["timestamp"] == ["nan", None]


def test_chunk_mapping_preserves_admitted_samples():
    audio = np.arange(16000, dtype=np.float32)
    segments = [{"start": 123, "end": 4000}, {"start": 4500, "end": 15999}]
    chunks = transformers_pipeline._build_chunks(audio, segments)
    assert [(a, b) for a, b, _ in chunks] == [(123, 4000), (4500, 15999)]
    for a, b, chunk in chunks:
        np.testing.assert_array_equal(chunk, audio[a:b])
        assert np.shares_memory(chunk, audio)


@pytest.mark.parametrize(
    ("interval", "expected"), [((19.99, 29.0), True), ((20.0, 29.0), False), ((None, 29.0), True)]
)
def test_retention_is_independent_of_timing_label(interval, expected):
    for label in ["model", "overlap", "missing", "invalid"]:
        retained, reason = transformers_pipeline._retention_decision(
            {"text": "same words", "timestamp": interval, "timing_status": label}, [(160000, 320000)]
        )
        assert retained is expected
        assert reason


def test_malformed_offset_preserves_text_and_raw_debug():
    model = FakeModel(timestamp=("bad", None))
    processor = FakeProcessor()
    processor.tokenizer.decode = Mock(return_value="spoken words")
    diagnostics = []
    result = transformers_pipeline._transcribe_speech_segments(
        model, processor, [np.ones(16000, dtype=np.float32)], "cpu", "en", diagnostics
    )
    assert result[0][0]["text"] == "spoken words"
    assert result[0][0]["timestamp"] == (None, None)
    assert diagnostics[0]["segments"][0]["timestamp"] == ["bad", None]
    assert diagnostics[0]["token_ids"] == [1000, 1, 1025]


def test_decoding_without_offsets_preserves_full_text():
    processor = FakeProcessor()
    processor.tokenizer.decode = Mock(return_value="no timing available")
    model = FakeModel()
    model.generate = Mock(return_value={"sequences": [torch.tensor([1])], "segments": [[]]})
    result = transformers_pipeline._transcribe_speech_segments(
        model, processor, [np.ones(16000, dtype=np.float32)], "cpu", "ru"
    )
    assert result[0][0]["text"] == "no timing available"
    assert result[0][0]["timestamp"] == (None, None)


def test_short_speech_real_postprocessing_and_debug_parity(monkeypatch):
    from silero_vad import get_speech_timestamps

    class ProbabilityModel:
        def reset_states(self):
            self.index = 0

        def __call__(self, chunk, rate):
            value = 0.99 if 10 <= self.index < 15 else 0.0
            self.index += 1
            return torch.tensor(value)

    audio = np.zeros(16000 * 4, dtype=np.float32)
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", ProbabilityModel)
    monkeypatch.setattr(transformers_pipeline, "get_speech_timestamps", get_speech_timestamps)
    normal = transformers_pipeline._get_speech_segments(audio, "cpu")
    diagnostics = []
    debug = transformers_pipeline._get_speech_segments(audio, "cpu", diagnostics)
    assert normal == debug == ([], [])
    assert len(diagnostics[0]["probabilities"]) == 125
    assert sum(p > 0.5 for p in diagnostics[0]["probabilities"]) == 5


def test_rejected_admission_records_scores(monkeypatch, audio_classifier):
    audio_classifier.model.side_effect = None
    audio_classifier.model.return_value = SimpleNamespace(logits=torch.tensor([[-4.0, 4.0]]))
    monkeypatch.setattr(transformers_pipeline, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        transformers_pipeline, "get_speech_timestamps", lambda *args, **kwargs: [{"start": 0, "end": 16000}]
    )
    diagnostics = []
    assert transformers_pipeline._get_speech_segments(np.ones(16000, dtype=np.float32), "cpu", diagnostics) == ([], [])
    window = next(item for item in diagnostics if item["kind"] == "audioset_window")
    candidate = next(item for item in diagnostics if item["kind"] == "candidate")
    assert not window["retained"] and window["speech_probability"] < 0.4
    assert not candidate["retained"] and candidate["reason"] == "no_admitted_window"


def test_structured_timing_preserves_skipped_windows_and_single_source_shift():
    from transformers.models.whisper.generation_whisper import WhisperGenerationMixin, _pad_to_max_length

    sequences = [torch.tensor([1000, 1, 1250]), torch.tensor([1000, 2, 1250])]
    segments = []
    for seek, tokens in zip((0.0, 60.0), sequences, strict=True):
        retrieved, _ = WhisperGenerationMixin._retrieve_segment(
            seek_sequence=tokens,
            seek_outputs=[{}],
            time_offset=torch.tensor([seek]),
            timestamp_begin=1000,
            seek_num_frames=torch.tensor([3000]),
            time_precision=0.02,
            time_precision_features=0.01,
            input_stride=2,
            prev_idx=0,
            idx=0,
            return_token_timestamps=False,
            decoder_input_ids=torch.zeros((1, 1), dtype=torch.long),
        )
        segments.extend(retrieved)
    flat = _pad_to_max_length([segments], 100, "cpu")
    model = FakeModel()
    model.generate = Mock(return_value={"sequences": flat, "segments": [segments]})
    debug = []
    offsets = transformers_pipeline._transcribe_chunks(
        model,
        FakeProcessor(),
        [(160000, 1600000, np.ones(1440000, dtype=np.float32))],
        1,
        "cpu",
        "en",
        debug,
    )
    assert [(o["text"], o["timestamp"]) for o in offsets] == [
        ("Segment 1", (10.0, 15.0)),
        ("Segment 2", (70.0, 75.0)),
    ]
    assert model.generate.call_args.kwargs["return_segments"] is True
    assert debug[0]["segments"][1]["timestamp"] == [60.0, 65.0]


@pytest.mark.parametrize(
    "tokens, expected",
    [
        ([1000, 1], (0.0, None)),
        ([1, 1250], (None, 5.0)),
        ([1], (None, None)),
    ],
)
def test_generator_fallback_bounds_do_not_claim_timestamp_tokens(tokens, expected):
    model = FakeModel()
    model.generate = Mock(
        return_value={
            "sequences": [torch.tensor(tokens)],
            "segments": [[{"tokens": torch.tensor(tokens), "start": torch.tensor(0.0), "end": torch.tensor(5.0)}]],
        }
    )
    debug = []
    result = transformers_pipeline._transcribe_speech_segments(
        model,
        FakeProcessor(),
        [np.ones(16000, dtype=np.float32)],
        "cpu",
        "ru",
        debug,
    )
    assert result[0][0]["text"] == "Segment 1"
    assert result[0][0]["timestamp"] == expected
    assert debug[0]["segments"][0]["timestamp"] == [0.0, 5.0]
