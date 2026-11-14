"""Whisper ASR backend with local speech and music detection."""

import gc
import math
import logging
from typing import Any, TypedDict, NotRequired, cast
from functools import lru_cache

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import (
    WhisperProcessor,
    ASTFeatureExtractor,
    ASTForAudioClassification,
    WhisperForConditionalGeneration,
)

from textplease.utils.time_utils import format_time_precise as format_time
from textplease.utils.audio_utils import TARGET_SAMPLE_RATE, load_pcm_wav


logger = logging.getLogger(__name__)


class _WhisperOffset(TypedDict):
    previous_end: NotRequired[float]
    chunk_id: NotRequired[int]
    span_id: NotRequired[str]
    text: str
    timestamp: tuple[float | None, float | None]
    timing_status: NotRequired[str]
    source_timestamp: NotRequired[tuple[float, float]]
    local_timestamp: NotRequired[tuple[float | None, float | None]]


class _RecordingVAD:
    def __init__(self, model):
        self.model = model
        self.probabilities: list[float] = []

    def reset_states(self):
        self.model.reset_states()
        self.probabilities.clear()

    def __call__(self, chunk, sampling_rate):
        probability = self.model(chunk, sampling_rate)
        self.probabilities.append(probability.item())
        return probability


def _build_chunks(audio: np.ndarray, segments: list[dict[str, int]]) -> list[tuple[int, int, np.ndarray]]:
    return [(item["start"], item["end"], audio[item["start"] : item["end"]]) for item in segments]


def _retention_decision(offset: _WhisperOffset, windows: list[tuple[int, int]]) -> tuple[bool, str]:
    start, end = offset["timestamp"]
    # Unresolved timing cannot establish that text falls outside admitted speech.
    if start is None or end is None or not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        return True, "unresolved_timing"
    bounds = offset.get("source_timestamp")
    if bounds is not None and (start < bounds[0] or end > bounds[1]):
        return True, "outside_source_bounds"
    if start < offset.get("previous_end", 0.0):
        return True, "overlapping_timing_review"
    retained = any(end > a / TARGET_SAMPLE_RATE and start < b / TARGET_SAMPLE_RATE for a, b in windows)
    return retained, "audioset_overlap" if retained else "no_audioset_overlap"


@lru_cache(maxsize=1)
def _load_model_and_processor(
    model_name: str,
    device: str,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load Whisper model and processor."""
    logger.info(f"Loading Transformers model '{model_name}' on device: {device}")
    processor = WhisperProcessor.from_pretrained(model_name)
    torch_dtype = torch.float16 if device not in ("cpu", "mps") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model = model.to(torch.device(device))
    return model, processor


def _get_speech_segments(
    audio_array: np.ndarray,
    device: str,
    diagnostics: list[dict] | None = None,
) -> tuple[list[dict[str, int]], list[tuple[int, int]]]:
    """Run speech detection and return bounded intervals in source samples."""
    vad_model = load_silero_vad()
    recorder = _RecordingVAD(vad_model) if diagnostics is not None else None
    audio_tensor = torch.from_numpy(audio_array)

    detected = get_speech_timestamps(
        audio_tensor,
        recorder if recorder is not None else vad_model,
        threshold=0.5,
        sampling_rate=TARGET_SAMPLE_RATE,
        min_speech_duration_ms=250,
        min_silence_duration_ms=2000,
        speech_pad_ms=100,
        return_seconds=False,
    )

    if diagnostics is not None:
        diagnostics.append(
            {
                "kind": "vad",
                "id": "vad",
                "sample_rate": TARGET_SAMPLE_RATE,
                "step_samples": 512,
                "probabilities": recorder.probabilities if recorder else [],
                "candidates": detected,
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 100,
            }
        )
    segments: list[dict[str, int]] = []
    for segment in detected:
        start = max(0, int(segment["start"]))
        end = min(len(audio_array), int(segment["end"]))
        if end > start:
            segments.append({"start": start, "end": end})

    if not segments:
        return [], []

    classifier_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    classifier_revision = "f826b80d28226b62986cc218e5cec390b1096902"
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        classifier_name,
        revision=classifier_revision,
    )
    classifier = ASTForAudioClassification.from_pretrained(
        classifier_name,
        revision=classifier_revision,
        use_safetensors=True,
    ).to(torch.device(device))
    classifier.eval()
    speech_label = classifier.config.label2id.get("Speech")
    music_label = classifier.config.label2id.get("Music")
    if not isinstance(speech_label, int) or not isinstance(music_label, int):
        raise ValueError("Audio classifier must provide Speech and Music labels")

    windows: list[tuple[int, int, int]] = []
    window_samples = 10 * TARGET_SAMPLE_RATE
    for segment_index, segment in enumerate(segments):
        window_starts = list(range(segment["start"], segment["end"], window_samples))
        # Kaldi fbank needs one 25 ms frame; keep the existing policy of leaving shorter tails unclassified.
        if len(window_starts) > 1 and segment["end"] - window_starts[-1] < TARGET_SAMPLE_RATE // 40:
            tail_start = window_starts.pop()
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "kind": "unclassified_tail",
                        "candidate_id": segment_index,
                        "start_sample": tail_start,
                        "end_sample": segment["end"],
                        "reason": "shorter_than_fbank_frame",
                    }
                )
        for window_start in window_starts:
            window_end = min(window_start + window_samples, segment["end"])
            windows.append((segment_index, window_start, window_end))

    retained_segments: set[int] = set()
    speech_windows: list[tuple[int, int]] = []
    for batch_start in range(0, len(windows), 16):
        batch = windows[batch_start : batch_start + 16]
        inputs = feature_extractor(
            [audio_array[window_start:window_end] for _, window_start, window_end in batch],
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = classifier(input_values=inputs.input_values.to(device)).logits
        probabilities = torch.sigmoid(logits).cpu()
        for (segment_index, window_start, window_end), scores in zip(batch, probabilities, strict=True):
            retained = bool(scores[speech_label] >= 0.4 or scores[music_label] < 0.5)
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "kind": "audioset_window",
                        "candidate_id": segment_index,
                        "start_sample": window_start,
                        "end_sample": window_end,
                        "speech_probability": float(scores[speech_label]),
                        "music_probability": float(scores[music_label]),
                        "retained": retained,
                    }
                )
            if retained:
                retained_segments.add(segment_index)
                speech_windows.append((window_start, window_end))

    if diagnostics is not None:
        diagnostics.extend(
            {
                "kind": "candidate",
                "id": index,
                **segment,
                "retained": index in retained_segments,
                "reason": "audioset_admission" if index in retained_segments else "no_admitted_window",
            }
            for index, segment in enumerate(segments)
        )
    speech_segments = [
        dict(segment, candidate_id=index) for index, segment in enumerate(segments) if index in retained_segments
    ]

    total_s = len(audio_array) / TARGET_SAMPLE_RATE
    retained_s = sum(segment["end"] - segment["start"] for segment in speech_segments) / TARGET_SAMPLE_RATE
    logger.info(
        f"Speech detection: {len(speech_segments)}/{len(segments)} VAD candidates retained — "
        f"{retained_s:.1f}s / {total_s:.1f}s total "
        f"({100 * retained_s / max(total_s, 1):.0f}% candidate audio)"
    )
    return speech_segments, speech_windows


def _transcribe_speech_segments(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_chunks: list[np.ndarray],
    device: str,
    language: str | None,
    diagnostics: list[dict] | None = None,
) -> list[list[_WhisperOffset]]:
    """Transcribe speech chunks and return decoded offsets for each chunk."""
    is_multilingual = getattr(model.generation_config, "is_multilingual", True)
    if not is_multilingual and language not in (None, "en", "english"):
        raise ValueError(f"This model is English-only and cannot transcribe language '{language}'")
    torch_dtype = torch.float16 if device not in ("cpu", "mps") else torch.float32

    inputs = processor(
        audio_chunks if len(audio_chunks) > 1 else audio_chunks[0],
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        sampling_rate=TARGET_SAMPLE_RATE,
    )
    input_features = inputs.input_features.to(device=device, dtype=torch_dtype)
    attention_mask = inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = None

    with torch.no_grad():
        generated: object = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            language=language if is_multilingual else None,
            task="transcribe" if is_multilingual else None,
            return_timestamps=True,
            return_segments=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

    if not isinstance(generated, dict):
        raise ValueError("Whisper did not return structured segments")
    generated = cast(dict[str, Any], generated)
    if len(generated["sequences"]) != len(audio_chunks) or len(generated["segments"]) != len(audio_chunks):
        raise ValueError("Whisper returned an unexpected number of decoded inputs")
    decoded_offsets: list[list[_WhisperOffset]] = []
    timestamp_begin = processor.tokenizer.convert_tokens_to_ids("<|0.00|>")
    for token_ids, segments in zip(generated["sequences"], generated["segments"], strict=True):
        offsets: list[_WhisperOffset] = []
        raw_segments = []
        for segment in segments:
            tokens = segment["tokens"].detach().cpu().tolist()
            text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            raw_times = [segment.get("start"), segment.get("end")]
            raw_times = [
                value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value for value in raw_times
            ]
            text_positions = [i for i, token in enumerate(tokens) if token < processor.tokenizer.eos_token_id]
            # Generator fallback bounds are not timestamp predictions when the corresponding token is absent.
            supported = (
                bool(text_positions) and any(token >= timestamp_begin for token in tokens[: text_positions[0]]),
                bool(text_positions) and any(token >= timestamp_begin for token in tokens[text_positions[-1] + 1 :]),
            )
            times = [
                float(value) if present and isinstance(value, (int, float)) and not isinstance(value, bool) else None
                for value, present in zip(raw_times, supported, strict=True)
            ]
            offsets.append(
                {
                    "text": text,
                    "timestamp": (times[0], times[1]),
                    "timing_status": "model" if all(supported) else "missing",
                }
            )
            if diagnostics is not None:
                raw_segments.append(
                    {"token_ids": tokens, "timestamp": raw_times, "timestamp_token_present": supported}
                )
        if not offsets:
            text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            if text.strip():
                offsets.append({"text": text, "timestamp": (None, None), "timing_status": "missing"})
        if diagnostics is not None:
            diagnostics.append(
                {
                    "kind": "decode",
                    "id": len(diagnostics),
                    "token_ids": token_ids.detach().cpu().tolist(),
                    "segments": raw_segments,
                    "decoded": {"offsets": offsets},
                    "timestamp_source": "generation_segments",
                }
            )
        decoded_offsets.append(offsets)
    return decoded_offsets


def _transcribe_chunks(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    chunks: list[tuple[int, int, np.ndarray]],
    batch_size: int,
    device: str,
    language: str | None,
    diagnostics: list[dict] | None = None,
) -> list[_WhisperOffset]:
    all_offsets: list[_WhisperOffset] = []
    previous_end_s = 0.0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        first_start, _, _ = batch[0]
        _, last_end, _ = batch[-1]
        last_index = batch_start + len(batch)
        logger.info(
            f"Transcribing speech segment {batch_start + 1}/{len(chunks)}: "
            f"batch {batch_start + 1}-{last_index} "
            f"[{first_start / TARGET_SAMPLE_RATE:.2f}s → {last_end / TARGET_SAMPLE_RATE:.2f}s]"
        )

        audio_chunks = [chunk for _, _, chunk in batch]
        debug_start = len(diagnostics) if diagnostics is not None else 0
        debug_arguments = {} if diagnostics is None else {"diagnostics": diagnostics}
        try:
            batch_offsets = _transcribe_speech_segments(
                model, processor, audio_chunks, device, language, **debug_arguments
            )
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            logger.warning("Whisper batch exhausted accelerator memory; retrying one segment at a time")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_offsets = [
                _transcribe_speech_segments(model, processor, [chunk], device, language, **debug_arguments)[0]
                for chunk in audio_chunks
            ]

        if diagnostics is not None:
            for index, record in enumerate(item for item in diagnostics[debug_start:] if item.get("kind") == "decode"):
                record["chunk_id"] = batch_start + index
        for chunk_index, ((start, end, _), offsets) in enumerate(zip(batch, batch_offsets, strict=True), batch_start):
            start_s = start / TARGET_SAMPLE_RATE
            end_s = end / TARGET_SAMPLE_RATE
            for offset_index, offset in enumerate(offsets):
                local_start, local_end = offset["timestamp"]
                absolute_start = None if local_start is None else local_start + start_s
                absolute_end = None if local_end is None else local_end + start_s
                status = "model"
                if absolute_start is None or absolute_end is None:
                    status = "missing"
                elif not (math.isfinite(absolute_start) and math.isfinite(absolute_end)):
                    status = "invalid"
                elif absolute_start < start_s or absolute_end > end_s or absolute_end <= absolute_start:
                    status = "out_of_bounds"
                elif absolute_start < previous_end_s:
                    status = "overlap"
                prior_end_s = previous_end_s
                if absolute_end is not None and math.isfinite(absolute_end):
                    previous_end_s = max(previous_end_s, absolute_end)
                all_offsets.append(
                    {
                        "text": offset["text"],
                        "previous_end": prior_end_s,
                        "chunk_id": chunk_index,
                        "span_id": f"{chunk_index}:{offset_index}",
                        "timestamp": (absolute_start, absolute_end),
                        "timing_status": status,
                        "source_timestamp": (start_s, end_s),
                        "local_timestamp": (local_start, local_end),
                    }
                )

    return all_offsets


def _offsets_to_segments(offsets: list[_WhisperOffset]) -> list[dict[str, str]]:
    """Convert decoded timestamp offsets to the standard {start_time, end_time, text} format."""
    segments: list[dict[str, str]] = []
    for chunk in offsets:
        text = chunk.get("text", "")
        ts = chunk.get("timestamp", (0.0, 0.0))
        if not text.strip() or len(ts) != 2:
            continue
        segments.append(
            {
                "start_time": format_time(ts[0]) if ts[0] is not None and math.isfinite(ts[0]) and ts[0] >= 0 else "",
                "end_time": format_time(ts[1]) if ts[1] is not None and math.isfinite(ts[1]) and ts[1] >= 0 else "",
                "text": text,
            }
        )
    return segments


def transcribe(
    audio_path: str,
    model_name: str,
    device: str,
    *,
    language: str | None = "en",
    batch_size: int = 1,
    diagnostics: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Transcribe a normalized mono 16 kHz PCM16 WAV."""
    if batch_size < 1:
        raise ValueError("Whisper batch size must be positive")

    audio_array = load_pcm_wav(audio_path)
    debug_arguments = {} if diagnostics is None else {"diagnostics": diagnostics}
    speech_segments, speech_windows = _get_speech_segments(audio_array, device, **debug_arguments)
    if not speech_segments:
        logger.info("No speech detected")
        return []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "mps":
        torch.mps.empty_cache()

    model, processor = _load_model_and_processor(model_name, device)
    chunks = _build_chunks(audio_array, speech_segments)
    if diagnostics is not None:
        diagnostics.extend(
            {
                "kind": "chunk",
                "id": index,
                "candidate_id": segment.get("candidate_id", index),
                "start_sample": start,
                "end_sample": end,
            }
            for index, (segment, (start, end, _)) in enumerate(zip(speech_segments, chunks, strict=True))
        )
    offsets = _transcribe_chunks(
        model,
        processor,
        chunks,
        batch_size,
        device,
        language,
        **debug_arguments,
    )
    retained_offsets: list[_WhisperOffset] = []
    for offset in offsets:
        retained, reason = _retention_decision(offset, speech_windows)
        if diagnostics is not None:
            record = {"kind": "span", **offset, "retained": retained, "reason": reason}
            for field in ("timestamp", "local_timestamp", "source_timestamp"):
                record[field] = [
                    str(value) if value is not None and not math.isfinite(value) else value
                    for value in offset.get(field, ())
                ]
            diagnostics.append(record)
        if retained:
            retained_offsets.append(offset)
    uncertain = sum(offset.get("timing_status", "model") != "model" for offset in retained_offsets)
    if uncertain:
        logger.warning("%d transcript spans have timestamps that need review", uncertain)
    offsets = retained_offsets
    segments = _offsets_to_segments(offsets)
    if segments:
        logger.info(f"Generated {len(segments)} segments via VAD + model.generate()")
        return segments

    logger.info("Whisper returned no usable timestamped text for detected speech")
    return []
