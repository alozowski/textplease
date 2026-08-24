"""Whisper ASR backend with local speech and music detection."""

import gc
import logging
from typing import TypedDict
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
    text: str
    timestamp: tuple[float | None, float | None]


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
) -> tuple[list[dict[str, int]], list[tuple[int, int]]]:
    """Run speech detection and return bounded intervals in source samples."""
    vad_model = load_silero_vad()
    audio_tensor = torch.from_numpy(audio_array)

    detected = get_speech_timestamps(
        audio_tensor,
        vad_model,
        threshold=0.5,
        sampling_rate=TARGET_SAMPLE_RATE,
        min_speech_duration_ms=250,
        min_silence_duration_ms=2000,
        speech_pad_ms=100,
        return_seconds=False,
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
        # Kaldi fbank needs one 25 ms frame; the previous complete window covers anything shorter.
        if len(window_starts) > 1 and segment["end"] - window_starts[-1] < TARGET_SAMPLE_RATE // 40:
            window_starts.pop()
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
            if scores[speech_label] >= 0.4 or scores[music_label] < 0.5:
                retained_segments.add(segment_index)
                speech_windows.append((window_start, window_end))

    speech_segments = [segment for index, segment in enumerate(segments) if index in retained_segments]

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
) -> list[list[_WhisperOffset]]:
    """Transcribe speech chunks and return decoded offsets for each chunk."""
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

    is_multilingual = getattr(model.generation_config, "is_multilingual", True)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            language=language if is_multilingual else None,
            task="transcribe" if is_multilingual else None,
            return_timestamps=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

    time_precision = processor.feature_extractor.chunk_length / model.config.max_source_positions
    decoded_offsets: list[list[_WhisperOffset]] = []
    for token_ids in generated_ids:
        decoded: object = processor.tokenizer.decode(
            token_ids,
            output_offsets=True,
            time_precision=time_precision,
        )
        offsets: list[_WhisperOffset] = []
        raw_offsets: object = decoded.get("offsets") if isinstance(decoded, dict) else None
        if isinstance(raw_offsets, list):
            for raw_offset in raw_offsets:
                if not isinstance(raw_offset, dict):
                    continue
                text = raw_offset.get("text")
                timestamp = raw_offset.get("timestamp")
                if not isinstance(text, str) or not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                    continue
                start, end = timestamp
                if start is not None and not isinstance(start, (int, float)):
                    continue
                if end is not None and not isinstance(end, (int, float)):
                    continue
                offsets.append(
                    {
                        "text": text,
                        "timestamp": (
                            float(start) if start is not None else None,
                            float(end) if end is not None else None,
                        ),
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
        try:
            batch_offsets = _transcribe_speech_segments(model, processor, audio_chunks, device, language)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            logger.warning("Whisper batch exhausted accelerator memory; retrying one segment at a time")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_offsets = [
                _transcribe_speech_segments(model, processor, [chunk], device, language)[0] for chunk in audio_chunks
            ]

        for (start, end, _), offsets in zip(batch, batch_offsets, strict=True):
            start_s = start / TARGET_SAMPLE_RATE
            end_s = end / TARGET_SAMPLE_RATE
            for offset_index, offset in enumerate(offsets):
                local_start, local_end = offset["timestamp"]
                if local_start is None:
                    continue
                if local_end is None and offset_index != len(offsets) - 1:
                    continue

                bounded_start = max(start_s, local_start + start_s, previous_end_s)
                bounded_end = end_s if local_end is None else min(end_s, local_end + start_s)
                if bounded_end <= bounded_start:
                    continue

                all_offsets.append(
                    {
                        "text": offset["text"],
                        "timestamp": (bounded_start, bounded_end),
                    }
                )
                previous_end_s = bounded_end

    return all_offsets


def _offsets_to_segments(offsets: list[_WhisperOffset]) -> list[dict[str, str]]:
    """Convert decoded timestamp offsets to the standard {start_time, end_time, text} format."""
    segments: list[dict[str, str]] = []
    for chunk in offsets:
        text = chunk.get("text", "")
        ts = chunk.get("timestamp", (0.0, 0.0))
        if not text.strip() or len(ts) != 2 or ts[0] is None or ts[1] is None:
            continue
        segments.append(
            {
                "start_time": format_time(float(ts[0])),
                "end_time": format_time(float(ts[1])),
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
) -> list[dict[str, str]]:
    """Transcribe a normalized mono 16 kHz PCM16 WAV."""
    if batch_size < 1:
        raise ValueError("Whisper batch size must be positive")

    audio_array = load_pcm_wav(audio_path)
    speech_segments, speech_windows = _get_speech_segments(audio_array, device)
    if not speech_segments:
        logger.info("No speech detected")
        return []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "mps":
        torch.mps.empty_cache()

    model, processor = _load_model_and_processor(model_name, device)
    chunks = [
        (segment["start"], segment["end"], audio_array[segment["start"] : segment["end"]])
        for segment in speech_segments
    ]
    offsets = _transcribe_chunks(
        model,
        processor,
        chunks,
        batch_size,
        device,
        language,
    )
    offsets = [
        offset
        for offset in offsets
        if offset["timestamp"][0] is not None
        and offset["timestamp"][1] is not None
        and any(
            offset["timestamp"][1] > window_start / TARGET_SAMPLE_RATE
            and offset["timestamp"][0] < window_end / TARGET_SAMPLE_RATE
            for window_start, window_end in speech_windows
        )
    ]
    segments = _offsets_to_segments(offsets)
    if segments:
        logger.info(f"Generated {len(segments)} segments via VAD + model.generate()")
        return segments

    logger.info("Whisper returned no usable timestamped text for detected speech")
    return []
