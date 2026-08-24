"""Whisper ASR backend: Silero-VAD preprocessing + model.generate() per speech segment."""

import gc
import re
import logging
import warnings
from typing import TypedDict
from functools import lru_cache

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from textplease.utils.time_utils import format_time_precise as format_time
from textplease.utils.audio_utils import TARGET_SAMPLE_RATE, load_pcm_wav


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*Whisper did not predict an ending timestamp.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")


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
    processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
    torch_dtype = torch.float16 if device not in ("cpu", "mps") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_safetensors=True,
    )
    model = model.to(torch.device(device))
    return model, processor


def _get_speech_segments(
    audio_array: np.ndarray,
) -> list[dict[str, int]]:
    """Run Silero VAD and return bounded speech intervals in source samples."""
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

    total_s = len(audio_array) / TARGET_SAMPLE_RATE
    speech_s = sum(segment["end"] - segment["start"] for segment in segments) / TARGET_SAMPLE_RATE
    logger.info(
        f"VAD: {len(segments)} speech segments — "
        f"{speech_s:.1f}s / {total_s:.1f}s total ({100 * speech_s / max(total_s, 1):.0f}% speech)"
    )
    return segments


def _transcribe_speech_segments(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_chunks: list[np.ndarray],
    device: str,
    language: str,
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

    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            language=language,
            task="transcribe",
            return_timestamps=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
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
    language: str,
) -> list[_WhisperOffset]:
    all_offsets: list[_WhisperOffset] = []
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

        for (start, _, _), offsets in zip(batch, batch_offsets, strict=True):
            start_s = start / TARGET_SAMPLE_RATE
            for offset in offsets:
                ts = offset.get("timestamp", (0.0, 0.0))
                if len(ts) == 2 and ts[0] is not None and ts[1] is not None:
                    offset["timestamp"] = (ts[0] + start_s, ts[1] + start_s)
            all_offsets.extend(offsets)

    return all_offsets


def _offsets_to_segments(offsets: list[_WhisperOffset]) -> list[dict[str, str]]:
    """Convert decoded timestamp offsets to the standard {start_time, end_time, text} format."""
    segments: list[dict[str, str]] = []
    for chunk in offsets:
        text = chunk.get("text", "").strip()
        ts = chunk.get("timestamp", (0.0, 0.0))
        if not text or len(ts) != 2 or ts[0] is None or ts[1] is None:
            continue
        segments.extend(_split_chunk_by_sentences(text, float(ts[0]), float(ts[1])))
    return segments


def transcribe(
    audio_path: str,
    model_name: str,
    device: str,
    *,
    language: str = "en",
    batch_size: int = 1,
) -> list[dict[str, str]]:
    """Transcribe a normalized mono 16 kHz PCM16 WAV."""
    if batch_size < 1:
        raise ValueError("Whisper batch size must be positive")

    audio_array = load_pcm_wav(audio_path)
    speech_segments = _get_speech_segments(audio_array)
    if not speech_segments:
        logger.info("No speech detected")
        return []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    segments = _offsets_to_segments(offsets)
    if segments:
        logger.info(f"Generated {len(segments)} segments via VAD + model.generate()")
        return segments

    logger.info("Whisper returned no usable timestamped text for detected speech")
    return []


def _split_chunk_by_sentences(text: str, start_time: float, end_time: float) -> list[dict[str, str]]:
    """Split a segment's text at sentence boundaries, distributing duration proportionally."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return [{"start_time": format_time(start_time), "end_time": format_time(end_time), "text": text}]

    duration = end_time - start_time
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return [{"start_time": format_time(start_time), "end_time": format_time(end_time), "text": text}]

    segments: list[dict[str, str]] = []
    current_time = start_time
    for sentence in sentences:
        end = min(current_time + (len(sentence) / total_chars) * duration, end_time)
        segments.append(
            {
                "start_time": format_time(current_time),
                "end_time": format_time(end),
                "text": sentence,
            }
        )
        current_time = end
    return segments
