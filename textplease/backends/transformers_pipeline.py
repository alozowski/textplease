"""Whisper ASR backend: Silero-VAD preprocessing + model.generate() per speech segment."""

import gc
import re
import logging
import warnings
from typing import TypedDict
from functools import lru_cache

import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from textplease.utils.time_utils import format_time_precise as format_time


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Padding (seconds) added around each VAD speech boundary to avoid clipping words.
_SPEECH_PAD_S = 0.1

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


def _load_audio(audio_path: str) -> np.ndarray:
    """Load audio file and return a mono 16 kHz numpy array."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
    return waveform.squeeze(0).numpy()


def _get_speech_segments(
    audio_array: np.ndarray,
    pause_threshold: float,
) -> list[dict[str, float]]:
    """Run Silero-VAD and return speech segment boundaries in seconds."""
    from silero_vad import load_silero_vad, get_speech_timestamps

    vad_model = load_silero_vad()
    audio_tensor = torch.from_numpy(audio_array)

    segments = get_speech_timestamps(
        audio_tensor,
        vad_model,
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=250,
        min_silence_duration_ms=int(pause_threshold * 1000),
        speech_pad_ms=int(_SPEECH_PAD_S * 1000),
        return_seconds=True,
    )

    total_s = len(audio_array) / SAMPLE_RATE
    speech_s = sum(s["end"] - s["start"] for s in segments)
    logger.info(
        f"VAD: {len(segments)} speech segments — "
        f"{speech_s:.1f}s / {total_s:.1f}s total ({100 * speech_s / max(total_s, 1):.0f}% speech)"
    )
    return [{"start": float(segment["start"]), "end": float(segment["end"])} for segment in segments]


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
        sampling_rate=SAMPLE_RATE,
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
    chunks: list[tuple[int, float, float, np.ndarray]],
    total_chunks: int,
    batch_size: int,
    device: str,
    language: str,
    progress_label: str,
) -> list[_WhisperOffset]:
    all_offsets: list[_WhisperOffset] = []
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        first_i, first_start_s, _, _ = batch[0]
        last_i, _, last_end_s, _ = batch[-1]
        logger.info(
            f"{progress_label} {first_i + 1}/{total_chunks}: "
            f"batch {first_i + 1}-{last_i + 1} "
            f"[{first_start_s:.2f}s → {last_end_s:.2f}s]"
        )

        audio_chunks = [chunk for _, _, _, chunk in batch]
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

        for (_, start_s, _, _), offsets in zip(batch, batch_offsets, strict=True):
            for offset in offsets:
                ts = offset.get("timestamp", (0.0, 0.0))
                if len(ts) == 2 and ts[0] is not None and ts[1] is not None:
                    offset["timestamp"] = (ts[0] + start_s, ts[1] + start_s)
            all_offsets.extend(offsets)

    return all_offsets


def _transcribe_long_form(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_array: np.ndarray,
    device: str,
    language: str,
    pause_threshold: float,
    batch_size: int,
) -> list[_WhisperOffset]:
    """Run VAD segmentation and batched transcription with offset timestamps."""
    speech_segments = _get_speech_segments(audio_array, pause_threshold)
    if not speech_segments:
        logger.warning("VAD found no speech in audio — returning empty transcript")
        return []

    total_s = len(audio_array) / SAMPLE_RATE
    n = len(speech_segments)

    chunks: list[tuple[int, float, float, np.ndarray]] = []
    for i, seg in enumerate(speech_segments):
        start_s = max(0.0, float(seg["start"]))
        end_s = min(total_s, float(seg["end"]))

        start_sample = int(start_s * SAMPLE_RATE)
        end_sample = int(end_s * SAMPLE_RATE)
        chunk = audio_array[start_sample:end_sample]

        # Skip chunks shorter than 0.5 s — too short for Whisper to process reliably.
        if len(chunk) < SAMPLE_RATE * 0.5:
            logger.debug(f"Skipping sub-0.5s chunk at {start_s:.2f}s")
            continue

        chunks.append((i, start_s, end_s, chunk))

    return _transcribe_chunks(
        model,
        processor,
        chunks,
        n,
        batch_size,
        device,
        language,
        "Transcribing speech segment",
    )


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


def _transcribe_with_fallbacks(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_array: np.ndarray,
    device: str,
    language: str,
    pause_threshold: float = 2.0,
    batch_size: int = 1,
) -> list[dict[str, str]]:
    """VAD + model.generate() with fallback to sentence splitting."""
    try:
        offsets = _transcribe_long_form(
            model,
            processor,
            audio_array,
            device,
            language,
            pause_threshold,
            batch_size,
        )
        segments = _offsets_to_segments(offsets)
        if segments:
            logger.info(f"Generated {len(segments)} segments via VAD + model.generate()")
            return segments
        logger.warning("VAD + model.generate() returned no segments; falling back to sentence splitting")
    except Exception as e:
        logger.warning(f"VAD + model.generate() failed ({e}); falling back to sentence splitting")

    return _fallback_sentence_segmentation(model, processor, audio_array, device, language, batch_size)


def transcribe(
    audio_path: str,
    model_name: str,
    device: str,
    pause_threshold: float = 2.0,
    language: str = "en",
    batch_size: int = 1,
) -> list[dict[str, str]]:
    """Transcribe audio using Silero-VAD + Whisper model.generate()."""
    if batch_size < 1:
        raise ValueError("Whisper batch size must be positive")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        model, processor = _load_model_and_processor(model_name, device)
        audio_array = _load_audio(audio_path)
        return _transcribe_with_fallbacks(
            model,
            processor,
            audio_array,
            device,
            language,
            pause_threshold,
            batch_size,
        )
    except Exception:
        _load_model_and_processor.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def _fallback_sentence_segmentation(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_array: np.ndarray,
    device: str,
    language: str = "en",
    batch_size: int = 1,
) -> list[dict[str, str]]:
    """Last resort when VAD finds no speech: fixed 28s chunks with timestamp offsets."""
    logger.info("Using fallback chunked segmentation (no VAD)")
    chunk_samples = int(28 * SAMPLE_RATE)
    n_chunks = max(1, (len(audio_array) + chunk_samples - 1) // chunk_samples)

    chunks: list[tuple[int, float, float, np.ndarray]] = []
    for i, start_sample in enumerate(range(0, len(audio_array), chunk_samples)):
        start_s = start_sample / SAMPLE_RATE
        end_sample = min(start_sample + chunk_samples, len(audio_array))
        chunk = audio_array[start_sample:end_sample]

        if len(chunk) < SAMPLE_RATE * 0.5:
            continue

        chunks.append((i, start_s, end_sample / SAMPLE_RATE, chunk))

    all_offsets = _transcribe_chunks(
        model,
        processor,
        chunks,
        n_chunks,
        batch_size,
        device,
        language,
        "Fallback chunk",
    )

    segments = _offsets_to_segments(all_offsets)
    if not segments:
        logger.warning("Fallback chunked segmentation returned no segments")
    return segments


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
