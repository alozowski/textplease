"""Whisper ASR backend: Silero-VAD preprocessing + model.generate() per speech segment."""

import gc
import re
import logging
import warnings
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from textplease.utils.time_utils import format_time_precise as format_time


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Padding (seconds) added around each VAD speech boundary to avoid clipping words.
_SPEECH_PAD_S = 0.1

warnings.filterwarnings("ignore", message=".*Whisper did not predict an ending timestamp.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")


def _load_model_and_processor(model_name: str, device: str) -> tuple[Any, Any]:
    """Load Whisper model and processor."""
    logger.info(f"Loading Transformers model '{model_name}' on device: {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    torch_dtype = torch.float16 if device not in ("cpu", "mps") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
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
    return segments  # type: ignore[return-value]


def _transcribe_speech_segment(
    model: Any,
    processor: Any,
    audio_chunk: np.ndarray,
    device: str,
    language: str,
) -> list[dict[str, Any]]:
    """Transcribe one speech chunk; returns decoded timestamp offsets."""
    torch_dtype = torch.float16 if device not in ("cpu", "mps") else torch.float32

    inputs = processor(
        audio_chunk,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        sampling_rate=SAMPLE_RATE,
    )
    input_features = inputs.input_features.to(device=device, dtype=torch_dtype)
    attention_mask = inputs.get("attention_mask")

    generate_kwargs: dict[str, Any] = {
        "input_features": input_features,
        "language": language,
        "task": "transcribe",
        "return_timestamps": True,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 1.35,
        "logprob_threshold": -1.0,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    time_precision = processor.feature_extractor.chunk_length / model.config.max_source_positions
    decoded = processor.tokenizer.decode(
        generated_ids[0],
        output_offsets=True,
        time_precision=time_precision,
    )
    return decoded.get("offsets", [])


def _transcribe_long_form(
    model: Any,
    processor: Any,
    audio_array: np.ndarray,
    device: str,
    language: str,
    pause_threshold: float,
) -> list[dict[str, Any]]:
    """Full pipeline: VAD segmentation → per-segment transcription → offset timestamps."""
    speech_segments = _get_speech_segments(audio_array, pause_threshold)
    if not speech_segments:
        logger.warning("VAD found no speech in audio — returning empty transcript")
        return []

    total_s = len(audio_array) / SAMPLE_RATE
    all_offsets: list[dict[str, Any]] = []
    n = len(speech_segments)

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

        logger.info(
            f"Transcribing speech segment {i + 1}/{n}: [{start_s:.2f}s → {end_s:.2f}s] ({end_s - start_s:.1f}s)"
        )
        offsets = _transcribe_speech_segment(model, processor, chunk, device, language)

        for offset in offsets:
            ts = offset.get("timestamp", (0.0, 0.0))
            if len(ts) == 2 and ts[0] is not None and ts[1] is not None:
                offset["timestamp"] = (ts[0] + start_s, ts[1] + start_s)
        all_offsets.extend(offsets)

    return all_offsets


def _offsets_to_segments(offsets: list[dict[str, Any]]) -> list[dict[str, str]]:
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
    model: Any,
    processor: Any,
    audio_array: np.ndarray,
    device: str,
    language: str,
    pause_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """VAD + model.generate() with fallback to sentence splitting."""
    try:
        offsets = _transcribe_long_form(model, processor, audio_array, device, language, pause_threshold)
        segments = _offsets_to_segments(offsets)
        if segments:
            logger.info(f"Generated {len(segments)} segments via VAD + model.generate()")
            return segments
        logger.warning("VAD + model.generate() returned no segments; falling back to sentence splitting")
    except Exception as e:
        logger.warning(f"VAD + model.generate() failed ({e}); falling back to sentence splitting")

    return _fallback_sentence_segmentation(model, processor, audio_array, device, language)


def transcribe(
    audio_path: str,
    model_name: str,
    device: str,
    chunk_duration_minutes: int = 10,
    pause_threshold: float = 2.0,
    batch_size: int = 1,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Transcribe audio using Silero-VAD + Whisper model.generate()."""
    if batch_size != 1:
        logger.warning(f"Ignoring batch_size={batch_size} — not applicable to model.generate()")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = None
    processor = None
    try:
        model, processor = _load_model_and_processor(model_name, device)
        audio_array = _load_audio(audio_path)
        return _transcribe_with_fallbacks(model, processor, audio_array, device, language, pause_threshold)
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _fallback_sentence_segmentation(
    model: Any,
    processor: Any,
    audio_array: np.ndarray,
    device: str,
    language: str = "en",
) -> list[dict[str, str]]:
    """Last resort when VAD finds no speech: fixed 28s chunks with timestamp offsets."""
    logger.info("Using fallback chunked segmentation (no VAD)")
    chunk_samples = int(28 * SAMPLE_RATE)
    n_chunks = max(1, (len(audio_array) + chunk_samples - 1) // chunk_samples)

    all_offsets: list[dict[str, Any]] = []
    for i, start_sample in enumerate(range(0, len(audio_array), chunk_samples)):
        start_s = start_sample / SAMPLE_RATE
        end_sample = min(start_sample + chunk_samples, len(audio_array))
        chunk = audio_array[start_sample:end_sample]

        if len(chunk) < SAMPLE_RATE * 0.5:
            continue

        logger.info(f"Fallback chunk {i + 1}/{n_chunks}: [{start_s:.1f}s → {end_sample / SAMPLE_RATE:.1f}s]")
        offsets = _transcribe_speech_segment(model, processor, chunk, device, language)
        for offset in offsets:
            ts = offset.get("timestamp", (0.0, 0.0))
            if len(ts) == 2 and ts[0] is not None and ts[1] is not None:
                offset["timestamp"] = (ts[0] + start_s, ts[1] + start_s)
        all_offsets.extend(offsets)

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
