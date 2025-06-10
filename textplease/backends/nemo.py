import gc
import os
import re
import logging
import tempfile
from typing import Any, Dict, List

import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment


logger = logging.getLogger(__name__)

FALLBACK_SEGMENT_DURATION_SEC = 60  # Used if timestamp info is missing


def transcribe(
    audio_path: str,
    model_name: str,
    device: str,
    chunk_duration_minutes: int = 10,
    pause_threshold: float = 2.0,
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """Transcribe audio using NeMo with optional chunking for long files.

    Args:
        audio_path: Path to the audio file (must be mono 16kHz WAV).
        model_name: Hugging Face model identifier.
        device: Device for inference: 'cuda', 'cpu', or 'mps'.
        chunk_duration_minutes: Maximum chunk size in minutes.
        pause_threshold: Silence duration in seconds for segment splitting.
        batch_size: Batch size for ASR processing.

    Returns:
        List of transcription segments with 'text', 'start_time', and 'end_time'.
    """
    logger.info(f"Loading NeMo model '{model_name}' on device: {device}")
    _clear_torch_memory()

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model.to(device).eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

    try:
        audio = AudioSegment.from_wav(audio_path)
        duration_sec = len(audio) / 1000.0

        if duration_sec <= chunk_duration_minutes * 60:
            return _transcribe_whole_audio(model, audio_path, pause_threshold, batch_size)
        return _transcribe_audio_in_chunks(model, audio, chunk_duration_minutes, pause_threshold, batch_size)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise
    finally:
        del model
        _clear_torch_memory()


def _transcribe_whole_audio(
    model: Any, audio_path: str, pause_threshold: float, batch_size: int
) -> List[Dict[str, Any]]:
    """Transcribe the entire audio file without chunking"""
    logger.info(f"Transcribing single audio: {audio_path}")
    output = model.transcribe([audio_path], batch_size=batch_size, timestamps=True)
    return _process_output(output, offset=0.0, pause_threshold=pause_threshold)


def _transcribe_audio_in_chunks(
    model: Any,
    audio: AudioSegment,
    chunk_duration_minutes: int,
    pause_threshold: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Split and transcribe long audio into smaller chunks"""
    chunk_ms = chunk_duration_minutes * 60 * 1000
    segments = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, start_ms in enumerate(range(0, len(audio), chunk_ms)):
            end_ms = min(start_ms + chunk_ms, len(audio))
            chunk = audio[start_ms:end_ms]
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03}.wav")
            chunk.export(chunk_path, format="wav")

            offset_sec = start_ms / 1000.0
            logger.info(f"Processing chunk {i + 1}: {offset_sec:.1f}s - {end_ms / 1000.0:.1f}s")

            try:
                output = model.transcribe([chunk_path], batch_size=batch_size, timestamps=True)
                segments.extend(_process_output(output, offset=offset_sec, pause_threshold=pause_threshold))
            except Exception as e:
                logger.error(f"Chunk {i + 1} ({chunk_path}) failed: {e}")
            _clear_torch_memory()

    logger.info(f"Transcribed {len(segments)} segments from {i + 1} chunks")
    return segments


def _process_output(output: List[Any], offset: float, pause_threshold: float) -> List[Dict[str, Any]]:
    """Convert raw ASR model output into timestamped text segments"""
    segments = []
    try:
        result = output[0] if output else None
        if hasattr(result, "timestamp") and result.timestamp and result.timestamp.get("word"):
            raw_segments = _group_words_into_segments(result.timestamp["word"], offset, pause_threshold)
            segments = [s for s in raw_segments if s["text"].strip()]
        elif hasattr(result, "text") and result.text.strip():
            segments.append(_fallback_text_segment(result.text, offset))
    except Exception as e:
        logger.error(f"Error processing ASR output: {e}")
        text = getattr(output[0], "text", str(output[0]))
        if text.strip():
            segments.append(_fallback_text_segment(text, offset))
    return segments


def _group_words_into_segments(
    words: List[Dict[str, Any]], offset: float, pause_threshold: float
) -> List[Dict[str, Any]]:
    """Group timestamped words into coherent segments based on pause and punctuation"""
    segments = []
    current_words = []
    start = end = None

    for i, word_info in enumerate(words):
        word = word_info["word"].strip()
        w_start = word_info["start"]
        w_end = word_info["end"]

        if start is None:
            start, end = w_start, w_end
            current_words = [word]
            continue

        pause = w_start - end
        should_split = pause > pause_threshold or _ends_sentence(words[i - 1]["word"]) or len(current_words) > 20

        if should_split:
            segments.append({
                "start_time": _format_time(start + offset),
                "end_time": _format_time(end + offset),
                "text": " ".join(current_words),
            })
            start, end = w_start, w_end
            current_words = [word]
        else:
            current_words.append(word)
            end = w_end

    if current_words:
        segments.append({
            "start_time": _format_time(start + offset),
            "end_time": _format_time(end + offset),
            "text": " ".join(current_words).strip(),
        })

    return segments


def _fallback_text_segment(text: str, offset: float) -> Dict[str, Any]:
    """Create a default segment when detailed word-level timestamps are unavailable"""
    return {
        "start_time": _format_time(offset),
        "end_time": _format_time(offset + FALLBACK_SEGMENT_DURATION_SEC),
        "text": text.strip(),
    }


def _ends_sentence(word: str) -> bool:
    """Determine if a word ends a sentence based on punctuation"""
    return bool(re.search(r"[.!?]$", word.strip()))


def _format_time(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) % 60 // 1)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def _clear_torch_memory() -> None:
    """Free unused PyTorch memory from CPU and GPU caches"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
