import gc
import os
import re
import logging
import tempfile
from typing import Any
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment

from textplease.utils.time_utils import format_time_precise as format_time


logger = logging.getLogger(__name__)

# Fallback duration when timestamps are unavailable
FALLBACK_SEGMENT_DURATION_SEC = 60


def transcribe(
    audio_path: str,
    model_name: str,
    device: str,
    chunk_duration_minutes: int = 10,
    pause_threshold: float = 2.0,
    batch_size: int = 1,
    max_segment_words: int = 100,
    min_segment_words: int = 3,
    min_segment_chars: int = 15,
) -> list[dict[str, str]]:
    """Transcribe audio using NeMo, with optional chunking for long files."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")

    logger.info(f"Loading NeMo model '{model_name}' on device: {device}")
    _clear_torch_memory()

    model = None
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model.to(device).eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load NeMo model '{model_name}': {e}") from e

    try:
        audio = AudioSegment.from_wav(audio_path)
        duration_sec = len(audio) / 1000.0
        logger.info(f"Audio duration: {duration_sec:.1f}s")

        if duration_sec <= chunk_duration_minutes * 60:
            return _transcribe_whole_audio(
                model, audio_path, pause_threshold, batch_size, max_segment_words, min_segment_words, min_segment_chars
            )
        return _transcribe_audio_in_chunks(
            model,
            audio,
            chunk_duration_minutes,
            pause_threshold,
            batch_size,
            max_segment_words,
            min_segment_words,
            min_segment_chars,
        )

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        if model is not None:
            del model
        _clear_torch_memory()


def _transcribe_whole_audio(
    model: Any,
    audio_path: str,
    pause_threshold: float,
    batch_size: int,
    max_segment_words: int,
    min_segment_words: int,
    min_segment_chars: int,
) -> list[dict[str, str]]:
    """Transcribe the entire audio file without chunking."""
    logger.info(f"Transcribing single audio: {audio_path}")

    output = model.transcribe([audio_path], batch_size=batch_size, timestamps=True)
    segments = _process_output(
        output,
        offset=0.0,
        pause_threshold=pause_threshold,
        max_segment_words=max_segment_words,
        min_segment_words=min_segment_words,
        min_segment_chars=min_segment_chars,
    )
    logger.info(f"Transcription complete: {len(segments)} segments")
    return segments


def _transcribe_audio_in_chunks(
    model: Any,
    audio: AudioSegment,
    chunk_duration_minutes: int,
    pause_threshold: float,
    batch_size: int,
    max_segment_words: int,
    min_segment_words: int,
    min_segment_chars: int,
) -> list[dict[str, str]]:
    """Split and transcribe long audio into smaller chunks."""
    chunk_ms = chunk_duration_minutes * 60 * 1000
    segments: list[dict[str, str]] = []
    total_chunks = (len(audio) + chunk_ms - 1) // chunk_ms

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, start_ms in enumerate(range(0, len(audio), chunk_ms)):
            end_ms = min(start_ms + chunk_ms, len(audio))
            chunk = audio[start_ms:end_ms]
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03}.wav")
            chunk.export(chunk_path, format="wav")

            offset_sec = start_ms / 1000.0
            logger.info(f"Processing chunk {i + 1}/{total_chunks}: {offset_sec:.1f}s - {end_ms / 1000.0:.1f}s")

            try:
                output = model.transcribe([chunk_path], batch_size=batch_size, timestamps=True)
                chunk_segments = _process_output(
                    output,
                    offset=offset_sec,
                    pause_threshold=pause_threshold,
                    max_segment_words=max_segment_words,
                    min_segment_words=min_segment_words,
                    min_segment_chars=min_segment_chars,
                )
                segments.extend(chunk_segments)
            except Exception as e:
                logger.error(f"Chunk {i + 1} ({chunk_path}) failed: {e}")
            finally:
                _clear_torch_memory()

    logger.info(f"Transcribed {len(segments)} segments from {total_chunks} chunks")
    return segments


def _process_output(
    output: list[Any],
    offset: float,
    pause_threshold: float,
    max_segment_words: int = 100,
    min_segment_words: int = 3,
    min_segment_chars: int = 15,
) -> list[dict[str, str]]:
    """Convert raw ASR model output into timestamped text segments."""
    segments: list[dict[str, str]] = []

    if not output:
        logger.warning("Empty output from ASR model")
        return segments

    try:
        result = output[0]

        if hasattr(result, "timestamp") and result.timestamp and result.timestamp.get("word"):
            words = result.timestamp["word"]
            if not words:
                logger.warning("Empty word list in timestamp data")
            else:
                logger.info(
                    f"Processing {len(words)} words with pause_threshold={pause_threshold}, max_words={max_segment_words}"
                )
                raw_segments = _group_words_into_segments(
                    words,
                    offset,
                    pause_threshold,
                    max_segment_words,
                    min_segment_words,
                    min_segment_chars,
                )
                logger.info(f"Created {len(raw_segments)} segments from word-level timestamps")
                segments = [s for s in raw_segments if s["text"].strip()]

        elif hasattr(result, "text") and result.text:
            logger.warning("No word-level timestamps available, using text-only fallback")
            text = result.text.strip()
            if text:
                segments.append(_fallback_text_segment(text, offset))
        else:
            logger.warning("No text or timestamp data found in result")

    except Exception as e:
        logger.error(f"Error processing ASR output: {e}", exc_info=True)
        try:
            text = getattr(output[0], "text", str(output[0])).strip()
            if text:
                segments.append(_fallback_text_segment(text, offset))
        except Exception as inner_e:
            logger.error(f"Fallback text extraction also failed: {inner_e}")

    return segments


def _group_words_into_segments(
    words: list[dict[str, Any]],
    offset: float,
    pause_threshold: float,
    max_segment_words: int = 100,
    min_segment_words: int = 3,
    min_segment_chars: int = 15,
) -> list[dict[str, str]]:
    """Group timestamped words into coherent segments based on pause and punctuation."""
    segments = []
    current_words = []
    start = end = None

    for i, word_info in enumerate(words):
        if not isinstance(word_info, dict):
            logger.warning(f"Invalid word_info at index {i}: {word_info}")
            continue

        if "word" not in word_info or "start" not in word_info or "end" not in word_info:
            logger.warning(f"Missing required fields in word_info: {word_info}")
            continue

        word = word_info["word"].strip()
        if not word:
            continue

        w_start = word_info["start"]
        w_end = word_info["end"]

        if w_start is None or w_end is None or w_start < 0 or w_end < 0 or w_end < w_start:
            logger.warning(f"Invalid timestamps for word '{word}': start={w_start}, end={w_end}")
            continue

        # Detect silence encoded as abnormally long words
        word_duration = w_end - w_start
        is_silence_word = len(word) <= 2 and word_duration > 2.0

        if is_silence_word:
            logger.debug(
                f"Detected silence word: '{word}' (duration={word_duration:.2f}s, "
                f"start={w_start:.2f}s, end={w_end:.2f}s)"
            )

        if start is None:
            start, end = float(w_start), float(w_end)
            current_words = [word]
            continue

        assert end is not None, "end should be set when start is not None"
        pause = float(w_start) - end
        current_text = " ".join(current_words)
        current_word_count = len(current_words)
        current_char_count = len(current_text)

        is_too_short = current_word_count < min_segment_words or current_char_count < min_segment_chars
        should_split = (
            (pause > pause_threshold and not is_too_short)
            or (is_silence_word and not is_too_short)
            or _ends_sentence(words[i - 1]["word"])
            or current_word_count >= max_segment_words
        )

        if should_split:
            segments.append(
                {
                    "start_time": format_time(start + offset),
                    "end_time": format_time(end + offset),
                    "text": " ".join(current_words),
                }
            )
            start, end = float(w_start), float(w_end)
            current_words = [word]
        else:
            current_words.append(word)
            end = float(w_end)

    if current_words and start is not None and end is not None:
        segments.append(
            {
                "start_time": format_time(start + offset),
                "end_time": format_time(float(end) + offset),
                "text": " ".join(current_words).strip(),
            }
        )

    return segments


def _fallback_text_segment(text: str, offset: float) -> dict[str, str]:
    """Create a default segment when detailed word-level timestamps are unavailable."""
    return {
        "start_time": format_time(offset),
        "end_time": format_time(offset + FALLBACK_SEGMENT_DURATION_SEC),
        "text": text.strip(),
    }


def _ends_sentence(word: str) -> bool:
    """Determine if a word ends a sentence based on punctuation."""
    return bool(re.search(r"[.!?]$", word.strip()))


def _clear_torch_memory() -> None:
    """Free unused PyTorch memory from CPU and GPU caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
