import gc
import re
import logging
import warnings
from typing import Any

import numpy as np
import torch
import torchaudio  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import pipeline as hf_pipeline

from textplease.utils.time_utils import format_time_precise as format_time


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Suppress repetitive Whisper warnings about timestamps
warnings.filterwarnings("ignore", message=".*Whisper did not predict an ending timestamp.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")


def _load_model_and_processor(model_name: str, device: str) -> tuple[Any, Any]:
    """Load model and processor for transcription.

    Args:
        model_name: Name of the model to load
        device: Target device for model loading

    Returns:
        Tuple of (model, processor)

    """
    logger.info(f"Loading Transformers model '{model_name}' on device: {device}")

    processor = AutoProcessor.from_pretrained(model_name)

    # Use float32 for CPU/MPS, float16 for CUDA (MPS has limited float16 support)
    if device == "cpu":
        torch_dtype = torch.float32
    elif device == "mps":
        torch_dtype = torch.float32
    else:  # CUDA
        torch_dtype = torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    return model, processor


def _load_audio(audio_path: str) -> np.ndarray:
    """Load and preprocess audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Preprocessed audio array

    """
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def _transcribe_with_fallbacks(
    model: Any,
    processor: Any,
    audio_array: np.ndarray,
    device: str,
    language: str,
    pause_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Orchestrate fallback attempts for transcription.

    Args:
        model: Loaded model
        processor: Loaded processor
        audio_array: Preprocessed audio
        device: Target device
        language: Language for transcription
        pause_threshold: Minimum pause duration to split segments

    Returns:
        List of transcription segments

    """
    segments = []

    # Try word-level timestamps first (skip on CPU for memory reasons)
    if device != "cpu":
        segments = _word_level_segmentation(
            model, processor, audio_array, device, language, pause_threshold
        )

    if not segments:
        logger.warning("Word-level timestamps failed or skipped, trying chunk-level with smart splitting")
        segments = _chunk_level_segmentation(model, processor, audio_array, device, language)

    if not segments:
        logger.warning("All timestamp methods failed, using sentence-based splitting")
        segments = _fallback_sentence_segmentation(model, processor, audio_array, device, language)

    logger.info(f"Generated {len(segments)} segments from transformers pipeline")
    return segments


def transcribe(
    audio_path: str,
    model_name: str,
    device: str,
    chunk_duration_minutes: int = 10,
    pause_threshold: float = 2.0,
    batch_size: int = 1,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Transcribe audio using transformers pipeline with fallback mechanisms."""
    logger.warning(f"Ignoring batch_size={batch_size} in transformers pipeline")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = None
    processor = None

    try:
        model, processor = _load_model_and_processor(model_name, device)
        audio_array = _load_audio(audio_path)
        segments = _transcribe_with_fallbacks(
            model, processor, audio_array, device, language, pause_threshold
        )
        return segments

    except Exception as e:
        logger.error(f"Transformers pipeline transcription failed for model '{model_name}': {e}")
        raise

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _create_pipeline(model, processor, device, return_timestamps, language="en"):
    """Create ASR pipeline with specified configuration.

    Args:
        model: Loaded model
        processor: Loaded processor
        device: Target device
        return_timestamps: Timestamp configuration
        language: Language for transcription

    Returns:
        Configured ASR pipeline

    """
    # Map device string to pipeline device format
    if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
        pipeline_device = 0
    elif device == "mps":
        pipeline_device = device
    else:
        pipeline_device = -1

    torch_dtype = torch.float16 if device not in ["cpu", "mps"] else torch.float32

    # Configure generate_kwargs with proper attention and timestamp handling
    generate_kwargs = {
        "language": language,
        "task": "transcribe",
    }

    # Add timestamp-specific generation parameters for word-level timestamps
    if return_timestamps == "word":
        generate_kwargs.update({
            "return_timestamps": True,
            "return_token_timestamps": True,
        })

    return hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=pipeline_device,
        torch_dtype=torch_dtype,
        return_timestamps=return_timestamps,
        chunk_length_s=None,  # Disable to avoid experimental seq2seq chunking warnings
        stride_length_s=None,
        batch_size=1,
        generate_kwargs=generate_kwargs,
    )


def _process_audio_chunks(
    asr_pipe,
    audio_array: np.ndarray,
    chunk_length_s: int,
    stride_length_s: int,
    return_timestamps,
) -> dict[str, Any]:
    """Process audio in chunks using the ASR pipeline.

    Args:
        asr_pipe: Configured ASR pipeline
        audio_array: Preprocessed audio data
        chunk_length_s: Chunk length in seconds
        stride_length_s: Stride length in seconds
        return_timestamps: Whether to return timestamps

    Returns:
        Dictionary with transcription results

    """
    sample_rate = SAMPLE_RATE
    chunk_samples = int(chunk_length_s * sample_rate)
    stride_samples = int(stride_length_s * sample_rate)
    step = chunk_samples - stride_samples

    merged_chunks = []
    results = []

    total_samples = len(audio_array)
    chunk_starts = list(range(0, total_samples, step))

    # Single progress bar that updates in place
    for i, start in enumerate(tqdm(chunk_starts, desc="Transcribing", unit="chunk", ncols=80)):
        end = min(start + chunk_samples, total_samples)
        chunk = audio_array[start:end]

        # Skip very short chunks (less than 0.5 seconds)
        if len(chunk) < sample_rate * 0.5:
            continue

        with torch.no_grad():
            try:
                result = asr_pipe(chunk)
            except Exception as e:
                logger.warning(f"Failed to transcribe chunk {i}: {e}")
                continue

        if return_timestamps:
            offset_sec = start / sample_rate
            chunk_chunks = result.get("chunks", [])
            for entry in chunk_chunks:
                ts = entry.get("timestamp", [0.0, 0.0])
                # Check for valid timestamps (not None)
                if len(ts) == 2 and ts[0] is not None and ts[1] is not None:
                    entry["timestamp"] = [ts[0] + offset_sec, ts[1] + offset_sec]
            merged_chunks.extend(chunk_chunks)
        else:
            text = result.get("text", "").strip()
            if text:
                results.append(text)

    return {"chunks": merged_chunks} if return_timestamps else {"text": " ".join(results)}


def _merge_timestamp_results(merged_chunks: list[dict], offset_sec: float) -> list[dict]:
    """Merge timestamp results with offset adjustment.

    Args:
        merged_chunks: List of chunks with timestamps
        offset_sec: Offset in seconds

    Returns:
        List of adjusted chunks

    """
    for entry in merged_chunks:
        ts = entry.get("timestamp", [0.0, 0.0])
        if len(ts) == 2 and ts[0] is not None and ts[1] is not None:
            entry["timestamp"] = [ts[0] + offset_sec, ts[1] + offset_sec]
    return merged_chunks


def _run_pipeline(
    model, processor, audio_array, device, return_timestamps, chunk_length_s=30, stride_length_s=3, language="en"
):
    """Run the ASR pipeline with chunk processing."""
    asr_pipe = None

    try:
        asr_pipe = _create_pipeline(model, processor, device, return_timestamps, language)

        # For short audio, process directly without manual chunking
        duration_seconds = len(audio_array) / SAMPLE_RATE
        if duration_seconds <= chunk_length_s:
            logger.info(f"Audio duration ({duration_seconds:.1f}s) is short, processing directly")
            with torch.no_grad():
                result = asr_pipe(audio_array)
            return result

        # For long audio, use manual chunking (our implementation handles it better than pipeline's)
        logger.info(f"Audio duration ({duration_seconds:.1f}s) exceeds {chunk_length_s}s, using manual chunking")
        return _process_audio_chunks(asr_pipe, audio_array, chunk_length_s, stride_length_s, return_timestamps)

    finally:
        if asr_pipe is not None:
            del asr_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _word_level_segmentation(
    model, processor, audio_array, device, language="en", pause_threshold=2.0
):
    try:
        logger.info("Attempting word-level timestamp segmentation")
        result = _run_pipeline(
            model,
            processor,
            audio_array,
            device,
            return_timestamps="word",
            chunk_length_s=30,
            stride_length_s=5,
            language=language
        )

        chunks = result.get("chunks", [])
        if not chunks:
            logger.warning("No chunks returned from word-level segmentation")
            return []

        logger.debug(f"Received {len(chunks)} word-level chunks")

        # Filter out chunks with invalid timestamps (None values)
        valid_chunks = [
            chunk for chunk in chunks
            if chunk.get("timestamp") and
            len(chunk["timestamp"]) == 2 and
            chunk["timestamp"][0] is not None and
            chunk["timestamp"][1] is not None
        ]

        if not valid_chunks:
            logger.warning("No valid chunks with timestamps")
            return []

        return _group_words_into_segments(valid_chunks, pause_threshold=pause_threshold)

    except RuntimeError as e:
        logger.warning(f"Word-level timestamp extraction failed: {e}")
        return []
    except Exception as e:
        logger.warning(f"Word-level segmentation failed: {e}")
        return []


def _chunk_level_segmentation(model, processor, audio_array, device, language):
    try:
        logger.info("Attempting chunk-level timestamp segmentation")
        result = _run_pipeline(
            model,
            processor,
            audio_array,
            device,
            return_timestamps=True,
            chunk_length_s=15,
            stride_length_s=3,
            language=language,
        )

        chunks = result.get("chunks", [])
        if not chunks:
            logger.warning("No chunks returned from chunk-level segmentation")
            return []

        segments = []
        for chunk in tqdm(chunks, desc="Chunk-level transcription", unit="chunk"):
            timestamp = chunk.get("timestamp", [0, 0])
            if len(timestamp) != 2 or timestamp[0] is None or timestamp[1] is None:
                continue

            start, end = timestamp
            text = chunk.get("text", "").strip()

            if not text:
                continue

            segments.extend(_split_chunk_by_sentences(text, start, end))

        return segments

    except Exception as e:
        logger.warning(f"Chunk-level segmentation failed: {e}")
        return []


def _fallback_sentence_segmentation(model, processor, audio_array, device, language="en"):
    try:
        logger.info("Using fallback sentence-based segmentation")
        result = _run_pipeline(
            model,
            processor,
            audio_array,
            device,
            return_timestamps=False,
            chunk_length_s=30,
            language=language
        )

        text = result.get("text", "").strip()
        if not text:
            logger.warning("No text returned from fallback segmentation")
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # If no sentence boundaries found, treat entire text as one segment
        if not sentences:
            duration_seconds = len(audio_array) / SAMPLE_RATE
            return [{
                "start_time": format_time(0.0),
                "end_time": format_time(duration_seconds),
                "text": text,
            }]

        duration_seconds = len(audio_array) / SAMPLE_RATE
        total_chars = sum(len(s) for s in sentences)

        segments = []
        current_time = 0.0
        for sentence in sentences:
            sentence_duration = (len(sentence) / total_chars) * duration_seconds
            end_time = min(current_time + sentence_duration, duration_seconds)
            segments.append({
                "start_time": format_time(current_time),
                "end_time": format_time(end_time),
                "text": sentence,
            })
            current_time = end_time

        return segments

    except Exception as e:
        logger.error(f"Fallback segmentation failed: {e}")
        return []


def _should_end_segment(
    current_duration: float,
    text: str,
    next_pause: float,
    max_segment_duration: float,
    min_segment_duration: float,
    pause_threshold: float = 1.5,
) -> bool:
    """Determine if a segment should end based on various criteria.

    Args:
        current_duration: Current segment duration
        text: Current text content
        next_pause: Pause to next segment
        max_segment_duration: Maximum allowed duration
        min_segment_duration: Minimum required duration
        pause_threshold: Minimum pause duration to end segment

    Returns:
        True if segment should end

    """
    return current_duration >= max(min_segment_duration, 1.0) and (
        _ends_with_sentence_boundary(text) or
        current_duration >= max_segment_duration or
        next_pause > pause_threshold
    )


def _create_segment(
    words: list[str],
    start_time: float,
    end_time: float,
) -> dict[str, str] | None:
    """Create a segment dictionary from word list and timestamps.

    Args:
        words: List of words in segment
        start_time: Segment start time
        end_time: Segment end time

    Returns:
        Segment dictionary

    """
    segment_text = " ".join(words).strip()
    if segment_text:
        return {
            "start_time": format_time(start_time),
            "end_time": format_time(end_time),
            "text": segment_text,
        }
    return None


def _group_words_into_segments(
        word_chunks,
        max_segment_duration=12.0,
        min_segment_duration=3.0,
        pause_threshold=1.5
        ) -> list[dict]:
    """Group word-level chunks into coherent segments."""
    segments = []
    current_segment_words = []
    current_start_time = None
    current_end_time = None

    for i, chunk in enumerate(word_chunks):
        timestamp = chunk.get("timestamp", [0, 0])
        if len(timestamp) != 2 or timestamp[0] is None or timestamp[1] is None:
            continue

        start, end = timestamp
        text = chunk.get("text", "").strip()
        if not text:
            continue

        if current_start_time is None:
            current_start_time = start

        current_end_time = end
        current_segment_words.append(text)

        current_duration = current_end_time - current_start_time

        # Calculate pause to next word (0 if this is the last word or next word has no timestamp)
        next_pause = (
            word_chunks[i + 1]["timestamp"][0] - end
            if i + 1 < len(word_chunks) and
               len(word_chunks[i + 1].get("timestamp", [])) == 2 and
               word_chunks[i + 1]["timestamp"][0] is not None
            else 0
        )

        if _should_end_segment(current_duration, text, next_pause, max_segment_duration, min_segment_duration, pause_threshold):
            segment = _create_segment(current_segment_words, current_start_time, current_end_time)
            if segment:
                segments.append(segment)
            current_segment_words = []
            current_start_time = None
            current_end_time = None

    # Handle remaining words
    if current_segment_words and current_start_time is not None and current_end_time is not None:
        segment = _create_segment(current_segment_words, current_start_time, current_end_time)
        if segment:
            segments.append(segment)

    return segments


def _split_chunk_by_sentences(text, start_time, end_time):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return [
            {
                "start_time": format_time(start_time),
                "end_time": format_time(end_time),
                "text": text,
            }
        ]

    duration = end_time - start_time
    total_chars = sum(len(s) for s in sentences)

    # Handle edge case where all sentences are empty (total_chars = 0)
    if total_chars == 0:
        return [{
            "start_time": format_time(start_time),
            "end_time": format_time(end_time),
            "text": text,
        }]

    segments = []
    current_time = start_time

    for sentence in sentences:
        sentence_duration = (len(sentence) / total_chars) * duration
        sentence_end_time = min(current_time + sentence_duration, end_time)
        segments.append({
            "start_time": format_time(current_time),
            "end_time": format_time(sentence_end_time),
            "text": sentence,
        })
        current_time = sentence_end_time

    return segments


def _ends_with_sentence_boundary(text):
    return bool(re.search(r"[.!?,;:]$", text.strip()))
