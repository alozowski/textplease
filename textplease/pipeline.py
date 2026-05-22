import os
import re
import time
import logging
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from textplease.segmenter import segment_transcript, post_process_segments
from textplease.transcriber import transcribe_audio
from textplease.utils.audio_utils import cleanup_temp, extract_audio
from textplease.utils.device_utils import detect_device
from textplease.utils.deduplicate_segments import deduplicate_segments


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Whisper hallucinates these phrases into silence regions.
# Patterns are checked case-insensitively against the full segment text.
_HALLUCINATION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # English
        r"thank\s+you\s+for\s+watching",
        r"thanks\s+for\s+watching",
        r"please\s+subscribe",
        r"subtitles?\s+by",
        r"transcribed\s+by",
        r"translation\s+by",
        # Russian
        r"субтитры\s+(созданы|создавал|сделаны|сде[а-яё]*)",
        r"продолжение\s+следует",
        r"подписывайтесь\s+на\s+канал",
    ]
]

# Phrase repeated ≥3 times consecutively within one segment = hallucination loop.
_REPEATED_PHRASE = re.compile(r'(.{4,40}?)(\s+\1){2,}', re.IGNORECASE)


def save_to_csv(segments: list, output_path: str) -> str:
    """Save segments to a tab-separated CSV file.

    Args:
        segments: List of segment dictionaries to save.
        output_path: Path where the CSV file should be saved.

    Returns:
        Path to the saved CSV file.

    Raises:
        ValueError: If segments is empty or output_path is invalid.
        IOError: If file writing fails.

    """
    if not segments:
        raise ValueError("Cannot save empty segments list")

    if not output_path or not isinstance(output_path, str):
        raise ValueError(f"Invalid output_path: {output_path}")

    try:
        df = pd.DataFrame(segments)
        df = df[df["text"].astype(str).str.strip() != ""]

        if df.empty:
            raise ValueError("No valid segments to save (all were empty)")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False, sep="\t")
        logger.info(f"Saved {len(df)} segments to {output_path}")
        return output_path

    except (ValueError, IOError):
        raise
    except Exception as e:
        logger.error(f"Failed to save CSV to {output_path}: {e}")
        raise IOError(f"Failed to save CSV: {e}") from e


def estimate_processing_time(num_segments: int) -> str:
    """Estimate segmentation processing time based on segment count."""
    if num_segments < 100:
        return "< 1 minute"
    elif num_segments < 500:
        return "1–3 minutes"
    elif num_segments < 1000:
        return "3–5 minutes"
    elif num_segments < 5000:
        return "5–15 minutes"
    else:
        return "15+ minutes"


def _validate_pipeline_config(config: dict) -> None:
    """Validate configuration for the transcription pipeline.

    Raises:
        ValueError: If required config keys are missing or invalid.
        FileNotFoundError: If input file does not exist.

    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")

    missing = {"input_path", "output_path", "model_name"} - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    input_path = config["input_path"]
    if not input_path or not isinstance(input_path, str):
        raise ValueError(f"Invalid input_path: {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")


def _extract_config_params(config: dict) -> dict:
    """Extract and normalise pipeline parameters from config."""
    return {
        "input_path": config["input_path"],
        "output_path": config["output_path"],
        "model_name": config["model_name"],
        "device": config.get("device", "cpu"),
        "batch_size": config.get("max_batch_size", 1),
        "chunk_duration": config.get("chunk_duration_minutes", 10),
        "pause_threshold": config.get("pause_threshold", 2.0),
        "similarity_threshold": config.get("similarity_threshold", 0.75),
        "embedding_model_name": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "min_segment_words": config.get("min_segment_words", 3),
        "min_segment_chars": config.get("min_segment_chars", 15),
        "max_segment_words": config.get("max_segment_words", 100),
        "language": config.get("language", "en"),
        "similarity_batch_size": config.get("performance", {}).get("similarity_batch_size", 32),
        "chunk_size": config.get("performance", {}).get("chunk_size", 1000),
    }


def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """Load the SentenceTransformer model for semantic segmentation."""
    start = time.time()
    logger.info(f"Loading SentenceTransformer '{model_name}' on: {device}")
    model = SentenceTransformer(model_name, device=device)
    logger.info(f"SentenceTransformer loaded in {time.time() - start:.2f}s")
    return model


def _filter_hallucinations(segments: list[dict]) -> list[dict]:
    """Remove or clean segments that contain Whisper hallucination patterns.

    Two cases handled:
    1. Known hallucination phrase found in segment text → strip it out.
       If stripping leaves the segment empty, drop the segment entirely.
    2. A phrase is repeated ≥3 consecutive times within one segment
       (hallucination loop) → drop the segment.
    """
    cleaned = []
    for seg in segments:
        text = seg["text"]

        # Hallucination loop: strip the repetition, keep any trailing real content.
        # E.g. "Thank you. Thank you. Thank you. Well," → "Well,"
        text = _REPEATED_PHRASE.sub(lambda m: m.group(1), text).strip()

        # Known hallucination phrases — strip the phrase from the text
        for pattern in _HALLUCINATION_PATTERNS:
            text = pattern.sub("", text).strip()

        if not text:
            logger.debug(f"Dropping empty-after-hallucination-strip segment [{seg['start_time']}]")
            continue

        cleaned.append({**seg, "text": text})

    removed = len(segments) - len(cleaned)
    if removed:
        logger.info(f"Hallucination filter removed {removed} segments")
    return cleaned


def _execute_transcription_stage(params: dict, audio_path: str) -> list[dict]:
    """Transcribe audio, deduplicate chunk boundaries, and filter hallucinations."""
    t0 = time.time()
    segments = transcribe_audio(
        audio_path,
        params["model_name"],
        params["device"],
        params["chunk_duration"],
        params["pause_threshold"],
        batch_size=params["batch_size"],
        language=params["language"],
    )
    logger.info(f"Transcription: {len(segments)} segments in {time.time() - t0:.2f}s")

    t1 = time.time()
    # Use a wider window (15 words) — the 5-second stride at typical speech rate
    # produces 10-20 words of overlap, which a 5-word window silently misses.
    segments = deduplicate_segments(segments, overlap_words=15)
    logger.info(f"Deduplication: {len(segments)} segments remaining in {time.time() - t1:.2f}s")

    segments = _filter_hallucinations(segments)
    return segments


def _execute_segmentation_stage(segments: list, params: dict, model: SentenceTransformer) -> list[dict]:
    """Merge segments semantically using sentence embeddings."""
    logger.info(f"Estimated segmentation time: {estimate_processing_time(len(segments))}")
    t0 = time.time()
    coherent = segment_transcript(
        segments,
        similarity_threshold=params["similarity_threshold"],
        pause_threshold=params["pause_threshold"],
        model=model,
        max_words=params["max_segment_words"],
        min_words=params["min_segment_words"],
        min_chars=params["min_segment_chars"],
        embedding_model_name=params["embedding_model_name"],
        preferred_device=params["device"],
        batch_size=params["similarity_batch_size"],
        chunk_size=params["chunk_size"],
    )
    logger.info(f"Segmentation: {len(coherent)} segments in {time.time() - t0:.2f}s")
    return coherent


def _execute_post_processing(segments: list, params: dict) -> list[dict]:
    """Enforce min/max segment length constraints."""
    t0 = time.time()
    final = post_process_segments(
        segments,
        min_words=params["min_segment_words"],
        min_chars=params["min_segment_chars"],
        max_words=params["max_segment_words"],
    )
    logger.info(f"Post-processing: {len(final)} final segments in {time.time() - t0:.2f}s")
    return final


def _save_and_cleanup(final_segments: list, params: dict, audio_path: str) -> None:
    """Persist results and remove temporary audio file."""
    t0 = time.time()

    if os.path.exists(params["output_path"]):
        try:
            os.remove(params["output_path"])
        except Exception as e:
            logger.warning(f"Could not delete existing output file: {e}")

    save_to_csv(final_segments, params["output_path"])
    logger.info(f"Save: {time.time() - t0:.2f}s")

    if cleanup_temp(params["input_path"], audio_path):
        try:
            os.remove(audio_path)
            logger.info(f"Deleted temporary audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")


def run_transcription_pipeline(config: dict) -> None:
    """Run the complete transcription pipeline.

    Args:
        config: Configuration dictionary with required keys.

    Raises:
        ValueError: If required config keys are missing.
        Exception: If any pipeline stage fails.

    """
    start = time.time()
    _validate_pipeline_config(config)
    params = _extract_config_params(config)

    logger.info(f"Input: {params['input_path']} → Output: {params['output_path']}")
    logger.info(
        f"ASR: {params['model_name']} | Device: {params['device']} | "
        f"Chunk: {params['chunk_duration']}min | Pause: {params['pause_threshold']}s"
    )
    logger.info(
        f"Embedding: {params['embedding_model_name']} | "
        f"Similarity threshold: {params['similarity_threshold']}"
    )

    t0 = time.time()
    audio_path = extract_audio(params["input_path"])
    logger.info(f"Audio extraction: {time.time() - t0:.2f}s")

    resolved_device = detect_device(params["device"])
    embedding_model = _load_embedding_model(params["embedding_model_name"], resolved_device)

    segments = _execute_transcription_stage(params, audio_path)
    coherent = _execute_segmentation_stage(segments, params, embedding_model)
    final = _execute_post_processing(coherent, params)
    _save_and_cleanup(final, params, audio_path)

    logger.info(f"Total processing time: {time.time() - start:.2f}s")
