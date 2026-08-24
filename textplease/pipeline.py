import os
import re
import time
import logging
import tempfile
from pathlib import Path
from functools import lru_cache
from contextlib import nullcontext

import pandas as pd
from sentence_transformers import SentenceTransformer

from textplease.segmenter import segment_transcript, post_process_segments
from textplease.transcriber import transcribe_audio
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
_REPEATED_PHRASE = re.compile(r"(.{4,40}?)(\s+\1){2,}", re.IGNORECASE)


def save_to_csv(segments: list, output_path: str, temporary_directory: str | Path) -> str:
    """Save segments to a tab-separated CSV file."""
    if segments:
        df = pd.DataFrame(segments)
        df = df[df["text"].astype(str).str.strip() != ""]
    else:
        df = pd.DataFrame(columns=["start_time", "end_time", "text"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(temporary_directory) / "transcript.tsv"
    try:
        df.to_csv(temporary_output, index=False, sep="\t")
        temporary_output.chmod(0o600)
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    if df.empty:
        logger.info(f"No speech was transcribed; saved an empty transcript to {output_path}")
    else:
        logger.info(f"Saved {len(df)} segments to {output_path}")
    return output_path


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
    """Validate configuration for the transcription pipeline."""
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

    output_path = config["output_path"]
    if not output_path or not isinstance(output_path, str):
        raise ValueError(f"Invalid output_path: {output_path}")

    input_file = Path(input_path)
    output_file = Path(output_path)
    if input_file.resolve() == output_file.resolve() or (output_file.exists() and input_file.samefile(output_file)):
        raise ValueError("Input and output paths must refer to different files")


def _extract_config_params(config: dict) -> dict:
    """Extract and normalise pipeline parameters from config."""
    return {
        "input_path": config["input_path"],
        "output_path": config["output_path"],
        "model_name": config["model_name"],
        "device": config.get("device", "cpu"),
        "pause_threshold": config.get("pause_threshold", 2.0),
        "similarity_threshold": config.get("similarity_threshold", 0.75),
        "embedding_model_name": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "min_segment_words": config.get("min_segment_words", 3),
        "min_segment_chars": config.get("min_segment_chars", 15),
        "max_segment_words": config.get("max_segment_words", 100),
        "language": config.get("language", "en"),
        "whisper_batch_size": config.get("performance", {}).get("whisper_batch_size"),
        "similarity_batch_size": config.get("performance", {}).get("similarity_batch_size", 32),
        "chunk_size": config.get("performance", {}).get("chunk_size", 1000),
    }


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    logger.info(f"Loading SentenceTransformer '{model_name}' on: {device}")
    return SentenceTransformer(model_name, device=device)


def _filter_hallucinations(segments: list[dict]) -> list[dict]:
    """Strip known Whisper hallucination phrases and drop repetition-loop segments."""
    cleaned = []
    for seg in segments:
        text = seg["text"]

        # Collapse repetition loops but keep any trailing real content.
        text = _REPEATED_PHRASE.sub(lambda m: m.group(1), text).strip()

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


def _execute_transcription_stage(params: dict, temporary_directory: str | Path) -> list[dict]:
    """Transcribe original media, deduplicate boundaries, and filter hallucinations."""
    t0 = time.time()
    segments = transcribe_audio(
        params["input_path"],
        params["model_name"],
        params["device"],
        temporary_directory=temporary_directory,
        language=params["language"],
        batch_size=params["whisper_batch_size"],
    )
    logger.info(f"Transcription: {len(segments)} segments in {time.time() - t0:.2f}s")

    t1 = time.time()
    # Use a wider window (15 words) — the 5-second stride at typical speech rate
    # produces 10-20 words of overlap, which a 5-word window silently misses.
    segments = deduplicate_segments(segments, overlap_words=15)
    logger.info(f"Deduplication: {len(segments)} segments remaining in {time.time() - t1:.2f}s")

    segments = _filter_hallucinations(segments)
    return segments


def _execute_segmentation_stage(segments: list, params: dict, model: SentenceTransformer | None) -> list[dict]:
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


def run_transcription_pipeline(config: dict) -> None:
    """Run the complete transcription pipeline."""
    start = time.time()
    _validate_pipeline_config(config)
    params = _extract_config_params(config)
    params["device"] = detect_device(params["device"])
    if params["whisper_batch_size"] is None:
        params["whisper_batch_size"] = 4 if params["device"] == "cuda" else 1

    logger.info(f"Input: {params['input_path']} → Output: {params['output_path']}")
    logger.info(f"ASR: {params['model_name']} | Device: {params['device']}")
    logger.info(
        f"Segmentation: {params['embedding_model_name']} | "
        f"Similarity threshold: {params['similarity_threshold']} | Grouping pause: {params['pause_threshold']}s"
    )

    output_parent = Path(params["output_path"]).resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    provided_directory = config.get("_temporary_directory")
    if provided_directory is not None and (
        not isinstance(provided_directory, Path) or provided_directory.resolve().parent != output_parent
    ):
        raise ValueError("Invalid managed temporary directory")
    temporary_directory = (
        nullcontext(provided_directory)
        if provided_directory is not None
        else tempfile.TemporaryDirectory(prefix=".textplease-", dir=output_parent)
    )
    with temporary_directory as work_directory:
        segments = _execute_transcription_stage(params, work_directory)

        embedding_model = None
        if len(segments) > 1 and params["similarity_threshold"] < 1.0:
            t0 = time.time()
            embedding_model = _load_embedding_model(params["embedding_model_name"], params["device"])
            logger.info(f"SentenceTransformer loaded in {time.time() - t0:.2f}s")

        coherent = _execute_segmentation_stage(segments, params, embedding_model)
        final = _execute_post_processing(coherent, params)
        t0 = time.time()
        save_to_csv(final, params["output_path"], work_directory)
        logger.info(f"Save: {time.time() - t0:.2f}s")

    logger.info(f"Total processing time: {time.time() - start:.2f}s")
