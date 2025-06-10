import logging
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer, util

from textplease.utils.device_utils import detect_device


logger = logging.getLogger(__name__)


def parse_time_str(time_str: str) -> int:
    """Convert a time string in HH:MM:SS format to total seconds"""
    try:
        h, m, s = map(int, time_str.split(":"))
        return h * 3600 + m * 60 + s
    except Exception as e:
        logger.error(f"Invalid time string '{time_str}': {e}")
        raise


def is_segment_too_short(text: str, min_words: int = 3, min_chars: int = 15) -> bool:
    """Check if a segment is too short based on word and character count"""
    words = len(text.strip().split())
    chars = len(text.strip())
    return words < min_words or chars < min_chars


def should_merge(
    current_text: str,
    next_text: str,
    pause: int,
    model: SentenceTransformer,
    threshold: float,
    pause_limit: int,
) -> bool:
    """Determine whether two segments should be merged based on similarity and pause.

    Args:
        current_text: Text from the current segment.
        next_text: Text from the next segment.
        pause: Pause between segments in seconds.
        model: SentenceTransformer model used to compute similarity.
        threshold: Cosine similarity threshold to trigger merging.
        pause_limit: Maximum allowed pause to consider merging.

    Returns:
        True if the segments should be merged.
    """
    if not current_text.strip() or not next_text.strip():
        return False

    try:
        embeddings = model.encode([current_text, next_text], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        logger.warning(f"Error computing similarity: {e}")
        similarity = 0.0

    return similarity > threshold and pause <= pause_limit


def merge_segments_if_short(
    segments: List[Dict], current: Dict, max_words: int, min_words: int, min_chars: int
) -> bool:
    """Try to merge a too-short current segment with the last processed one.

    Args:
        segments: List of already merged segments.
        current: The current short segment to merge.
        max_words: Maximum words allowed in a merged segment.
        min_words: Minimum words required for standalone segments.
        min_chars: Minimum characters required for standalone segments.

    Returns:
        True if the merge was performed, else False.
    """
    if is_segment_too_short(current["text"], min_words, min_chars) and segments:
        prev = segments[-1]
        combined_length = len(prev["text"].split()) + len(current["text"].split())
        if combined_length <= max_words:
            prev["text"] += " " + current["text"]
            prev["end_time"] = current["end_time"]
            logger.debug(f"Merged short segment with previous: '{current['text'][:30]}...'")
            return True
    return False


def segment_coherently(
    segments: List[Dict],
    similarity_threshold: float = 0.7,
    pause_threshold: int = 8,
    model: Optional[SentenceTransformer] = None,
    max_words: int = 80,
    min_words: int = 3,
    min_chars: int = 15,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    preferred_device: str = "cpu",
) -> List[Dict]:
    """Merge segments based on semantic similarity and pause durations.

    Args:
        segments: Initial list of timestamped transcription segments.
        similarity_threshold: Cosine similarity threshold for merging.
        pause_threshold: Max allowed pause in seconds between segments to merge.
        model: Pre-loaded SentenceTransformer model, or None to load internally.
        max_words: Max number of words allowed per segment.
        min_words: Min words required for a segment to stand alone.
        min_chars: Min characters required for a segment to stand alone.
        embedding_model_name: Model name to use if loading internally.
        preferred_device: Device hint for model loading.

    Returns:
        List of merged transcription segments.
    """
    if not segments:
        return []

    if model is None:
        device = detect_device(preferred_device)
        model = SentenceTransformer(embedding_model_name, device=device)

    merged_segments = []
    current = segments[0].copy()

    for next_seg in segments[1:]:
        next_text = next_seg["text"]
        pause = parse_time_str(next_seg["start_time"]) - parse_time_str(current["end_time"])
        current_len = len(current["text"].split())
        next_len = len(next_text.split())

        merge_conditions = (
            is_segment_too_short(current["text"], min_words, min_chars)
            or is_segment_too_short(next_text, min_words, min_chars)
            or (
                should_merge(current["text"], next_text, pause, model, similarity_threshold, pause_threshold)
                and current_len < max_words
            )
        )

        if merge_conditions and current_len + next_len <= max_words:
            current["end_time"] = next_seg["end_time"]
            current["text"] += " " + next_text
        else:
            if not merge_segments_if_short(merged_segments, current, max_words, min_words, min_chars):
                merged_segments.append(current)
            current = next_seg.copy()

    if not merge_segments_if_short(merged_segments, current, max_words, min_words, min_chars):
        merged_segments.append(current)

    return merged_segments


def post_process_segments(
    segments: List[Dict], min_words: int = 3, min_chars: int = 15, max_words: int = 80
) -> List[Dict]:
    """Post-process segments to handle remaining short ones.

    Args:
        segments: List of segments after initial merging.
        min_words: Minimum words allowed in a standalone segment.
        min_chars: Minimum characters allowed.
        max_words: Max total words for merged segments.

    Returns:
        Final list of post-processed segments
    """
    if not segments:
        return segments

    processed = []
    i = 0

    while i < len(segments):
        current = segments[i].copy()
        text = current["text"].strip()

        if is_segment_too_short(text, min_words, min_chars):
            merged = False

            if merge_segments_if_short(processed, current, max_words, min_words, min_chars):
                merged = True
            elif i + 1 < len(segments):
                next_segment = segments[i + 1]
                combined_length = len(text.split()) + len(next_segment["text"].strip().split())
                if combined_length <= max_words:
                    current["text"] = text + " " + next_segment["text"].strip()
                    current["end_time"] = next_segment["end_time"]
                    processed.append(current)
                    logger.debug(f"Post-process: merged short segment '{text[:30]}...' with next")
                    i += 1  # Skip next
                    merged = True

            if not merged:
                processed.append(current)
                logger.warning(f"Could not merge short segment: '{text}'")
        else:
            processed.append(current)

        i += 1

    return processed
