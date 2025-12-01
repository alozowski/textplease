import gc
import logging
from typing import Any

import numpy as np
import torch
import psutil  # type: ignore
from sentence_transformers import SentenceTransformer, util

from textplease.utils.time_utils import parse_time_str, format_time_precise
from textplease.utils.device_utils import detect_device


logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during processing."""

    def __init__(self, warn_threshold_gb: float = 2.0):
        """Initialize memory monitor with warning threshold."""
        self.warn_threshold_bytes = warn_threshold_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return self.process.memory_info().rss / (1024 * 1024 * 1024)

    def check_memory(self, context: str = ""):
        """Check memory usage and warn if high."""
        memory_gb = self.get_memory_usage()
        if memory_gb * 1024 * 1024 * 1024 > self.warn_threshold_bytes:
            logger.warning(f"High memory usage: {memory_gb:.2f}GB {context}")
            return True
        return False

    def force_cleanup(self):
        """Force garbage collection and clear GPU cache if available."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BatchSimilarityComputer:
    """Efficiently compute similarities in batches."""

    def __init__(self, model: SentenceTransformer, batch_size: int = 32):
        """Initialize batch similarity computer."""
        self.model = model
        self.batch_size = batch_size
        self.embedding_cache: dict[str, Any] = {}
        self.memory_monitor = MemoryMonitor()

    def _get_embedding_cached(self, text: str) -> Any:
        """Get cached embedding for text with automatic cache size management."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        embedding = self.model.encode([text], convert_to_tensor=True)[0]

        # Cache management - remove oldest entries if cache gets too large
        if len(self.embedding_cache) >= 500:
            # Remove 100 oldest entries (FIFO)
            keys_to_remove = list(self.embedding_cache.keys())[:100]
            for key in keys_to_remove:
                self.embedding_cache.pop(key, None)

        self.embedding_cache[text] = embedding
        return embedding

    def compute_similarity_batch(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """Compute similarities for multiple text pairs efficiently."""
        if not text_pairs:
            return []

        similarities = []

        # Process in batches to manage memory
        for i in range(0, len(text_pairs), self.batch_size):
            batch_pairs = text_pairs[i : i + self.batch_size]

            # Collect unique texts to avoid duplicate encodings
            unique_texts = set()
            for text1, text2 in batch_pairs:
                unique_texts.add(text1)
                unique_texts.add(text2)

            # Get embeddings for unique texts
            unique_texts_list = list(unique_texts)
            try:
                embeddings = self.model.encode(unique_texts_list, convert_to_tensor=True)
                text_to_embedding = dict(zip(unique_texts_list, embeddings))

                # Compute similarities for this batch
                for text1, text2 in batch_pairs:
                    if not text1.strip() or not text2.strip():
                        similarities.append(0.0)
                        continue

                    emb1 = text_to_embedding[text1]
                    emb2 = text_to_embedding[text2]
                    similarity = util.pytorch_cos_sim(emb1, emb2).item()
                    similarities.append(similarity)

            except Exception as e:
                logger.warning(f"Error computing similarity batch: {e}")
                similarities.extend([0.0] * len(batch_pairs))

            # Check memory after each batch
            if self.memory_monitor.check_memory(f"after batch {i // self.batch_size + 1}"):
                self.memory_monitor.force_cleanup()

        return similarities

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts with caching."""
        if not text1.strip() or not text2.strip():
            return 0.0

        try:
            # Use cached embeddings if available
            emb1 = self._get_embedding_cached(text1)
            emb2 = self._get_embedding_cached(text2)
            return util.pytorch_cos_sim(emb1, emb2).item()
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return 0.0


def is_segment_too_short(text: str, min_words: int = 3, min_chars: int = 15) -> bool:
    """Check if a segment is too short based on word and character count."""
    word_count = len(text.strip().split())
    char_count = len(text.strip())
    return word_count < min_words or char_count < min_chars


def should_merge_optimized(
    current_segment_text: str,
    next_segment_text: str,
    pause_duration: float,
    similarity_computer: BatchSimilarityComputer,
    similarity_threshold: float,
    max_pause_for_merge: float,
) -> bool:
    """Optimized version using batch similarity computer."""
    if not current_segment_text.strip() or not next_segment_text.strip():
        return False

    # Quick pause check first (cheaper than similarity)
    if pause_duration > max_pause_for_merge:
        return False

    similarity = similarity_computer.compute_similarity(current_segment_text, next_segment_text)
    return similarity > similarity_threshold


def merge_segments_if_short(
    processed_segments: list[dict],
    current_segment: dict,
    max_words_per_segment: int,
    min_words_threshold: int,
    min_chars_threshold: int,
) -> bool:
    """Try to merge a too-short current segment with the last processed one."""
    if is_segment_too_short(current_segment["text"], min_words_threshold, min_chars_threshold) and processed_segments:
        previous_segment = processed_segments[-1]
        combined_word_count = len(previous_segment["text"].split()) + len(current_segment["text"].split())
        if combined_word_count <= max_words_per_segment:
            previous_segment["text"] += " " + current_segment["text"]
            previous_segment["end_time"] = current_segment["end_time"]
            logger.debug(f"Merged short segment with previous: '{current_segment['text'][:30]}...'")
            return True
    return False


def split_long_segment(segment: dict, max_words_per_chunk: int) -> list[dict]:
    """Split overly long segments into multiple chunks of max_words_per_chunk."""
    words = segment["text"].split()
    total_words = len(words)
    if total_words <= max_words_per_chunk:
        return [segment]

    split_segments = []
    start_time_seconds = parse_time_str(segment["start_time"])
    end_time_seconds = parse_time_str(segment["end_time"])
    total_duration_seconds = end_time_seconds - start_time_seconds
    seconds_per_word = total_duration_seconds / total_words if total_words else 1

    for word_index in range(0, total_words, max_words_per_chunk):
        chunk_words = words[word_index : word_index + max_words_per_chunk]
        chunk_start_seconds = start_time_seconds + word_index * seconds_per_word
        chunk_end_seconds = chunk_start_seconds + len(chunk_words) * seconds_per_word
        split_segments.append({
            "start_time": format_time_precise(chunk_start_seconds),
            "end_time": format_time_precise(chunk_end_seconds),
            "text": " ".join(chunk_words),
        })

    return split_segments


def _process_single_chunk(
    chunk_segments: list[dict],
    similarity_computer: BatchSimilarityComputer,
    similarity_threshold: float,
    pause_threshold: float,
    max_words: int,
    min_words: int,
    min_chars: int,
) -> list[dict]:
    """Process a single chunk of segments.

    Args:
        chunk_segments: List of segments in this chunk
        similarity_computer: Similarity computation utility
        similarity_threshold: Threshold for semantic similarity
        pause_threshold: Maximum pause duration for merging
        max_words: Maximum words per segment
        min_words: Minimum words per segment
        min_chars: Minimum characters per segment

    Returns:
        List of processed segments for this chunk

    """
    merged_segments: list[dict[str, str]] = []
    current_segment = chunk_segments[0].copy()

    for next_segment in chunk_segments[1:]:
        next_segment_text = next_segment["text"]
        pause_duration = parse_time_str(next_segment["start_time"]) - parse_time_str(current_segment["end_time"])
        current_word_count = len(current_segment["text"].split())
        next_word_count = len(next_segment_text.split())

        should_merge_segments = (
            (
                is_segment_too_short(current_segment["text"], min_words, min_chars)
                or is_segment_too_short(next_segment_text, min_words, min_chars)
            ) and pause_duration <= pause_threshold
            or (
                should_merge_optimized(
                    current_segment["text"],
                    next_segment_text,
                    pause_duration,
                    similarity_computer,
                    similarity_threshold,
                    pause_threshold,
                )
                and current_word_count < max_words
            )
        )

        if should_merge_segments and current_word_count + next_word_count <= max_words:
            current_segment["end_time"] = next_segment["end_time"]
            current_segment["text"] += " " + next_segment_text
        else:
            if not merge_segments_if_short(merged_segments, current_segment, max_words, min_words, min_chars):
                merged_segments.append(current_segment)
            current_segment = next_segment.copy()

    if not merge_segments_if_short(merged_segments, current_segment, max_words, min_words, min_chars):
        merged_segments.append(current_segment)

    return merged_segments


def _merge_chunk_boundaries(
    prev_chunk_last: dict,
    curr_chunk_first: dict,
    similarity_computer: BatchSimilarityComputer,
    similarity_threshold: float,
    pause_threshold: float,
    max_words: int,
) -> bool:
    """Check if segments from adjacent chunks should be merged.

    Args:
        prev_chunk_last: Last segment from previous chunk
        curr_chunk_first: First segment from current chunk
        similarity_computer: Similarity computation utility
        similarity_threshold: Threshold for semantic similarity
        pause_threshold: Maximum pause duration for merging
        max_words: Maximum words per segment

    Returns:
        True if segments should be merged

    """
    pause_duration = parse_time_str(curr_chunk_first["start_time"]) - parse_time_str(prev_chunk_last["end_time"])

    if should_merge_optimized(
        prev_chunk_last["text"],
        curr_chunk_first["text"],
        pause_duration,
        similarity_computer,
        similarity_threshold,
        pause_threshold,
    ):
        combined_words = len(prev_chunk_last["text"].split()) + len(curr_chunk_first["text"].split())
        return combined_words <= max_words

    return False


def _process_segments_in_chunks(
    segments: list[dict[str, str]],
    similarity_computer: BatchSimilarityComputer,
    similarity_threshold: float,
    pause_threshold: float,
    max_words: int,
    min_words: int,
    min_chars: int,
    chunk_size: int,
    memory_monitor: MemoryMonitor,
) -> list[dict[str, str]]:
    """Process segments in chunks to manage memory usage."""
    all_merged_segments: list[dict[str, str]] = []

    for chunk_start in range(0, len(segments), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(segments))
        chunk_segments = segments[chunk_start:chunk_end]

        logger.info(f"Processing chunk {chunk_start // chunk_size + 1}: segments {chunk_start}-{chunk_end}")

        # Process this chunk
        merged_segments = _process_single_chunk(
            chunk_segments,
            similarity_computer,
            similarity_threshold,
            pause_threshold,
            max_words,
            min_words,
            min_chars,
        )

        # Try to merge the last segment of previous chunk with first of current chunk
        if all_merged_segments and merged_segments:
            last_prev = all_merged_segments[-1]
            first_curr = merged_segments[0]

            if _merge_chunk_boundaries(
                last_prev,
                first_curr,
                similarity_computer,
                similarity_threshold,
                pause_threshold,
                max_words,
            ):
                last_prev["text"] += " " + first_curr["text"]
                last_prev["end_time"] = first_curr["end_time"]
                merged_segments = merged_segments[1:]  # Remove merged segment

        all_merged_segments.extend(merged_segments)

        # Force cleanup after each chunk
        memory_monitor.force_cleanup()
        logger.info(f"Chunk processed. Memory: {memory_monitor.get_memory_usage():.2f}GB")

    return all_merged_segments


def _handle_short_segment(
    current: dict,
    processed: list[dict],
    segments: list[dict],
    index: int,
    max_words: int,
    min_words: int,
    min_chars: int,
) -> tuple[bool, int]:
    """Handle a short segment by attempting to merge with adjacent segments.

    Args:
        current: Current segment to process
        processed: Already processed segments
        segments: All original segments
        index: Current segment index
        max_words: Maximum words per segment
        min_words: Minimum words per segment
        min_chars: Minimum characters per segment

    Returns:
        Tuple of (was_merged, index_increment)

    """
    text = current["text"].strip()

    # Try merging with previous segment
    if merge_segments_if_short(processed, current, max_words, min_words, min_chars):
        return True, 0

    # Try merging with next segment
    if index + 1 < len(segments):
        next_seg = segments[index + 1]
        combined_text = text + " " + next_seg["text"].strip()
        combined_word_count = len(combined_text.split())
        combined_char_count = len(combined_text)

        if (
            combined_word_count >= min_words and
            combined_char_count >= min_chars and
            combined_word_count <= max_words
        ):
            current["text"] = combined_text
            current["end_time"] = next_seg["end_time"]
            processed.append(current)
            logger.debug(f"Merged short segment '{text[:30]}...' with next")
            return True, 1

    logger.warning(f"Forcing retention of short segment: '{text[:30]}...'")
    processed.append(current)
    return False, 0


def _handle_long_segment(current: dict, max_words: int) -> list[dict]:
    """Handle an overly long segment by splitting it.

    Args:
        current: Segment to process
        max_words: Maximum words per segment

    Returns:
        List of split segments

    """
    return split_long_segment(current, max_words)


def _try_merge_with_next(
    current: dict,
    next_seg: dict,
    max_words: int,
    min_words: int,
    min_chars: int,
) -> bool:
    """Attempt to merge current segment with next segment.

    Args:
        current: Current segment
        next_seg: Next segment
        max_words: Maximum words per segment
        min_words: Minimum words per segment
        min_chars: Minimum characters per segment

    Returns:
        True if segments were merged

    """
    combined_text = current["text"].strip() + " " + next_seg["text"].strip()
    combined_word_count = len(combined_text.split())
    combined_char_count = len(combined_text)

    if (
        combined_word_count >= min_words and
        combined_char_count >= min_chars and
        combined_word_count <= max_words
    ):
        current["text"] = combined_text
        current["end_time"] = next_seg["end_time"]
        return True

    return False


def post_process_segments(
    segments: list[dict[str, str]],
    min_words: int = 3,
    min_chars: int = 15,
    max_words: int = 80
) -> list[dict[str, str]]:
    """Post-process segments to enforce min/max length constraints and reduce memory usage."""
    if not segments:
        return []

    logger.info(f"Post-processing {len(segments)} segments...")
    memory_monitor = MemoryMonitor()
    processed: list[dict[str, str]] = []
    i = 0

    while i < len(segments):
        if i % 500 == 0:
            memory_monitor.check_memory(f"post-processing segment {i}")

        current = segments[i].copy()
        text = current["text"].strip()
        word_count = len(text.split())
        char_count = len(text)

        # Handle short segments
        if word_count < min_words or char_count < min_chars:
            merged, index_increment = _handle_short_segment(
                current, processed, segments, i, max_words, min_words, min_chars
            )
            i += index_increment

        # Handle overly long segments
        elif word_count > max_words:
            split_segments = _handle_long_segment(current, max_words)
            processed.extend(split_segments)

        # Normal case
        else:
            processed.append(current)

        i += 1

    memory_monitor.force_cleanup()
    logger.info(f"Post-processing complete. Memory: {memory_monitor.get_memory_usage():.2f}GB")

    return processed


def _initialize_segmentation(
    segments: list[dict],
    model: SentenceTransformer | None,
    embedding_model_name: str,
    preferred_device: str,
    batch_size: int,
    chunk_size: int,
) -> tuple[BatchSimilarityComputer, MemoryMonitor, SentenceTransformer]:
    """Initialize segmentation components.

    Args:
        segments: Initial transcription segments
        model: Pre-loaded model or None
        embedding_model_name: Model name for loading
        preferred_device: Target device
        batch_size: Batch size for similarity computation
        chunk_size: Chunk size for processing

    Returns:
        Tuple of (similarity_computer, memory_monitor, model)

    """
    memory_monitor = MemoryMonitor()
    logger.info(
        f"Starting segmentation with {len(segments)} segments. Memory: {memory_monitor.get_memory_usage():.2f}GB"
    )

    if model is None:
        device = detect_device(preferred_device)
        model = SentenceTransformer(embedding_model_name, device=device)

    similarity_computer = BatchSimilarityComputer(model, batch_size)
    return similarity_computer, memory_monitor, model


def _merge_sequential_segments(
    segments: list[dict[str, str]],
    similarity_computer: BatchSimilarityComputer,
    similarity_threshold: float,
    pause_threshold: float,
    max_words: int,
    min_words: int,
    min_chars: int,
    memory_monitor: MemoryMonitor,
) -> list[dict[str, str]]:
    """Merge sequential segments based on semantic similarity.

    Args:
        segments: Initial segments to merge
        similarity_computer: Similarity computation utility
        similarity_threshold: Threshold for semantic similarity
        pause_threshold: Maximum pause duration for merging
        max_words: Maximum words per segment
        min_words: Minimum words per segment
        min_chars: Minimum characters per segment
        memory_monitor: Memory monitoring utility

    Returns:
        List of merged segments

    """
    merged_segments: list[dict[str, str]] = []
    current_segment = segments[0].copy()

    for i, next_segment in enumerate(segments[1:], 1):
        if i % 100 == 0:  # Check memory every 100 segments
            memory_monitor.check_memory(f"processing segment {i}")

        next_segment_text = next_segment["text"]
        pause_duration = parse_time_str(next_segment["start_time"]) - parse_time_str(current_segment["end_time"])
        current_word_count = len(current_segment["text"].split())
        next_word_count = len(next_segment_text.split())

        should_merge_segments = (
            (
                is_segment_too_short(current_segment["text"], min_words, min_chars)
                or is_segment_too_short(next_segment_text, min_words, min_chars)
            ) and pause_duration <= pause_threshold
            or (
                should_merge_optimized(
                    current_segment["text"],
                    next_segment_text,
                    pause_duration,
                    similarity_computer,
                    similarity_threshold,
                    pause_threshold,
                )
                and current_word_count < max_words
            )
        )

        if should_merge_segments and current_word_count + next_word_count <= max_words:
            current_segment["end_time"] = next_segment["end_time"]
            current_segment["text"] += " " + next_segment_text
        else:
            if not merge_segments_if_short(merged_segments, current_segment, max_words, min_words, min_chars):
                merged_segments.append(current_segment)
            current_segment = next_segment.copy()

    if not merge_segments_if_short(merged_segments, current_segment, max_words, min_words, min_chars):
        merged_segments.append(current_segment)

    return merged_segments


def segment_transcript(
    segments: list[dict[str, str]],
    similarity_threshold: float = 0.7,
    pause_threshold: float = 2.0,
    model: SentenceTransformer | None = None,
    max_words: int = 80,
    min_words: int = 3,
    min_chars: int = 15,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    preferred_device: str = "cpu",
    batch_size: int = 32,
    chunk_size: int = 1000,
) -> list[dict[str, str]]:
    """Public API for segmenting transcript with semantic similarity.

    Args:
        segments: Initial list of timestamped transcription segments.
        similarity_threshold: Cosine similarity threshold for merging (0.0-1.0).
        pause_threshold: Max allowed pause in seconds between segments to merge.
        model: Pre-loaded SentenceTransformer model, or None to load internally.
        max_words: Max number of words allowed per segment.
        min_words: Min words required for a segment to stand alone.
        min_chars: Min characters required for a segment to stand alone.
        embedding_model_name: Model name to use if loading internally.
        preferred_device: Device hint for model loading ('cpu', 'cuda', 'mps').
        batch_size: Batch size for similarity computations.
        chunk_size: Process segments in chunks to manage memory (0 = no chunking).

    Returns:
        List of merged transcription segments with improved coherence.

    """
    if not segments:
        return []

    # Initialize segmentation components
    similarity_computer, memory_monitor, model = _initialize_segmentation(
        segments, model, embedding_model_name, preferred_device, batch_size, chunk_size
    )

    # For very large inputs, process in chunks
    if len(segments) > chunk_size:
        logger.info(f"Processing {len(segments)} segments in chunks of {chunk_size}")
        return _process_segments_in_chunks(
            segments,
            similarity_computer,
            similarity_threshold,
            pause_threshold,
            max_words,
            min_words,
            min_chars,
            chunk_size,
            memory_monitor,
        )

    # Regular processing for smaller inputs
    merged_segments = _merge_sequential_segments(
        segments,
        similarity_computer,
        similarity_threshold,
        pause_threshold,
        max_words,
        min_words,
        min_chars,
        memory_monitor,
    )

    # Final cleanup
    memory_monitor.force_cleanup()
    logger.info(f"Segmentation complete. Final memory: {memory_monitor.get_memory_usage():.2f}GB")

    return merged_segments
