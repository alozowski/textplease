import gc
import logging

import torch
import psutil
from sentence_transformers import SentenceTransformer, util

from textplease.utils.time_utils import parse_time_str, format_time_precise
from textplease.utils.device_utils import detect_device


logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor RSS memory usage during processing."""

    def __init__(self, warn_threshold_gb: float = 2.0):
        """Initialise with a warning threshold in GB."""
        self.warn_threshold_gb = warn_threshold_gb
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """Return current RSS memory usage in GB."""
        return self.process.memory_info().rss / (1024**3)

    def check_memory(self, context: str = "") -> bool:
        """Warn and return True if memory exceeds threshold."""
        memory_gb = self.get_memory_usage()
        if memory_gb > self.warn_threshold_gb:
            logger.warning(f"High memory usage: {memory_gb:.2f}GB {context}")
            return True
        return False

    def force_cleanup(self) -> None:
        """Run garbage collection and clear GPU cache if available."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SimilarityComputer:
    """Compute semantic similarity between segment texts using pre-encoded cached embeddings."""

    def __init__(self, model: SentenceTransformer, batch_size: int = 32):
        """Initialise with a loaded SentenceTransformer and batch size."""
        self.model = model
        self.batch_size = batch_size
        self.embedding_cache: dict[str, torch.Tensor] = {}

    def precompute_embeddings(self, texts: list[str]) -> None:
        """Encode all unique texts in one batched call and store in the cache."""
        unique = list({t for t in texts if t.strip()})
        if not unique:
            return
        logger.info(f"Pre-encoding {len(unique)} unique segment texts (batch_size={self.batch_size})")
        try:
            embeddings = self.model.encode(unique, batch_size=self.batch_size, convert_to_tensor=True)
            if not isinstance(embeddings, torch.Tensor):
                raise TypeError("SentenceTransformer did not return tensor embeddings")
            self.embedding_cache.update(dict(zip(unique, embeddings)))
        except Exception as e:
            logger.warning(f"Batch pre-encoding failed: {e}. Similarity will fall back to 0.0.")

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Return cosine similarity for a cached pair of texts."""
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            emb1 = self.embedding_cache.get(text1)
            emb2 = self.embedding_cache.get(text2)
            if emb1 is None or emb2 is None:
                # Fallback for texts not in cache (e.g. merged segment text)
                texts_to_encode = [t for t, e in [(text1, emb1), (text2, emb2)] if e is None]
                encoded = self.model.encode(texts_to_encode, convert_to_tensor=True)
                if not isinstance(encoded, torch.Tensor):
                    raise TypeError("SentenceTransformer did not return tensor embeddings")
                for t, e in zip(texts_to_encode, encoded):
                    self.embedding_cache[t] = e
                emb1 = self.embedding_cache[text1]
                emb2 = self.embedding_cache[text2]
            return util.pytorch_cos_sim(emb1, emb2).item()
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return 0.0


def is_segment_too_short(text: str, min_words: int = 3, min_chars: int = 15) -> bool:
    """Return True if text is below word or character threshold."""
    words = text.strip().split()
    return len(words) < min_words or len(text.strip()) < min_chars


def _should_merge(
    current_text: str,
    next_text: str,
    pause_duration: float,
    similarity_computer: SimilarityComputer | None,
    similarity_threshold: float,
    max_pause: float,
) -> bool:
    """Return True if two segments are semantically similar within the pause limit."""
    if not current_text.strip() or not next_text.strip():
        return False
    if similarity_computer is None or pause_duration > max_pause:
        return False
    return similarity_computer.compute_similarity(current_text, next_text) > similarity_threshold


def merge_segments_if_short(
    processed: list[dict],
    current: dict,
    max_words: int,
    min_words: int,
    min_chars: int,
) -> bool:
    """Merge a too-short segment into the previous one if within word limit."""
    if is_segment_too_short(current["text"], min_words, min_chars) and processed:
        prev = processed[-1]
        if len(prev["text"].split()) + len(current["text"].split()) <= max_words:
            prev["text"] += " " + current["text"]
            prev["end_time"] = current["end_time"]
            return True
    return False


def split_long_segment(segment: dict, max_words_per_chunk: int) -> list[dict]:
    """Split a segment that exceeds max_words into equal-duration chunks."""
    words = segment["text"].split()
    total_words = len(words)
    if total_words <= max_words_per_chunk:
        return [segment]

    start_sec = parse_time_str(segment["start_time"])
    end_sec = parse_time_str(segment["end_time"])
    secs_per_word = (end_sec - start_sec) / total_words if total_words else 1.0

    result = []
    for i in range(0, total_words, max_words_per_chunk):
        chunk = words[i : i + max_words_per_chunk]
        chunk_start = start_sec + i * secs_per_word
        chunk_end = chunk_start + len(chunk) * secs_per_word
        result.append(
            {
                "start_time": format_time_precise(chunk_start),
                "end_time": format_time_precise(chunk_end),
                "text": " ".join(chunk),
            }
        )
    return result


def _merge_segments(
    segments: list[dict],
    similarity_computer: SimilarityComputer | None,
    similarity_threshold: float,
    pause_threshold: float,
    max_words: int,
    min_words: int,
    min_chars: int,
    memory_monitor: MemoryMonitor | None = None,
) -> list[dict]:
    """Merge a sequence of segments by semantic similarity and pause duration."""
    merged: list[dict] = []
    current = segments[0].copy()

    for i, nxt in enumerate(segments[1:], 1):
        if memory_monitor and i % 100 == 0:
            memory_monitor.check_memory(f"at segment {i}")

        nxt_text = nxt["text"]
        pause = parse_time_str(nxt["start_time"]) - parse_time_str(current["end_time"])
        cur_words = len(current["text"].split())
        nxt_words = len(nxt_text.split())

        do_merge = (
            (
                is_segment_too_short(current["text"], min_words, min_chars)
                or is_segment_too_short(nxt_text, min_words, min_chars)
            )
            and pause <= pause_threshold
            or (
                _should_merge(
                    current["text"],
                    nxt_text,
                    pause,
                    similarity_computer,
                    similarity_threshold,
                    pause_threshold,
                )
                and cur_words < max_words
            )
        )

        if do_merge and cur_words + nxt_words <= max_words:
            current["end_time"] = nxt["end_time"]
            current["text"] += " " + nxt_text
        else:
            if not merge_segments_if_short(merged, current, max_words, min_words, min_chars):
                merged.append(current)
            current = nxt.copy()

    if not merge_segments_if_short(merged, current, max_words, min_words, min_chars):
        merged.append(current)

    return merged


def _merge_chunk_boundaries(
    prev_last: dict,
    curr_first: dict,
    similarity_computer: SimilarityComputer | None,
    similarity_threshold: float,
    pause_threshold: float,
    max_words: int,
) -> bool:
    """Return True if the boundary segments from adjacent chunks should merge."""
    pause = parse_time_str(curr_first["start_time"]) - parse_time_str(prev_last["end_time"])
    if not _should_merge(
        prev_last["text"],
        curr_first["text"],
        pause,
        similarity_computer,
        similarity_threshold,
        pause_threshold,
    ):
        return False
    return len(prev_last["text"].split()) + len(curr_first["text"].split()) <= max_words


def _process_segments_in_chunks(
    segments: list[dict],
    similarity_computer: SimilarityComputer | None,
    similarity_threshold: float,
    pause_threshold: float,
    max_words: int,
    min_words: int,
    min_chars: int,
    chunk_size: int,
    memory_monitor: MemoryMonitor,
) -> list[dict]:
    """Process segments in chunks to bound peak memory usage."""
    all_merged: list[dict] = []

    for chunk_start in range(0, len(segments), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(segments))
        chunk = segments[chunk_start:chunk_end]
        chunk_num = chunk_start // chunk_size + 1
        logger.info(f"Processing chunk {chunk_num}: segments {chunk_start}–{chunk_end}")

        merged = _merge_segments(
            chunk,
            similarity_computer,
            similarity_threshold,
            pause_threshold,
            max_words,
            min_words,
            min_chars,
            memory_monitor=memory_monitor,
        )

        if all_merged and merged:
            if _merge_chunk_boundaries(
                all_merged[-1],
                merged[0],
                similarity_computer,
                similarity_threshold,
                pause_threshold,
                max_words,
            ):
                all_merged[-1]["text"] += " " + merged[0]["text"]
                all_merged[-1]["end_time"] = merged[0]["end_time"]
                merged = merged[1:]

        all_merged.extend(merged)
        gc.collect()  # cheap; GPU cache cleared once after the full loop
        logger.info(f"Chunk {chunk_num} done. Memory: {memory_monitor.get_memory_usage():.2f}GB")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_merged


def _handle_short_segment(
    current: dict,
    processed: list[dict],
    segments: list[dict],
    index: int,
    max_words: int,
    min_words: int,
    min_chars: int,
) -> int:
    """Absorb a too-short segment into a neighbour; return the extra index increment."""
    text = current["text"].strip()

    if merge_segments_if_short(processed, current, max_words, min_words, min_chars):
        return 0

    if index + 1 < len(segments):
        nxt = segments[index + 1]
        combined = text + " " + nxt["text"].strip()
        cw = len(combined.split())
        if cw >= min_words and len(combined) >= min_chars and cw <= max_words:
            current["text"] = combined
            current["end_time"] = nxt["end_time"]
            processed.append(current)
            return 1

    logger.warning(f"Forcing retention of short segment: '{text[:30]}...'")
    processed.append(current)
    return 0


def post_process_segments(
    segments: list[dict],
    min_words: int = 3,
    min_chars: int = 15,
    max_words: int = 80,
) -> list[dict]:
    """Enforce min/max length constraints on all segments."""
    if not segments:
        return []

    logger.info(f"Post-processing {len(segments)} segments...")
    memory_monitor = MemoryMonitor()
    processed: list[dict] = []
    i = 0

    while i < len(segments):
        if i % 500 == 0:
            memory_monitor.check_memory(f"post-processing segment {i}")

        current = segments[i].copy()
        text = current["text"].strip()
        word_count = len(text.split())
        char_count = len(text)

        if word_count < min_words or char_count < min_chars:
            i += _handle_short_segment(current, processed, segments, i, max_words, min_words, min_chars)
        elif word_count > max_words:
            processed.extend(split_long_segment(current, max_words))
        else:
            processed.append(current)

        i += 1

    memory_monitor.force_cleanup()
    logger.info(f"Post-processing complete. Memory: {memory_monitor.get_memory_usage():.2f}GB")
    return processed


def segment_transcript(
    segments: list[dict],
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
) -> list[dict]:
    """Merge transcript segments by semantic similarity."""
    if not segments:
        return []

    memory_monitor = MemoryMonitor()
    logger.info(f"Starting segmentation: {len(segments)} segments. Memory: {memory_monitor.get_memory_usage():.2f}GB")

    similarity_computer = None
    if len(segments) > 1 and similarity_threshold < 1.0:
        if model is None:
            device = detect_device(preferred_device)
            model = SentenceTransformer(embedding_model_name, device=device)
        similarity_computer = SimilarityComputer(model, batch_size)
        similarity_computer.precompute_embeddings([seg["text"] for seg in segments])

    effective_chunk_size = chunk_size if chunk_size > 0 else len(segments) + 1

    if len(segments) > effective_chunk_size:
        logger.info(f"Processing in chunks of {effective_chunk_size}")
        result = _process_segments_in_chunks(
            segments,
            similarity_computer,
            similarity_threshold,
            pause_threshold,
            max_words,
            min_words,
            min_chars,
            effective_chunk_size,
            memory_monitor,
        )
    else:
        result = _merge_segments(
            segments,
            similarity_computer,
            similarity_threshold,
            pause_threshold,
            max_words,
            min_words,
            min_chars,
            memory_monitor=memory_monitor,
        )

    memory_monitor.force_cleanup()
    logger.info(f"Segmentation complete. Memory: {memory_monitor.get_memory_usage():.2f}GB")
    return result
