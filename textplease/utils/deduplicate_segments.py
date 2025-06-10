import logging
from typing import Dict, List


logger = logging.getLogger(__name__)


def deduplicate_segments(segments: List[Dict[str, str]], overlap_words: int = 5) -> List[Dict[str, str]]:
    """Removes repeated words from consecutive segments due to model chunk overlap.

    Args:
        segments: List of dicts with keys like 'text', 'start_time', 'end_time'.
        overlap_words: Max number of overlapping words to search for at segment edges.

    Returns:
        A new list of segments with text deduplicated at boundaries.
    """
    if not segments:
        return []

    deduplicated = [segments[0].copy()]

    for current in segments[1:]:
        previous = deduplicated[-1]
        prev_tail = previous["text"].split()[-overlap_words:]
        curr_head = current["text"].split()

        # Detect maximum overlap
        max_overlap = 0
        for i in range(1, min(len(prev_tail), len(curr_head), overlap_words) + 1):
            if prev_tail[-i:] == curr_head[:i]:
                max_overlap = i

        current_copy = current.copy()
        if max_overlap:
            logger.debug(f"Overlap of {max_overlap} words detected between segments.")
            current_copy["text"] = " ".join(curr_head[max_overlap:]).strip()

            if not current_copy["text"]:
                logger.warning(f"Segment became empty after deduplication: start={current['start_time']}")

        deduplicated.append(current_copy)

    return deduplicated
