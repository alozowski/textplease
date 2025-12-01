import logging


logger = logging.getLogger(__name__)


def deduplicate_segments(segments: list[dict[str, str]], overlap_words: int = 5) -> list[dict[str, str]]:
    """Remove repeated words from consecutive segments due to model chunk overlap.

    Args:
        segments: List of dicts with keys like 'text', 'start_time', 'end_time'.
        overlap_words: Max number of overlapping words to search for at segment edges.

    Returns:
        A new list of segments with text deduplicated at boundaries.

    Raises:
        ValueError: If segments is not a list or overlap_words is invalid
        KeyError: If required keys are missing from segment dictionaries

    """
    if not isinstance(segments, list):
        raise ValueError(f"segments must be a list, got {type(segments)}")

    if not isinstance(overlap_words, int) or overlap_words < 1:
        raise ValueError(f"overlap_words must be a positive integer, got {overlap_words}")

    if not segments:
        return []

    try:
        # Validate first segment has required keys
        required_keys = {"text", "start_time", "end_time"}
        if not all(key in segments[0] for key in required_keys):
            missing = required_keys - set(segments[0].keys())
            raise KeyError(f"Segment missing required keys: {missing}")

        deduplicated = [segments[0].copy()]

        for current in segments[1:]:
            # Validate current segment has required keys
            if not all(key in current for key in required_keys):
                missing = required_keys - set(current.keys())
                logger.error(f"Segment missing required keys: {missing}")
                raise KeyError(f"Segment missing required keys: {missing}")

            previous = deduplicated[-1]
            prev_text = previous.get("text", "")
            curr_text = current.get("text", "")

            if not isinstance(prev_text, str) or not isinstance(curr_text, str):
                logger.warning("Non-string text found in segment, skipping deduplication")
                deduplicated.append(current.copy())
                continue

            prev_tail = prev_text.split()[-overlap_words:]
            curr_head = curr_text.split()

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
                    logger.warning(f"Segment became empty after deduplication: start={current.get('start_time', 'unknown')}")

            deduplicated.append(current_copy)

        return deduplicated

    except (KeyError, ValueError) as e:
        logger.error(f"Deduplication failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during deduplication: {e}")
        raise
