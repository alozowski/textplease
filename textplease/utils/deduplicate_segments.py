import logging


logger = logging.getLogger(__name__)


def deduplicate_segments(segments: list[dict[str, str]], overlap_words: int = 5) -> list[dict[str, str]]:
    """Remove words repeated across consecutive segments from model chunk overlap."""
    if not segments:
        return []

    required_keys = {"text", "start_time", "end_time"}
    deduplicated: list[dict[str, str]] = []

    for current in segments:
        missing = required_keys - current.keys()
        if missing:
            raise KeyError(f"Segment missing required keys: {missing}")

        current_copy = current.copy()
        if deduplicated:
            prev_tail = deduplicated[-1]["text"].split()[-overlap_words:]
            curr_head = current["text"].split()

            max_overlap = 0
            for i in range(1, min(len(prev_tail), len(curr_head), overlap_words) + 1):
                if prev_tail[-i:] == curr_head[:i]:
                    max_overlap = i

            if max_overlap:
                current_copy["text"] = " ".join(curr_head[max_overlap:]).strip()
                if not current_copy["text"]:
                    logger.warning(f"Segment empty after deduplication: start={current.get('start_time', 'unknown')}")

        deduplicated.append(current_copy)

    return deduplicated
